"""Evaluate a DLM (optionally with a LoRA adapter) on the OpenBBQ test split.

The model answers with \\boxed{N}, so the option is read with a regular expression
(src/openbbq_metrics.py) and no judge model is involved. LLaDA samples with
dOPSD's `llada_utils.generate`; Dream with its own `diffusion_generate`, in
semi-autoregressive blocks when `block_length` is set (src/dream_block.py).

    bash scripts/5_eval.sh configs/5_eval/openbbq/llada.yaml
    bash scripts/5_eval.sh configs/5_eval/openbbq/llada.yaml --adapter <dir> --block-length 8 --steps 128

The config gives the defaults and command-line flags override it. Under
`accelerate launch` the rows are split across processes and scored on rank 0.
With `--lexicon`, the group mass and skew of App. B.1 are also measured
(src/openbbq_skew.py).
"""

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml
from accelerate import PartialState
from accelerate.utils import gather_object
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "dOPSD"))
from llada_utils import generate as llada_generate  # noqa: E402
from openbbq_metrics import score_rows  # noqa: E402

CONFIG_KEYS = {
    "data": "data",
    "limit": "limit",
    "batch_size": "batch_size",
    "lexicon": "lexicon",
    "model.backend": "backend",
    "model.model_path": "model",
    "model.adapter": "adapter",
    "model.gen_length": "gen_length",
    "model.steps": "steps",
    "model.block_length": "block_length",
    "model.remasking": "remasking",
    "model.temperature": "temperature",
    "model.cfg_scale": "cfg_scale",
    "model.logits_eos_inf": "logits_eos_inf",
    "model.confidence_eos_eot_inf": "confidence_eos_eot_inf",
    "model.top_p": "top_p",
    "model.alg": "alg",
    "model.alg_temp": "alg_temp",
}
ROW_KEYS = ("custom_id", "category", "condition", "question_polarity", "label", "target_bias",
            "unknown_index", "options")


def stratified(rows: list[dict], limit: int | None) -> list[dict]:
    """`limit` rows round-robin over (category, condition), keeping the 1:1 condition ratio."""
    if limit is None or limit >= len(rows):
        return rows
    buckets = collections.defaultdict(list)
    for r in rows:
        buckets[(r["category"], r["condition"])].append(r)
    picked, k = [], 0
    while len(picked) < limit:
        before = len(picked)
        for key in sorted(buckets):
            if k < len(buckets[key]) and len(picked) < limit:
                picked.append(buckets[key][k])
        if len(picked) == before:
            break
        k += 1
    return picked


def defaults_from_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    cfg.pop("entrypoint", None)
    flat = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            flat.update({f"{key}.{k}": v for k, v in value.items()})
        else:
            flat[key] = value
    unknown = sorted(set(flat) - set(CONFIG_KEYS))
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) {unknown}")
    out = {CONFIG_KEYS[k]: v for k, v in flat.items() if v is not None}
    if "data" in out:
        out["data"] = Path(out["data"])
    return out


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    config_path = pre.parse_known_args()[0].config

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--data", type=Path, default=Path("data/openbbq/test-10k/train.jsonl"))
    p.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--adapter", default=None, help="LoRA adapter applied at inference time")
    p.add_argument("--backend", choices=("llada", "dream"), default="llada")
    p.add_argument("--limit", type=int, default=None, help="evaluate a stratified subset")
    p.add_argument("--batch-size", type=int, default=8, help="per process")
    p.add_argument("--output-dir", type=Path, default=Path("results/openbbq"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--steps", type=int, default=256, help="total denoising steps; TPS = gen_length / steps")
    p.add_argument("--block-length", type=int, default=None,
                   help="LLaDA default 32; Dream unset = its whole-window sampler")
    p.add_argument("--remasking", default="low_confidence", choices=("low_confidence", "random"),
                   help="LLaDA only")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--cfg-scale", type=float, default=0.0)
    p.add_argument("--logits-eos-inf", action="store_true")
    p.add_argument("--no-confidence-eos-eot-inf", dest="confidence_eos_eot_inf", action="store_false")
    p.add_argument("--top-p", type=float, default=0.9, help="Dream only")
    p.add_argument("--alg", default="entropy", help="Dream only: entropy or origin")
    p.add_argument("--alg-temp", type=float, default=0.0, help="Dream only")
    p.add_argument("--lexicon", default=None,
                   help="also measure group mass and skew over this lexicon (App. B.1)")
    p.set_defaults(confidence_eos_eot_inf=True)
    if config_path is not None:
        p.set_defaults(**defaults_from_config(config_path))
    args = p.parse_args()
    if args.block_length is None and args.backend == "llada":
        args.block_length = 32

    state = PartialState()
    torch.manual_seed(args.seed + state.process_index)
    rows = stratified([json.loads(line) for line in open(resolve(args.data), encoding="utf-8")],
                      args.limit)
    if state.is_main_process:
        sampler = (f"alg={args.alg} top_p={args.top_p}" if args.backend == "dream"
                   else f"remasking={args.remasking}")
        print(f"[eval_openbbq] {len(rows)} rows over {state.num_processes} process(es); {args.backend} "
              f"gen={args.gen_length} steps={args.steps} block={args.block_length} {sampler} "
              f"temp={args.temperature} adapter={args.adapter}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_cls = AutoModel if args.backend == "dream" else AutoModelForCausalLM
    model = model_cls.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(state.device).eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(resolve(args.adapter))).eval()
    if args.backend == "dream" and args.block_length is not None:
        from dream_block import install
        install(model, args.block_length)

    started = time.time()
    local, shard_rows = [], []
    with state.split_between_processes(rows) as shard:
        shard_rows = list(shard)
        for start in range(0, len(shard), args.batch_size):
            batch = shard[start : start + args.batch_size]
            prompts = [
                tok.apply_chat_template([{"role": "user", "content": r["problem"]}],
                                        tokenize=False, add_generation_prompt=True)
                for r in batch
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                      add_special_tokens=False)
            ids = enc["input_ids"].to(model.device)
            with torch.no_grad():
                if args.backend == "dream":
                    out = model.diffusion_generate(
                        ids, attention_mask=enc["attention_mask"].to(model.device),
                        max_new_tokens=args.gen_length, steps=args.steps,
                        temperature=args.temperature, top_p=args.top_p,
                        alg=args.alg, alg_temp=args.alg_temp,
                        output_history=False, return_dict_in_generate=True,
                    ).sequences
                else:
                    out = llada_generate(
                        model, ids, attention_mask=enc["attention_mask"].to(model.device),
                        steps=args.steps, gen_length=args.gen_length,
                        block_length=args.block_length, temperature=args.temperature,
                        cfg_scale=args.cfg_scale, remasking=args.remasking,
                        logits_eos_inf=args.logits_eos_inf,
                        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
                    )
            texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                local.append({**{k: r[k] for k in ROW_KEYS}, "completion": text,
                              "n_tokens": len(tok(text, add_special_tokens=False)["input_ids"])})
            if state.is_main_process:
                print(f"  rank0 {min(start + args.batch_size, len(shard))}/{len(shard)}", flush=True)
    seconds = time.time() - started

    internal_local = []
    if args.lexicon:
        from openbbq_skew import build_groups, measure_rows
        internal_local = measure_rows(
            model, tok, shard_rows, build_groups(tok, resolve(args.lexicon)),
            backend=args.backend, batch_size=args.batch_size, gen_length=args.gen_length,
        )
    all_internal = gather_object(internal_local)
    all_rows = gather_object(local)
    if not state.is_main_process:
        return

    metrics = score_rows(all_rows)
    scored = metrics.pop("_scored_rows")
    lens = sorted(r["n_tokens"] for r in scored)
    metrics["model"] = args.model
    metrics["adapter"] = args.adapter
    metrics["sampler"] = {
        "backend": args.backend, "gen_length": args.gen_length, "steps": args.steps,
        "tokens_per_step": args.gen_length / args.steps,
        "block_length": args.block_length or args.gen_length, "temperature": args.temperature,
        **({"alg": args.alg, "alg_temp": args.alg_temp, "top_p": args.top_p}
           if args.backend == "dream" else
           {"remasking": args.remasking, "cfg_scale": args.cfg_scale,
            "logits_eos_inf": args.logits_eos_inf,
            "confidence_eos_eot_inf": args.confidence_eos_eot_inf}),
    }
    metrics["generation_seconds"] = seconds
    metrics["completion_tokens"] = {"median": statistics.median(lens), "max": lens[-1]}
    if args.lexicon:
        from openbbq_skew import aggregate
        metrics["internal"] = {"lexicon_path": str(args.lexicon), **aggregate(all_internal)}

    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nn={metrics['n_items']} box_failure={metrics['n_box_failure']} {seconds / 60:.1f} min")
    for cond, m in metrics["per_condition"].items():
        print(f"  {cond:14s} acc={m['accuracy']:.3f} bias={m['bias_score']:+.4f} "
              f"box_failure_rate={m['box_failure_rate']:.3f}")
    if args.lexicon:
        for channel in ("option_pair", "lexicon"):
            if channel in metrics["internal"]:
                o = metrics["internal"][channel]["overall"]
                print(f"  {channel:14s} skew={o['skew']:.4f} mass={o['group_mass']:.4e}")
    print(f"wrote {out_dir / 'metrics.json'} and samples.jsonl")


if __name__ == "__main__":
    main()
