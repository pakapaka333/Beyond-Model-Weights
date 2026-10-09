"""Run lm-evaluation-harness tasks (GSM8K, IFEval, MMLU) against a DLM, driven by a YAML config.

    bash scripts/5_eval.sh configs/5_eval/lm_eval/llada_gsm8k.yaml --adapter <dir>

`model.backend` picks the wrapper (`llada` or `dream`); the rest of the `model`
block is that backend's own sampler. A LoRA adapter is applied at inference time.
"""

import argparse
import datetime
import json
import shutil
import time
from pathlib import Path

import yaml

import lm_eval
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_output_dir(cfg: dict, config_path: Path, override: str | None) -> Path:
    if override:
        output_dir = Path(override)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/lm_eval") / f"{config_path.stem}_{timestamp}"
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter", default=None, help="LoRA adapter; overrides model.adapter")
    parser.add_argument("--limit", type=int, default=None, help="for a smoke test")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    apply_chat_template = bool(cfg.get("apply_chat_template", False))
    model_kwargs = dict(cfg["model"])
    backend = str(model_kwargs.pop("backend", "llada")).lower()
    if backend not in ("llada", "dream"):
        raise SystemExit(f"{args.config}: model.backend must be 'llada' or 'dream', got {backend!r}.")
    if apply_chat_template and model_kwargs.get("self_chat_template"):
        raise SystemExit(f"{args.config}: set only one of apply_chat_template and model.self_chat_template.")
    if args.adapter:
        model_kwargs["adapter"] = args.adapter
    model_kwargs.setdefault("add_special_tokens", not apply_chat_template)

    tasks = cfg.get("tasks") or cfg.get("task")
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    missing = [t for t in tasks if t not in TaskManager().all_tasks]
    if missing:
        raise SystemExit(f"unknown task(s) {missing}; mmlu and ifeval are fetched by "
                         "scripts/3_fetch_lm_eval_tasks.sh")
    eval_kwargs = dict(
        tasks=tasks,
        limit=args.limit if args.limit is not None else cfg.get("limit"),
        apply_chat_template=apply_chat_template,
        log_samples=True,
    )
    if cfg.get("num_fewshot") is not None:
        eval_kwargs["num_fewshot"] = int(cfg["num_fewshot"])

    output_dir = resolve_output_dir(cfg, args.config, args.output_dir)
    if not (output_dir / args.config.name).exists():
        shutil.copy2(args.config, output_dir / args.config.name)
    print(f"[run_lm_eval] backend={backend} model={model_kwargs['model_path']} "
          f"adapter={model_kwargs.get('adapter')} tasks={','.join(tasks)} output={output_dir}")

    # Both eval directories ship a top-level package named `model`, so only the
    # wrapper of the backend in use may be imported.
    if backend == "dream":
        from dream_lm import DreamLM
        lm = DreamLM(**model_kwargs)
    else:
        from llada_lm import LLaDALM
        lm = LLaDALM(**model_kwargs)
    start_time = time.monotonic()
    results = lm_eval.simple_evaluate(model=lm, **eval_kwargs)
    seconds = time.monotonic() - start_time
    if results is None:
        return

    samples = results.pop("samples", {})
    results["generation_seconds"] = seconds
    results["total_nfe"] = lm.total_nfe
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    for task_name, task_samples in samples.items():
        with open(output_dir / f"samples_{task_name}.jsonl", "w") as f:
            for sample in task_samples:
                f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
    print(make_table(results))
    print(f"[run_lm_eval] saved results to {output_dir}")


if __name__ == "__main__":
    main()
