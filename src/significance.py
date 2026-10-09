"""Paired cluster bootstrap of tuned models against the base model.

Every model answered the same items, so each comparison is paired: one resample
of the clusters is scored for both models, and the 95% percentile interval of the
difference d = tuned - base decides significance.

  OpenBBQ  the ambiguous and disambiguated variants of a BBQ template share the
           numeric suffix of `custom_id`, so (category, suffix) is the cluster
  IFEval   the prompt is the cluster (instruction-level loose accuracy)
  MMLU     the question, resampled within its subject
  GSM8K    the question

Bias scores are tested on |bias|, so a move toward zero is an improvement.

    uv run --no-sync python src/significance.py --spec configs/6_significance/example.yaml

A spec names the result directories of the base model and of each tuned model:

    base:
      openbbq: results/openbbq/llada/base/default
      gsm8k:   results/lm_eval/llada_gsm8k_base
    models:
      - name: SFT
        openbbq: results/openbbq/llada/sft/default
        gsm8k:   results/lm_eval/llada_gsm8k_sft
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openbbq_metrics import boxed_to_choice, extract_boxed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENBBQ_METRICS = [
    # (key, condition, statistic, higher_is_better)
    ("amb_acc", "ambiguous", "acc", True),
    ("amb_bias", "ambiguous", "bias", False),
    ("dis_acc", "disambiguated", "acc", True),
    ("dis_bias", "disambiguated", "bias", False),
]
UTILITY_TASKS = ["gsm8k", "ifeval", "mmlu"]
_N, _CORRECT, _NONUNK, _TARGET = range(4)


# Each loader returns (keys, counts, strata): one row of per-cluster sums per cluster.

def load_openbbq(d):
    per = collections.defaultdict(lambda: np.zeros((2, 4)))
    for line in open(Path(d) / "samples.jsonl", encoding="utf-8"):
        r = json.loads(line)
        choice = r["choice"] if "choice" in r else boxed_to_choice(
            extract_boxed(r["completion"]), r.get("options"))
        cond = 0 if r["condition"] == "ambiguous" else 1
        c = per[(r["category"], r["custom_id"].rsplit("-", 1)[1])][cond]
        c[_N] += 1
        c[_CORRECT] += choice == r["label"]
        if choice is not None and choice != r["unknown_index"]:
            c[_NONUNK] += 1
            c[_TARGET] += choice == r["target_bias"]
    keys = sorted(per)
    return keys, np.stack([per[k].reshape(-1) for k in keys]), [k[0] for k in keys]


def openbbq_stat(cond, which):
    off = 0 if cond == "ambiguous" else 4

    def f(s):
        acc = s[..., off + _CORRECT] / s[..., off + _N]
        nonunk = s[..., off + _NONUNK]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = np.where(nonunk > 0, 2 * s[..., off + _TARGET] / nonunk - 1, np.nan)
        if which == "acc":
            return acc
        return (1 - acc) * np.nan_to_num(raw) if cond == "ambiguous" else raw
    return f


def _lm_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_gsm8k(d):
    per = {}
    for r in _lm_rows(Path(d) / "samples_gsm8k.jsonl"):
        if r.get("filter") == "flexible-extract":
            per[str(r["doc_id"])] = [1.0, float(r["exact_match"])]
    keys = sorted(per, key=int)
    return keys, np.array([per[k] for k in keys]), ["gsm8k"] * len(keys)


def load_ifeval(d):
    per = {}
    for r in _lm_rows(Path(d) / "samples_ifeval.jsonl"):
        v = r["inst_level_loose_acc"]
        per[str(r["doc_id"])] = [float(len(v)), float(sum(bool(x) for x in v))]
    keys = sorted(per, key=int)
    return keys, np.array([per[k] for k in keys]), ["ifeval"] * len(keys)


def load_mmlu(d):
    per = {}
    for path in sorted(Path(d).glob("samples_mmlu_*.jsonl")):
        subject = re.sub(r"^samples_mmlu_", "", path.stem)
        for r in _lm_rows(path):
            per[(subject, str(r["doc_id"]))] = [1.0, float(r["acc"])]
    keys = sorted(per)
    return keys, np.array([per[k] for k in keys]), [k[0] for k in keys]


def mean_stat(s):
    return s[..., 1] / s[..., 0]


LOADERS = {"openbbq": load_openbbq, "gsm8k": load_gsm8k, "ifeval": load_ifeval, "mmlu": load_mmlu}


def align(base, model):
    bk, bc, bs = base
    ak, ac, _ = model
    pos = {k: i for i, k in enumerate(ak)}
    keep = [i for i, k in enumerate(bk) if k in pos]
    if len(keep) < len(bk) or len(keep) < len(ak):
        print(f"  ! {len(bk)} / {len(ak)} clusters, {len(keep)} shared", file=sys.stderr)
    return bc[keep], ac[[pos[bk[i]] for i in keep]], [bs[i] for i in keep]


def cluster_indices(strata, n_boot, rng):
    """(n_boot x clusters) resampled cluster indices, drawn within each stratum."""
    strata = np.asarray(strata)
    cols = []
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        cols.append(idx[rng.integers(0, len(idx), size=(n_boot, len(idx)))])
    return np.concatenate(cols, axis=1)


def resampled_sums(counts_list, indices, chunk=512):
    """Column sums of each counts array under every resample."""
    n_boot, n_clusters = indices.shape
    outs = [np.empty((n_boot, c.shape[1])) for c in counts_list]
    w = np.empty((chunk, n_clusters))
    for i in range(0, n_boot, chunk):
        rows = indices[i:i + chunk]
        for j, row in enumerate(rows):
            w[j] = np.bincount(row, minlength=n_clusters)
        for out, c in zip(outs, counts_list):
            out[i:i + len(rows)] = w[:len(rows)] @ c
    return outs


def paired_diff(stat, base_counts, model_counts, base_sums, model_sums, absolute):
    def value(c):
        v = stat(c)
        return np.abs(v) if absolute else v
    point = float(value(model_counts.sum(0)) - value(base_counts.sum(0)))
    return point, value(model_sums) - value(base_sums)


def summarize(point, boot, higher_is_better):
    boot = boot[~np.isnan(boot)]
    if len(boot) == 0:
        boot = np.array([np.nan])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    tail = min((boot <= 0).mean(), (boot >= 0).mean())
    return {
        "delta": point,
        "ci95": [float(lo), float(hi)],
        "p": float(min(1.0, 2 * tail)),
        "improved": bool(lo > 0 if higher_is_better else hi < 0),
        "degraded": bool(hi < 0 if higher_is_better else lo > 0),
    }


def compare(spec, n_boot, seed):
    rng = np.random.default_rng(seed)
    base_runs = {t: LOADERS[t](REPO_ROOT / spec["base"][t]) for t in spec["base"] if t in LOADERS}
    out = []
    for model in spec["models"]:
        row = {"name": model["name"], "tests": {}}
        for task, loader in LOADERS.items():
            if task not in model or task not in base_runs:
                continue
            b, a, strata = align(base_runs[task], loader(REPO_ROOT / model[task]))
            idx = cluster_indices(strata, n_boot, rng)
            bs, as_ = resampled_sums([b, a], idx)
            if task == "openbbq":
                for key, cond, which, hib in OPENBBQ_METRICS:
                    pt, boot = paired_diff(openbbq_stat(cond, which), b, a, bs, as_,
                                           absolute=which != "acc")
                    row["tests"][key] = {"n_clusters": len(strata), **summarize(pt, boot, hib)}
            else:
                pt, boot = paired_diff(mean_stat, b, a, bs, as_, absolute=False)
                row["tests"][task] = {"n_clusters": len(strata), **summarize(pt, boot, True)}
        out.append(row)
    return out


def markdown(rows):
    keys = [k for k in [m[0] for m in OPENBBQ_METRICS] + UTILITY_TASKS
            if any(k in r["tests"] for r in rows)]
    mark = lambda t: "†" if t["improved"] else ("‡" if t["degraded"] else "")  # noqa: E731
    lines = ["| model | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for r in rows:
        cells = []
        for k in keys:
            t = r["tests"].get(k)
            cells.append("—" if t is None else f"{t['delta']:+.4f}{mark(t)}")
        lines.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Δ = tuned - base (|bias| for the bias columns). †/‡: significant improvement / "
                 "degradation (95% interval excludes zero).")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="JSON with every interval")
    args = ap.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    rows = compare(spec, args.n_boot, args.seed)
    print(markdown(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"spec": str(args.spec), "n_boot": args.n_boot,
                                        "seed": args.seed, "models": rows}, indent=1))


if __name__ == "__main__":
    main()
