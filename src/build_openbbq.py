"""Build the OpenBBQ evaluation and training splits.

Source: https://github.com/zhaoliu0914/LLM-Bias-Benchmark
    data/<category>_<condition>_short_answer_gpt4o.jsonl  context (system turn) + question
    metadata/<category>_<condition>_metadata.jsonl        options, gold label, bias target

The split unit is a context group: one context x {neg, nonneg} question x
{ambiguous, disambiguated}. Keeping groups whole means no context is shared across
splits, and every split is exactly 1:1 ambiguous:disambiguated. Categories are
allocated in proportion to OpenBBQ by largest remainder.

Outputs (each as <split>/train.jsonl):
    test-10k    the evaluation split
    train-10k   the training split
    train-pool  every row not in test-10k (read by src/build_term_table.py)
"""

import argparse
import collections
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openbbq_prompts import prompt_fingerprint, render_problem  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "62cca6a863e83fcd0f176e6dae1ca8a63e7f3a24"
RAW_BASE = f"https://raw.githubusercontent.com/zhaoliu0914/LLM-Bias-Benchmark/{COMMIT}"
SOURCE_MODEL = "gpt4o"
CONDITIONS = ("ambiguous", "disambiguated")
CATEGORY_ROWS = {
    "age": 1840,
    "disability_status": 778,
    "gender_identity": 2836,
    "nationality": 1540,
    "physical_appearance": 788,
    "race_ethnicity": 3440,
    "race_x_gender": 7980,
    "race_x_ses": 5580,
    "religion": 600,
    "ses": 3432,
    "sexual_orientation": 432,
}
SHORT_ANSWER_SUFFIX = (
    " Could you give us the most possible answer and provide your chain of thought?"
)


def fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_rows(raw_dir: Path) -> list[dict]:
    rows = []
    for category, expected in CATEGORY_ROWS.items():
        for condition in CONDITIONS:
            data = read_jsonl(fetch(
                f"{RAW_BASE}/data/{category}_{condition}_short_answer_{SOURCE_MODEL}.jsonl",
                raw_dir / f"{category}_{condition}_short_answer.jsonl",
            ))
            meta = read_jsonl(fetch(
                f"{RAW_BASE}/metadata/{category}_{condition}_metadata.jsonl",
                raw_dir / f"{category}_{condition}_metadata.jsonl",
            ))
            if len(data) != expected or len(meta) != expected:
                raise SystemExit(f"{category}/{condition}: expected {expected} rows, "
                                 f"got data={len(data)} meta={len(meta)}")
            for d, m in zip(data, meta):
                if d["custom_id"] != m["custom_id"]:
                    raise SystemExit(f"{category}/{condition}: data and metadata are not aligned "
                                     f"({d['custom_id']} vs {m['custom_id']})")
                messages = d["body"]["messages"]
                context = messages[0]["content"]
                question = messages[1]["content"]
                if not question.endswith(SHORT_ANSWER_SUFFIX):
                    raise SystemExit(f"{d['custom_id']}: unexpected user turn")
                options = [m["answer_info"][f"ans{i}"][0] for i in range(3)]
                label = int(m["label"])
                rows.append({
                    "problem": render_problem(context, question, options),
                    "solution": f"\\boxed{{{label}}}",
                    "target": str(label),
                    "custom_id": m["custom_id"],
                    "category": category,
                    "condition": condition,
                    "question_polarity": m["question_polarity"],
                    "label": label,
                    "target_bias": int(m["target_bias"]),
                    "options": options,
                    "unknown_index": next(
                        i for i in range(3) if m["answer_info"][f"ans{i}"][1] == "unknown"
                    ),
                    "context": context,
                })
        print(f"  {category}: {2 * expected} rows")
    return rows


def group_rows(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group by (category, ambiguous context).

    The disambiguated context of an item extends its ambiguous one, so both
    conditions are keyed by the ambiguous context of the same custom_id. OpenBBQ
    reuses a few context strings across items; those groups merge.
    """
    ambiguous_context = {
        (r["category"], r["custom_id"].rsplit("-", 1)[-1], r["question_polarity"]): r["context"]
        for r in rows if r["condition"] == "ambiguous"
    }
    groups: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in rows:
        key = (r["category"], r["custom_id"].rsplit("-", 1)[-1], r["question_polarity"])
        groups[(r["category"], ambiguous_context[key])].append(r)

    for key, members in groups.items():
        conditions = collections.Counter(m["condition"] for m in members)
        polarities = collections.Counter(m["question_polarity"] for m in members)
        if (len(members) % 4
                or conditions["ambiguous"] != conditions["disambiguated"]
                or polarities["neg"] != polarities["nonneg"]):
            raise SystemExit(f"context group {key[0]} is not balanced: n={len(members)} "
                             f"{dict(conditions)} {dict(polarities)}")

    for gid, (_, members) in enumerate(sorted(groups.items(), key=lambda kv: kv[0])):
        for m in members:
            m["group_id"] = gid
    return groups


def allocate(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` rows across categories, in units of 4."""
    if total % 4:
        raise SystemExit(f"total must be a multiple of 4, got {total}")
    units, pool = total // 4, sum(sizes.values()) // 4
    if units > pool:
        raise SystemExit(f"cannot allocate {total} rows from a pool of {pool * 4}")
    exact = {c: (n // 4) * units / pool for c, n in sizes.items()}
    out = {c: int(v) for c, v in exact.items()}
    for c in sorted(sizes, key=lambda c: (-(exact[c] - out[c]), c))[: units - sum(out.values())]:
        out[c] += 1
    for c, n in out.items():
        if n * 4 > sizes[c]:
            raise SystemExit(f"{c}: allocated {n * 4} rows > available {sizes[c]}")
    return {c: n * 4 for c, n in out.items()}


def take(pools: dict[str, list], quota: dict[str, int]) -> list:
    """Pop whole groups off each pre-shuffled category pool up to its row quota."""
    picked = []
    for category, want in quota.items():
        pool, keep, got = pools[category], [], 0
        for group in pool:
            if got + len(group) <= want:
                picked.append(group)
                got += len(group)
            else:
                keep.append(group)
        if got != want:
            raise SystemExit(f"{category}: could not fill the quota exactly ({got}/{want} rows)")
        pools[category] = keep
    return picked


def write_split(name: str, out_dir: Path, groups: list[list[dict]]) -> dict:
    rows = [r for g in groups for r in g]
    rows.sort(key=lambda r: (r["category"], r["custom_id"], r["condition"]))
    path = out_dir / name / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    conditions = collections.Counter(r["condition"] for r in rows)
    print(f"  {name:11s} groups={len(groups):6d} rows={len(rows):6d} "
          f"amb/dis={conditions['ambiguous']}/{conditions['disambiguated']}")
    return {
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "groups": len(groups),
        "rows": len(rows),
        "by_condition": dict(conditions),
        "by_category": dict(collections.Counter(r["category"] for r in rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "openbbq")
    p.add_argument("--test-rows", type=int, default=10_000)
    p.add_argument("--train-rows", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("[1/4] loading OpenBBQ")
    rows = load_rows(args.out / "_raw")
    print(f"      {len(rows)} rows")

    print("[2/4] grouping by context")
    groups = group_rows(rows)
    by_category: dict[str, list[list[dict]]] = collections.defaultdict(list)
    for members in groups.values():
        by_category[members[0]["category"]].append(members)

    print("[3/4] splitting")
    rng = random.Random(args.seed)
    pools = {c: sorted(g, key=lambda m: m[0]["custom_id"]) for c, g in by_category.items()}
    for c in pools:
        rng.shuffle(pools[c])
    sizes = {c: sum(len(g) for g in pool) for c, pool in pools.items()}

    test = take(pools, allocate(sizes, args.test_rows))
    remaining = {c: sum(len(g) for g in pool) for c, pool in pools.items()}
    rest = [g for c in pools for g in pools[c]]
    train = take({c: list(g) for c, g in pools.items()}, allocate(remaining, args.train_rows))

    print("[4/4] writing")
    manifest = {
        "source": RAW_BASE,
        "prompt_fingerprint": prompt_fingerprint(),
        "seed": args.seed,
        "total_rows": len(rows),
        "splits": {
            "test-10k": write_split("test-10k", args.out, test),
            "train-10k": write_split("train-10k", args.out, train),
            "train-pool": write_split("train-pool", args.out, rest),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test_ctx = {r["context"] for g in test for r in g}
    rest_ctx = {r["context"] for g in rest for r in g}
    assert not test_ctx & rest_ctx, "a context is shared between test-10k and the training pool"
    print(f"      manifest: {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
