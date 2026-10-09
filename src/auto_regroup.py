"""Regroup the source groups of each category (App. A.3, stage 2).

1. Pool the category: discard the group boundaries; case variants fold onto the
   spelling the corpus sees more often.
2. Admit a term if ' ' + term is one token, the corpus counted it, and its
   adhesion is at most `--adhesion-max`.
3. Split the admitted terms into cells by their majority part-of-speech tag.
4. Cut each cell into frequency bands (`--band-width`, as in build_generated_groups.py).
5. A band of one joins the neighbouring band that keeps the spread within
   `--merge-cap`; otherwise it is dropped.

The bands are the groups. The output is read by `build_bias_lexicon.py --regrouped`.

    uv run --no-sync python src/auto_regroup.py --lexicon data/lexicon/llada/source/groups.json \
        --stats data/corpus_stats/llada --out data/lexicon/llada/regrouped_board.json
"""

import argparse
import collections
import json
from pathlib import Path

from build_generated_groups import Stats, cut_bands, single_token

GRANULARITIES = ("tuple", "name", "axis", "place")


def read_board(path):
    """category -> term -> the granularities the term arrived under."""
    doc = json.loads(Path(path).read_text())
    bag = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for axis, groups in doc["axes"].items():
        for g in groups:
            for t in g["terms"]:
                bag[axis][t][g["granularity"]] += 1
    return bag


def fold_casing(words, stats):
    by_key = collections.defaultdict(list)
    for term in words:
        by_key[term.casefold()].append(term)
    out = {}
    for spellings in by_key.values():
        term = max(spellings, key=lambda t: (stats.term_per_million(t), t))
        out[term] = sum((words[t] for t in spellings), collections.Counter())
    return out


def admit(words, tok, stats, adhesion_max):
    kept, why = [], collections.Counter()
    for term in sorted(words):
        if single_token(tok, term) is None:
            why["not one token"] += 1
            continue
        a = stats.adhesion(term)
        if a is None:
            why["not counted"] += 1
            continue
        if a > adhesion_max:
            why[f"adhesion > {adhesion_max:.0%}"] += 1
            continue
        kept.append({"term": term, "per_million": stats.term_per_million(term),
                     "adhesion": a, "granularity": words[term]})
    kept.sort(key=lambda r: -r["per_million"])
    return kept, why


def absorb_singletons(bands, cap):
    """A band of one joins the neighbour whose merged spread is smallest and at most `cap`."""
    bands = [list(b) for b in bands]
    i = 0
    while i < len(bands):
        if len(bands[i]) > 1:
            i += 1
            continue
        best = None
        for j in (i - 1, i + 1):
            if not 0 <= j < len(bands) or j == i:
                continue
            merged = bands[j] + bands[i]
            rates = [r["per_million"] for r in merged]
            spread = max(rates) / max(min(rates), 1e-12)
            if spread <= cap and (best is None or spread < best[0]):
                best = (spread, j)
        if best is None:
            i += 1
            continue
        bands[best[1]] += bands[i]
        bands[best[1]].sort(key=lambda r: -r["per_million"])
        bands.pop(i)
    return bands


def granularity_of(band):
    counts = collections.Counter()
    for r in band:
        counts.update(r["granularity"])
    return max(GRANULARITIES, key=lambda g: (counts[g], -GRANULARITIES.index(g)))


def regroup(bag, tok, stats, args, fallback):
    kept, discarded = [], []
    for axis in sorted(bag):
        folded = fold_casing(bag[axis], stats)
        rows, why = admit(folded, tok, stats, args.adhesion_max)
        cells = collections.defaultdict(list)
        for r in rows:
            cells[stats.majority_pos(r["term"]) or fallback(r["term"])].append(r)
        made = 0
        for tag in sorted(cells):
            for band in absorb_singletons(cut_bands(cells[tag], args.band_width), args.merge_cap):
                if len(band) < 2:
                    discarded.append({"axis": axis, "terms": [r["term"] for r in band],
                                      "why": f"alone in its {tag} band"})
                    continue
                kept.append({"axis": axis, "terms": [r["term"] for r in band],
                             "source": "regrouped", "granularity": granularity_of(band)})
                made += 1
        print(f"  {axis:<20} {len(folded):>5} terms -> {len(rows):>4} admitted -> {made:>3} groups")
    return kept, discarded


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", required=True, help="groups.json of the source groups")
    ap.add_argument("--stats", required=True, help="directory written by src/build_corpus_stats.py")
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--adhesion-max", type=float, default=0.20)
    ap.add_argument("--band-width", type=float, default=3.0)
    ap.add_argument("--merge-cap", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import nltk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stats = Stats(args.stats, args.corpus, tok)
    fallback = lambda t: nltk.pos_tag([t])[0][1]  # noqa: E731
    kept, discarded = regroup(read_board(args.lexicon), tok, stats, args, fallback)
    doc = {
        "thresholds": {"adhesion_max": args.adhesion_max, "band_width": args.band_width,
                       "merge_cap": args.merge_cap},
        "kept": kept,
        "discarded": discarded,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {out}  ({len(kept)} groups, {len(discarded)} terms dropped)")


if __name__ == "__main__":
    main()
