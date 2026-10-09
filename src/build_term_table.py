"""The terms whose corpus statistics src/build_corpus_stats.py collects.

Every term of the source groups mined from all sources (OpenBBQ's option pairs
included, the generated groups excluded), every alphabetic cell of BBQ's
proper-name vocabulary and of Czarnowska et al.'s country list, and the
additional terms below. Counting matches the longest term at each position, so
a term's count depends on the other terms in the table; this table is the one
the paper's statistics were collected over.

    uv run --no-sync python src/build_term_table.py --out data/lexicon/terms.json
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import build_bias_lexicon as bbl

EXTRA_TERMS = ("adults",)


def alphabetic_cells(path):
    return {v.strip() for r in csv.DictReader(path.open()) for v in r.values()
            if v and v.strip().isalpha()}


def term_table():
    with contextlib.redirect_stdout(io.StringIO()):
        groups = bbl.collect()
    terms = {t for g in groups for t in g["terms"]}
    terms |= alphabetic_cells(bbl.RAW / "bbq" / "vocabulary_proper_names.csv")
    terms |= alphabetic_cells(bbl.RAW / "czarnowska" / "countries.csv")
    terms |= set(EXTRA_TERMS)
    return sorted(terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    terms = term_table()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"terms": terms}, indent=0, ensure_ascii=False))
    print(f"wrote {out}  ({len(terms)} terms)")


if __name__ == "__main__":
    main()
