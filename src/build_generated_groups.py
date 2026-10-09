"""Cut the person-name and place groups from the corpus statistics (App. A.3, stage 1).

Names come from BBQ's proper-name vocabulary (surnames by ethnicity, given names
by gender and by ethnicity), places from the country names and country
adjectives of Czarnowska et al. (2021). A word is admitted when ' ' + word is one
token and, for names, its adhesion is at most `--adhesion-max`. Admitted words
are sorted by frequency and cut into bands: a band closes when the next word
falls below 1/`--band-width` of the band's first word. A band of names that does
not span at least two cells (ethnicities or genders) has nothing to be equalized
against and is dropped.

    uv run --no-sync python src/build_generated_groups.py --stats data/corpus_stats/llada \
        --out data/lexicon/llada/generated_groups.json
"""

import argparse
import collections
import csv
import functools
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "lexicon" / "_raw"

FAMILIES = {
    "surname_ethnicity": ("race_ethnicity", "name"),
    "first_name_gender": ("gender_identity", "name"),
    "first_name_ethnicity": ("race_ethnicity", "name"),
    "country": ("nationality", "place"),
    "country_adjective": ("nationality", "place"),
}


def _stemmer():
    from nltk.stem import PorterStemmer
    return functools.lru_cache(maxsize=None)(PorterStemmer().stem)


def pos_table(z, meta):
    """word -> {tag: count}, from the tagging done during the corpus pass."""
    terms, tags = meta["terms"], meta["pos_tags"]
    out = collections.defaultdict(dict)
    for (tid, tag), n in zip(z["pos_pairs"].tolist(), z["pos_counts"].tolist()):
        out[terms[tid]][tags[tag]] = n
    return dict(out)


class Stats:
    """The counts of one src/build_corpus_stats.py pass, addressed by word."""

    def __init__(self, stats_dir, corpus, tok):
        d = Path(stats_dir)
        z = np.load(d / f"{corpus}.npz", allow_pickle=True)
        meta = json.loads((d / f"{corpus}.meta.json").read_text())
        if meta["tokenizer"] != tok.name_or_path:
            raise SystemExit(f"{d} was counted with {meta['tokenizer']}, not {tok.name_or_path}")
        self.meta = meta
        self.tokens = meta["tokens"]
        self.unigram = z["unigram"]
        self.term = np.asarray(z["term"])
        self.term_cont = np.asarray(z["term_cont"])
        self.term_next = np.asarray(z["term_next"])
        self.index = {t: i for i, t in enumerate(meta["terms"])}
        self.ctx_indptr = z["ctx_right_indptr"]
        self.ctx_indices = z["ctx_right_indices"]
        self.ctx_data = z["ctx_right_data"]
        self.pos = pos_table(z, meta)
        # An entry continues the word before it when its text starts with a letter or digit.
        vocab = tok.get_vocab()
        self.continues = np.zeros(max(max(vocab.values()) + 1, len(tok)), dtype=bool)
        self.text = [""] * self.continues.size
        for piece, i in vocab.items():
            text = tok.convert_tokens_to_string([piece])
            self.text[i] = text
            if text and text[0].isalnum():
                self.continues[i] = True
        self.stem = _stemmer()

    def per_million(self, token_id):
        return self.unigram[token_id] / self.tokens * 1e6

    def term_per_million(self, word):
        i = self.index.get(word)
        return self.term[i] / self.tokens * 1e6 if i is not None else 0.0

    def majority_pos(self, word):
        """The commonest tag of the word (ties broken by tag name), or None."""
        tags = self.pos.get(word)
        return max(tags, key=lambda t: (tags[t], t)) if tags else None

    def adhesion(self, word):
        """Share of the word's occurrences whose next token continues it into a different word.

        `term_cont / term_next`, both counted on the space-prefixed form, minus the
        continuations that are inflections of the same word (same Porter stem),
        whose share is read off the right contexts. None if the word was not counted.
        """
        i = self.index.get(word)
        if i is None or self.term_next[i] == 0:
            return None
        cont = float(self.term_cont[i])
        if not cont:
            return 0.0
        return cont * (1.0 - self.inflected_share(i, word)) / float(self.term_next[i])

    def inflected_share(self, i, word):
        lo, hi = self.ctx_indptr[i], self.ctx_indptr[i + 1]
        nbr, cnt = self.ctx_indices[lo:hi], self.ctx_data[lo:hi]
        if not nbr.size:
            return 0.0
        keep = self.continues[nbr]
        nbr, cnt = nbr[keep].tolist(), cnt[keep].tolist()
        total = float(sum(cnt))
        if not total:
            return 0.0
        base = self.stem(word)
        same = sum(c for n, c in zip(nbr, cnt) if self.stem(word + self.text[n]) == base)
        return same / total


def single_token(tokenizer, word):
    ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def admit(words, tokenizer, stats, adhesion_max, need_adhesion):
    seen, unique = set(), []
    for w in words:
        if w.casefold() not in seen:
            seen.add(w.casefold())
            unique.append(w)
    kept, why = [], collections.Counter()
    for w in unique:
        tid = single_token(tokenizer, w)
        if tid is None:
            why["not one token"] += 1
            continue
        a = stats.adhesion(w) if need_adhesion else 0.0
        if need_adhesion and a is None:
            why["not counted"] += 1
            continue
        if a > adhesion_max:
            why[f"adhesion > {adhesion_max:.0%}"] += 1
            continue
        kept.append({"term": w, "token_id": int(tid),
                     "per_million": stats.per_million(tid), "adhesion": a})
    kept.sort(key=lambda r: -r["per_million"])
    return kept, why


def spread(rows):
    """Each member's frequency over the group's geometric mean."""
    logs = [math.log(max(r["per_million"], 1e-12)) for r in rows]
    mean = math.exp(sum(logs) / len(logs))
    return [r["per_million"] / mean for r in rows]


def cut_bands(rows, width, top_width=None):
    """Greedy bands from the top: a band holds everything within `width` of its first word."""
    rows = sorted(rows, key=lambda r: -r["per_million"])
    bands, i = [], 0
    while i < len(rows):
        head = rows[i]["per_million"]
        w = top_width if (not bands and top_width) else width
        j = i
        while j < len(rows) and rows[j]["per_million"] >= head / w:
            j += 1
        bands.append(rows[i:j])
        i = j
    return bands


def bands_over_cells(cells, width, top_width=None):
    """Bands over every cell's words at once; a word two cells claim stays with the first."""
    label, rows = {}, []
    for cell, words in cells.items():
        for r in words:
            if r["term"] not in label:
                label[r["term"]] = cell
                rows.append(r)
    return [(b, collections.Counter(label[r["term"]] for r in b))
            for b in cut_bands(rows, width, top_width)]


def bbq_names():
    """BBQ's names by slot; a word that is both a surname and a given name is dropped from both."""
    rows = list(csv.DictReader((RAW / "bbq" / "vocabulary_proper_names.csv").open()))
    first, last = collections.defaultdict(list), collections.defaultdict(list)
    gender = collections.defaultdict(list)
    for r in rows:
        name = r["Name"].strip()
        if r["First_last"] == "last":
            last[r["ethnicity"]].append(name)
        else:
            gender[r["gender"]].append(name)
            if r["ethnicity"]:
                first[r["ethnicity"]].append(name)
    both = {n for ns in first.values() for n in ns} | {n for ns in gender.values() for n in ns}
    both &= {n for ns in last.values() for n in ns}
    strip = lambda d: {k: [n for n in v if n not in both] for k, v in d.items()}  # noqa: E731
    return strip(first), strip(last), strip(gender), sorted(both)


def czarnowska_places():
    rows = list(csv.DictReader((RAW / "czarnowska" / "countries.csv").open()))
    names = [r["TERM"].strip() for r in rows if r["TERM"].strip()]
    adjs = [r["COUNTRY_ADJ"].strip() for r in rows if r["COUNTRY_ADJ"].strip()]
    return names, adjs


def emit(groups, family, members, cells):
    axis, granularity = FAMILIES[family]
    rows = list(members)
    groups.append({"axis": axis, "terms": [r["term"] for r in rows],
                   "source": "generated", "granularity": granularity,
                   "family": family, "cells": dict(cells),
                   "per_million": [round(r["per_million"], 3) for r in rows],
                   "ratio_to_mean": [round(x, 2) for x in spread(rows)]})


def cut_family(groups, family, cells, width, top_width=None, order=None):
    order = order or sorted(cells)
    admitted = {c: cells[c][0] for c in order}
    # Countries have no sides, so a band of them only needs two words.
    need_cells = 2 if len(order) > 1 else 1
    made = 0
    for band, comp in bands_over_cells(admitted, width, top_width):
        if len(comp) < need_cells or len(band) < 2:
            continue
        emit(groups, family, band, comp)
        made += 1
    n_admitted = sum(len(v) for v in admitted.values())
    print(f"  {family:<22} {n_admitted:>4} admitted -> {made} groups")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, help="directory written by src/build_corpus_stats.py")
    ap.add_argument("--corpus", default="c4")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--adhesion-max", type=float, default=0.20)
    ap.add_argument("--band-width", type=float, default=3.0)
    ap.add_argument("--place-band-width", type=float, default=2.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stats = Stats(args.stats, args.corpus, tok)
    print(f"stats: {args.stats}/{args.corpus}.npz  {stats.tokens:,} tokens")

    first, last, gender, both_slots = bbq_names()
    countries, adjectives = czarnowska_places()

    def cells_of(raw, need_adhesion):
        return {c: admit(ws, tok, stats, args.adhesion_max, need_adhesion) for c, ws in raw.items()}

    groups = []
    ethnicities = ["Arab", "Asian", "Black", "Latino", "White"]
    cut_family(groups, "surname_ethnicity", cells_of(last, True), args.band_width, order=ethnicities)
    cut_family(groups, "first_name_ethnicity", cells_of(first, True), args.band_width, order=ethnicities)
    cut_family(groups, "first_name_gender", cells_of(gender, True), args.band_width, order=["F", "M"])
    cut_family(groups, "country", cells_of({"country": countries}, False), args.place_band_width)
    cut_family(groups, "country_adjective", cells_of({"adjective": adjectives}, False),
               args.place_band_width)

    doc = {"tokenizer": args.tokenizer,
           "thresholds": {"adhesion_max": args.adhesion_max, "band_width": args.band_width,
                          "place_band_width": args.place_band_width},
           "dropped_both_name_slots": both_slots,
           "groups": groups}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {out}  ({len(groups)} groups)")


if __name__ == "__main__":
    main()
