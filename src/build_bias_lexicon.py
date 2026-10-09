"""Build the bias group lexicon (App. A.3).

A group is a set of two or more terms that can replace one another in the same
slot. Without `--regrouped` this builds the source groups of stage 1: groups
mined from the lexical resources, with the person-name and place groups replaced
by those of src/build_generated_groups.py, and the race, gender and SES groups
copied onto the two cross categories. With `--regrouped` it builds the lexicon
from a board written by src/auto_regroup.py (stage 2).

Sources (fetched by scripts/3_fetch_lexicon_sources.sh into data/lexicon/_raw/):
  openbbq         OpenBBQ's own option pairs (only for the corpus term list)
  bbq_vocab       BBQ templates/vocabulary.csv, one group per (Category, SubCat)
  bias_bench      the CDA word tuples of Meade et al. (2022)
  debiaswe        the gender pairs of Bolukbasi et al. (2016)
  gn_glove        the gender pairs of Zhao et al. (2018)
  crows_pairs     the minimal edits of CrowS-Pairs (Nangia et al., 2020)
  czarnowska      the identity terms of Czarnowska et al. (2021), by (POS, type)
  holistic_nouns  HolisticBias person nouns (Smith et al., 2022)
  holistic_bias   HolisticBias descriptors, one group per axis

Writes `groups.json` (the groups) and `lexicon.json` (the groups after
single_dedup_partial, resolved against `--tokenizer`), which the trainers read.

    uv run --no-sync python src/build_bias_lexicon.py --out data/lexicon/llada/source \
        --generated data/lexicon/llada/generated_groups.json --skip-source openbbq
"""

import argparse
import collections
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "lexicon" / "_raw"

AXES = (
    "age", "disability_status", "gender_identity", "nationality",
    "physical_appearance", "race_ethnicity", "race_x_gender", "race_x_ses",
    "religion", "ses", "sexual_orientation",
)
CROSSES = {
    "race_x_gender": ("race_ethnicity", "gender_identity"),
    "race_x_ses": ("race_ethnicity", "ses"),
}
# Earlier source wins a casing conflict when two sources emit the same term set.
PRIORITY = ("regrouped", "generated", "openbbq", "bbq_vocab", "bias_bench", "debiaswe",
            "gn_glove", "crows_pairs", "czarnowska", "holistic_nouns", "holistic_bias")
SOURCES = ("openbbq", "bbq_vocab", "bias_bench", "debiaswe", "gn_glove",
           "crows_pairs", "czarnowska", "holistic_nouns", "holistic_bias")

PRONOUNS = frozenset("he she they him her them his hers their theirs himself "
                     "herself themselves it its".split())
FUNCTION_WORDS = frozenset(
    "a an the of in on at to for with without and or but is are was were be been "
    "being has have had do does did not no this that these those very quite so as "
    "by from into than then there here".split())

# Misspellings and nonstandard spellings found in the sources.
SPELLING = {
    "cathloic": "catholic", "whibte": "white", "Mexians": "Mexicans",
    "imigrants": "immigrants", "rAmericans": "Americans", "Hetrosexual": "Heterosexual",
    "Hetrosexuals": "Heterosexuals", "etero": "hetero", "Pentacostal": "Pentecostal",
    "Scandanavian": "Scandinavian", "mentleman": "gentleman", "godess": "goddess",
    "stailor": "tailor", "barnoesses": "baronesses", "caucasain": "caucasian",
    "millenial": "millennial", "monther": "mother", "adultor": "adulterer",
    "adultors": "adulterers", "adultress": "adulteress", "adultresses": "adulteresses",
}

WS = re.compile(r"\s+")
TOKEN = re.compile(r"[\w'-]+|[^\w\s]")


def norm(term):
    """Whitespace and Unicode only; case and inflection are kept."""
    return WS.sub(" ", unicodedata.normalize("NFKC", str(term))).strip()


def key(term):
    return term.casefold()


def prim(group):
    """The source that first emitted the group (`dedupe` appends `+other`, crossing `@axis`)."""
    return group["source"].split("@")[0].split("+")[0]


def is_coreference(word, repl):
    """A pronoun standing in for a non-pronoun rewrites the reference, not the group named."""
    return key(repl) in PRONOUNS and key(word) not in PRONOUNS


def emit(groups, axis, terms, source, granularity):
    terms = [SPELLING.get(t, t) for t in (norm(x) for x in terms)]
    terms = [t for t in terms if t and len(t) < 40]
    seen, uniq = set(), []
    for t in terms:
        if key(t) not in seen:
            seen.add(key(t))
            uniq.append(t)
    if len(uniq) >= 2:
        groups.append({"axis": axis, "terms": uniq, "source": source,
                       "granularity": granularity})


# --------------------------------------------------------------------------
# sources

def from_openbbq(groups):
    path = REPO / "data" / "openbbq" / "train-pool" / "train.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing; run scripts/2_build_openbbq.sh first.")
    seen = set()
    for line in path.open():
        r = json.loads(line)
        opts, unk = r.get("options"), r.get("unknown_index")
        if not opts or unk is None:
            continue
        bias = [o for i, o in enumerate(opts) if i != unk]
        if len(bias) < 2:
            continue
        k = (r["category"], tuple(sorted(key(o) for o in bias)))
        if k in seen:
            continue
        seen.add(k)
        emit(groups, r["category"], bias, "openbbq", "tuple")
    print(f"  openbbq: {len(seen)} option pairs")


def proper_names():
    path = RAW / "bbq" / "vocabulary_proper_names.csv"
    return {key(r["Name"]) for r in csv.DictReader(path.open())}


def from_bbq_vocab(groups):
    m = {"Gender_identity": "gender_identity", "Nationality": "nationality",
         "Race_ethnicity": "race_ethnicity", "Religion": "religion",
         "SES": "ses", "Sexual_orientation": "sexual_orientation"}
    buckets = collections.defaultdict(list)
    for r in csv.DictReader((RAW / "bbq" / "vocabulary.csv").open()):
        axis = m.get(r["Category"].strip())
        if axis:
            buckets[(axis, r["SubCat"].strip())].append(r["Name"])
    for (axis, _), names in buckets.items():
        emit(groups, axis, names, "bbq_vocab", "axis")
    print(f"  bbq_vocab: {len(buckets)} slots")


def from_bias_bench(groups):
    m = {"gender": "gender_identity", "race": "race_ethnicity", "religion": "religion"}
    data = json.loads((RAW / "bias_bench" / "bias_attribute_words.json").read_text())
    n = 0
    for k, tuples in data.items():
        for t in tuples:
            emit(groups, m[k], t, "bias_bench", "tuple")
            n += 1
    print(f"  bias_bench: {n} tuples")


def from_holistic_bias(groups):
    m = {"ability": "disability_status", "age": "age",
         "body_type": "physical_appearance",
         "gender_and_sex": "gender_identity", "nationality": "nationality",
         "race_ethnicity": "race_ethnicity", "religion": "religion",
         "sexual_orientation": "sexual_orientation",
         "socioeconomic_class": "ses"}
    data = json.loads((RAW / "holistic_bias" / "descriptors.json").read_text())
    per_axis = collections.defaultdict(list)
    # `cultural` mixes generation labels (age) with subculture labels (no category).
    buckets = [(axis, b) for hb_axis, axis in m.items() for b in data.get(hb_axis, {}).values()]
    buckets.append(("age", data["cultural"]["generations"]))
    for axis, bucket in buckets:
        for d in bucket:
            per_axis[axis].append(d["descriptor"] if isinstance(d, dict) else d)
    for axis, terms in per_axis.items():
        emit(groups, axis, terms, "holistic_bias", "axis")
    print(f"  holistic_bias: {len(per_axis)} axis sets")


def from_czarnowska(groups):
    m = {"age": "age", "disability": "disability_status",
         "gender": "gender_identity", "race": "race_ethnicity",
         "religion": "religion", "sexuality": "sexual_orientation",
         "countries": "nationality"}
    n = 0
    for f, axis in m.items():
        slots = collections.defaultdict(list)
        for r in csv.DictReader((RAW / "czarnowska" / f"{f}.csv").open()):
            slots[(r.get("POS", ""), r.get("TYPE", ""), r.get("SEM", ""))].append(r["TERM"])
            # A country name and its adjective fill different slots.
            if r.get("COUNTRY_ADJ"):
                slots[("adj", "", "")].append(r["COUNTRY_ADJ"])
        for terms in slots.values():
            emit(groups, axis, terms, "czarnowska", "axis")
            n += 1
    print(f"  czarnowska: {n} slots")


def from_gn_glove(groups):
    """Line i of the male file and line i of the female file are one pair."""
    d = RAW / "gn_glove"
    m = (d / "male_word_file.txt").read_text().split()
    f = (d / "female_word_file.txt").read_text().split()
    n = 0
    for a, b in zip(m, f):
        if "_" in a or "_" in b:
            continue
        emit(groups, "gender_identity", [a, b], "gn_glove", "tuple")
        n += 1
    print(f"  gn_glove: {n} pairs")


def from_debiaswe(groups):
    d = RAW / "debiaswe"
    n = 0
    for f in ("equalize_pairs.json", "definitional_pairs.json"):
        for a, b in json.loads((d / f).read_text()):
            if "_" in a or "_" in b:
                continue
            emit(groups, "gender_identity", [a, b], "debiaswe", "tuple")
            n += 1
    print(f"  debiaswe: {n} pairs")


def from_holistic_nouns(groups):
    d = json.loads((RAW / "holistic_bias" / "nouns.json").read_text())
    for i in (0, 1):  # singular, plural
        emit(groups, "gender_identity", [p[i] for v in d.values() for p in v],
             "holistic_nouns", "axis")
    print("  holistic_nouns: 2 slots")


def from_crows_pairs(groups):
    """The edit of each minimal pair: shared prefix and suffix trimmed off.

    Kept only if it is one word on each side, both words start with a letter,
    neither is a function word, and it is not a pronoun on one side and a
    non-pronoun on the other.
    """
    axes = {"race-color": "race_ethnicity", "gender": "gender_identity",
            "socioeconomic": "ses", "nationality": "nationality",
            "religion": "religion", "age": "age",
            "sexual-orientation": "sexual_orientation",
            "physical-appearance": "physical_appearance",
            "disability": "disability_status"}
    seen = {}
    for r in csv.DictReader((RAW / "crows_pairs" / "crows_pairs_anonymized.csv").open()):
        axis = axes.get(r["bias_type"].strip())
        if axis is None:
            continue
        a, b = TOKEN.findall(r["sent_more"]), TOKEN.findall(r["sent_less"])
        i = 0
        while i < len(a) and i < len(b) and a[i] == b[i]:
            i += 1
        j = 0
        while (j < len(a) - i and j < len(b) - i
               and a[len(a) - 1 - j] == b[len(b) - 1 - j]):
            j += 1
        src, dst = a[i:len(a) - j], b[i:len(b) - j]
        if len(src) != 1 or len(dst) != 1:
            continue
        src, dst = src[0], dst[0]
        if (key(src) == key(dst)
                or not (src[:1].isalpha() and dst[:1].isalpha())
                or key(src) in FUNCTION_WORDS or key(dst) in FUNCTION_WORDS
                or is_coreference(src, dst) or is_coreference(dst, src)):
            continue
        seen[(axis, tuple(sorted((src, dst), key=key)))] = True
    for axis, pair in seen:
        emit(groups, axis, list(pair), "crows_pairs", "tuple")
    print(f"  crows_pairs: {len(seen)} pairs")


# --------------------------------------------------------------------------
# names, merging, crosses

def place_terms():
    out = set()
    for r in csv.DictReader((RAW / "czarnowska" / "countries.csv").open()):
        for col in ("TERM", "COUNTRY_ADJ"):
            if r.get(col):
                out.add(key(norm(r[col])))
    return out


def mark_name_groups(groups):
    """Mark as `name` the pairs that are proper names rather than demographic terms.

    A pair is a name pair when every term contains a BBQ name, or when every term
    is a capitalised single word that no curated vocabulary lists (in any number).
    """
    names = proper_names()
    curated = {key(t) for g in groups if g["source"] in
               ("bbq_vocab", "bias_bench", "czarnowska", "holistic_bias")
               for t in g["terms"]}
    curated |= {w + suf for w in list(curated) for suf in ("s", "es")}
    curated |= {w[:-2] for w in list(curated) if w.endswith("es")}
    curated |= {w[:-1] for w in list(curated) if w.endswith("s")}
    for g in groups:
        terms = g["terms"]
        if g["granularity"] != "tuple":
            continue
        if all(any(key(w) in names for w in t.split()) for t in terms):
            g["granularity"] = "name"
        elif (all(t[:1].isupper() and " " not in t for t in terms)
              and not any(key(t) in curated for t in terms)):
            g["granularity"] = "name"
    return groups


def merge_generated(groups, path):
    """Replace every name group, and every nationality group of countries, by the generated ones."""
    doc = json.loads(Path(path).read_text())
    places = place_terms()
    kept = []
    for g in groups:
        if g["granularity"] == "name":
            continue
        if g["axis"] == "nationality" and all(key(t) in places for t in g["terms"]):
            continue
        kept.append(g)
    for g in doc["groups"]:
        emit(kept, g["axis"], g["terms"], g["source"], g["granularity"])
    return kept


def dedupe(groups):
    """One group per term set; a strict subset of another group is dropped."""
    rank = {s: i for i, s in enumerate(PRIORITY)}
    pri = lambda g: rank[prim(g)]  # noqa: E731
    best = {}
    for g in groups:
        k = (g["axis"], frozenset(key(t) for t in g["terms"]))
        if k not in best or pri(g) < pri(best[k]):
            if k in best:
                g = dict(g, source=g["source"] + "+" + best[k]["source"].split("+")[0])
            best[k] = g
    by_axis = collections.defaultdict(list)
    for (axis, ks), g in best.items():
        by_axis[(axis, g["granularity"])].append((ks, g))
    out = []
    for axis, items in by_axis.items():
        items.sort(key=lambda x: -len(x[0]))
        supersets = []
        for ks, g in items:
            if not any(ks < bigger for bigger in supersets):
                out.append(g)
            supersets.append(ks)
    return out


def add_crosses(groups):
    extra = []
    for cross, parts in CROSSES.items():
        for g in groups:
            if g["axis"] in parts:
                extra.append(dict(g, axis=cross, source=g["source"] + "@" + g["axis"]))
    return groups + extra


def single_dedup_partial(groups):
    """A term contested within a category goes to the smallest group; the groups that lose it are dropped."""
    order = sorted(range(len(groups)), key=lambda i: (len(groups[i]["terms"]), i))
    owned, kept = set(), []
    for i in order:
        g = groups[i]
        ks = {(g["axis"], key(t)) for t in g["terms"]}
        if ks & owned:
            continue
        owned |= ks
        kept.append(i)
    return [groups[i] for i in sorted(kept)]


def resolve(groups, tokenizer):
    """Give each term the vocabulary entry that tells it apart within its group.

    Every term is tokenized with a leading space; the prefix all of them share is
    skipped and the next token of each is taken. A term with nothing left after
    the prefix, or whose next token repeats another's, is dropped. `exact` marks a
    group whose every term is exactly one vocabulary entry.
    """
    cache = {}

    def seq(t):
        if t not in cache:
            cache[t] = tuple(tokenizer(" " + t, add_special_tokens=False)["input_ids"])
        return cache[t]

    out = []
    for g in groups:
        terms = list(g["terms"])
        while len(terms) >= 2:
            seqs = [seq(t) for t in terms]
            k = 0
            while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
                k += 1
            short = [t for t, s in zip(terms, seqs) if len(s) <= k]
            if short:
                terms = [t for t in terms if t not in short]
                continue
            ids, kept = {}, []
            for t, s in zip(terms, seqs):
                if s[k] not in ids:
                    ids[s[k]] = t
                    kept.append(t)
            terms = kept
            break
        if len(terms) >= 2:
            seqs = [seq(t) for t in terms]
            k = 0
            while all(len(s) > k for s in seqs) and len({s[k] for s in seqs}) == 1:
                k += 1
            out.append(dict(g, terms=terms,
                            token_ids=[seq(t)[k] for t in terms],
                            divergence_index=k,
                            exact=all(len(seq(t)) == k + 1 for t in terms)))
    return out


# --------------------------------------------------------------------------

def collect(skip=(), generated=None):
    """Stage 1: the source groups, crossed onto the intersectional categories."""
    groups = []
    for name in SOURCES:
        if name not in skip:
            globals()[f"from_{name}"](groups)
    mark_name_groups(groups)
    if generated:
        groups = merge_generated(groups, generated)
    return dedupe(add_crosses(dedupe(groups)))


def from_board(path):
    """Stage 2: the groups of an auto_regroup.py board, which already covers every category."""
    groups = []
    for g in json.loads(Path(path).read_text())["kept"]:
        emit(groups, g["axis"], g["terms"], g.get("source", "regrouped"),
             g.get("granularity", "tuple"))
    return dedupe(groups)


def by_axis(groups):
    out = collections.defaultdict(list)
    for g in groups:
        out[g["axis"]].append({k: v for k, v in g.items() if k != "axis"})
    return {"axes": {a: out.get(a, []) for a in AXES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tokenizer", default="GSAI-ML/LLaDA-8B-Instruct")
    ap.add_argument("--skip-source", default="", help="comma-separated sources not to mine")
    ap.add_argument("--generated", help="src/build_generated_groups.py's output")
    ap.add_argument("--regrouped", help="src/auto_regroup.py's output; replaces the sources")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.regrouped:
        groups = from_board(args.regrouped)
    else:
        skip = {x.strip() for x in args.skip_source.split(",") if x.strip()}
        unknown = skip - set(SOURCES)
        if unknown:
            sys.exit(f"unknown source(s) {sorted(unknown)}")
        groups = collect(skip, args.generated)
    (out / "groups.json").write_text(json.dumps(by_axis(groups), indent=1, ensure_ascii=False))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    resolved = resolve(single_dedup_partial(groups), tok)
    doc = {"tokenizer": args.tokenizer, "vocab_size": len(tok), **by_axis(resolved)}
    (out / "lexicon.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    exact = sum(g["exact"] for g in resolved)
    print(f"  {len(groups)} groups -> {out / 'groups.json'}")
    print(f"  {len(resolved)} groups after single_dedup_partial ({exact} exact) -> {out / 'lexicon.json'}")


if __name__ == "__main__":
    main()
