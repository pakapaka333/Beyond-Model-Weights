"""Count the lexicon's terms in C4 with the models' own tokenizers (App. A.3).

One pass over the first `--shards` training shards of allenai/c4 (en) records, for
each tokenizer (the part-of-speech tags do not depend on the tokenizer and are
computed once for all of them):

  unigram    the count of every vocabulary entry
  term       each term's count as the token sequence of ' ' + term
             (`term_bare`: of the term without the leading space)
  term_cont  occurrences of ' ' + term whose next token continues it into a longer word
  term_next  occurrences of ' ' + term that have a next token
  ctx_right  each term's commonest right-neighbour tokens
  pos        each single-word term's part-of-speech tags (nltk), over every document

Every whitespace-separated part of a multi-word term is counted as well. At each
position the longest matching term is counted.

    uv run --no-sync python src/build_corpus_stats.py --terms data/lexicon/terms.json \
        --tokenizer GSAI-ML/LLaDA-8B-Instruct Dream-org/Dream-v0-Instruct-7B \
        --out data/corpus_stats/llada data/corpus_stats/dream
"""

import argparse
import collections
import gzip
import itertools
import json
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

C4 = {"repo": "allenai/c4", "files": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(1024)]}
CTX_KEEP = 256
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|[0-9]+|[^\sA-Za-z0-9]")

_W = {}


def term_table(path):
    terms = set(json.loads(Path(path).read_text())["terms"])
    terms |= {w for t in terms for w in t.split() if w != t}
    return sorted(terms)


def build_trie(terms, tokenizer):
    """first token -> [(remaining tokens, term id, bare?), ...], longest first."""
    by_first = collections.defaultdict(list)
    for i, t in enumerate(terms):
        for bare, text in ((False, " " + t), (True, t)):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                by_first[ids[0]].append((tuple(ids[1:]), i, bare))
    for v in by_first.values():
        v.sort(key=lambda x: -len(x[0]))
    return dict(by_first)


def _tokenizer_state(tokenizer_name, terms):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    vocab = max(len(tok), max(tok.get_vocab().values()) + 1)
    # A bare match counts only after a token that ends a word.
    ends_word = np.ones(vocab, dtype=bool)
    continues_word = np.zeros(vocab, dtype=bool)
    for piece, i in tok.get_vocab().items():
        text = tok.convert_tokens_to_string([piece])
        if text and (text[-1].isalnum() or text[-1] == "'"):
            ends_word[i] = False
        if text and text[0].isalnum():
            continues_word[i] = True
    trie = build_trie(terms, tok)
    is_first = np.zeros(vocab, dtype=bool)
    for t in trie:
        is_first[t] = True
    return dict(tok=tok, vocab=vocab, ends_word=ends_word, continues_word=continues_word,
                trie=trie, is_first=is_first)


def _init(tokenizer_names, terms_path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import nltk
    terms = term_table(terms_path)
    _W.update(n_terms=len(terms), tagger=nltk.pos_tag,
              pos_watch={t: i for i, t in enumerate(terms) if " " not in t},
              tokenizers=[_tokenizer_state(name, terms) for name in tokenizer_names])


def _scan(texts):
    pos = collections.Counter()
    for text in texts:
        for w, tag in _W["tagger"](WORD_RE.findall(text)):
            j = _W["pos_watch"].get(w)
            if j is not None:
                pos[(j, tag)] += 1
    return [_scan_tokens(texts, T) for T in _W["tokenizers"]], pos, len(texts)


def _scan_tokens(texts, T):
    tok, trie, is_first = T["tok"], T["trie"], T["is_first"]
    ends_word, continues_word = T["ends_word"], T["continues_word"]
    unigram = np.zeros(T["vocab"], dtype=np.int64)
    term = np.zeros(_W["n_terms"], dtype=np.int64)
    term_bare = np.zeros(_W["n_terms"], dtype=np.int64)
    term_cont = np.zeros(_W["n_terms"], dtype=np.int64)
    term_next = np.zeros(_W["n_terms"], dtype=np.int64)
    right = collections.Counter()
    n_tok = 0
    for ids in tok(texts, add_special_tokens=False)["input_ids"]:
        if not ids:
            continue
        arr = np.asarray(ids, dtype=np.int64)
        n_tok += arr.size
        unigram += np.bincount(arr, minlength=unigram.size)[: unigram.size]
        for i in np.flatnonzero(is_first[arr]).tolist():
            for rest, tid, bare in trie[ids[i]]:
                if bare and i and not ends_word[ids[i - 1]]:
                    continue
                end = i + 1 + len(rest)
                if end <= len(ids) and tuple(ids[i + 1:end]) == rest:
                    (term_bare if bare else term)[tid] += 1
                    if not bare and end < len(ids):
                        term_next[tid] += 1
                        if continues_word[ids[end]]:
                            term_cont[tid] += 1
                    if end < len(ids):
                        right[(tid, ids[end])] += 1
                    break
    return unigram, term, term_bare, term_cont, term_next, right, n_tok


def batches(paths, size):
    for p in paths:
        buf = []
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                buf.append(json.loads(line)["text"])
                if len(buf) >= size:
                    yield buf
                    buf = []
        if buf:
            yield buf


def download(shards):
    """Fetch the shards once into the Hugging Face cache."""
    from huggingface_hub import hf_hub_download
    out = []
    for name in C4["files"][:shards]:
        out.append(hf_hub_download(C4["repo"], name, repo_type="dataset",
                                   token=os.environ.get("HF_TOKEN")))
        print(f"  {name}", flush=True)
    return out


def trim(counter, keep):
    """Keep each term's `keep` commonest contexts, to bound memory during the pass."""
    per = collections.defaultdict(list)
    for (tid, ctx), c in counter.items():
        per[tid].append((c, ctx))
    out = collections.Counter()
    for tid, rows in per.items():
        rows.sort(reverse=True)
        for c, ctx in rows[:keep]:
            out[(tid, ctx)] = c
    return out


def prune(counter, n_terms, keep):
    """Per-term top-`keep` contexts, as a CSR triple."""
    per = collections.defaultdict(list)
    for (tid, ctx), c in counter.items():
        per[tid].append((c, ctx))
    indptr = np.zeros(n_terms + 1, dtype=np.int64)
    indices, data = [], []
    for tid in range(n_terms):
        rows = sorted(per.get(tid, ()), reverse=True)[:keep]
        indptr[tid + 1] = indptr[tid] + len(rows)
        for c, ctx in rows:
            indices.append(ctx)
            data.append(c)
    return indptr, np.asarray(indices, dtype=np.int64), np.asarray(data, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terms", required=True, help="written by src/build_term_table.py")
    ap.add_argument("--tokenizer", required=True, nargs="+")
    ap.add_argument("--out", required=True, nargs="+", help="one directory per tokenizer")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--batch-size", type=int, default=1024, help="documents per unit of work")
    ap.add_argument("--limit-batches", type=int, default=0, help="for a smoke test")
    ap.add_argument("--download-only", action="store_true")
    args = ap.parse_args()
    if len(args.out) != len(args.tokenizer):
        ap.error("--out needs one directory per --tokenizer")

    print(f"[corpus_stats] c4: {args.shards} shard(s)", flush=True)
    paths = download(args.shards)
    if args.download_only:
        return
    terms = term_table(args.terms)
    print(f"[corpus_stats] {len(terms)} terms; {args.workers} workers", flush=True)

    src = batches(paths, args.batch_size)
    if args.limit_batches:
        src = itertools.islice(src, args.limit_batches)
    K = len(args.tokenizer)
    counts = [None] * K
    right = [collections.Counter() for _ in range(K)]
    n_tok = [0] * K
    pos = collections.Counter()
    n_doc = n_batch = 0
    t0 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(args.tokenizer, args.terms)) as pool:
        for per_tok, pc, nd in pool.imap_unordered(_scan, src, chunksize=1):
            for k, (u, t, tb, tc, tn, r, nt) in enumerate(per_tok):
                arrays = (u, t, tb, tc, tn)
                counts[k] = arrays if counts[k] is None else tuple(a + b for a, b in zip(counts[k], arrays))
                right[k].update(r)
                n_tok[k] += nt
            pos.update(pc)
            n_doc += nd
            n_batch += 1
            if n_batch % 200 == 0:
                dt = time.time() - t0
                print(f"  {n_doc:>10,} docs  {n_tok[0]:>13,} tokens  {n_tok[0] / dt / 1e6:.1f}M tok/s",
                      flush=True)
            if n_batch % 500 == 0:
                right = [trim(r, CTX_KEEP * 4) for r in right]
    dt = time.time() - t0
    print(f"[corpus_stats] {n_doc:,} docs in {dt:.0f}s", flush=True)

    tag_ids = {t: i for i, t in enumerate(sorted({tag for _, tag in pos}))}
    pos_pairs = np.array([(tid, tag_ids[tag]) for tid, tag in sorted(pos)],
                         dtype=np.int64).reshape(-1, 2)
    pos_counts = np.array([pos[k] for k in sorted(pos)], dtype=np.int64)
    for k, out in enumerate(args.out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        unigram, term, term_bare, term_cont, term_next = counts[k]
        rp, ri, rd = prune(right[k], len(terms), CTX_KEEP)
        np.savez_compressed(
            out / "c4.npz",
            unigram=unigram, term=term, term_bare=term_bare, term_cont=term_cont, term_next=term_next,
            ctx_right_indptr=rp, ctx_right_indices=ri, ctx_right_data=rd,
            pos_pairs=pos_pairs, pos_counts=pos_counts,
        )
        meta = {
            "corpus": "c4", "repo": C4["repo"], "files": C4["files"][: args.shards],
            "tokenizer": args.tokenizer[k], "documents": n_doc, "tokens": n_tok[k],
            "context_keep": CTX_KEEP, "pos_tags": sorted(tag_ids, key=tag_ids.get),
            "seconds": round(dt, 1), "terms": terms,
        }
        (out / "c4.meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        print(f"  wrote {out / 'c4.npz'} and {out / 'c4.meta.json'}")


if __name__ == "__main__":
    main()
