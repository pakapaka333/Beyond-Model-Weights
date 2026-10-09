"""Bias groups and the fairness term shared by the SFT and dOPSD trainers.

A row's groups are its own option pair followed by every lexicon group registered
under its `category`. The option pair is represented by the first token at which
the two options' tokenizations (with a leading space) diverge:

    ' Black woman' -> [' Black', ' woman']
    ' Black man'   -> [' Black', ' man']     -> (' woman', ' man')

A pair where one option's tokens are a prefix of the other's has no such token
and is skipped. Groups are padded to a fixed [groups, terms] shape with a
per-term mask, so every statistic below runs over the real terms only.
"""

import json

import torch

FAIR_COLUMNS = ("options", "unknown_index", "condition", "category")
FAIR_TERMS = ("log_ratio", "js_div", "logprob_var")
AMBIGUOUS = "ambiguous"


class OptionPair:
    """Resolves an OpenBBQ row to the vocabulary entries of its two non-unknown options."""

    num_pairs = 1

    def __init__(self, tokenizer, ambiguous_only: bool = True):
        self.tokenizer = tokenizer
        self.ambiguous_only = bool(ambiguous_only)
        self.pad_id = tokenizer.pad_token_id
        self._cache = {}

    def _seq(self, option: str):
        if option not in self._cache:
            self._cache[option] = self.tokenizer(" " + option, add_special_tokens=False)["input_ids"]
        return self._cache[option]

    def enabled(self, feature) -> bool:
        # In a disambiguated row one option is the gold answer, so it is not equalized.
        if self.ambiguous_only and feature.get("condition") != AMBIGUOUS:
            return False
        options = feature.get("options")
        unknown_index = feature.get("unknown_index")
        if not options or unknown_index is None:
            return False
        return len([i for i in range(len(options)) if i != unknown_index]) == 2

    def __call__(self, feature) -> list:
        if not self.enabled(feature):
            return [None]
        options, unknown_index = feature["options"], feature["unknown_index"]
        idx_a, idx_b = [i for i in range(len(options)) if i != unknown_index]
        a, b = self._seq(options[idx_a]), self._seq(options[idx_b])
        k = 0
        while k < len(a) and k < len(b) and a[k] == b[k]:
            k += 1
        if k >= len(a) or k >= len(b):
            return [None]
        pair = (a[k], b[k])
        if pair[0] == pair[1] or (self.pad_id is not None and self.pad_id in pair):
            return [None]
        return [pair]


class BiasLexicon:
    """The groups of a lexicon built by src/build_bias_lexicon.py, keyed by category.

    Only groups whose every term is a single vocabulary entry are kept.
    """

    def __init__(self, path, tokenizer):
        doc = json.loads(open(path).read())
        if "axes" not in doc or "tokenizer" not in doc:
            raise ValueError(f"{path} is not a resolved lexicon (no token ids).")
        if doc.get("vocab_size") != len(tokenizer):
            name = getattr(tokenizer, "name_or_path", "")
            raise ValueError(
                f"{path} was built for {doc['tokenizer']!r} (vocab {doc.get('vocab_size')}), "
                f"but the tokenizer is {name!r} (vocab {len(tokenizer)}). "
                f"Rebuild it with `--tokenizer {name}`."
            )
        self.path = str(path)
        self.by_axis = {}
        for axis, groups in doc["axes"].items():
            kept = []
            for g in groups:
                if not g["exact"]:
                    continue
                ids = tuple(dict.fromkeys(g["token_ids"]))
                if len(ids) >= 2 and tokenizer.pad_token_id not in ids:
                    kept.append(ids)
            self.by_axis[axis] = kept
        self.max_groups = max((len(v) for v in self.by_axis.values()), default=0)
        self.max_terms = max((len(g) for v in self.by_axis.values() for g in v), default=0)
        self.num_groups = sum(len(v) for v in self.by_axis.values())

    def __call__(self, feature) -> list:
        return self.by_axis.get(feature.get("category"), ())


class BiasGroups:
    """One row's group list: the option pair first, then its category's lexicon groups."""

    def __init__(self, pairs: OptionPair, lexicon: BiasLexicon | None = None):
        self.pairs = pairs
        self.lexicon = lexicon
        self.pad_id = pairs.pad_id
        self.num_groups = pairs.num_pairs + (lexicon.max_groups if lexicon else 0)
        self.max_terms = max(2, lexicon.max_terms if lexicon else 2)

    def __call__(self, feature) -> list:
        groups = list(self.pairs(feature))
        if self.lexicon is not None and self.pairs.enabled(feature):
            groups.extend(self.lexicon(feature))
        groups.extend([None] * (self.num_groups - len(groups)))
        return groups[: self.num_groups]


def build_fair_groups(tokenizer, lexicon_path=None, ambiguous_only=True) -> BiasGroups:
    lexicon = BiasLexicon(lexicon_path, tokenizer) if lexicon_path else None
    return BiasGroups(OptionPair(tokenizer, ambiguous_only), lexicon)


def group_tensors(features, groups: BiasGroups):
    """[B, P, K] token ids, [B, P] group mask and [B, P, K] term mask."""
    B, P, K = len(features), groups.num_groups, groups.max_terms
    ids = torch.full((B, P, K), groups.pad_id, dtype=torch.long)
    gmask = torch.zeros((B, P), dtype=torch.long)
    tmask = torch.zeros((B, P, K), dtype=torch.long)
    for i, feat in enumerate(features):
        for p, group in enumerate(groups(feat)):
            if group is None:
                continue
            n = len(group)
            ids[i, p, :n] = torch.tensor(group, dtype=torch.long)
            gmask[i, p] = 1
            tmask[i, p, :n] = 1
    return ids, gmask, tmask


def fair_penalty_term(logits, rows, ids, mask, term_mask, term="logprob_var"):
    """The fairness term V of the paper, averaged over groups and then over positions.

    `logits` [M, V] are the student's logits at the M scored positions and `rows`
    [M] maps each position to its example.

      log_ratio    Qian et al. (2019): mean |log p_i - log p_j| over the group's pairs
      js_div       Zhao et al. (2024): JS(r || uniform), r = the within-group shares
      logprob_var  the variance of the group's log-probabilities
    """
    M = max(1, logits.shape[0])
    idx = ids[rows]
    m = mask[rows].bool()
    tm = term_mask[rows].float()
    Mm, P, K = idx.shape
    flat = idx.reshape(Mm, P * K)
    lp = logits.gather(1, flat).reshape(Mm, P, K) - logits.logsumexp(-1)[:, None, None]
    nt = tm.sum(-1).clamp(min=1)

    real = tm.bool()
    if term == "logprob_var":
        centred = lp - (lp * tm).sum(-1, True) / nt.unsqueeze(-1)
        val = (centred.pow(2) * tm).sum(-1) / nt
    elif term == "log_ratio":
        # sum_{i<j} |x_j - x_i| = sum_k x_(k) (2k - n - 1) over the sorted real terms.
        xs = torch.where(real, lp, lp.new_full((), float("inf"))).sort(-1).values
        k = torch.arange(1, K + 1, device=lp.device, dtype=lp.dtype)
        cnt = tm.sum(-1, keepdim=True)
        w = torch.where(k <= cnt, 2 * k - cnt - 1, torch.zeros_like(k))
        xs = torch.where(k <= cnt, xs, torch.zeros_like(xs))
        val = (xs * w).sum(-1) / (nt * (nt - 1) / 2).clamp(min=1)
    elif term == "js_div":
        # Padded slots stay finite so no NaN reaches the backward pass through the mask.
        lse = torch.where(real, lp, lp.new_full((), float("-inf"))).logsumexp(-1, keepdim=True)
        lse = torch.where(torch.isfinite(lse), lse, torch.zeros_like(lse))
        lq = lp - lse
        q = lq.exp() * tm
        u = 1.0 / nt.unsqueeze(-1)
        lm = torch.log(((q + u * tm) / 2).clamp_min(1e-30))
        kl_q = (q * (lq - lm) * tm).sum(-1)
        kl_u = (u * (-torch.log(nt.unsqueeze(-1)) - lm) * tm).sum(-1)
        val = 0.5 * (kl_q + kl_u)
    else:
        raise ValueError(f"fair_term must be one of {FAIR_TERMS}, got {term!r}.")

    w = m.float()
    wsum = w.sum(-1)
    per_pos = (val * w).sum(-1) / wsum.clamp(min=1)
    return per_pos.sum() / float(M)
