"""Group mass and skew: how much probability the model puts on a group's terms,
and how unevenly it splits it (App. B.1).

The answer block of `gen_length` positions is fully masked and read once,
conditioned on the prompt alone. For a measured position i and a group G,

    m_i(G) = sum_g p_i(g)                 r_i(g) = p_i(g) / m_i(G)
    mass   = mean over (i, G) of m_i(G)
    skew   = sum_{i,G} m_i(G) TV(r_i, u) / (1 - 1/|G|)  /  sum_{i,G} m_i(G)

The row's option pair and the lexicon groups of its category are reported apart.
"""

import torch

from openbbq_fairness import BiasGroups, BiasLexicon, OptionPair, group_tensors

TINY = 1e-30
MASK_IDS = {"llada": 126336, "dream": 151666}
LOGIT_CHUNK = 256
FIELDS = ("active", "mass", "mass_skew")


def position_stats(logprobs, rows, ids, group_mask, term_mask):
    """Per (measured position, group) mass and mass-weighted skew."""
    idx = ids[rows]
    active = group_mask[rows].float()
    tf = term_mask[rows].float()
    N, P, K = idx.shape
    p = logprobs.gather(1, idx.reshape(N, P * K)).reshape(N, P, K).exp() * tf
    mass = p.sum(-1)
    nt = tf.sum(-1).clamp(min=1)
    share = p / mass.clamp_min(TINY).unsqueeze(-1)
    deviation = (share - tf / nt.unsqueeze(-1)).abs().sum(-1) / 2
    skew = deviation / (1 - 1 / nt).clamp_min(TINY)
    return {"active": active, "mass": mass * active, "mass_skew": mass * skew * active}


def row_totals(stats, rows, n_rows, groups=slice(None), prefix=""):
    out = {}
    for key, value in stats.items():
        acc = value.new_zeros(n_rows)
        acc.index_add_(0, rows, value[:, groups].sum(-1))
        out[prefix + key] = acc
    return out


def build_groups(tokenizer, lexicon_path=None) -> BiasGroups:
    """Disambiguated rows are measured too; the two conditions are reported apart."""
    lexicon = BiasLexicon(lexicon_path, tokenizer) if lexicon_path else None
    return BiasGroups(OptionPair(tokenizer, ambiguous_only=False), lexicon)


def forward_logits(model, input_ids, attention_mask, backend):
    """Logits aligned so `out[:, i]` is the distribution at position i."""
    if backend == "dream":
        raw = model(input_ids=input_ids, attention_mask=attention_mask[:, None, None, :].bool()).logits
        return torch.cat([raw[:, :1], raw[:, :-1]], dim=1)
    return model(input_ids, attention_mask=attention_mask).logits


@torch.no_grad()
def measure_rows(model, tokenizer, rows, groups, *, backend="llada", batch_size=8, gen_length=256,
                 progress=None):
    """Per-row sums of the `position_stats` fields, for the option pair and the lexicon."""
    device = model.device
    mask_id = MASK_IDS[backend]
    n_pair = groups.pairs.num_pairs
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": r["problem"]}],
                                          tokenize=False, add_generation_prompt=True)
            for r in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left",
                        add_special_tokens=False)
        prompt_ids = enc["input_ids"].to(device)
        prompt_mask = enc["attention_mask"].to(device)
        block = torch.full((len(batch), gen_length), mask_id, dtype=torch.long, device=device)
        block_mask = torch.ones_like(block)

        ids, gmask, tmask = (t.to(device) for t in group_tensors(batch, groups))
        measured = block_mask.bool() & gmask[:, :n_pair].any(-1)[:, None]
        totals = None
        if bool(measured.any()):
            logits = forward_logits(model, torch.cat([prompt_ids, block], 1),
                                    torch.cat([prompt_mask, block_mask], 1), backend)
            at = logits[:, prompt_ids.shape[1]:, :][measured]
            where = measured.nonzero(as_tuple=False)[:, 0]
            del logits
            for i in range(0, at.shape[0], LOGIT_CHUNK):
                sel = at[i : i + LOGIT_CHUNK].float()
                lp = sel - sel.logsumexp(-1, keepdim=True)
                at_rows = where[i : i + LOGIT_CHUNK]
                stats = position_stats(lp, at_rows, ids, gmask, tmask)
                chunk = row_totals(stats, at_rows, len(batch), slice(0, n_pair))
                if groups.lexicon is not None:
                    chunk.update(row_totals(stats, at_rows, len(batch), slice(n_pair, None), "lex_"))
                totals = chunk if totals is None else {k: totals[k] + v for k, v in chunk.items()}
            del at

        for i, row in enumerate(batch):
            entry = {k: row[k] for k in ("custom_id", "category", "condition")}
            entry.update({k: float(v[i]) for k, v in (totals or {}).items()})
            out.append(entry)
        if progress:
            progress(min(start + batch_size, len(rows)), len(rows))
    return out


def _aggregate(per_row: list[dict], prefix: str) -> dict:
    buckets = {"per_condition": {}, "overall": {}}
    for row in per_row:
        for where, key in (("per_condition", row["condition"]), ("overall", "all")):
            acc = buckets[where].setdefault(key, {})
            acc["n_rows"] = acc.get("n_rows", 0) + 1
            for field in (prefix + f for f in FIELDS):
                if field in row:
                    acc[field] = acc.get(field, 0.0) + row[field]

    def finish(acc: dict) -> dict:
        n_active, mass = acc.get(prefix + "active", 0.0), acc.get(prefix + "mass", 0.0)
        return {
            "n_rows": acc["n_rows"],
            "n_active": n_active,
            "group_mass": mass / n_active if n_active else 0.0,
            "skew": acc.get(prefix + "mass_skew", 0.0) / mass if mass else 0.0,
        }

    out = {where: {key: finish(acc) for key, acc in sorted(group.items())}
           for where, group in buckets.items()}
    out["overall"] = out["overall"]["all"]
    return out


def aggregate(per_row: list[dict]) -> dict:
    out = {"option_pair": _aggregate(per_row, "")}
    if any(k.startswith("lex_") for row in per_row for k in row):
        out["lexicon"] = _aggregate(per_row, "lex_")
    return out
