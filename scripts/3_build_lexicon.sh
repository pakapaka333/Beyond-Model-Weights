#!/usr/bin/env bash
# Build the bias group lexicon for LLaDA-8B-Instruct / LLaDA-1.5 (which share a
# tokenizer) and Dream-v0-Instruct-7B (App. A.3):
#   terms.json -> C4 statistics -> name/place groups -> source groups (stage 1)
#   -> regrouping (stage 2) -> data/lexicon/<llada|dream>/regrouped/lexicon.json
# The C4 pass downloads 64 shards (about 20 GB), needs only CPUs, counts both
# tokenizers at once, and is skipped when its output exists.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
PY=(uv run --no-sync python)

[ -d data/lexicon/_raw ] || bash scripts/3_fetch_lexicon_sources.sh
[ -f data/openbbq/train-pool/train.jsonl ] || bash scripts/2_build_openbbq.sh
"${PY[@]}" -m nltk.downloader -q averaged_perceptron_tagger_eng

[ -f data/lexicon/terms.json ] || "${PY[@]}" src/build_term_table.py --out data/lexicon/terms.json
if [ ! -f data/corpus_stats/llada/c4.npz ] || [ ! -f data/corpus_stats/dream/c4.npz ]; then
    "${PY[@]}" src/build_corpus_stats.py --terms data/lexicon/terms.json \
        --tokenizer GSAI-ML/LLaDA-8B-Instruct Dream-org/Dream-v0-Instruct-7B \
        --out data/corpus_stats/llada data/corpus_stats/dream
fi

for M in llada dream; do
    case "${M}" in
        llada) TOKENIZER=GSAI-ML/LLaDA-8B-Instruct ;;
        dream) TOKENIZER=Dream-org/Dream-v0-Instruct-7B ;;
    esac
    STATS=data/corpus_stats/${M}
    LEX=data/lexicon/${M}
    "${PY[@]}" src/build_generated_groups.py --stats "${STATS}" --tokenizer "${TOKENIZER}" \
        --out "${LEX}/generated_groups.json"
    "${PY[@]}" src/build_bias_lexicon.py --out "${LEX}/source" --tokenizer "${TOKENIZER}" \
        --generated "${LEX}/generated_groups.json" --skip-source openbbq
    "${PY[@]}" src/auto_regroup.py --lexicon "${LEX}/source/groups.json" --stats "${STATS}" \
        --tokenizer "${TOKENIZER}" --out "${LEX}/regrouped_board.json"
    "${PY[@]}" src/build_bias_lexicon.py --out "${LEX}/regrouped" --tokenizer "${TOKENIZER}" \
        --regrouped "${LEX}/regrouped_board.json"
    echo "[build_lexicon] ${LEX}/regrouped/lexicon.json"
done
