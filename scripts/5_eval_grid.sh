#!/usr/bin/env bash
# Usage: bash scripts/5_eval_grid.sh configs/5_eval/openbbq/<model>.yaml [flags ...]
#
# OpenBBQ over the 24 sampling configurations of the paper: block length
# B in {8, 32, 128} x tokens per step TPS in {1, 2, 4, 8} x two remasking
# strategies (LLaDA: low_confidence / random; Dream: entropy / origin), over a
# 256-token window (steps = 256 / TPS). Flags such as --adapter are passed on.
#
# Each point is written to $OUTPUT_ROOT/<remasking>_b<B>_tps<TPS>/, by default
# results/openbbq_grid/<config>[_<adapter name>]/. A point that is finished, or
# that another invocation has claimed, is skipped, so the grid can be split over
# GPUs by starting the script several times with different CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:?usage: $0 <config.yaml> [flags ...]}"
shift

if [ -z "${OUTPUT_ROOT:-}" ]; then
    OUTPUT_ROOT="results/openbbq_grid/$(basename "${CONFIG}" .yaml)"
    for ((i = 1; i <= $#; i++)); do
        [ "${!i}" = "--adapter" ] && { j=$((i + 1)); OUTPUT_ROOT+="_$(basename "${!j}")"; }
    done
fi

BACKEND=$(uv run --no-sync python -c \
    "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['model']['backend'])" "${CONFIG}")
if [ "${BACKEND}" = dream ]; then
    STRATEGIES=(entropy origin); FLAG=--alg
else
    STRATEGIES=(low_confidence random); FLAG=--remasking
fi

for strategy in "${STRATEGIES[@]}"; do
    for block in 8 32 128; do
        for tps in 1 2 4 8; do
            out="${OUTPUT_ROOT}/${strategy}_b${block}_tps${tps}"
            [ -f "${out}/metrics.json" ] && continue
            mkdir -p "${OUTPUT_ROOT}"
            mkdir "${out}.claimed" 2>/dev/null || continue
            status=0
            OUTPUT_DIR="${out}" bash scripts/5_eval.sh "${CONFIG}" \
                "${FLAG}" "${strategy}" --block-length "${block}" --steps $((256 / tps)) "$@" || status=$?
            rmdir "${out}.claimed"
            [ "${status}" -eq 0 ] || exit "${status}"
        done
    done
done
