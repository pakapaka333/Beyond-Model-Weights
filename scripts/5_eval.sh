#!/usr/bin/env bash
# Usage: bash scripts/5_eval.sh configs/5_eval/<benchmark>/<name>.yaml [flags ...]
#
# Runs src/<entrypoint>.py named by the config (eval_openbbq or run_lm_eval) on
# every visible GPU. Results go to $OUTPUT_DIR, by default
# results/<entrypoint>/<config>/<timestamp>; flags such as --adapter override the config.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
[ -f .env ] && set -a && . ./.env && set +a

CONFIG="${1:?usage: $0 <config.yaml> [flags ...]}"
shift
ENTRYPOINT=$(uv run --no-sync python -c \
    "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['entrypoint'])" "${CONFIG}")

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
fi
[ "${NUM_GPUS:-0}" -gt 0 ] || NUM_GPUS=1
if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCHER=(accelerate launch --num_processes "${NUM_GPUS}" --main_process_port "${MAIN_PROCESS_PORT:-13380}")
else
    LAUNCHER=(python)
fi

OUTPUT_DIR="${OUTPUT_DIR:-results/${ENTRYPOINT}/$(basename "${CONFIG}" .yaml)/$(date +%Y%m%d_%H%M%S)}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=true
echo "[5_eval.sh] ${ENTRYPOINT} ${CONFIG} gpus=${NUM_GPUS} -> ${OUTPUT_DIR}"
uv run --no-sync "${LAUNCHER[@]}" "src/${ENTRYPOINT}.py" \
    --config "${CONFIG}" --output-dir "${OUTPUT_DIR}" "$@"
