#!/usr/bin/env bash
# Usage: bash scripts/4_train.sh configs/4_train/<model>/<arm>.yaml [--flag value ...]
#
# The config's file name picks dOPSD's entry point: sft*.yaml -> sft_train_dllm.py,
# dopsd*.yaml -> opsd_dllm_trajectory_train.py. Extra flags override the config.
# Uses every visible GPU; the LoRA adapter is written to outputs/<model>/<arm>/.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"
[ -f .env ] && set -a && . ./.env && set +a

CONFIG="${1:?usage: $0 <config.yaml> [--flag value ...]}"
CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
shift
case "$(basename "${CONFIG}")" in
    sft*)   ENTRY_POINT=sft_train_dllm.py ;;
    dopsd*) ENTRY_POINT=opsd_dllm_trajectory_train.py ;;
    *) echo "$0: the config name must start with sft or dopsd" >&2; exit 1 ;;
esac

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
fi
[ "${NUM_GPUS:-0}" -gt 0 ] || NUM_GPUS=1
GRAD_ACCUM=$(uv run --no-sync python -c \
    "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('gradient_accumulation_steps', 1))" "${CONFIG}")

export PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1 TOKENIZERS_PARALLELISM=false
echo "[4_train.sh] ${ENTRY_POINT} ${CONFIG} gpus=${NUM_GPUS}"
cd third_party/dOPSD
uv run --project "${REPO_ROOT}" --no-sync accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "${NUM_GPUS}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --main_process_port "${MAIN_PROCESS_PORT:-13379}" \
    "${ENTRY_POINT}" --config "${CONFIG}" "$@"
