#!/usr/bin/env bash
# One-time setup:
#   1. dOPSD at the pinned commit, with patches/dOPSD.patch applied
#   2. Dream's generation_utils.py with patches/dream_generation_utils.patch (block decoding)
#   3. the Python environment (uv)
#   4. the MMLU and IFEval tasks of lm-eval
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

git submodule update --init third_party/dOPSD
if git -C third_party/dOPSD apply --reverse --check ../../patches/dOPSD.patch 2>/dev/null; then
    echo "[1_setup] patches/dOPSD.patch already applied"
else
    git -C third_party/dOPSD apply ../../patches/dOPSD.patch
    echo "[1_setup] applied patches/dOPSD.patch"
fi

uv sync --group eval

DREAM_REVISION=05334cb9faaf763692dcf9d8737c642be2b2a6ae
if [ ! -f third_party/dream/generation_utils.py ]; then
    mkdir -p third_party/dream
    uv run --no-sync python - "${DREAM_REVISION}" <<'PY'
import shutil, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download("Dream-org/Dream-v0-Instruct-7B", "generation_utils.py", revision=sys.argv[1])
shutil.copy(path, "third_party/dream/generation_utils.py")
PY
    git apply --directory=third_party/dream patches/dream_generation_utils.patch
    echo "[1_setup] third_party/dream/generation_utils.py patched for block decoding"
fi

bash scripts/3_fetch_lm_eval_tasks.sh
echo "[1_setup] done"
