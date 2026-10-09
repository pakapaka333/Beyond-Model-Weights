#!/usr/bin/env bash
# Download OpenBBQ and build the evaluation and training splits under data/openbbq/.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
uv run --no-sync python src/build_openbbq.py "$@"
