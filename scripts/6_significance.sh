#!/usr/bin/env bash
# Usage: bash scripts/6_significance.sh configs/6_significance/<spec>.yaml [--out <json>]
# Paired cluster bootstrap of tuned models against the base model (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")/.."
SPEC="${1:?usage: $0 <spec.yaml> [--out <json>]}"
shift
uv run --no-sync python src/significance.py --spec "${SPEC}" "$@"
