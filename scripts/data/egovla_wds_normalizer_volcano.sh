#!/bin/bash
# Compute production normalization statistics from train VLA shards only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
DATA_CONFIG="${DATA_CONFIG:-$PROJECT_DIR/groot/vla/configs/data/dreamzero/egovla_wds_fingertips_relative.yaml}"
OUTPUT="${OUTPUT:-$PROJECT_DIR/artifacts/egovla_wds_metadata_full.json}"

exec "$PYTHON_BIN" -u "$SCRIPT_DIR/compute_egovla_wds_metadata.py" \
    --data-config "$DATA_CONFIG" \
    --output "$OUTPUT" \
    --action-horizon 24 \
    --anchor-stride 1 \
    --reservoir-size 500000 \
    --seed 42
