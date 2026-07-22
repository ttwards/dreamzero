#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
NORMALIZER_PID="${NORMALIZER_PID:-8622}"
POLL_SECONDS="${POLL_SECONDS:-60}"
METADATA="${METADATA:-$PROJECT_DIR/artifacts/egovla_wds_metadata_full.json}"
NORMALIZER_LOG="${NORMALIZER_LOG:-$PROJECT_DIR/artifacts/egovla_wds_normalizer.log}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/efs-exp/agent-workspace/xuwenxi/checkpoints/umt5-xxl}"
TRAIN_SHARD="${TRAIN_SHARD:-/efs-exp/yeyuyao/real_world_wds_chest/real_world_with_cn_0520/train/shard-Put_glasses_into_case-w0013-000003.tar}"
VAL_SHARD="${VAL_SHARD:-/efs-exp/yeyuyao/real_world_wds_chest/real_world_with_cn_0520/val/shard-Cork_wine_test_tube-w0015-000000.tar}"

while kill -0 "$NORMALIZER_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
done

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -s "$METADATA" ]]; then
    echo "Normalizer exited without producing non-empty metadata: $METADATA" >&2
    exit 1
fi
if grep -Eiq "traceback|error|exception|killed|oom|out of memory" "$NORMALIZER_LOG"; then
    echo "Normalizer log contains an error marker: $NORMALIZER_LOG" >&2
    grep -Ein "traceback|error|exception|killed|oom|out of memory" "$NORMALIZER_LOG" >&2
    exit 1
fi

"$PYTHON_BIN" scripts/data/validate_egovla_wds_metadata.py "$METADATA"
"$PYTHON_BIN" scripts/data/smoke_test_egovla_wds.py \
    --metadata "$METADATA" \
    --train-shard "$TRAIN_SHARD" \
    --val-shard "$VAL_SHARD" \
    --tokenizer-path "$TOKENIZER_PATH"

echo "EgoVLA full-metadata postcheck passed."
