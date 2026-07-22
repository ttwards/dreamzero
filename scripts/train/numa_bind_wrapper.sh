#!/bin/bash
# Per-rank NUMA binding adapted from ../EgoVLA/scripts/numa_bind_wrapper.sh.
set -euo pipefail

LOCAL_RANK="${LOCAL_RANK:-0}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.11}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 127
fi

GPU_PCI="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$LOCAL_RANK" \
    | tr 'A-Z' 'a-z' | cut -c5-)"
NUMA_PATH="/sys/bus/pci/devices/${GPU_PCI}/numa_node"
if [ -r "$NUMA_PATH" ]; then
    NUMA_NODE="$(<"$NUMA_PATH")"
else
    NUMA_NODE=-1
fi
if [ "$NUMA_NODE" -lt 0 ]; then
    NUMA_NODE=$((LOCAL_RANK / 4))
fi

if command -v numactl >/dev/null 2>&1; then
    echo "[rank $LOCAL_RANK] NUMA $NUMA_NODE (GPU $GPU_PCI)" >&2
    exec numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
        "$PYTHON_BIN" -u "$@"
fi

echo "[rank $LOCAL_RANK] numactl unavailable; running without NUMA binding" >&2
exec "$PYTHON_BIN" -u "$@"
