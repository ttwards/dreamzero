#!/usr/bin/env bash
# One-command Volcano/MLP launcher for the 8-node x 8-GPU EgoSteer LeRobot run.
# Volcano starts this script once in every worker container and injects the
# rendezvous topology consumed below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-}}"
export MACHINE_RANK="${MACHINE_RANK:-${MLP_ROLE_INDEX:-}}"
export NNODES="${NNODES:-${MLP_WORKER_NUM:-}}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-}}"
export RDMA_IFNAME="${RDMA_IFNAME:-${MLP_IFNAME:-eth0}}"

: "${MASTER_ADDR:?Volcano did not provide MLP_WORKER_0_HOST}"
: "${MASTER_PORT:?Volcano did not provide MLP_WORKER_0_PORT}"
: "${MACHINE_RANK:?Volcano did not provide MLP_ROLE_INDEX}"
: "${NNODES:?Volcano did not provide MLP_WORKER_NUM}"
: "${GPUS_PER_NODE:?Volcano did not provide MLP_WORKER_GPU}"

if [[ "$NNODES" != "8" || "$GPUS_PER_NODE" != "8" ]]; then
    echo "Expected exactly 8 nodes x 8 GPUs, received ${NNODES} x ${GPUS_PER_NODE}" >&2
    exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the Volcano Secret/environment}"

# Production data/model configuration. The user-facing launch command only
# needs WANDB_API_KEY; defaults remain overridable for controlled tests.
export NUM_GPUS="$GPUS_PER_NODE"
export DATA_CONFIG="${DATA_CONFIG:-dreamzero/dual_arm_dexterous_hand_mixture_relative}"
export DATA_ROOT_CONFIG_KEY="${DATA_ROOT_CONFIG_KEY:-egosteer_lerobot_root}"
export EGO_STEER_DATA_ROOT="${EGO_STEER_DATA_ROOT:-/efs-exp/agent-workspace/xuwenxi/datasets/realworld-dreamzero-lerobot}"
export OUTPUT_DIR="${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egosteer_lerobot_64gpu}"
export WAN_CKPT_DIR="${WAN_CKPT_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/Wan2.1-I2V-14B-480P}"
export TOKENIZER_DIR="${TOKENIZER_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/umt5-xxl}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/efs-exp/agent-workspace/xuwenxi/checkpoints/DreamZero-AgiBot}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero3_hpz8.json}"

# One fixed four-chunk context per GPU. Global batch 128 gives two gradient
# accumulation microsteps across 64 ranks, matching the prior production run.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
export DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"

export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
export DO_EVAL="${DO_EVAL:-true}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
export EVAL_STEPS="${EVAL_STEPS:-500}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
export SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
export SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-false}"

# Compile only repeated Wan blocks. ZeRO-3 communication hooks remain eager at
# module boundaries, preserving the node-local hpZ partition of eight ranks.
export TORCH_COMPILE="${TORCH_COMPILE:-true}"
export TORCH_COMPILE_SCOPE="${TORCH_COMPILE_SCOPE:-wan_blocks}"
export TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}"
export TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-auto}"
export TORCH_COMPILE_FULLGRAPH="${TORCH_COMPILE_FULLGRAPH:-true}"
export TORCH_COMPILE_DIAGNOSTICS="${TORCH_COMPILE_DIAGNOSTICS:-false}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/dreamzero-inductor-cache}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-12}"
export TORCHINDUCTOR_FALLBACK_RANDOM="${TORCHINDUCTOR_FALLBACK_RANDOM:-true}"
export TORCHINDUCTOR_AUTOTUNE_POINTWISE="${TORCHINDUCTOR_AUTOTUNE_POINTWISE:-false}"

# Profile rank 0 exactly once after compilation and steady-state warmup.
# ProfCallback uses repeat=1 and removes itself after writing the trace.
export TORCH_PROFILE="${TORCH_PROFILE:-true}"
export PROFILE_START_STEP="${PROFILE_START_STEP:-100}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
export PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-2}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export PROFILE_DIR="${PROFILE_DIR:-$OUTPUT_DIR/profiling}"
export TEARDOWN_CUPTI="${TEARDOWN_CUPTI:-1}"

export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="$RDMA_IFNAME"
export TP_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
# Keep the legacy project knob and set the PyTorch ProcessGroupNCCL watchdog
# knob explicitly; the latter is what controls heartbeat timeout in torch 2.8.
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export MALLOC_TRIM_THRESHOLD_=0
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_ARENA_MAX=2

echo "DreamZero EgoSteer LeRobot Volcano launch"
echo "topology: ${NNODES} nodes x ${GPUS_PER_NODE} GPUs = $((NNODES * GPUS_PER_NODE)) ranks"
echo "machine rank: $MACHINE_RANK"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "network interface: $RDMA_IFNAME"
echo "data mixture root: $EGO_STEER_DATA_ROOT"
echo "output: $OUTPUT_DIR"
echo "global batch: $GLOBAL_BATCH_SIZE"
echo "compile: $TORCH_COMPILE scope=$TORCH_COMPILE_SCOPE"
echo "one-shot profiler: $TORCH_PROFILE start=$PROFILE_START_STEP warmup=$PROFILE_WARMUP_STEPS active=$PROFILE_ACTIVE_STEPS"

exec bash "$SCRIPT_DIR/egosteer_lerobot_training.sh"
