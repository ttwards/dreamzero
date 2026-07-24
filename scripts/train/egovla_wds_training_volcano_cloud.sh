#!/bin/bash
# One-command Volcano/MLP launcher for the 8-node x 8-GPU DreamZero run.
#
# This script is started once inside every worker container. Volcano MLP
# supplies the topology variables; the underlying launcher starts 8 local
# torchrun processes and forms one 64-rank process group.
#
# Required user configuration:
#   WANDB_API_KEY=<secret> bash scripts/train/egovla_wds_training_volcano_cloud.sh
#
# The W&B key must be injected into every worker container, preferably by a
# Volcano Secret. It is intentionally never printed by this script.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Volcano MLP topology. Explicit values still take precedence, which makes
# this wrapper usable with an equivalent container-cloud job definition.
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

if [[ -z "$GPUS_PER_NODE" ]]; then
    GPUS_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    export GPUS_PER_NODE
fi
: "${GPUS_PER_NODE:?Could not determine the number of GPUs per node}"

if [[ "$NNODES" != "8" || "$GPUS_PER_NODE" != "8" ]]; then
    echo "This launcher is for exactly 8 nodes x 8 GPUs (received ${NNODES} x ${GPUS_PER_NODE})." >&2
    echo "Use scripts/train/egovla_wds_training_volcano.sh for a different topology." >&2
    exit 2
fi

: "${WANDB_API_KEY:?Set WANDB_API_KEY in the Volcano Secret/environment before starting the job}"

# Paper-style DreamZero full fine-tuning defaults.
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export MAX_STEPS="${MAX_STEPS:-100000}"
export TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
export SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-false}"
# ZeRO++ HPZ partitions parameters over the eight local ranks and replicates
# those shards across nodes. Gradients and optimizer states remain globally
# partitioned. Keep the optimizer on GPU for the 64-rank production run;
# zero3_hpz8_offload.json remains available for memory-constrained smoke tests.
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero3_hpz8.json}"
export NVIMGCODEC_DECODE="${NVIMGCODEC_DECODE:-true}"
# "grouped" keeps PyTorch's A800-optimized FlashAttention kernel while
# batching query groups that share K/V. "flex" remains available for A/B.
export TEACHER_FORCING_ATTN_BACKEND="${TEACHER_FORCING_ATTN_BACKEND:-grouped}"
# Compile the complete VLA module through TrainingArguments -> Accelerate ->
# DeepSpeedEngine.compile(). Keep graph breaks enabled because preprocessing
# and logging contain Python control flow; the fixed training shapes remain
# specialized for Inductor.
export TORCH_COMPILE="${TORCH_COMPILE:-true}"
export TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
export TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}"
export TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-false}"
export TORCH_COMPILE_FULLGRAPH="${TORCH_COMPILE_FULLGRAPH:-false}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/dreamzero-inductor-cache}"
# This setting is per rank. Eight local ranks times 12 workers gives a
# 96-worker whole-node compilation budget without spawning the default
# 8 x 32 = 256 mostly-idle workers.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-12}"
# PyTorch 2.8 + ZeRO-3 DeepCompile can leave symbolic randn nodes that miss
# Inductor's random rewrite. Only those small RNG kernels use ATen fallback;
# the complete VLA graph still goes through Inductor.
export TORCHINDUCTOR_FALLBACK_RANDOM="${TORCHINDUCTOR_FALLBACK_RANDOM:-true}"
# Runtime autotuning clones mutated pointwise inputs.  The Wan MLP bias-add
# output is roughly 420 MiB, which is enough to OOM the compiled forward on an
# 80 GiB A800.  Use the fixed pointwise launch heuristic; GEMM selection and
# the rest of Inductor remain enabled.
export TORCHINDUCTOR_AUTOTUNE_POINTWISE="${TORCHINDUCTOR_AUTOTUNE_POINTWISE:-false}"
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
export DREAMZERO_RUNTIME_DIR="${DREAMZERO_RUNTIME_DIR:-/opt/dreamzero-runtime}"
# Capture one short rank-0 trace by default. Disable with TORCH_PROFILE=false.
export TORCH_PROFILE="${TORCH_PROFILE:-true}"
export PROFILE_START_STEP="${PROFILE_START_STEP:-50}"
export PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
export PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-2}"
export PROFILE_RANKS="${PROFILE_RANKS:-0}"
export PROFILE_UPLOAD_WANDB="${PROFILE_UPLOAD_WANDB:-true}"

if [[ ! -d "$DREAMZERO_RUNTIME_DIR/train-site" ||
      ! -d "$DREAMZERO_RUNTIME_DIR/torch-site" ||
      ! -d "$DREAMZERO_RUNTIME_DIR/local-site" ]]; then
    echo "The worker image is missing the local DreamZero runtime: $DREAMZERO_RUNTIME_DIR" >&2
    echo "Rebuild the worker image from the prepared development machine before launching." >&2
    exit 2
fi

if (( GLOBAL_BATCH_SIZE % 64 != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by 64 when per-device batch size is 1." >&2
    exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / 64))

if [[ "$REPORT_TO" == "wandb" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY is required when REPORT_TO=wandb}"
fi

echo "DreamZero Volcano cloud launch"
echo "topology: ${NNODES} nodes x ${GPUS_PER_NODE} GPUs = 64 ranks"
echo "machine rank: ${MACHINE_RANK}"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "network interface: ${RDMA_IFNAME}"
echo "global batch size: ${GLOBAL_BATCH_SIZE}"
echo "per-device batch size: 1"
echo "gradient accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "architecture: ${TRAIN_ARCHITECTURE}"
echo "deepspeed config: ${DEEPSPEED_CONFIG}"
echo "nvImageCodec decode: ${NVIMGCODEC_DECODE}"
echo "teacher-forcing attention: ${TEACHER_FORCING_ATTN_BACKEND}"
echo "torch compile: ${TORCH_COMPILE} (${TORCH_COMPILE_BACKEND}/${TORCH_COMPILE_MODE}, dynamic=${TORCH_COMPILE_DYNAMIC}, fullgraph=${TORCH_COMPILE_FULLGRAPH})"
echo "torch inductor compile workers: ${TORCHINDUCTOR_COMPILE_THREADS}/rank ($((TORCHINDUCTOR_COMPILE_THREADS * GPUS_PER_NODE))/node)"
echo "torch inductor random fallback: ${TORCHINDUCTOR_FALLBACK_RANDOM}"
echo "torch inductor pointwise autotune: ${TORCHINDUCTOR_AUTOTUNE_POINTWISE}"
echo "local runtime: ${DREAMZERO_RUNTIME_DIR}"
echo "torch extensions: ${TORCH_EXTENSIONS_DIR:-$DREAMZERO_RUNTIME_DIR/torch-extensions}"
echo "output: ${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egovla_wds}"

exec bash "$SCRIPT_DIR/egovla_wds_training_volcano.sh"
