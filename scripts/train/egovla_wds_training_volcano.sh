#!/bin/bash
# Volcano multi-node launcher for native EgoVLA WDS training in DreamZero.
# It reuses EgoVLA's torchrun/RDMA/NUMA topology. DreamZero itself uses
# Transformers + DeepSpeed ZeRO-2 rather than EgoVLA's FSDP2/HSDP DeviceMesh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
MACHINE_RANK="${MACHINE_RANK:-${MLP_ROLE_INDEX:-0}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-}}"
RDMA_IFNAME="${RDMA_IFNAME:-${MLP_IFNAME:-eth0}}"
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    DEFAULT_PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
elif [ -x /usr/bin/python3.11 ]; then
    DEFAULT_PYTHON_BIN=/usr/bin/python3.11
else
    DEFAULT_PYTHON_BIN="$(command -v python3)"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

WDS_SHARDS="${WDS_SHARDS:-}"
VAL_WDS_SHARDS="${VAL_WDS_SHARDS:-}"
WDS_METADATA="${WDS_METADATA:-$PROJECT_DIR/artifacts/egovla_wds_metadata.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egovla_wds}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:?Set WAN_CKPT_DIR to Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:?Set TOKENIZER_DIR to the umt5-xxl tokenizer}"
DREAMZERO_CKPT_DIR="${DREAMZERO_CKPT_DIR:?Set DREAMZERO_CKPT_DIR to DreamZero-AgiBot}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-5000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"

if [ ! -f "$WDS_METADATA" ]; then
    echo "Missing WDS metadata: $WDS_METADATA" >&2
    echo "Run scripts/data/compute_egovla_wds_metadata.py first." >&2
    exit 2
fi
for required_dir in "$WAN_CKPT_DIR" "$TOKENIZER_DIR" "$DREAMZERO_CKPT_DIR"; do
    if [ ! -d "$required_dir" ]; then
        echo "Missing model directory: $required_dir" >&2
        exit 2
    fi
done

export PYTHON_BIN
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1
export NCCL_SOCKET_FAMILY=AF_INET
export GLOO_SOCKET_IFNAME="$RDMA_IFNAME"
export TP_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_SOCKET_IFNAME="$RDMA_IFNAME"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT

TRAIN_COMMAND=(
    groot/vla/experiment/experiment.py
    "report_to=$REPORT_TO"
    "wandb_project=$WANDB_PROJECT"
    data=dreamzero/egovla_wds_fingertips_relative
    model=dreamzero/vla
    model/dreamzero/action_head=wan_flow_matching_action_tf
    model/dreamzero/transform=dreamzero_cotrain
    train_architecture=lora
    num_frames=33
    action_horizon=24
    num_frame_per_block=2
    num_action_per_block=24
    num_state_per_block=1
    num_views=2
    max_state_dim=64
    max_action_dim=48
    max_chunk_size=4
    frame_seqlen=880
    "egovla_wds_metadata_path=$WDS_METADATA"
    wds_keep_ratio=0.1
    wds_shuffle_buffer=256
    wds_shuffle_initial=32
    "output_dir=$OUTPUT_DIR"
    "pretrained_model_path=$DREAMZERO_CKPT_DIR"
    "dit_version=$WAN_CKPT_DIR"
    "text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    "image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    "tokenizer_path=$TOKENIZER_DIR"
    "global_batch_size=$GLOBAL_BATCH_SIZE"
    "max_steps=$MAX_STEPS"
    per_device_train_batch_size=1
    learning_rate=1e-5
    weight_decay=1e-5
    warmup_ratio=0.05
    bf16=true
    tf32=true
    eval_bf16=true
    do_eval=true
    eval_strategy=steps
    eval_steps=500
    per_device_eval_batch_size=1
    dataloader_num_workers=4
    dataloader_pin_memory=false
    dataloader_persistent_workers=true
    save_strategy=steps
    save_steps=500
    save_total_limit=10
    save_lora_only=true
    upload_checkpoints=false
    training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json
    ++action_head_cfg.config.skip_component_loading=true
    ++action_head_cfg.config.defer_lora_injection=true
)

if [ -n "$WDS_SHARDS" ]; then
    TRAIN_COMMAND+=("egovla_wds_shards=$WDS_SHARDS")
fi
if [ -n "$VAL_WDS_SHARDS" ]; then
    TRAIN_COMMAND+=("egovla_wds_val_shards=$VAL_WDS_SHARDS")
fi

if [ -n "${TRAIN_ARGS:-}" ]; then
    read -r -a EXTRA_TRAIN_ARGS <<< "$TRAIN_ARGS"
    TRAIN_COMMAND+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "DreamZero Volcano launch: rank $MACHINE_RANK/$NNODES, $GPUS_PER_NODE GPUs/node"
echo "master=$MASTER_ADDR:$MASTER_PORT interface=$RDMA_IFNAME"
echo "train shards=${WDS_SHARDS:-<config defaults>}"
echo "val shards=${VAL_WDS_SHARDS:-<config defaults>}"
echo "metadata=$WDS_METADATA"
echo "output=$OUTPUT_DIR"

exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$MACHINE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --no-python \
    bash "$SCRIPT_DIR/numa_bind_wrapper.sh" \
    "${TRAIN_COMMAND[@]}"
