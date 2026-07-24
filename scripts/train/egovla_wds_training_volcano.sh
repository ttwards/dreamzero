#!/bin/bash
# Volcano multi-node launcher for native EgoVLA WDS training in DreamZero.
# It reuses EgoVLA's torchrun/RDMA/NUMA topology. DreamZero itself uses
# Transformers + DeepSpeed rather than EgoVLA's FSDP2/HSDP DeviceMesh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-}}"
MACHINE_RANK="${MACHINE_RANK:-${MLP_ROLE_INDEX:-}}"
NNODES="${NNODES:-${MLP_WORKER_NUM:-}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${MLP_WORKER_GPU:-}}"
RDMA_IFNAME="${RDMA_IFNAME:-${MLP_IFNAME:-eth0}}"

: "${MASTER_ADDR:?Set MASTER_ADDR or use a Volcano MLP worker task}"
: "${MASTER_PORT:?Set MASTER_PORT or use a Volcano MLP worker task}"
: "${MACHINE_RANK:?Set MACHINE_RANK or use a Volcano MLP worker task}"
: "${NNODES:?Set NNODES or use a Volcano MLP worker task}"
if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi

LOCAL_RUNTIME_DIR="${DREAMZERO_RUNTIME_DIR:-/opt/dreamzero-runtime}"
if [[ -d "$LOCAL_RUNTIME_DIR/train-site" &&
      -d "$LOCAL_RUNTIME_DIR/torch-site" &&
      -d "$LOCAL_RUNTIME_DIR/local-site" ]]; then
    # Volcano workers are cloned from the development image. Keep the large
    # Python packages on the image's local disk instead of importing them from
    # EFS. Ignore a generic inherited PYTHON_BIN here: some cloud images set it
    # to a bare Conda interpreter that does not contain torch.
    DEFAULT_PYTHON_BIN=/root/miniconda3/bin/python3
    PYTHON_BIN="${DREAMZERO_PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
    export PYTHONPATH="$LOCAL_RUNTIME_DIR/train-site:$LOCAL_RUNTIME_DIR/torch-site:$LOCAL_RUNTIME_DIR/local-site:${PYTHONPATH:-}"
    export LD_LIBRARY_PATH="$LOCAL_RUNTIME_DIR/torch-site/nvidia/nccl/lib:$LOCAL_RUNTIME_DIR/torch-site/nvidia/cusparselt/lib:$LOCAL_RUNTIME_DIR/local-site/nvidia/nvjpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
    DEFAULT_PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
    PYTHON_BIN="${DREAMZERO_PYTHON_BIN:-${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}}"
else
    echo "Missing local DreamZero runtime: $LOCAL_RUNTIME_DIR" >&2
    echo "The worker image must contain the runtime installed on the development machine." >&2
    exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
fi

if ! "$PYTHON_BIN" -c \
    'import torch, torchvision, transformers, accelerate, deepspeed, wandb; from nvidia import nvimgcodec'; then
    echo "DreamZero training dependency preflight failed with: $PYTHON_BIN" >&2
    echo "runtime=$LOCAL_RUNTIME_DIR" >&2
    exit 2
fi

WDS_SHARDS="${WDS_SHARDS:-}"
VAL_WDS_SHARDS="${VAL_WDS_SHARDS:-}"
WDS_METADATA="${WDS_METADATA:-$PROJECT_DIR/artifacts/egovla_wds_metadata_full.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egovla_wds}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/umt5-xxl}"
DREAMZERO_CKPT_DIR="${DREAMZERO_CKPT_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/DreamZero-AgiBot}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-100000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
TRAIN_ARCHITECTURE="${TRAIN_ARCHITECTURE:-full}"
SAVE_LORA_ONLY="${SAVE_LORA_ONLY:-false}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-groot/vla/configs/deepspeed/zero2_offload.json}"
TEACHER_FORCING_ATTN_BACKEND="${TEACHER_FORCING_ATTN_BACKEND:-fragmented}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}"
TORCH_COMPILE_DYNAMIC="${TORCH_COMPILE_DYNAMIC:-false}"
TORCH_COMPILE_FULLGRAPH="${TORCH_COMPILE_FULLGRAPH:-false}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/dreamzero-inductor-cache}"
TORCH_PROFILE="${TORCH_PROFILE:-false}"
PROFILE_START_STEP="${PROFILE_START_STEP:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-2}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_UPLOAD_WANDB="${PROFILE_UPLOAD_WANDB:-true}"
PROFILE_DIR="${PROFILE_DIR:-/tmp/dreamzero-profiler/${OUTPUT_DIR##*/}}"

if [[ "$TEACHER_FORCING_ATTN_BACKEND" == "flex" ]] && \
   ! "$PYTHON_BIN" -c \
       'import jinja2; from torch.nn.attention.flex_attention import flex_attention, create_block_mask'; then
    echo "FlexAttention preflight failed with: $PYTHON_BIN" >&2
    echo "Install jinja2 in the local DreamZero runtime or set TEACHER_FORCING_ATTN_BACKEND=fragmented." >&2
    exit 2
fi

if [ "$REPORT_TO" = "wandb" ]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY=... when launching the Volcano job}"
fi

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
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "Missing DeepSpeed config: $DEEPSPEED_CONFIG" >&2
    exit 2
fi
if [ "${PATCH_DEEPSPEED_HPZ_SMALL_PARAM:-true}" = "true" ]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/patch_deepspeed_hpz_small_param.py" \
        --config "$DEEPSPEED_CONFIG"
fi
required_files=(
    "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth"
    "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    "$WAN_CKPT_DIR/Wan2.1_VAE.pth"
    "$TOKENIZER_DIR/spiece.model"
    "$DREAMZERO_CKPT_DIR/model.safetensors.index.json"
)
for required_file in "${required_files[@]}"; do
    if [ ! -f "$required_file" ]; then
        echo "Missing model file: $required_file" >&2
        exit 2
    fi
done

export PYTHON_BIN
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
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
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export WANDB_PROJECT
if [ "$TORCH_COMPILE" = "true" ]; then
    # TrainingArguments configures the backend/mode. These two Accelerate
    # options control the remaining torch.compile arguments before
    # DeepSpeedEngine.compile() wraps the complete VLA model.
    export ACCELERATE_DYNAMO_USE_DYNAMIC="$TORCH_COMPILE_DYNAMIC"
    export ACCELERATE_DYNAMO_USE_FULLGRAPH="$TORCH_COMPILE_FULLGRAPH"
    export TORCHINDUCTOR_CACHE_DIR
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
fi

TRAIN_COMMAND=(
    groot/vla/experiment/experiment.py
    "report_to=$REPORT_TO"
    "wandb_project=$WANDB_PROJECT"
    data=dreamzero/egovla_wds_fingertips_relative
    model=dreamzero/vla
    model/dreamzero/action_head=wan_flow_matching_action_tf
    model/dreamzero/transform=dreamzero_cotrain
    "train_architecture=$TRAIN_ARCHITECTURE"
    num_frames=33
    action_horizon=24
    num_frame_per_block=2
    num_action_per_block=24
    num_state_per_block=1
    "teacher_forcing_attn_backend=$TEACHER_FORCING_ATTN_BACKEND"
    num_views=2
    max_state_dim=64
    max_action_dim=48
    max_chunk_size=4
    image_resolution_width=320
    image_resolution_height=176
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
    "torch_compile=$TORCH_COMPILE"
    "torch_compile_backend=$TORCH_COMPILE_BACKEND"
    "torch_compile_mode=$TORCH_COMPILE_MODE"
    do_eval=true
    eval_strategy=steps
    eval_steps=500
    per_device_eval_batch_size=1
    dataloader_num_workers=4
    dataloader_pin_memory=true
    dataloader_persistent_workers=true
    dataloader_prefetch_factor=2
    dataloader_non_blocking=true
    "nvimgcodec_decode=${NVIMGCODEC_DECODE:-false}"
    save_strategy=steps
    save_steps=500
    save_total_limit=10
    "save_lora_only=$SAVE_LORA_ONLY"
    upload_checkpoints=false
    "training_args.deepspeed=$DEEPSPEED_CONFIG"
    ++action_head_cfg.config.skip_component_loading=true
)

if [ "$TRAIN_ARCHITECTURE" = "lora" ]; then
    TRAIN_COMMAND+=("++action_head_cfg.config.defer_lora_injection=true")
fi

if [ "$TORCH_PROFILE" = "true" ]; then
    TRAIN_COMMAND+=(
        trainer.enable_prof_callback=true
        "trainer.profile_start_step=$PROFILE_START_STEP"
        "trainer.profile_warmup_steps=$PROFILE_WARMUP_STEPS"
        "trainer.profile_active_steps=$PROFILE_ACTIVE_STEPS"
        "trainer.profile_ranks=[$PROFILE_RANKS]"
        trainer.profile_record_shapes=false
        trainer.profile_with_stack=false
        trainer.profile_memory=false
        trainer.profile_with_flops=true
        "trainer.profile_upload_wandb=$PROFILE_UPLOAD_WANDB"
        "profile_dir=$PROFILE_DIR"
    )
fi

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
echo "python=$PYTHON_BIN"
echo "runtime=$LOCAL_RUNTIME_DIR"
echo "deepspeed config=$DEEPSPEED_CONFIG"
echo "teacher-forcing attention=$TEACHER_FORCING_ATTN_BACKEND"
echo "torch compile=$TORCH_COMPILE backend=$TORCH_COMPILE_BACKEND mode=$TORCH_COMPILE_MODE dynamic=$TORCH_COMPILE_DYNAMIC fullgraph=$TORCH_COMPILE_FULLGRAPH"
if [ "$TORCH_COMPILE" = "true" ]; then
    echo "torch inductor cache=$TORCHINDUCTOR_CACHE_DIR"
fi
echo "torch profiler=$TORCH_PROFILE"
if [ "$TORCH_PROFILE" = "true" ]; then
    echo "profile window=start:$PROFILE_START_STEP warmup:$PROFILE_WARMUP_STEPS active:$PROFILE_ACTIVE_STEPS ranks:[$PROFILE_RANKS]"
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$MACHINE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    --no-python \
    bash "$SCRIPT_DIR/numa_bind_wrapper.sh" \
    "${TRAIN_COMMAND[@]}"
