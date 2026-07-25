#!/usr/bin/env bash
# Standalone single-node launcher for the EgoSteer LeRobot v3 48D export.
#
# Defaults target Wan2.1-I2V-14B full fine-tuning.  The production dataset
# provides text_embs/<sha1(raw task text)[:16]>.pt; the data config enables
# that cache and falls back to T5 only for cache misses or mixed batches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

RUNTIME_DIR="${DREAMZERO_RUNTIME_DIR:-/opt/dreamzero-runtime}"
if [[ -d "$RUNTIME_DIR/train-site" && -d "$RUNTIME_DIR/torch-site" && -d "$RUNTIME_DIR/local-site" ]]; then
    PYTHON_BIN="${DREAMZERO_PYTHON_BIN:-/root/miniconda3/bin/python3}"
    export PYTHONPATH="$RUNTIME_DIR/train-site:$RUNTIME_DIR/torch-site:$RUNTIME_DIR/local-site:$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="$RUNTIME_DIR/torch-site/nvidia/nccl/lib:$RUNTIME_DIR/torch-site/nvidia/cusparselt/lib:$RUNTIME_DIR/local-site/nvidia/nvjpeg/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "Missing DreamZero runtime under $RUNTIME_DIR" >&2
    exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter is not executable: $PYTHON_BIN" >&2
    exit 2
fi

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
DATA_ROOT="${EGO_STEER_DATA_ROOT:-/efs-exp/agent-workspace/xuwenxi/datasets/realworld-dreamzero-lerobot/dagger}"
OUTPUT_DIR="${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egosteer_lerobot}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/umt5-xxl}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/efs-exp/agent-workspace/xuwenxi/checkpoints/DreamZero-AgiBot}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$NUM_GPUS}"
MAX_STEPS="${MAX_STEPS:-100000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
if [[ -z "$DEEPSPEED_CONFIG" ]]; then
    if [[ "$NUM_GPUS" == "8" ]]; then
        DEEPSPEED_CONFIG="groot/vla/configs/deepspeed/zero3_hpz8_offload.json"
    else
        DEEPSPEED_CONFIG="groot/vla/configs/deepspeed/zero2_offload.json"
    fi
fi

DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_NUM_STEPS_PER_SHARD="${DATASET_NUM_STEPS_PER_SHARD:-}"
DATASET_NUM_SHARDS_TO_SAMPLE="${DATASET_NUM_SHARDS_TO_SAMPLE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
TORCH_PROFILE="${TORCH_PROFILE:-false}"
PROFILE_START_STEP="${PROFILE_START_STEP:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-2}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_DIR="${PROFILE_DIR:-$OUTPUT_DIR/profiling}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-false}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"

for required in \
    "$DATA_ROOT/meta/info.json" \
    "$DATA_ROOT/meta/modality.json" \
    "$DATA_ROOT/meta/relative_stats_dreamzero.json" \
    "$DATA_ROOT/text_embs" \
    "$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
    "$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    "$WAN_CKPT_DIR/Wan2.1_VAE.pth" \
    "$TOKENIZER_DIR/spiece.model" \
    "$PRETRAINED_MODEL_PATH/model.safetensors.index.json" \
    "$DEEPSPEED_CONFIG"; do
    if [[ ! -e "$required" ]]; then
        echo "Missing required path: $required" >&2
        exit 2
    fi
done

if [[ "$REPORT_TO" == "wandb" ]]; then
    : "${WANDB_API_KEY:?Set WANDB_API_KEY or use REPORT_TO=none}"
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "NUM_GPUS must be positive" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS" >&2
    exit 2
fi

if [[ "${PATCH_DEEPSPEED_COMPAT:-true}" == "true" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/patch_deepspeed_compat.py" --config "$DEEPSPEED_CONFIG"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_COMPILE

if [[ "$DATALOADER_NUM_WORKERS" == "0" ]]; then
    DATALOADER_PERSISTENT_WORKERS=false
    # Transformers rejects a prefetch factor without worker subprocesses.
    DATALOADER_PREFETCH_FACTOR=null
else
    DATALOADER_PERSISTENT_WORKERS=true
fi

TRAIN_COMMAND=(
    groot/vla/experiment/experiment.py
    "report_to=$REPORT_TO"
    "wandb_project=$WANDB_PROJECT"
    data=dreamzero/dual_arm_dexterous_hand_relative
    model=dreamzero/vla
    model/dreamzero/action_head=wan_flow_matching_action_tf
    model/dreamzero/transform=dreamzero_cotrain
    train_architecture=full
    num_frames=33
    action_horizon=24
    num_frame_per_block=2
    num_action_per_block=24
    num_state_per_block=1
    num_views=2
    max_state_dim=64
    max_action_dim=48
    max_chunk_size=4
    image_resolution_width=320
    image_resolution_height=176
    frame_seqlen=880
    teacher_forcing_attn_backend=fragmented
    "dual_arm_dexterous_hand_data_root=$DATA_ROOT"
    "dataset_shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE"
    "output_dir=$OUTPUT_DIR"
    "pretrained_model_path=$PRETRAINED_MODEL_PATH"
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
    do_eval=false
    "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    dataloader_pin_memory=true
    "dataloader_persistent_workers=$DATALOADER_PERSISTENT_WORKERS"
    "dataloader_prefetch_factor=$DATALOADER_PREFETCH_FACTOR"
    dataloader_non_blocking=true
    "torch_compile=$TORCH_COMPILE"
    "save_strategy=$SAVE_STRATEGY"
    save_steps=500
    save_total_limit=8
    "skip_final_save=$SKIP_FINAL_SAVE"
    save_lora_only=false
    upload_checkpoints=false
    "training_args.deepspeed=$DEEPSPEED_CONFIG"
    ++action_head_cfg.config.skip_component_loading=true
)

if [[ -n "$DATASET_NUM_STEPS_PER_SHARD" ]]; then
    TRAIN_COMMAND+=("+train_dataset.dataset_kwargs.num_steps_per_shard=$DATASET_NUM_STEPS_PER_SHARD")
fi
if [[ -n "$DATASET_NUM_SHARDS_TO_SAMPLE" ]]; then
    TRAIN_COMMAND+=("+train_dataset.mixture_kwargs.num_shards_to_sample=$DATASET_NUM_SHARDS_TO_SAMPLE")
fi
if [[ "$TORCH_PROFILE" == "true" ]]; then
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
        trainer.profile_upload_wandb=false
        "profile_dir=$PROFILE_DIR"
    )
fi
if [[ -n "${TRAIN_ARGS:-}" ]]; then
    read -r -a EXTRA_TRAIN_ARGS <<< "$TRAIN_ARGS"
    TRAIN_COMMAND+=("${EXTRA_TRAIN_ARGS[@]}")
fi

echo "DreamZero LeRobot launch: Wan2.1-I2V-14B full fine-tune"
echo "gpus=$NUM_GPUS global_batch=$GLOBAL_BATCH_SIZE data=$DATA_ROOT"
echo "deepspeed=$DEEPSPEED_CONFIG output=$OUTPUT_DIR"
echo "shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE shard_steps=${DATASET_NUM_STEPS_PER_SHARD:-default}"
echo "torch_profiler=$TORCH_PROFILE profile_dir=$PROFILE_DIR"

exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    "${TRAIN_COMMAND[@]}"
