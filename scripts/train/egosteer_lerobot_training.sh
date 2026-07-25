#!/usr/bin/env bash
# Single- or multi-node launcher for the EgoSteer LeRobot v3 48D export.
#
# Defaults target Wan2.1-I2V-14B full fine-tuning.  The production dataset
# provides text_embs/<sha1(raw task text)[:16]>.pt. The data loader keeps only
# complete four-chunk contexts and the action head consumes the cached T5
# embedding directly.
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
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$NUM_GPUS}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
DATA_ROOT="${EGO_STEER_DATA_ROOT:-/efs-exp/agent-workspace/xuwenxi/datasets/realworld-dreamzero-lerobot/dagger}"
DATA_CONFIG="${DATA_CONFIG:-dreamzero/dual_arm_dexterous_hand_relative}"
DATA_ROOT_CONFIG_KEY="${DATA_ROOT_CONFIG_KEY:-dual_arm_dexterous_hand_data_root}"
OUTPUT_DIR="${OUTPUT_DIR:-/efs-exp/agent-workspace/xuwenxi/outputs/dreamzero_egosteer_lerobot}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/efs-exp/agent-workspace/xuwenxi/checkpoints/umt5-xxl}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/efs-exp/agent-workspace/xuwenxi/checkpoints/DreamZero-AgiBot}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$WORLD_SIZE}"
MAX_STEPS="${MAX_STEPS:-100000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-dreamzero}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
if [[ -z "$DEEPSPEED_CONFIG" ]]; then
    if [[ "$NUM_GPUS" == "8" ]]; then
        # This is the measured ~30% MFU path: GPU AdamW plus tuned HPZ
        # communication. Use zero3_hpz8_offload.json explicitly as the
        # lower-memory fallback.
        DEEPSPEED_CONFIG="groot/vla/configs/deepspeed/zero3_hpz8.json"
    else
        DEEPSPEED_CONFIG="groot/vla/configs/deepspeed/zero2_offload.json"
    fi
fi

DATASET_SHARD_SAMPLING_RATE="${DATASET_SHARD_SAMPLING_RATE:-0.1}"
DATASET_NUM_STEPS_PER_SHARD="${DATASET_NUM_STEPS_PER_SHARD:-}"
DATASET_NUM_SHARDS_TO_SAMPLE="${DATASET_NUM_SHARDS_TO_SAMPLE:-}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
DO_EVAL="${DO_EVAL:-false}"
EVAL_STRATEGY="${EVAL_STRATEGY:-no}"
EVAL_STEPS="${EVAL_STEPS:-500}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
TORCH_PROFILE="${TORCH_PROFILE:-false}"
PROFILE_START_STEP="${PROFILE_START_STEP:-50}"
PROFILE_WARMUP_STEPS="${PROFILE_WARMUP_STEPS:-1}"
PROFILE_ACTIVE_STEPS="${PROFILE_ACTIVE_STEPS:-2}"
PROFILE_RANKS="${PROFILE_RANKS:-0}"
PROFILE_DIR="${PROFILE_DIR:-$OUTPUT_DIR/profiling}"
TORCH_COMPILE="${TORCH_COMPILE:-false}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-null}"
SKIP_FINAL_SAVE="${SKIP_FINAL_SAVE:-false}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"

if [[ "$DATA_CONFIG" == "dreamzero/dual_arm_dexterous_hand_mixture_relative" ]]; then
    DATASET_ROOTS=(
        "$DATA_ROOT/dagger"
        "$DATA_ROOT/multitask"
        "$DATA_ROOT/singletask"
        "$DATA_ROOT/val"
    )
else
    DATASET_ROOTS=("$DATA_ROOT")
fi
for dataset_root in "${DATASET_ROOTS[@]}"; do
    for required in \
        "$dataset_root/meta/info.json" \
        "$dataset_root/meta/modality.json" \
        "$dataset_root/meta/relative_stats_dreamzero.json" \
        "$dataset_root/text_embs"; do
        if [[ ! -e "$required" ]]; then
            echo "Missing required path: $required" >&2
            exit 2
        fi
    done
done

for required in \
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
if (( GLOBAL_BATCH_SIZE % WORLD_SIZE != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by world size $WORLD_SIZE" >&2
    exit 2
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" ]]; then
    echo "Multi-node training requires MASTER_ADDR from the container platform" >&2
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
export PYTHON_BIN

# Transformers treats a non-null backend or mode as an implicit whole-model
# compile request.  Clear both when compile is disabled so a smoke run stays
# eager instead of compiling every rank on its first batch.
TRAINER_TORCH_COMPILE_BACKEND="$TORCH_COMPILE_BACKEND"
TRAINER_TORCH_COMPILE_MODE="$TORCH_COMPILE_MODE"
if [[ "$TORCH_COMPILE" != "true" ]]; then
    TRAINER_TORCH_COMPILE_BACKEND=null
    TRAINER_TORCH_COMPILE_MODE=null
fi

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
    "data=$DATA_CONFIG"
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
    # Fixed physical single-context sequence:
    # 2 * (4 chunks * 2 latent frames + 1 first frame) * 880
    # + 4 * (24 action + 1 state) = 15,940 transformer tokens.
    performance_tokens_per_sample=15940
    teacher_forcing_attn_backend=fragmented
    "$DATA_ROOT_CONFIG_KEY=$DATA_ROOT"
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
    "do_eval=$DO_EVAL"
    "eval_strategy=$EVAL_STRATEGY"
    "eval_steps=$EVAL_STEPS"
    "per_device_eval_batch_size=$PER_DEVICE_EVAL_BATCH_SIZE"
    "dataloader_num_workers=$DATALOADER_NUM_WORKERS"
    dataloader_pin_memory=true
    "dataloader_persistent_workers=$DATALOADER_PERSISTENT_WORKERS"
    "dataloader_prefetch_factor=$DATALOADER_PREFETCH_FACTOR"
    dataloader_non_blocking=true
    "torch_compile=$TORCH_COMPILE"
    "torch_compile_backend=$TRAINER_TORCH_COMPILE_BACKEND"
    "torch_compile_mode=$TRAINER_TORCH_COMPILE_MODE"
    "save_strategy=$SAVE_STRATEGY"
    "save_steps=$SAVE_STEPS"
    "save_total_limit=$SAVE_TOTAL_LIMIT"
    "skip_final_save=$SKIP_FINAL_SAVE"
    save_lora_only=false
    upload_checkpoints=false
    "training_args.deepspeed=$DEEPSPEED_CONFIG"
    ++action_head_cfg.config.skip_component_loading=true
)

if [[ -n "$DATASET_NUM_STEPS_PER_SHARD" ]]; then
    TRAIN_COMMAND+=("++train_dataset.dataset_kwargs.num_steps_per_shard=$DATASET_NUM_STEPS_PER_SHARD")
fi
if [[ -n "$DATASET_NUM_SHARDS_TO_SAMPLE" ]]; then
    TRAIN_COMMAND+=("++train_dataset.mixture_kwargs.num_shards_to_sample=$DATASET_NUM_SHARDS_TO_SAMPLE")
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
echo "topology=$NNODES nodes x $GPUS_PER_NODE GPUs = $WORLD_SIZE ranks"
echo "machine_rank=$MACHINE_RANK master=$MASTER_ADDR:$MASTER_PORT"
echo "global_batch=$GLOBAL_BATCH_SIZE data=$DATA_ROOT config=$DATA_CONFIG"
echo "deepspeed=$DEEPSPEED_CONFIG output=$OUTPUT_DIR"
echo "shard_sampling_rate=$DATASET_SHARD_SAMPLING_RATE shard_steps=${DATASET_NUM_STEPS_PER_SHARD:-default}"
echo "torch_compile=$TORCH_COMPILE backend=$TRAINER_TORCH_COMPILE_BACKEND mode=$TRAINER_TORCH_COMPILE_MODE"
echo "torch_profiler=$TORCH_PROFILE profile_dir=$PROFILE_DIR"

if (( NNODES > 1 )); then
    exec "$PYTHON_BIN" -m torch.distributed.run \
        --nnodes="$NNODES" \
        --node_rank="$MACHINE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        --no-python \
        bash "$SCRIPT_DIR/numa_bind_wrapper.sh" \
        "${TRAIN_COMMAND[@]}"
fi

exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    "${TRAIN_COMMAND[@]}"
