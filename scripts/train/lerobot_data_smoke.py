#!/usr/bin/env python3
"""Data-side smoke for the EgoSteer LeRobot mixture config.

Instantiates the production three-way train mixture and val dataset, decodes
one train and one val sample through PyAV, and checks the fixed four-chunk
modalities plus the cached T5 task embedding. No model weights, no GPU.

Run from the repository root with the training runtime on PYTHONPATH:

    python scripts/train/lerobot_data_smoke.py
"""
import os

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get(
    "EGO_STEER_DATA_ROOT",
    "/efs-exp/agent-workspace/xuwenxi/datasets/realworld-dreamzero-lerobot",
)
CONFIG_DIR = os.path.join(PROJECT_DIR, "groot/vla/configs")

with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(
        config_name="conf",
        overrides=[
            "data=dreamzero/dual_arm_dexterous_hand_mixture_relative",
            f"egosteer_lerobot_root={ROOT}",
            "output_dir=/tmp/dreamzero-data-smoke",
            "dataset_shard_sampling_rate=0.1",
        ],
    )

print("val_use_train_metadata:", cfg.val_use_train_metadata)


def describe(sample, tag):
    assert isinstance(sample, dict), f"{tag}: sample is {type(sample)}"
    for key in sorted(sample):
        value = sample[key]
        shape = getattr(value, "shape", None)
        if shape is None and isinstance(value, (list, tuple)):
            shape = f"list[{len(value)}]"
        print(f"  {tag} {key}: {shape}")
    video_keys = [k for k in sample if k.startswith("video.")]
    state_keys = [k for k in sample if k.startswith("state.")]
    action_keys = [k for k in sample if k.startswith("action.")]
    assert video_keys, f"{tag}: no video keys"
    assert state_keys, f"{tag}: no state keys"
    assert action_keys, f"{tag}: no action keys"
    assert "task_embedding" in sample, f"{tag}: missing task_embedding"
    emb = np.asarray(sample["task_embedding"])
    assert emb.shape == (512, 4096), f"{tag}: task_embedding {emb.shape}"
    return sample


print("=== instantiating train mixture (dagger/multitask/singletask) ===")
train_ds = instantiate(cfg.train_dataset)
print("train mixture instantiated:", type(train_ds).__name__)
train_iter = iter(train_ds)
train_sample = next(train_iter)
print("=== train sample (decoded via PyAV) ===")
describe(train_sample, "train")

print("=== instantiating val mixture ===")
val_ds = instantiate(cfg.val_dataset)
print("val mixture instantiated:", type(val_ds).__name__)
val_samples = list(iter(val_ds))
print(f"val yielded {len(val_samples)} samples (fixed per-rank budget)")
assert 0 < len(val_samples) <= 4, f"unexpected val budget {len(val_samples)}"
describe(val_samples[0], "val")

print("=== collator check ===")
collator = instantiate(cfg.data_collator)
batch = collator([train_sample])
emb = batch.get("task_embedding") if isinstance(batch, dict) else None
if emb is None and hasattr(batch, "task_embedding"):
    emb = batch.task_embedding
assert emb is not None, f"collated batch lacks task_embedding: {type(batch)}"
emb = torch.as_tensor(emb)
assert emb.shape == (1, 512, 4096), f"collated task_embedding {tuple(emb.shape)}"
print("collated task_embedding:", tuple(emb.shape))
print("DATA SMOKE OK")
