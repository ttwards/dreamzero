#!/usr/bin/env python3
"""Run one real train and validation sample through EgoVLA transform+collate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import numpy as np
from omegaconf import open_dict


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "groot" / "vla" / "configs"
sys.path.insert(0, str(REPO_ROOT))


def _shape(value: object) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"Expected an array-like value, got {type(value).__name__}")
    return tuple(shape)


def _assert_finite(name: str, value: object) -> None:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_sample(split: str, sample: dict, batch: dict, action_horizon: int) -> None:
    action_shape = _shape(sample["action"])
    state_shape = _shape(sample["state"])
    image_shape = _shape(sample["images"])
    if len(action_shape) != 2 or action_shape[1] != 48:
        raise ValueError(f"{split} sample action shape is {action_shape}, expected [T, 48]")
    if action_shape[0] == 0 or action_shape[0] % action_horizon:
        raise ValueError(
            f"{split} sample action length {action_shape[0]} is not a positive "
            f"multiple of horizon {action_horizon}"
        )
    block_count = action_shape[0] // action_horizon
    if state_shape != (block_count, 64):
        raise ValueError(
            f"{split} sample state shape is {state_shape}, expected {(block_count, 64)}"
        )
    expected_frames = block_count * 8 + 1
    if image_shape != (expected_frames, 352, 640, 3):
        raise ValueError(
            f"{split} sample image shape is {image_shape}, expected "
            f"{(expected_frames, 352, 640, 3)}"
        )

    expected_batch_shapes = {
        "action": (1, *action_shape),
        "state": (1, *state_shape),
        "images": (1, *image_shape),
        "text": (1, 512),
    }
    for key, expected in expected_batch_shapes.items():
        actual = _shape(batch[key])
        if actual != expected:
            raise ValueError(f"{split} batch {key} shape is {actual}, expected {expected}")
    if "__key__" in batch or "__url__" in batch:
        raise ValueError(f"{split} batch leaked WebDataset source metadata")

    for key in ("action", "state", "images"):
        _assert_finite(f"{split} sample {key}", sample[key])
        _assert_finite(f"{split} batch {key}", batch[key])
    print(
        f"Validated {split}: sample_action={action_shape}, sample_state={state_shape}, "
        f"sample_images={image_shape}, batch_text={_shape(batch['text'])}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train-shard", required=True)
    parser.add_argument("--val-shard", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    args = parser.parse_args()

    overrides = [
        "data=dreamzero/egovla_wds_fingertips_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "max_state_dim=64",
        "max_action_dim=48",
        "num_views=2",
        "action_horizon=24",
        f"tokenizer_path={args.tokenizer_path}",
        f"egovla_wds_metadata_path={args.metadata}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="conf", overrides=overrides)

    with open_dict(cfg):
        cfg.train_dataset.shard_urls = [args.train_shard]
        cfg.train_dataset.keep_ratio = 1.0
        cfg.train_dataset.shuffle_buffer = 1
        cfg.train_dataset.shuffle_initial = 1
        cfg.train_dataset.max_samples = 1
        cfg.val_dataset.shard_urls = [args.val_shard]
        cfg.val_dataset.max_samples = 1

    collator = instantiate(cfg.data_collator)
    train_dataset = instantiate(cfg.train_dataset)
    val_dataset = instantiate(cfg.val_dataset)
    if not train_dataset.training or val_dataset.training:
        raise ValueError("Expected training=True for train and training=False for val")
    if train_dataset.shard_urls == val_dataset.shard_urls:
        raise ValueError("Train and validation resolved to the same shard")

    for split, dataset in (("train", train_dataset), ("val", val_dataset)):
        sample = next(iter(dataset))
        batch = collator([sample])
        _validate_sample(split, sample, batch, cfg.action_horizon)


if __name__ == "__main__":
    main()
