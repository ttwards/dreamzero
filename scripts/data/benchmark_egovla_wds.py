#!/usr/bin/env python3
"""Benchmark the native EgoVLA WDS input pipeline without starting training.

The benchmark measures three boundaries with the real Hydra configuration:

* ``decoded_block``: tar read, JPEG decode, block assembly, and WDS filtering;
* ``transform``: the previous stage plus video/state/action transforms;
* ``transform_collate``: the previous stage plus the real VLA text/numeric collator.

Each result excludes a configurable warmup window from steady-state throughput,
while also reporting first-batch latency (which includes worker startup).
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import open_dict
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "groot" / "vla" / "configs"
sys.path.insert(0, str(REPO_ROOT))


def make_config(metadata: str, tokenizer: str, keep_ratio: float):
    overrides = [
        "data=dreamzero/egovla_wds_fingertips_relative",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "max_state_dim=64",
        "max_action_dim=48",
        "num_views=2",
        "action_horizon=24",
        f"tokenizer_path={tokenizer}",
        f"egovla_wds_metadata_path={metadata}",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="conf", overrides=overrides)
    with open_dict(cfg):
        cfg.train_dataset.keep_ratio = keep_ratio
        cfg.train_dataset.shuffle_buffer = 256
        cfg.train_dataset.shuffle_initial = 32
        cfg.train_dataset.training = True
    return cfg


def identity_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError(f"Expected batch size 1, got {len(samples)}")
    return samples[0]


def run_one(
    *,
    stage: str,
    workers: int,
    samples: int,
    warmup: int,
    metadata: str,
    tokenizer: str,
    keep_ratio: float,
    pin_memory: bool,
    prefetch_factor: int,
) -> dict[str, Any]:
    cfg = make_config(metadata, tokenizer, keep_ratio)
    transformed = stage != "decoded_block"
    collator: Callable | None = None

    with open_dict(cfg):
        cfg.train_dataset.max_samples = (samples + warmup + 8) * max(1, workers)
        if not transformed:
            cfg.train_dataset.transforms = None
        elif stage == "transform_collate":
            collator = instantiate(cfg.data_collator)

    dataset = instantiate(cfg.train_dataset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collator or identity_collate,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        **({"prefetch_factor": prefetch_factor} if workers > 0 else {}),
    )

    iterator = iter(loader)
    first_start = time.perf_counter()
    next(iterator)
    first_seconds = time.perf_counter() - first_start

    for _ in range(max(0, warmup - 1)):
        next(iterator)

    measured_start = time.perf_counter()
    measured = 0
    for _ in range(samples):
        next(iterator)
        measured += 1
    measured_seconds = time.perf_counter() - measured_start

    result = {
        "stage": stage,
        "workers": workers,
        "pin_memory": pin_memory,
        "prefetch_factor": prefetch_factor if workers > 0 else None,
        "warmup_batches": warmup,
        "measured_batches": measured,
        "first_batch_seconds": first_seconds,
        "steady_seconds": measured_seconds,
        "steady_batches_per_second": measured / measured_seconds,
        "steady_batch_ms": measured_seconds / measured * 1000.0,
    }
    print(
        f"{stage:18s} workers={workers:2d} "
        f"first={first_seconds:.3f}s "
        f"steady={result['steady_batches_per_second']:.2f} batch/s "
        f"({result['steady_batch_ms']:.1f}ms/batch)",
        flush=True,
    )

    del iterator, loader, dataset, cfg, collator
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--keep-ratio", type=float, default=0.1)
    parser.add_argument("--workers", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for workers in args.workers:
        for stage in ("decoded_block", "transform", "transform_collate"):
            results.append(
                run_one(
                    stage=stage,
                    workers=workers,
                    samples=args.samples,
                    warmup=args.warmup,
                    metadata=args.metadata,
                    tokenizer=args.tokenizer,
                    keep_ratio=args.keep_ratio,
                    pin_memory=args.pin_memory,
                    prefetch_factor=args.prefetch_factor,
                )
            )

    payload = {
        "metadata": args.metadata,
        "tokenizer": args.tokenizer,
        "samples": args.samples,
        "warmup": args.warmup,
        "keep_ratio": args.keep_ratio,
        "results": results,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
