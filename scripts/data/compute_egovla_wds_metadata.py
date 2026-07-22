#!/usr/bin/env python3
"""Compute DreamZero normalization metadata from EgoVLA WDS shards."""

from __future__ import annotations

import argparse
import glob
import io
from pathlib import Path
import random
import re
import sys
import tarfile
from typing import Iterator, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from groot.vla.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatisticalValues,
    DatasetStatistics,
    EmbodimentTag,
    RotationType,
    StateActionMetadata,
    VideoMetadata,
)

FIELDS = (
    ("left_wrist_position", slice(0, 3)),
    ("right_wrist_position", slice(3, 6)),
    ("left_wrist_rotation_6d", slice(6, 12)),
    ("right_wrist_rotation_6d", slice(12, 18)),
    ("left_fingertip_position", slice(18, 33)),
    ("right_fingertip_position", slice(33, 48)),
)
KEY_PATTERN = re.compile(r"^(?P<dataset>.+)_ep(?P<episode>\d+)_f(?P<frame>\d+)$")


class RunningStatistics:
    """Exact moments/extrema plus reservoir-sampled quantiles."""

    def __init__(self, dim: int, reservoir_size: int, seed: int):
        self.dim = dim
        self.reservoir_size = reservoir_size
        self.rng = random.Random(seed)
        self.count = 0
        self.total = np.zeros(dim, dtype=np.float64)
        self.total_square = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)
        self.reservoir = np.empty((reservoir_size, dim), dtype=np.float32)
        self.reservoir_count = 0

    def add(self, values: np.ndarray) -> None:
        rows = np.asarray(values, dtype=np.float32).reshape(-1, self.dim)
        if rows.size == 0:
            return
        rows64 = rows.astype(np.float64)
        self.total += rows64.sum(axis=0)
        self.total_square += np.square(rows64).sum(axis=0)
        self.minimum = np.minimum(self.minimum, rows64.min(axis=0))
        self.maximum = np.maximum(self.maximum, rows64.max(axis=0))
        for row in rows:
            self.count += 1
            if self.reservoir_count < self.reservoir_size:
                self.reservoir[self.reservoir_count] = row
                self.reservoir_count += 1
            else:
                replacement = self.rng.randrange(self.count)
                if replacement < self.reservoir_size:
                    self.reservoir[replacement] = row

    def values(self) -> dict[str, np.ndarray]:
        if self.count == 0:
            raise ValueError("No valid rows were found for statistics")
        mean = self.total / self.count
        variance = np.maximum(self.total_square / self.count - np.square(mean), 0.0)
        quantiles = self.reservoir[: self.reservoir_count]
        return {
            "max": self.maximum.astype(np.float32),
            "min": self.minimum.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "q01": np.quantile(quantiles, 0.01, axis=0).astype(np.float32),
            "q99": np.quantile(quantiles, 0.99, axis=0).astype(np.float32),
        }


def expand_shards(patterns: Sequence[str]) -> list[str]:
    shards: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if matches:
            shards.extend(matches)
        elif not any(character in pattern for character in "*?["):
            shards.append(str(Path(pattern).expanduser()))
    if not shards:
        raise FileNotFoundError(f"No shards matched {patterns}")
    return shards


def iter_shard_episodes(shard: str) -> Iterator[np.ndarray]:
    episodes: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    with tarfile.open(shard, mode="r:*") as archive:
        for member in archive:
            suffix = ".lowdim.npy"
            if not member.name.endswith(suffix):
                continue
            key = Path(member.name).name[: -len(suffix)]
            match = KEY_PATTERN.match(key)
            if match is None:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            lowdim = np.asarray(
                np.load(io.BytesIO(extracted.read()), allow_pickle=False), dtype=np.float32
            ).reshape(-1)
            if len(lowdim) < 96:
                raise ValueError(f"{member.name}: expected at least 96 values")
            episode_key = (match.group("dataset"), int(match.group("episode")))
            episodes.setdefault(episode_key, {})[int(match.group("frame"))] = lowdim
    for frames in episodes.values():
        contiguous: list[np.ndarray] = []
        previous = None
        for index in sorted(frames):
            if previous is not None and index != previous + 1:
                if contiguous:
                    yield np.stack(contiguous)
                contiguous = []
            contiguous.append(frames[index])
            previous = index
        if contiguous:
            yield np.stack(contiguous)


def relative_actions(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    result = actions.copy()
    result[:, 0:6] -= anchor_state[None, 0:6]
    result[:, 18:48] -= anchor_state[None, 18:48]
    return result


def split_statistics(values: dict[str, np.ndarray]) -> dict:
    return {
        name: DatasetStatisticalValues(
            **{statistic: array[field_slice] for statistic, array in values.items()}
        )
        for name, field_slice in FIELDS
    }


def make_metadata(
    state_statistics: RunningStatistics,
    action_statistics: RunningStatistics,
    args: argparse.Namespace,
) -> DatasetMetadata:
    state_modalities = {}
    action_modalities = {}
    for name, field_slice in FIELDS:
        dim = field_slice.stop - field_slice.start
        rotation = RotationType.ROTATION_6D if name.endswith("rotation_6d") else None
        state_modalities[name] = StateActionMetadata(
            absolute=True, rotation_type=rotation, shape=(dim,), continuous=True
        )
        action_modalities[name] = StateActionMetadata(
            absolute=rotation is not None,
            rotation_type=rotation,
            shape=(dim,),
            continuous=True,
        )
    videos = {
        "head": VideoMetadata(
            resolution=(args.width, args.height), channels=3, fps=args.fps
        )
    }
    if not args.head_only:
        videos["chest"] = VideoMetadata(
            resolution=(args.width, args.height), channels=3, fps=args.fps
        )
    return DatasetMetadata(
        statistics=DatasetStatistics(
            state=split_statistics(state_statistics.values()),
            action=split_statistics(action_statistics.values()),
        ),
        modalities=DatasetModalities(
            video=videos, state=state_modalities, action=action_modalities
        ),
        embodiment_tag=EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    shard_source = parser.add_mutually_exclusive_group(required=True)
    shard_source.add_argument("--shards", nargs="+")
    shard_source.add_argument(
        "--data-config",
        help="YAML containing egovla_wds_shards; validation shards are ignored.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--anchor-stride", type=int, default=1)
    parser.add_argument("--reservoir-size", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()

    shard_patterns = args.shards
    if args.data_config is not None:
        from omegaconf import OmegaConf

        data_config = OmegaConf.load(args.data_config)
        shard_patterns = OmegaConf.to_container(
            data_config.egovla_wds_shards, resolve=True
        )
    if not isinstance(shard_patterns, list) or not shard_patterns:
        raise ValueError("No training shard patterns were configured")

    state_stats = RunningStatistics(48, args.reservoir_size, args.seed)
    action_stats = RunningStatistics(48, args.reservoir_size, args.seed + 1)
    episode_count = anchor_count = 0
    for shard in expand_shards(shard_patterns):
        print(f"Scanning {shard}")
        for episode in iter_shard_episodes(shard):
            episode_count += 1
            for anchor in range(0, max(0, len(episode) - args.action_horizon), args.anchor_stride):
                state = episode[anchor, :48]
                actions = episode[anchor : anchor + args.action_horizon, 48:96]
                state_stats.add(state)
                action_stats.add(relative_actions(actions, state))
                anchor_count += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_metadata(state_stats, action_stats, args).model_dump_json(indent=2) + "\n")
    print(
        f"Wrote {output} from {episode_count} contiguous episode segments "
        f"and {anchor_count} valid {args.action_horizon}-step anchors"
    )


if __name__ == "__main__":
    main()
