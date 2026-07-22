"""Streaming loader for EgoVLA frame-wise WebDataset shards.

Each tar sample is one control step and contains image.jpg, chest_image.jpg,
lowdim.npy, and meta.json. The 136D lowdim layout is documented in
EgoVLA/data/data.md; this loader consumes the 48D state and 48D action prefix.
"""

from __future__ import annotations

from collections import deque
import glob
import io
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import cv2
import numpy as np
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import webdataset as wds

from groot.vla.data.schema import DatasetMetadata, EmbodimentTag


STATE_DIM = 48
ACTION_DIM = 48
LOWDIM_MIN_DIM = 96
LOWDIM_FIELDS: tuple[tuple[str, slice], ...] = (
    ("left_wrist_position", slice(0, 3)),
    ("right_wrist_position", slice(3, 6)),
    ("left_wrist_rotation_6d", slice(6, 12)),
    ("right_wrist_rotation_6d", slice(12, 18)),
    ("left_fingertip_position", slice(18, 33)),
    ("right_fingertip_position", slice(33, 48)),
)


def expand_shard_patterns(shard_urls: str | Sequence[str]) -> list[str]:
    """Expand local globs while preserving explicit URLs."""
    patterns = [shard_urls] if isinstance(shard_urls, str) else list(shard_urls)
    urls: list[str] = []
    for pattern in patterns:
        if "://" in pattern:
            urls.append(pattern)
            continue
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            urls.extend(matches)
        elif not any(char in pattern for char in "*?["):
            urls.append(os.path.expanduser(pattern))
    if not urls:
        raise FileNotFoundError(f"No WDS shards matched: {patterns}")
    return urls


def split_state_action(prefix: str, values: np.ndarray) -> dict[str, np.ndarray]:
    """Split a 48D EgoVLA vector into DreamZero named modalities."""
    if values.shape[-1] != STATE_DIM:
        raise ValueError(f"Expected {STATE_DIM}D {prefix}, got {values.shape}")
    return {
        f"{prefix}.{name}": np.asarray(values[..., field_slice], dtype=np.float32)
        for name, field_slice in LOWDIM_FIELDS
    }


def make_relative_action(actions: np.ndarray, anchor_state: np.ndarray) -> np.ndarray:
    """Make wrist/fingertip translations relative to the block anchor."""
    relative = np.asarray(actions, dtype=np.float32).copy()
    relative[:, 0:6] -= anchor_state[None, 0:6]
    relative[:, 18:48] -= anchor_state[None, 18:48]
    return relative


def _decode_npy(value: bytes) -> np.ndarray:
    return np.asarray(np.load(io.BytesIO(value), allow_pickle=False), dtype=np.float32)


def _decode_small_members(sample: dict[str, Any]) -> dict[str, Any]:
    lowdim = _decode_npy(sample["lowdim.npy"]).reshape(-1)
    if lowdim.shape[0] < LOWDIM_MIN_DIM:
        raise ValueError(
            f"{sample.get('__key__')} has {lowdim.shape[0]} lowdim values; "
            f"expected at least {LOWDIM_MIN_DIM}"
        )
    return {
        "__key__": sample["__key__"],
        "__url__": sample.get("__url__"),
        "image.jpg": sample["image.jpg"],
        "chest_image.jpg": sample.get("chest_image.jpg"),
        "lowdim": lowdim,
        "meta": json.loads(sample["meta.json"]),
    }


def _episode_key(frame: dict[str, Any]) -> tuple[str, int]:
    meta = frame["meta"]
    return str(meta["dataset_name"]), int(meta["episode_index"])


def _frame_number(frame: dict[str, Any]) -> int:
    key = str(frame["__key__"])
    if "_f" not in key:
        raise ValueError(f"Cannot parse frame index from WDS key: {key}")
    return int(key.rsplit("_f", 1)[1])


def _choose_anchors(
    current_index: int,
    episode_start: int,
    episode_end: int,
    action_horizon: int,
    max_chunk_size: int,
) -> list[int]:
    candidates = [current_index]
    for distance in range(1, max_chunk_size):
        candidates.extend(
            [current_index - distance * action_horizon, current_index + distance * action_horizon]
        )
    valid = [
        index
        for index in candidates
        if index >= episode_start and index + action_horizon <= episode_end
    ]
    return sorted(valid[:max_chunk_size])


def _compose_sample(
    frames_by_index: dict[int, dict[str, Any]],
    current_index: int,
    episode_start: int,
    episode_end: int,
    *,
    action_horizon: int,
    max_chunk_size: int,
    video_offsets: Sequence[int],
    relative_action: bool,
) -> dict[str, Any] | None:
    if current_index + action_horizon > episode_end:
        return None
    anchors = _choose_anchors(
        current_index, episode_start, episode_end, action_horizon, max_chunk_size
    )
    if not anchors:
        return None

    head_images: list[bytes] = []
    chest_images: list[bytes] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for anchor in anchors:
        anchor_frame = frames_by_index[anchor]
        anchor_state = anchor_frame["lowdim"][:STATE_DIM]
        block_actions = np.stack(
            [
                frames_by_index[index]["lowdim"][STATE_DIM : STATE_DIM + ACTION_DIM]
                for index in range(anchor, anchor + action_horizon)
            ]
        )
        if relative_action:
            block_actions = make_relative_action(block_actions, anchor_state)
        states.append(np.asarray(anchor_state, dtype=np.float32))
        actions.append(block_actions)
        for offset in video_offsets:
            frame = frames_by_index[anchor + offset]
            head_images.append(frame["image.jpg"])
            if frame["chest_image.jpg"] is not None:
                chest_images.append(frame["chest_image.jpg"])

    boundary = frames_by_index[anchors[-1] + action_horizon]
    head_images.append(boundary["image.jpg"])
    if boundary["chest_image.jpg"] is not None:
        chest_images.append(boundary["chest_image.jpg"])

    meta = frames_by_index[current_index]["meta"]
    instructions = meta.get("instruction") or [""]
    if isinstance(instructions, str):
        instructions = [instructions]
    result: dict[str, Any] = {
        "__key__": frames_by_index[current_index]["__key__"],
        "video.head": head_images,
        "state": np.stack(states),
        "action": np.concatenate(actions),
        "annotation.task": instructions,
        "embodiment_id": EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND,
        "chunk_size": len(anchors),
        "block_anchors": np.asarray(anchors, dtype=np.int64),
    }
    if chest_images:
        if len(chest_images) != len(head_images):
            raise ValueError(f"Incomplete chest camera sequence at {result['__key__']}")
        result["video.chest"] = chest_images
    result.update(split_state_action("state", result.pop("state")))
    result.update(split_state_action("action", result.pop("action")))
    return result


def sliding_block_samples(
    source: Iterable[dict[str, Any]],
    *,
    action_horizon: int = 24,
    max_chunk_size: int = 4,
    video_offsets: Sequence[int] = (0, 3, 6, 9, 12, 15, 18, 21),
    relative_action: bool = True,
) -> Iterator[dict[str, Any]]:
    """Compose bounded-lookahead samples without loading an episode in memory."""
    if not video_offsets or min(video_offsets) < 0 or max(video_offsets) >= action_horizon:
        raise ValueError("video_offsets must lie inside one action block")
    lookbehind = (max_chunk_size - 1) * action_horizon
    lookahead = max_chunk_size * action_horizon
    frames: deque[dict[str, Any]] = deque()
    current_episode: tuple[str, int] | None = None
    episode_start = 0
    next_emit = 0
    previous_frame_number: int | None = None

    def drain(final: bool = False) -> Iterator[dict[str, Any]]:
        nonlocal next_emit
        while frames:
            frame_map = {_frame_number(frame): frame for frame in frames}
            episode_end = max(frame_map)
            if not final and next_emit + lookahead > episode_end:
                break
            sample = _compose_sample(
                frame_map,
                next_emit,
                episode_start,
                episode_end,
                action_horizon=action_horizon,
                max_chunk_size=max_chunk_size,
                video_offsets=video_offsets,
                relative_action=relative_action,
            )
            if sample is not None:
                yield sample
            next_emit += 1
            while frames and _frame_number(frames[0]) < next_emit - lookbehind:
                frames.popleft()
            if next_emit > episode_end:
                break

    for frame in source:
        frame_episode = _episode_key(frame)
        frame_number = _frame_number(frame)
        frame_sequence_discontinuous = (
            previous_frame_number is not None
            and frame_number != previous_frame_number + 1
        )
        if current_episode != frame_episode or frame_sequence_discontinuous:
            if current_episode is not None:
                yield from drain(final=True)
            frames.clear()
            current_episode = frame_episode
            episode_start = frame_number
            next_emit = episode_start
        frames.append(frame)
        previous_frame_number = frame_number
        yield from drain(final=False)
    if current_episode is not None:
        yield from drain(final=True)


def _decode_jpeg(value: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _materialize_media(sample: dict[str, Any]) -> dict[str, Any]:
    sample["video.head"] = np.stack([_decode_jpeg(value) for value in sample["video.head"]])
    if "video.chest" in sample:
        sample["video.chest"] = np.stack(
            [_decode_jpeg(value) for value in sample["video.chest"]]
        )
    instructions = sample["annotation.task"]
    sample["annotation.task"] = random.choice(instructions) if instructions else ""
    return sample


class EgoVLAWdsDataset(IterableDataset):
    """DreamZero-compatible iterable dataset over EgoVLA tar shards."""

    def __init__(
        self,
        shard_urls: str | Sequence[str],
        metadata_path: str,
        transforms: Any | None = None,
        training: bool = True,
        action_horizon: int = 24,
        max_chunk_size: int = 4,
        video_offsets: Sequence[int] = (0, 3, 6, 9, 12, 15, 18, 21),
        relative_action: bool = True,
        load_chest: bool = True,
        keep_ratio: float = 0.1,
        shuffle_buffer: int = 256,
        shuffle_initial: int = 32,
        seed: int = 42,
    ):
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        self.shard_urls = expand_shard_patterns(shard_urls)
        self.training = training
        self.action_horizon = action_horizon
        self.max_chunk_size = max_chunk_size
        self.video_offsets = tuple(video_offsets)
        self.relative_action = relative_action
        self.load_chest = load_chest
        self.keep_ratio = keep_ratio
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_initial = min(shuffle_initial, shuffle_buffer)
        self.seed = seed
        self.epoch = 0
        self.transforms = transforms

        metadata_json = json.loads(Path(metadata_path).read_text())
        if "embodiment_tag" not in metadata_json:
            metadata_json = metadata_json[EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND.value]
        self.metadata = DatasetMetadata.model_validate(metadata_json)
        self.tag = EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND
        if self.metadata.embodiment_tag != self.tag:
            raise ValueError(
                f"Metadata embodiment is {self.metadata.embodiment_tag}; expected {self.tag}"
            )
        self.merged_metadata = {self.tag.value: self.metadata}
        if self.transforms is not None:
            self.transforms.set_metadata(self.metadata)
            self.transforms.train() if training else self.transforms.eval()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def reset_seed(self) -> None:
        self.epoch = 0

    def _worker_seed(self) -> int:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return self.seed + self.epoch * 1_000_003 + rank * 10_007 + worker_id

    def build_pipeline(self, apply_transforms: bool = True) -> wds.DataPipeline:
        rng = random.Random(self._worker_seed())
        shard_source: Any
        if self.training:
            shard_source = wds.ResampledShards(self.shard_urls, seed=self._worker_seed())
        else:
            shard_source = wds.SimpleShardList(self.shard_urls)
        stages: list[Any] = [
            shard_source,
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.select(
                lambda sample: "lowdim.npy" in sample
                and "meta.json" in sample
                and "image.jpg" in sample
                and (not self.load_chest or "chest_image.jpg" in sample)
            ),
            wds.map(_decode_small_members, handler=wds.warn_and_continue),
            lambda source: sliding_block_samples(
                source,
                action_horizon=self.action_horizon,
                max_chunk_size=self.max_chunk_size,
                video_offsets=self.video_offsets,
                relative_action=self.relative_action,
            ),
        ]
        if self.training and self.keep_ratio < 1.0:
            stages.append(wds.select(lambda _: rng.random() < self.keep_ratio))
        if self.training and self.shuffle_buffer > 1:
            stages.append(wds.shuffle(self.shuffle_buffer, initial=self.shuffle_initial, rng=rng))
        stages.append(wds.map(_materialize_media, handler=wds.warn_and_continue))
        if apply_transforms and self.transforms is not None:
            stages.append(wds.map(self.transforms, handler=wds.warn_and_continue))
        return wds.DataPipeline(*stages)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.build_pipeline(apply_transforms=True))
