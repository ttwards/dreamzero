"""Export EgoSteer LeRobot v3 splits to the DreamZero 48D layout.

The source dataset is kept intact. For every split this script:

* rewrites only ``observation.state`` and ``action`` from 44D
  (position + quaternion + fingertip XYZ) to 48D (position + rotation-6D +
  fingertip XYZ);
* copies the ordinary LeRobot metadata and writes DreamZero modality metadata;
* symlinks the original MP4 directory and ``text_embs`` by default, avoiding a
  second copy of the large video/T5 stores.

It does not precompute CLIP by default. Use ``--estimate-clip`` first: the
current Wan CLIP consumes one 2-view composite per timestep and returns
257x1280 tokens, so caching every timestep is much larger than the source
videos themselves.

Example:
  python scripts/data/export_egosteer_v3_48d.py \
      --source-root /efs-exp/chenzhang/datasets/lerobot/egosteer-realworld \
      --output-root /efs-exp/agent-workspace/xuwenxi/datasets/realworld-dreamzero-lerobot

Estimate only:
  python scripts/data/export_egosteer_v3_48d.py \
      --source-root /efs-exp/chenzhang/datasets/lerobot/egosteer-realworld \
      --estimate-clip --clip-gpus 8 --clip-images-per-sec-per-gpu 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


SOURCE_NAMES = [
    "left_eef_x", "left_eef_y", "left_eef_z",
    "left_eef_qx", "left_eef_qy", "left_eef_qz", "left_eef_qw",
    "right_eef_x", "right_eef_y", "right_eef_z",
    "right_eef_qx", "right_eef_qy", "right_eef_qz", "right_eef_qw",
    "left_tip0_x", "left_tip0_y", "left_tip0_z",
    "left_tip1_x", "left_tip1_y", "left_tip1_z",
    "left_tip2_x", "left_tip2_y", "left_tip2_z",
    "left_tip3_x", "left_tip3_y", "left_tip3_z",
    "left_tip4_x", "left_tip4_y", "left_tip4_z",
    "right_tip0_x", "right_tip0_y", "right_tip0_z",
    "right_tip1_x", "right_tip1_y", "right_tip1_z",
    "right_tip2_x", "right_tip2_y", "right_tip2_z",
    "right_tip3_x", "right_tip3_y", "right_tip3_z",
    "right_tip4_x", "right_tip4_y", "right_tip4_z",
]

OUTPUT_NAMES = [
    "left_wrist_x", "left_wrist_y", "left_wrist_z",
    *[f"left_wrist_rotation_6d_{i}" for i in range(6)],
    "right_wrist_x", "right_wrist_y", "right_wrist_z",
    *[f"right_wrist_rotation_6d_{i}" for i in range(6)],
    *[f"left_fingertip_{i}_{axis}" for i in range(5) for axis in "xyz"],
    *[f"right_fingertip_{i}_{axis}" for i in range(5) for axis in "xyz"],
]

OUTPUT_FIELDS = (
    ("left_wrist_position", 0, 3, None),
    ("left_wrist_rotation_6d", 3, 9, "rotation_6d"),
    ("right_wrist_position", 9, 12, None),
    ("right_wrist_rotation_6d", 12, 18, "rotation_6d"),
    ("left_fingertip_position", 18, 33, None),
    ("right_fingertip_position", 33, 48, None),
)

RELATIVE_KEYS = {
    "left_wrist_position": (0, 3),
    "right_wrist_position": (9, 12),
    "left_fingertip_position": (18, 33),
    "right_fingertip_position": (33, 48),
}


def load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The export needs pandas and a Parquet engine (pyarrow or fastparquet). "
            "Run it in the LeRobot/training environment."
        ) from exc
    return pd


def quaternion_xyzw_to_rotation6d(quaternion: np.ndarray) -> np.ndarray:
    """Match pytorch3d quaternion_to_matrix + matrix_to_rotation_6d."""
    q = np.asarray(quaternion, dtype=np.float32)
    x, y, z, w = [q[..., i] for i in range(4)]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    matrix = np.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        axis=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    # pytorch3d matrix_to_rotation_6d flattens the first two columns.
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def convert_vector(values: object) -> np.ndarray:
    array = np.stack(values).astype(np.float32, copy=False)
    if array.ndim != 2 or array.shape[1] != 44:
        raise ValueError(f"Expected a [N,44] vector column, got {array.shape}")
    return np.concatenate(
        [
            array[:, 0:3],
            quaternion_xyzw_to_rotation6d(array[:, 3:7]),
            array[:, 7:10],
            quaternion_xyzw_to_rotation6d(array[:, 10:14]),
            array[:, 14:29],
            array[:, 29:44],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


class StreamingStats:
    """Mean/std/min/max plus a bounded reservoir for approximate quantiles."""

    def __init__(self, dim: int, reservoir_size: int, seed: int):
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.minimum = np.full(dim, np.inf, dtype=np.float64)
        self.maximum = np.full(dim, -np.inf, dtype=np.float64)
        self.reservoir = np.empty((reservoir_size, dim), dtype=np.float32)
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.size == 0:
            return
        previous_count = self.count
        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0, dtype=np.float64)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0, dtype=np.float64)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
            self.mean += delta * batch_count / total
        self.count += batch_count
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

        if self.reservoir_size == 0:
            return
        if previous_count < self.reservoir_size:
            fill_count = min(batch_count, self.reservoir_size - previous_count)
            self.reservoir[previous_count : previous_count + fill_count] = values[:fill_count]
            values = values[fill_count:]
            first_seen = previous_count + fill_count
            if values.size == 0:
                return
        else:
            first_seen = previous_count
        if values.size == 0:
            return
        # Reservoir sampling over the remaining rows. This is deliberately
        # approximate; exact quantiles would require holding tens of GB.
        for row_offset, row in enumerate(values):
            seen = first_seen + row_offset
            slot = int(self.rng.integers(0, seen + 1))
            if slot < self.reservoir_size:
                self.reservoir[slot] = row

    def as_dict(self) -> dict:
        if self.count == 0:
            raise ValueError("Cannot serialize empty statistics")
        sample_count = min(self.count, self.reservoir_size)
        sample = self.reservoir[:sample_count]
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "std": np.sqrt(self.m2 / self.count).tolist(),
            "count": int(self.count),
            "q01": np.quantile(sample, 0.01, axis=0).tolist(),
            "q10": np.quantile(sample, 0.10, axis=0).tolist(),
            "q50": np.quantile(sample, 0.50, axis=0).tolist(),
            "q90": np.quantile(sample, 0.90, axis=0).tolist(),
            "q99": np.quantile(sample, 0.99, axis=0).tolist(),
        }

    def merge(self, other: "StreamingStats") -> None:
        """Merge one worker's summary into this process."""
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            self.minimum = other.minimum.copy()
            self.maximum = other.maximum.copy()
            if self.reservoir_size:
                n = min(other.count, self.reservoir_size)
                self.reservoir[:n] = other.reservoir[:n]
            return

        left_count = self.count
        right_count = other.count
        total = left_count + right_count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * left_count * right_count / total
        self.mean += delta * right_count / total
        self.count = total
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)

        if self.reservoir_size:
            left_n = min(left_count, self.reservoir_size)
            right_n = min(right_count, self.reservoir_size)
            samples = np.concatenate((self.reservoir[:left_n], other.reservoir[:right_n]))
            if len(samples) > self.reservoir_size:
                indices = self.rng.choice(len(samples), self.reservoir_size, replace=False)
                samples = samples[indices]
            self.reservoir[: len(samples)] = samples



def build_modality() -> dict:
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for kind, original_key in (("state", "observation.state"), ("action", "action")):
        for name, start, end, rotation_type in OUTPUT_FIELDS:
            modality[kind][name] = {
                "original_key": original_key,
                "start": start,
                "end": end,
                "rotation_type": rotation_type,
                "absolute": True,
                "dtype": "float32",
                "range": None,
            }
    modality["video"] = {
        "head": {"original_key": "observation.images.head"},
        "chest": {"original_key": "observation.images.chest"},
    }
    modality["annotation"] = {"task": {"original_key": "task_index"}}
    return modality


def update_info(info: dict) -> dict:
    info = json.loads(json.dumps(info))
    for key in ("observation.state", "action"):
        feature = info["features"][key]
        feature["shape"] = [48]
        feature["names"] = OUTPUT_NAMES
    return info


def atomic_write_parquet(df, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        df.to_parquet(temp_path, index=False)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def convert_parquet_file(job: tuple[Path, Path, int]):
    """Convert one Parquet file in a worker process."""
    input_path, output_path, reservoir_size = job
    pd = load_pandas()
    df = pd.read_parquet(input_path)
    state = convert_vector(df["observation.state"].to_numpy())
    action = convert_vector(df["action"].to_numpy())
    state_stats = StreamingStats(48, reservoir_size, seed=17)
    action_stats = StreamingStats(48, reservoir_size, seed=19)
    relative_stats = {
        key: StreamingStats(end - start, reservoir_size, seed=i + 30)
        for i, (key, (start, end)) in enumerate(RELATIVE_KEYS.items())
    }
    state_stats.update(state)
    action_stats.update(action)
    for key, (start, end) in RELATIVE_KEYS.items():
        relative_stats[key].update(action[:, start:end] - state[:, start:end])
    df["observation.state"] = list(state)
    df["action"] = list(action)
    atomic_write_parquet(df, output_path)
    return state_stats, action_stats, relative_stats


def attach_link_or_copy(source: Path, target: Path, mode: str) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source, target_is_directory=source.is_dir())
    else:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


def estimate_clip(source_root: Path, splits: list[str], gpus: int, images_per_sec: float) -> None:
    total_frames = 0
    for split in splits:
        info = json.loads((source_root / split / "meta/info.json").read_text())
        frames = int(info["total_frames"])
        total_frames += frames
        print(f"{split}: {frames:,} timesteps")

    # Current WanImageEncoder returns [B,257,1280] from one composite image.
    elements = 257 * 1280
    bf16_bytes = total_frames * elements * 2
    print(f"total timesteps: {total_frames:,}")
    print(f"raw camera images decoded: {total_frames * 2:,} (head + chest)")
    print(f"CLIP cache, bf16 [257,1280] per timestep: {bf16_bytes / 2**40:.2f} TiB")
    print(f"CLIP cache, float32 [257,1280] per timestep: {bf16_bytes * 2 / 2**40:.2f} TiB")
    if images_per_sec > 0 and gpus > 0:
        hours = total_frames / (images_per_sec * gpus) / 3600
        print(
            f"compute-only estimate at {images_per_sec:g} composite images/s/GPU on "
            f"{gpus} GPUs: {hours:.1f} h"
        )


def export_split(source: Path, target: Path, reservoir_size: int, workers: int) -> None:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output split is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    source_info = json.loads((source / "meta/info.json").read_text())
    if source_info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected v3.0 at {source}, got {source_info.get('codebase_version')!r}")
    for key in ("observation.state", "action"):
        feature = source_info["features"][key]
        if feature.get("shape") != [44] or feature.get("names") != SOURCE_NAMES:
            raise ValueError(f"Unexpected {key} schema at {source}; refusing to guess conversion")

    state_stats = StreamingStats(48, reservoir_size, seed=17)
    action_stats = StreamingStats(48, reservoir_size, seed=19)
    relative_stats = {key: StreamingStats(end - start, reservoir_size, seed=i + 30)
                      for i, (key, (start, end)) in enumerate(RELATIVE_KEYS.items())}

    input_paths = sorted((source / "data").glob("*/*.parquet"))
    jobs = [
        (input_path, target / input_path.relative_to(source), reservoir_size)
        for input_path in input_paths
    ]
    if workers == 1:
        results = map(convert_parquet_file, jobs)
        for worker_state, worker_action, worker_relative in results:
            state_stats.merge(worker_state)
            action_stats.merge(worker_action)
            for key in RELATIVE_KEYS:
                relative_stats[key].merge(worker_relative[key])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for worker_state, worker_action, worker_relative in executor.map(convert_parquet_file, jobs):
                state_stats.merge(worker_state)
                action_stats.merge(worker_action)
                for key in RELATIVE_KEYS:
                    relative_stats[key].merge(worker_relative[key])

    meta_out = target / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    for path in (source / "meta").iterdir():
        if path.name in {"info.json", "stats.json", "modality.json", "embodiment.json", "relative_stats_dreamzero.json"}:
            continue
        if path.is_file():
            shutil.copy2(path, meta_out / path.name)
        elif path.is_dir():
            shutil.copytree(path, meta_out / path.name)

    info = update_info(source_info)
    (meta_out / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    source_stats = json.loads((source / "meta/stats.json").read_text())
    source_stats["observation.state"] = state_stats.as_dict()
    source_stats["action"] = action_stats.as_dict()
    (meta_out / "stats.json").write_text(json.dumps(source_stats, indent=2) + "\n")
    (meta_out / "modality.json").write_text(json.dumps(build_modality(), indent=2) + "\n")
    (meta_out / "embodiment.json").write_text(json.dumps({
        "robot_type": "dual_arm_dexterous_hand",
        "embodiment_tag": "dual_arm_dexterous_hand",
    }, indent=2) + "\n")
    (meta_out / "relative_stats_dreamzero.json").write_text(json.dumps(
        {key: stats.as_dict() for key, stats in relative_stats.items()}, indent=2
    ) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["dagger", "multitask", "singletask", "val"])
    parser.add_argument("--video-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--text-embedding-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--latent-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--reservoir-size", type=int, default=100_000)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parquet worker processes per split; 8-16 is a reasonable EFS starting point.",
    )
    parser.add_argument("--estimate-clip", action="store_true")
    parser.add_argument("--clip-gpus", type=int, default=8)
    parser.add_argument("--clip-images-per-sec-per-gpu", type=float, default=200.0)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    splits = args.splits
    if args.estimate_clip:
        estimate_clip(source_root, splits, args.clip_gpus, args.clip_images_per_sec_per_gpu)
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --estimate-clip is used")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for split in splits:
        source = source_root / split
        target = output_root / split
        export_split(source, target, args.reservoir_size, args.workers)
        attach_link_or_copy(source / "videos", target / "videos", args.video_mode)
        if (source / "text_embs").exists():
            attach_link_or_copy(source / "text_embs", target / "text_embs", args.text_embedding_mode)
        if (source / "latents").exists():
            attach_link_or_copy(source / "latents", target / "latents", args.latent_mode)
        print(f"exported {split} -> {target}")


if __name__ == "__main__":
    main()
