"""
Add the GEAR/DreamZero metadata sidecars for an EgoSteer LeRobot v3 split.

This is intentionally a metadata-only operation. It does not rewrite Parquet
files, videos, task embeddings, or any other raw data. The remote
``egosteer-realworld`` export stores wrist rotations as quaternions (44D
state/action vectors); the generated modality maps those slices to the
semantic 48D EgoVLA keys, and the training transform converts quaternion to
rotation-6D online.

Examples:
  # One split, in place:
  python scripts/data/add_lerobot_v3_egosteer_metadata.py \
      --dataset-path /path/to/egosteer-realworld/val

  # All split directories below the dataset root:
  python scripts/data/add_lerobot_v3_egosteer_metadata.py \
      --dataset-path /path/to/egosteer-realworld --recursive

  # Preview the files without writing them:
  python scripts/data/add_lerobot_v3_egosteer_metadata.py \
      --dataset-path /path/to/egosteer-realworld --recursive --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EMBODIMENT_TAG = "dual_arm_dexterous_hand"

# Raw EgoSteer v3 layout. The rotation fields are 4D quaternions in the
# parquet files; their output names retain the 6D semantic key expected by the
# existing DreamZero dual-arm config because the transform performs the
# representation conversion online.
VECTOR_FIELDS = (
    ("left_wrist_position", 0, 3, None),
    ("left_wrist_rotation_6d", 3, 7, "quaternion"),
    ("right_wrist_position", 7, 10, None),
    ("right_wrist_rotation_6d", 10, 14, "quaternion"),
    ("left_fingertip_position", 14, 29, None),
    ("right_fingertip_position", 29, 44, None),
)


def _feature_dim(feature: dict) -> int:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"Expected a one-dimensional feature shape, got {shape!r}")
    return int(shape[0])


def _validate_vector_feature(info: dict, key: str) -> dict:
    feature = info.get("features", {}).get(key)
    if feature is None:
        raise ValueError(f"Missing required feature {key!r}")
    if _feature_dim(feature) != 44:
        raise ValueError(f"Expected {key!r} to have shape [44], got {feature.get('shape')!r}")
    if feature.get("dtype") != "float32":
        raise ValueError(f"Expected {key!r} to have dtype float32, got {feature.get('dtype')!r}")
    expected_names = [
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
    if feature.get("names") != expected_names:
        raise ValueError(
            f"The {key!r} field names do not match the known EgoSteer v3 layout; "
            "refusing to guess slice semantics."
        )
    return feature


def build_modality(info: dict) -> dict:
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')!r}")

    state_feature = _validate_vector_feature(info, "observation.state")
    action_feature = _validate_vector_feature(info, "action")
    if state_feature.get("names") != action_feature.get("names"):
        raise ValueError("State and action field names differ; refusing to create a shared layout")

    modality = {"state": {}, "action": {}, "video": {}, "annotation": {}}
    for modality_name, feature_key in (
        ("state", "observation.state"),
        ("action", "action"),
    ):
        dtype = info["features"][feature_key]["dtype"]
        for name, start, end, rotation_type in VECTOR_FIELDS:
            modality[modality_name][name] = {
                "original_key": feature_key,
                "start": start,
                "end": end,
                "rotation_type": rotation_type,
                "absolute": True,
                "dtype": dtype,
                "range": None,
            }

    features = info.get("features", {})
    for camera in ("head", "chest"):
        key = f"observation.images.{camera}"
        if key not in features or features[key].get("dtype") != "video":
            raise ValueError(f"Missing required video feature {key!r}")
        modality["video"][camera] = {"original_key": key}

    if "task_index" not in features:
        raise ValueError("Missing required v3 task_index feature")
    modality["annotation"]["task"] = {"original_key": "task_index"}
    return modality


def dataset_paths(root: Path, recursive: bool) -> list[Path]:
    if (root / "meta" / "info.json").exists():
        return [root]
    if not recursive:
        raise FileNotFoundError(
            f"No meta/info.json under {root}; use --recursive for a directory containing splits"
        )
    return sorted({path.parent.parent for path in root.glob("*/meta/info.json")})


def add_metadata(dataset_path: Path, force: bool, dry_run: bool) -> None:
    info_path = dataset_path / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    modality = build_modality(info)
    sidecars = {
        dataset_path / "meta" / "modality.json": modality,
        dataset_path / "meta" / "embodiment.json": {
            "robot_type": EMBODIMENT_TAG,
            "embodiment_tag": EMBODIMENT_TAG,
        },
    }
    for path, payload in sidecars.items():
        if path.exists() and not force:
            print(f"{dataset_path.name}: exists, skip {path.name} (use --force to overwrite)")
            continue
        if dry_run:
            print(f"{dataset_path.name}: would write {path.name}")
            continue
        path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"{dataset_path.name}: wrote {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.dataset_path.resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = dataset_paths(root, args.recursive)
    if not paths:
        raise FileNotFoundError(f"No LeRobot v3 split found below {root}")
    for path in paths:
        add_metadata(path, args.force, args.dry_run)


if __name__ == "__main__":
    main()
