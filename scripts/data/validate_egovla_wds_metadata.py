#!/usr/bin/env python3
"""Validate DreamZero normalization metadata for the EgoVLA WDS embodiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from groot.vla.data.schema import DatasetMetadata, EmbodimentTag, RotationType


FIELD_DIMS = {
    "left_wrist_position": 3,
    "right_wrist_position": 3,
    "left_wrist_rotation_6d": 6,
    "right_wrist_rotation_6d": 6,
    "left_fingertip_position": 15,
    "right_fingertip_position": 15,
}
STATISTIC_NAMES = ("min", "max", "mean", "std", "q01", "q99")


def validate_metadata(path: Path) -> DatasetMetadata:
    raw = json.loads(path.read_text())
    if "embodiment_tag" not in raw:
        raw = raw[EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND.value]
    metadata = DatasetMetadata.model_validate(raw)

    expected_tag = EmbodimentTag.DUAL_ARM_DEXTEROUS_HAND
    if metadata.embodiment_tag != expected_tag:
        raise ValueError(
            f"Expected embodiment {expected_tag.value}, got {metadata.embodiment_tag.value}"
        )

    expected_fields = set(FIELD_DIMS)
    for modality_name in ("state", "action"):
        modalities = getattr(metadata.modalities, modality_name)
        statistics = getattr(metadata.statistics, modality_name)
        if set(modalities) != expected_fields:
            raise ValueError(
                f"{modality_name} modality fields differ: "
                f"expected={sorted(expected_fields)}, got={sorted(modalities)}"
            )
        if set(statistics) != expected_fields:
            raise ValueError(
                f"{modality_name} statistic fields differ: "
                f"expected={sorted(expected_fields)}, got={sorted(statistics)}"
            )

        for field, dim in FIELD_DIMS.items():
            modality = modalities[field]
            if modality.shape != (dim,):
                raise ValueError(
                    f"{modality_name}.{field} shape is {modality.shape}, expected {(dim,)}"
                )
            expected_rotation = (
                RotationType.ROTATION_6D if field.endswith("rotation_6d") else None
            )
            if modality.rotation_type != expected_rotation:
                raise ValueError(
                    f"{modality_name}.{field} rotation_type is "
                    f"{modality.rotation_type}, expected {expected_rotation}"
                )
            if not modality.continuous:
                raise ValueError(f"{modality_name}.{field} must be continuous")

            values = statistics[field]
            arrays = {}
            for statistic_name in STATISTIC_NAMES:
                array = np.asarray(getattr(values, statistic_name))
                if array.shape != (dim,):
                    raise ValueError(
                        f"{modality_name}.{field}.{statistic_name} shape is "
                        f"{array.shape}, expected {(dim,)}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(
                        f"{modality_name}.{field}.{statistic_name} contains non-finite values"
                    )
                arrays[statistic_name] = array

            tolerance = 1e-6
            ordered_pairs = (("min", "q01"), ("q01", "q99"), ("q99", "max"))
            for lower_name, upper_name in ordered_pairs:
                if np.any(arrays[lower_name] > arrays[upper_name] + tolerance):
                    raise ValueError(
                        f"{modality_name}.{field} violates "
                        f"{lower_name} <= {upper_name}"
                    )
            if np.any(arrays["mean"] < arrays["min"] - tolerance) or np.any(
                arrays["mean"] > arrays["max"] + tolerance
            ):
                raise ValueError(f"{modality_name}.{field}.mean lies outside min/max")
            if np.any(arrays["std"] < 0):
                raise ValueError(f"{modality_name}.{field}.std contains negative values")

    video_fields = set(metadata.modalities.video)
    if video_fields != {"head", "chest"}:
        raise ValueError(f"Expected head/chest video fields, got {sorted(video_fields)}")
    for field, video in metadata.modalities.video.items():
        if video.resolution != (640, 480):
            raise ValueError(
                f"video.{field} resolution is {video.resolution}, expected (640, 480)"
            )
        if video.channels != 3 or video.fps != 30.0:
            raise ValueError(
                f"video.{field} metadata is channels={video.channels}, fps={video.fps}"
            )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    metadata = validate_metadata(args.metadata)
    state_dim = sum(value.shape[0] for value in metadata.modalities.state.values())
    action_dim = sum(value.shape[0] for value in metadata.modalities.action.values())
    print(
        f"Validated {args.metadata}: embodiment={metadata.embodiment_tag.value}, "
        f"state_dim={state_dim}, action_dim={action_dim}, "
        f"video_fields={sorted(metadata.modalities.video)}"
    )


if __name__ == "__main__":
    main()
