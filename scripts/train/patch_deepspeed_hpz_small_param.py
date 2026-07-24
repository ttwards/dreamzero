#!/usr/bin/env python3
"""Patch DeepSpeed hpZ partitioning for parameters smaller than the hpZ group.

DeepSpeed's secondary parameter partitioner computes ``sec_numel == 0`` for
hpZ ranks that own no elements of a small parameter, but still calls
``Tensor.narrow`` with a start beyond the end of the parameter.  PyTorch
rejects that zero-length slice.  Skip the copy when the local partition is
empty.

The patch is intentionally source-checked and idempotent.  If DeepSpeed
changes the affected code, this script fails instead of modifying an unknown
version.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


VULNERABLE_BLOCK = """\
            # copy from full tensor to secondary tensor
            with torch.no_grad():
                # make sure param.ds_secondary_tensor requires_grad always be false
                param.ds_secondary_tensor.narrow(0, 0,
                                                 sec_numel).copy_(one_dim_param.narrow(0, secondary_start, sec_numel))
"""

PATCHED_BLOCK = """\
            # Ranks past the unpadded end own no elements.  Avoid an invalid
            # zero-length narrow whose start is greater than tensor.numel().
            if sec_numel > 0:
                with torch.no_grad():
                    # make sure param.ds_secondary_tensor requires_grad always be false
                    param.ds_secondary_tensor.narrow(0, 0, sec_numel).copy_(
                        one_dim_param.narrow(0, secondary_start, sec_numel))
"""


def _hpz_enabled(config_path: Path) -> bool:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    hpz_size = config.get("zero_optimization", {}).get("zero_hpz_partition_size", 1)
    return isinstance(hpz_size, int) and hpz_size > 1


def _deepspeed_partition_source() -> Path:
    spec = importlib.util.find_spec("deepspeed")
    if spec is None or spec.origin is None:
        raise RuntimeError("DeepSpeed is not importable from the selected training runtime")
    return Path(spec.origin).parent / "runtime" / "zero" / "partition_parameters.py"


def _apply_patch(source_path: Path) -> bool:
    source = source_path.read_text(encoding="utf-8")
    if PATCHED_BLOCK in source:
        return False
    if VULNERABLE_BLOCK not in source:
        raise RuntimeError(
            f"DeepSpeed hpZ source does not match the expected vulnerable code: {source_path}"
        )
    source_path.write_text(source.replace(VULNERABLE_BLOCK, PATCHED_BLOCK, 1), encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    if not _hpz_enabled(args.config):
        return

    source_path = _deepspeed_partition_source()
    changed = _apply_patch(source_path)
    state = "applied" if changed else "already applied"
    print(f"DeepSpeed hpZ small-parameter compatibility patch: {state} ({source_path})")


if __name__ == "__main__":
    main()
