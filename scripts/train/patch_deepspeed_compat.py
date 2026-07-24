#!/usr/bin/env python3
"""Apply narrow DreamZero compatibility fixes to the installed DeepSpeed.

DeepSpeed's secondary parameter partitioner computes ``sec_numel == 0`` for
hpZ ranks that own no elements of a small parameter, but still calls
``Tensor.narrow`` with a start beyond the end of the parameter. PyTorch
rejects that zero-length slice. Skip the copy when the local partition is
empty.

DeepCompile's ZeRO-3 setup also assumes every model parameter has an optimizer
gradient partition. Frozen text/image/VAE parameters intentionally are not in
the optimizer, so register those parameters with an empty gradient buffer.

The patches are source-checked and idempotent. If DeepSpeed changes either
affected block, this script fails instead of modifying an unknown version.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


HPZ_VULNERABLE_BLOCK = """\
            # copy from full tensor to secondary tensor
            with torch.no_grad():
                # make sure param.ds_secondary_tensor requires_grad always be false
                param.ds_secondary_tensor.narrow(0, 0,
                                                 sec_numel).copy_(one_dim_param.narrow(0, secondary_start, sec_numel))
"""

HPZ_PATCHED_BLOCK = """\
            # Ranks past the unpadded end own no elements.  Avoid an invalid
            # zero-length narrow whose start is greater than tensor.numel().
            if sec_numel > 0:
                with torch.no_grad():
                    # make sure param.ds_secondary_tensor requires_grad always be false
                    param.ds_secondary_tensor.narrow(0, 0, sec_numel).copy_(
                        one_dim_param.narrow(0, secondary_start, sec_numel))
"""

DEEPCOMPILE_VULNERABLE_BLOCK = """\
    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        if use_opt:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]
"""

DEEPCOMPILE_PATCHED_BLOCK = """\
    for p in engine.module.parameters():
        grad_buffer = torch.Tensor()
        if use_opt and p.requires_grad:
            grad_buffer = optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition[p.ds_id]
"""


def _load_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _hpz_enabled(config: dict) -> bool:
    hpz_size = config.get("zero_optimization", {}).get("zero_hpz_partition_size", 1)
    return isinstance(hpz_size, int) and hpz_size > 1


def _deepcompile_zero3_enabled(config: dict) -> bool:
    return (
        config.get("zero_optimization", {}).get("stage") == 3
        and config.get("compile", {}).get("deepcompile") is True
    )


def _deepspeed_source(relative_path: str) -> Path:
    spec = importlib.util.find_spec("deepspeed")
    if spec is None or spec.origin is None:
        raise RuntimeError("DeepSpeed is not importable from the selected training runtime")
    return Path(spec.origin).parent / relative_path


def _apply_patch(
    source_path: Path,
    vulnerable_block: str,
    patched_block: str,
    description: str,
) -> bool:
    source = source_path.read_text(encoding="utf-8")
    if patched_block in source:
        return False
    if vulnerable_block not in source:
        raise RuntimeError(
            f"DeepSpeed {description} source does not match the expected code: {source_path}"
        )
    source_path.write_text(
        source.replace(vulnerable_block, patched_block, 1),
        encoding="utf-8",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = _load_config(args.config)
    patches = []
    if _hpz_enabled(config):
        patches.append(
            (
                "hpZ small-parameter partition",
                _deepspeed_source("runtime/zero/partition_parameters.py"),
                HPZ_VULNERABLE_BLOCK,
                HPZ_PATCHED_BLOCK,
            )
        )
    if _deepcompile_zero3_enabled(config):
        patches.append(
            (
                "DeepCompile frozen-parameter registration",
                _deepspeed_source("compile/init_z3.py"),
                DEEPCOMPILE_VULNERABLE_BLOCK,
                DEEPCOMPILE_PATCHED_BLOCK,
            )
        )

    for description, source_path, vulnerable_block, patched_block in patches:
        changed = _apply_patch(
            source_path,
            vulnerable_block,
            patched_block,
            description,
        )
        state = "applied" if changed else "already applied"
        print(f"DeepSpeed {description} compatibility patch: {state} ({source_path})")


if __name__ == "__main__":
    main()
