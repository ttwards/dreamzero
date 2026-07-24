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

DeepCompile traces ZeRO-3 parameters using their full shapes, then releases
them before TorchDynamo builds tensor-match guards. Keep the full symbolic
shape for tracing while recording the stable released representation for
guards. This is a backport of the upstream DeepSpeed guard-stability fix.

DeepCompile also keeps forward inputs in a module-global FIFO. With multiple
graph breaks, AOTAutograd can invoke graph compiler closures out of FIFO order,
pairing one graph's parameter indices with another graph's inputs. Keep the
one-shot real inputs in each backend closure instead.

DeepCompile's ZeRO-3 partitioner forces parameter-derived activations to be
saved unless they are aliases or casts. Wan's per-block timestep modulation is
a cheap add involving a trainable parameter, but its output is nearly 1 GiB at
the training sequence length. Recompute that add in backward instead of saving
one copy for every transformer block.

The patches are source-checked and idempotent. If DeepSpeed changes an
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

DEEPCOMPILE_GUARD_VULNERABLE_BLOCK = """\
def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):
        data = torch.rand(t.ds_shape,
                          dtype=t.dtype,
                          layout=t.layout,
                          device=t.device,
                          pin_memory=t.is_pinned(),
                          requires_grad=t.requires_grad)
        if isinstance(t, torch.nn.Parameter):
            t = torch.nn.Parameter(data, requires_grad=t.requires_grad)
        else:
            t = data
    return t


def patch_fake_tensor():
    # dynamo tracer uses wrap_to_fake_tensor_and_record
    # Wrapping FakeTensorMode.from_tensor is not sufficient as dynamo generates SymbolicContext before calling from_tensor
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record

    def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
        dummy_tensor = wrap_if_ds_param(t)
        ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.tensor_to_context[t] = tracing_context.tensor_to_context.pop(dummy_tensor)
        return ret
"""

DEEPCOMPILE_GUARD_PATCHED_BLOCK = """\
def wrap_if_ds_param(t):
    if hasattr(t, 'ds_id'):
        data = torch.rand(t.ds_shape,
                          dtype=t.dtype,
                          layout=t.layout,
                          device=t.device,
                          pin_memory=t.is_pinned(),
                          requires_grad=t.requires_grad)
        if isinstance(t, torch.nn.Parameter):
            t = torch.nn.Parameter(data, requires_grad=t.requires_grad)
        else:
            t = data
    return t


def _get_guard_sizes_strides(t):
    if hasattr(t, "ds_id"):
        # ZeRO-3 may temporarily all-gather a parameter during tracing, but
        # Dynamo guards run after DeepSpeed releases it back to empty(0).
        released = torch.empty(0, dtype=t.dtype, device=t.device)
        return released.size(), released.stride()
    return t.size(), t.stride()


def patch_fake_tensor():
    # dynamo tracer uses wrap_to_fake_tensor_and_record
    # Wrapping FakeTensorMode.from_tensor is not sufficient as dynamo generates SymbolicContext before calling from_tensor
    original_wrap_to_fake_tensor_and_record = wrap_to_fake_tensor_and_record

    def wrap_to_fake_tensor_and_record_wrapper(t, *args, **kwargs):
        dummy_tensor = wrap_if_ds_param(t)
        ret = original_wrap_to_fake_tensor_and_record(dummy_tensor, *args, **kwargs)
        tx = kwargs.get("tx") if "tx" in kwargs else args[0]
        source = kwargs.get("source")
        if tracing_context := torch._guards.TracingContext.try_get():
            tracing_context.tensor_to_context[t] = tracing_context.tensor_to_context.pop(dummy_tensor)
        if source is not None:
            # Preserve the full ds_shape symbolic context, but guard against
            # the stable released ZeRO-3 representation.
            size, stride = _get_guard_sizes_strides(t)
            tx.output.input_source_to_sizes_strides[source] = {
                "size": size,
                "stride": stride,
            }
        return ret
"""

DEEPCOMPILE_INPUT_STORE_VULNERABLE_BLOCK = """\
        global fwd_real_inputs

        # Create an InputStorage instance for this specific graph
        # It will be captured by the make_fw_graph closure, eliminating the need for graph ID management
        input_storage = InputStorage(keep_int_input_tensors=compile_config.keep_int_input_tensors,
                                     keep_all_input_tensors=compile_config.keep_all_input_tensors)

        # Store in both list (for backward compatibility) and storage (for persistence)
        # The input_storage keeps tensor metadata to handle cases where
        # backend_fn is called once but make_fw_graph is called multiple times
        fwd_real_inputs.append(real_inputs)
        input_storage.put(real_inputs)
"""

DEEPCOMPILE_INPUT_STORE_PATCHED_BLOCK = """\
        # AOTAutograd may invoke graph compiler closures out of creation order.
        # Keep the first call's real inputs local to this graph instead of
        # sharing DeepSpeed's module-global FIFO across graph breaks.
        graph_real_inputs = [real_inputs]

        # Retain graph-local metadata for a repeated make_fw_graph invocation.
        input_storage = InputStorage(keep_int_input_tensors=compile_config.keep_int_input_tensors,
                                     keep_all_input_tensors=compile_config.keep_all_input_tensors)
        input_storage.put(real_inputs)
"""

DEEPCOMPILE_INPUT_LOAD_VULNERABLE_BLOCK = """\
            # Try to get real_inputs from the list first, then from storage
            if fwd_real_inputs:
                real_inputs = fwd_real_inputs.pop(0)
            elif input_storage.has_data():
                # Note: input_storage is captured from the enclosing backend_fn scope
                # Materialize tensors from storage when list is empty
                log_rank0(f"Retrieving real inputs from storage for graph_id={graph_id}", enable=debug_log)
                real_inputs = input_storage.get()
            else:
                raise RuntimeError(f"No real inputs available for graph_id {graph_id}. "
                                   f"List size: {len(fwd_real_inputs)}, Storage has data: {input_storage.has_data()}")
"""

DEEPCOMPILE_INPUT_LOAD_PATCHED_BLOCK = """\
            # Consume this graph's own real inputs on the first compiler call.
            # Repeated calls reconstruct inputs from the graph-local metadata.
            if graph_real_inputs:
                real_inputs = graph_real_inputs.pop(0)
            elif input_storage.has_data():
                log_rank0(f"Retrieving real inputs from storage for graph_id={graph_id}", enable=debug_log)
                real_inputs = input_storage.get()
            else:
                raise RuntimeError(f"No real inputs available for graph_id {graph_id}. "
                                   f"Queue size: {len(graph_real_inputs)}, Storage has data: {input_storage.has_data()}")
"""

DEEPCOMPILE_INPUT_CHECK_VULNERABLE_BLOCK = """\
            real_inputs = set_example_values_to_symints(real_inputs)

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)
"""

DEEPCOMPILE_INPUT_CHECK_PATCHED_BLOCK = """\
            real_inputs = set_example_values_to_symints(real_inputs)

            max_param_index = max((index for index, _, _ in param_indices), default=-1)
            if max_param_index >= len(real_inputs):
                raise RuntimeError(
                    f"DeepCompile graph-local input mismatch for graph_id={graph_id}: "
                    f"input_count={len(real_inputs)}, max_param_index={max_param_index}, "
                    f"parameter_count={len(param_indices)}")

            param_manager[graph_id] = DSGraphParamManager(gm.graph, real_inputs, param_indices)
"""

DEEPCOMPILE_PARAM_POINTWISE_VULNERABLE_BLOCK = """\
    no_copy_ops = get_no_copy_ops()

    def need_recompute(n: Node) -> bool:
        if n.op == "call_function":
            is_cast, _ = is_cast_op(n)
            return n.target in no_copy_ops or is_cast
        return False
"""

DEEPCOMPILE_PARAM_POINTWISE_PATCHED_BLOCK = """\
    no_copy_ops = get_no_copy_ops()
    # Saving the output of a cheap parameter-dependent add can be much more
    # expensive than recomputing it. Wan timestep modulation produces one
    # [batch, sequence, 6, hidden] tensor per transformer block (~934 MiB at
    # the production shape), otherwise defeating activation checkpointing.
    parameter_pointwise_ops = {
        torch.ops.aten.add.Tensor,
    }

    def need_recompute(n: Node) -> bool:
        if n.op == "call_function":
            is_cast, _ = is_cast_op(n)
            return (
                n.target in no_copy_ops
                or n.target in parameter_pointwise_ops
                or is_cast
            )
        return False
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
    patched_markers: tuple[str, ...] = (),
) -> bool:
    source = source_path.read_text(encoding="utf-8")
    if patched_block in source or (
        patched_markers and all(marker in source for marker in patched_markers)
    ):
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


def _apply_patch_set(
    source_path: Path,
    replacements: tuple[tuple[str, str], ...],
    description: str,
) -> bool:
    source = source_path.read_text(encoding="utf-8")
    patched = [patched_block in source for _, patched_block in replacements]
    if all(patched):
        return False
    if any(patched):
        raise RuntimeError(
            f"DeepSpeed {description} source is only partially patched: {source_path}"
        )

    missing = [
        index
        for index, (vulnerable_block, _) in enumerate(replacements)
        if vulnerable_block not in source
    ]
    if missing:
        raise RuntimeError(
            f"DeepSpeed {description} source does not match expected blocks "
            f"{missing}: {source_path}"
        )

    updated = source
    for vulnerable_block, patched_block in replacements:
        updated = updated.replace(vulnerable_block, patched_block, 1)
    source_path.write_text(updated, encoding="utf-8")
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
                (),
            )
        )
    if _deepcompile_zero3_enabled(config):
        patches.append(
            (
                "DeepCompile frozen-parameter registration",
                _deepspeed_source("compile/init_z3.py"),
                DEEPCOMPILE_VULNERABLE_BLOCK,
                DEEPCOMPILE_PATCHED_BLOCK,
                (),
            )
        )
        patches.append(
            (
                "DeepCompile ZeRO-3 guard stability",
                _deepspeed_source("compile/patch_fake_tensor.py"),
                DEEPCOMPILE_GUARD_VULNERABLE_BLOCK,
                DEEPCOMPILE_GUARD_PATCHED_BLOCK,
                (
                    "def _get_guard_sizes_strides(t):",
                    "tx.output.input_source_to_sizes_strides[source]",
                ),
            )
        )
        patches.append(
            (
                "DeepCompile parameter-derived pointwise recomputation",
                _deepspeed_source("compile/partitioner.py"),
                DEEPCOMPILE_PARAM_POINTWISE_VULNERABLE_BLOCK,
                DEEPCOMPILE_PARAM_POINTWISE_PATCHED_BLOCK,
                (),
            )
        )

    for description, source_path, vulnerable_block, patched_block, patched_markers in patches:
        changed = _apply_patch(
            source_path,
            vulnerable_block,
            patched_block,
            description,
            patched_markers,
        )
        state = "applied" if changed else "already applied"
        print(f"DeepSpeed {description} compatibility patch: {state} ({source_path})")

    if _deepcompile_zero3_enabled(config):
        source_path = _deepspeed_source("compile/backend.py")
        changed = _apply_patch_set(
            source_path,
            (
                (
                    DEEPCOMPILE_INPUT_STORE_VULNERABLE_BLOCK,
                    DEEPCOMPILE_INPUT_STORE_PATCHED_BLOCK,
                ),
                (
                    DEEPCOMPILE_INPUT_LOAD_VULNERABLE_BLOCK,
                    DEEPCOMPILE_INPUT_LOAD_PATCHED_BLOCK,
                ),
                (
                    DEEPCOMPILE_INPUT_CHECK_VULNERABLE_BLOCK,
                    DEEPCOMPILE_INPUT_CHECK_PATCHED_BLOCK,
                ),
            ),
            "DeepCompile graph-local forward inputs",
        )
        state = "applied" if changed else "already applied"
        print(
            f"DeepSpeed DeepCompile graph-local forward inputs compatibility "
            f"patch: {state} ({source_path})"
        )


if __name__ == "__main__":
    main()
