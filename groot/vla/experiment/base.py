# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/

from abc import ABC
import contextlib
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Optional
import warnings

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, Dataset, Sampler
import transformers
from transformers.integrations import WandbCallback
from transformers import TrainerCallback, set_seed
from transformers.trainer import (
    # ALL_LAYERNORM_LAYERS,  # ShardedDDPOption,  # Removed deprecated import
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)

import groot.vla.common.utils as U
from groot.vla.data.dataset.lerobot_sharded import ShardedLeRobotMixtureDataset
from groot.vla.data.nvimgcodec import (
    NVIMGCODEC_RAW_KEY,
    NvImageCodecBatchProcessor,
    raw_samples_collate,
)
from groot.vla.data.schema import EmbodimentTag
from groot.vla.data.transform import ComposedModalityTransform
from groot.vla.experiment.utils import (
    compute_grad_accum_to_match_global_bs,
    dtype_from_string,
    get_checkpoint_path,
    mprint,
    safe_save_model_for_hf_trainer,
)
from groot.vla.utils.timer import ContextTimer

# Fix resume: https://github.com/huggingface/transformers/pull/34632/files
np_core = np.core
allowlist = [np_core.multiarray._reconstruct, np.ndarray, np.dtype]
# numpy >1.25 defines numpy.dtypes.UInt32DType, but below works for
# all versions of numpy
allowlist += [type(np.dtype(np.uint32))]
torch.serialization.add_safe_globals(allowlist)

# Define LayerNorm classes locally to replace deprecated ALL_LAYERNORM_LAYERS
LAYERNORM_LAYERS = [
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LocalResponseNorm,
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
]


_EXPANDABLE_ACTION_ADAPTER_KEYS = (
    ".action_encoder.W1.W",
    ".action_decoder.layer2.W",
    ".action_decoder.layer2.b",
)


def expand_action_adapter_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Partially load action adapters when max_action_dim was increased."""
    for key, source in list(state_dict.items()):
        target = target_state_dict.get(key)
        if target is None or source.shape == target.shape:
            continue
        if not any(name in key for name in _EXPANDABLE_ACTION_ADAPTER_KEYS):
            continue
        if source.ndim != target.ndim:
            continue
        expanded = target.detach().clone()
        overlap = tuple(slice(0, min(old, new)) for old, new in zip(source.shape, target.shape))
        expanded[overlap] = source[overlap].to(device=expanded.device, dtype=expanded.dtype)
        state_dict[key] = expanded
        mprint(
            f"Expanded pretrained action adapter {key}: "
            f"{tuple(source.shape)} -> {tuple(target.shape)}"
        )
    return state_dict


class LossLoggerCallback(TrainerCallback):
    """Callback that writes per-step loss metrics to a JSONL file for offline analysis."""

    def __init__(self, output_path: str):
        self.output_path = output_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return
        entry = {"step": state.global_step}
        for key, value in logs.items():
            if key.startswith(
                (
                    "perf/",
                    "time/",
                    "train_loss/",
                    "val_loss/",
                    "weight_norm/",
                    "grad_norm/",
                    "optimizer/",
                )
            ):
                entry[key] = value
        if len(entry) > 1:  # more than just "step"
            with open(self.output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")


def namespace_wandb_logs(logs: dict) -> dict:
    """Normalize Trainer metrics into stable, top-level W&B namespaces.

    Hugging Face's stock W&B callback puts every non-eval metric below
    ``train/``.  Keeping the semantic namespace in the metric itself makes
    W&B create separate sections for performance, timings, losses, and norms.
    This function also accepts the legacy metric names so resumed/older code
    paths do not create a second set of charts.
    """

    exact_names = {
        "loss": "train_loss/total",
        "train_loss": "train_loss/run_mean",
        "dynamics_loss_avg": "train_loss/dynamics",
        "action_loss_avg": "train_loss/action",
        "eval_loss": "val_loss/total",
        "grad_norm": "grad_norm/global_l2",
        "norm/grad_l2": "grad_norm/global_l2",
        "norm/weight_l2": "weight_norm/global_l2",
        "norm/grad_to_weight": "grad_norm/to_weight_ratio",
        "learning_rate": "optimizer/learning_rate",
        "epoch": "progress/epoch",
        "train_runtime": "time/train_runtime_s",
        "train_samples_per_second": "perf/train_samples_per_s",
        "train_steps_per_second": "perf/train_steps_per_s",
        "total_flos": "perf/total_flops",
        "eval_runtime": "time/validation_runtime_s",
        "eval_samples_per_second": "perf/validation_samples_per_s",
        "eval_steps_per_second": "perf/validation_steps_per_s",
    }
    legacy_time_names = {
        "perf/step_time_s": "time/step_s",
        "perf/training_step_s": "time/training_step_s",
        "perf/model_forward_s": "time/model_forward_s",
        "perf/gpu_training_s": "time/cuda_training_span_s",
        "perf/gpu_forward_s": "time/cuda_forward_span_s",
        "perf/gpu_optimizer_s": "time/cuda_optimizer_wrapper_span_s",
        "perf/data_wait_s": "time/data_wait_s",
        "perf/data_wait_fraction": "time/data_wait_fraction",
        "perf/optimizer_and_host_s": "time/trainer_outer_step_s",
    }

    normalized = {}
    for key, value in logs.items():
        target = exact_names.get(key, legacy_time_names.get(key, key))
        if key.startswith("eval_") and key.endswith("_loss") and key != "eval_loss":
            component = key[len("eval_") : -len("_loss")]
            target = f"val_loss/{component}"
        normalized[target] = value
    return normalized


class NamespaceWandbCallback(WandbCallback):
    """W&B callback that preserves DreamZero's top-level metric namespaces."""

    def __init__(self):
        super().__init__()
        self._metrics_defined = False

    def _define_metrics(self, args, state, model):
        if self._metrics_defined:
            return
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero and self._wandb.run is not None:
            wandb = self._wandb
            # The parent setup installs ``train/global_step`` plus a wildcard
            # rule. Hide that compatibility metric and replace the wildcard so
            # it does not create an otherwise empty top-level ``train`` panel.
            wandb.define_metric(
                "train/global_step", hidden=True, overwrite=True
            )
            wandb.define_metric("global_step", hidden=True, overwrite=True)
            wandb.define_metric(
                "*", step_metric="global_step", step_sync=True, overwrite=True
            )
            for namespace in (
                "perf",
                "time",
                "train_loss",
                "val_loss",
                "weight_norm",
                "grad_norm",
                "optimizer",
                "progress",
            ):
                wandb.define_metric(
                    f"{namespace}/*",
                    step_metric="global_step",
                    overwrite=True,
                )
        self._metrics_defined = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        super().on_train_begin(args, state, control, model=model, **kwargs)
        self._define_metrics(args, state, model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is None or self._wandb is None:
            return
        self._define_metrics(args, state, model)
        if state.is_world_process_zero:
            self._wandb.log({**logs, "global_step": state.global_step})


class PerformanceMetricsCallback(TrainerCallback):
    """Emit low-overhead training and hardware metrics at optimizer-step cadence."""

    def __init__(self, trainer, log_steps=10, norm_steps=50):
        self.trainer = trainer
        self.log_steps = max(1, int(log_steps))
        self.norm_steps = max(1, int(norm_steps))
        self.pending_grad_norm = None

    def _model_for_norm(self, kwargs):
        """Prefer the accelerator/DeepSpeed wrapper over the callback's base model."""
        wrapped = getattr(self.trainer, "model_wrapped", None)
        callback_model = kwargs.get("model")
        if wrapped is not None and hasattr(wrapped, "get_global_grad_norm"):
            return wrapped
        return callback_model or wrapped

    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer._start_performance_window(global_step=state.global_step)
        self.pending_grad_norm = None

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = state.global_step + 1
        self.trainer._record_optimizer_event_start()
        if step % self.norm_steps != 0:
            return
        model = self._model_for_norm(kwargs)
        # DeepSpeed populates get_global_grad_norm() inside optimizer.step(),
        # so the pre-step value is from the previous update (and is zero for
        # the first update). For ordinary optimizers, gradients are available
        # here and can be sampled before they are cleared.
        if model is not None and hasattr(model, "get_global_grad_norm"):
            # DeepSpeed's ZeRO-2 engine does not populate get_global_grad_norm()
            # until optimizer.step().  On the first step that value is still 0,
            # so read the live partitioned gradients before they are cleared.
            grad_norm = self.trainer._read_grad_norm(model, live=True)
        else:
            grad_norm = self.trainer._read_grad_norm(model)
        # ZeRO's live norm is a collective and therefore must run on every
        # rank. Only rank 0 retains the value for W&B/JSONL logging.
        if state.is_world_process_zero:
            self.pending_grad_norm = grad_norm

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._record_optimizer_event_end()
        step = state.global_step + 1
        if step % self.norm_steps != 0:
            return
        model = self._model_for_norm(kwargs)
        if model is not None and hasattr(model, "get_global_grad_norm"):
            post_step_grad_norm = self.trainer._read_grad_norm(model)
            if state.is_world_process_zero and (
                post_step_grad_norm > 0.0 or self.pending_grad_norm is None
            ):
                self.pending_grad_norm = post_step_grad_norm

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_steps != 0:
            return
        metrics = self.trainer._finish_performance_window(
            global_step=state.global_step,
            grad_norm=self.pending_grad_norm,
            collect_weight_norm=state.global_step % self.norm_steps == 0,
        )
        if metrics:
            # Do not call Trainer.log() from rank 0 only. Trainer.log() invokes
            # DefaultFlowCallback.on_log(), which clears control.should_log.
            # Doing that on only one rank makes rank 0 skip Trainer's built-in
            # loss gather while the remaining ranks enter its all-gather.
            # Queue the metrics on every rank and merge them into the normal
            # Trainer log below, preserving one identical collective schedule.
            self.trainer._pending_performance_metrics = metrics
            control.should_log = True


class TargetedCompileCallback(TrainerCallback):
    """Compile selected DreamZero submodules after DeepSpeed has prepared them.

    ``TrainingArguments.torch_compile`` asks Accelerate to compile the complete
    model.  That is useful for a small, ordinary model but makes the Wan VLA
    graph unnecessarily large: frozen T5/CLIP/VAE paths and the trainable Wan
    DiT are traced together.  This callback keeps the module registration and
    ZeRO parameter handles unchanged, then replaces only selected ``forward``
    callables after the DeepSpeed engine is ready.

    Supported scopes are ``wan``, ``frozen``, and ``wan_frozen``.  The launcher
    uses ``all`` to request the original whole-model Accelerate path explicitly.
    """

    def __init__(self, trainer, scope: str):
        self.trainer = trainer
        self.scope = scope
        self.completed = False

    @staticmethod
    def _unwrap_model(model):
        """Reach the original VLA module without replacing its parameter tree."""
        current = model
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            # DeepSpeedEngine and DDP expose the actual VLA as ``module``.
            if hasattr(current, "module") and (
                current.__class__.__name__.endswith("Engine")
                or current.__class__.__name__.endswith("DistributedDataParallel")
            ):
                current = current.module
                continue
            break
        return current

    @staticmethod
    def _compile_forward(owner, attribute, label, compile_kwargs):
        forward = getattr(owner, attribute)
        if getattr(forward, "_dreamzero_target_compile", False):
            return False
        compiled = torch.compile(forward, **compile_kwargs)
        # Mark the bound callable before assigning it back to the instance.
        setattr(compiled, "_dreamzero_target_compile", True)
        setattr(owner, attribute, compiled)
        print(f"Targeted torch.compile enabled: {label}", flush=True)
        return True

    @staticmethod
    def _disable_zero3_collectives_for_dynamo():
        """Keep ZeRO-3 parameter communication eager while compiling compute.

        Targeted compilation happens after DeepSpeed has installed ZeRO-3
        parameter hooks.  A compiled submodule can therefore reach
        ``dist.all_gather_into_tensor`` while lazily materializing a parameter.
        Letting Dynamo trace that collective mixes communication buffers with
        the module's compute graph and is not shape-stable across parameters.
        A Dynamo-disabled wrapper creates a graph break around the collective;
        the all-gather still runs normally, but Inductor sees only the tensor
        returned by the eager communication path.
        """
        dynamo = getattr(torch, "_dynamo", None)
        distributed = getattr(torch, "distributed", None)
        if dynamo is None or distributed is None:
            return

        for name in (
            "all_gather_into_tensor",
            "all_gather",
            "all_gather_coalesced",
        ):
            function = getattr(distributed, name, None)
            if function is None or getattr(function, "_dreamzero_dynamo_disabled", False):
                continue
            disabled = dynamo.disable(function)
            setattr(disabled, "_dreamzero_dynamo_disabled", True)
            setattr(distributed, name, disabled)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.completed:
            return

        compile_kwargs = {
            "backend": os.environ.get("TORCH_COMPILE_BACKEND", "inductor"),
            "mode": os.environ.get("TORCH_COMPILE_MODE") or "default",
            "dynamic": os.environ.get("TORCH_COMPILE_DYNAMIC", "false").lower()
            in {"1", "true", "yes", "on"},
            "fullgraph": os.environ.get("TORCH_COMPILE_FULLGRAPH", "false").lower()
            in {"1", "true", "yes", "on"},
        }

        root = self._unwrap_model(
            getattr(self.trainer, "model_wrapped", None) or model or self.trainer.model
        )
        action_head = getattr(root, "action_head", None)
        if action_head is None:
            raise RuntimeError("Targeted compile could not find model.action_head")

        targets = []
        if self.scope in {"wan", "wan_frozen"}:
            targets.append((action_head.model, "forward", "Wan DiT"))
        if self.scope in {"frozen", "wan_frozen"}:
            targets.extend(
                [
                    (action_head.text_encoder, "forward", "frozen T5"),
                    (action_head.image_encoder.model.visual, "forward", "frozen CLIP visual"),
                    (action_head.vae.model, "encode", "frozen VAE encode"),
                ]
            )
        if not targets:
            raise ValueError(
                f"Unsupported targeted compile scope={self.scope!r}; "
                "use wan, frozen, wan_frozen, or all"
            )

        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                f"Targeted torch.compile scope={self.scope} kwargs={compile_kwargs}",
                flush=True,
            )
            print(
                "Targeted torch.compile: keeping ZeRO-3 all-gather collectives eager",
                flush=True,
            )
        self._disable_zero3_collectives_for_dynamo()
        for owner, attribute, label in targets:
            self._compile_forward(owner, attribute, label, compile_kwargs)
        self.completed = True


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ProfCallback(transformers.TrainerCallback):
    """Callback to manage PyTorch profiler during training.

    Dynamically starts/stops the profiler within a specified session step window.
    After profiling completes, triggers optional S3 upload and removes itself.

    Args:
        profile_dir: Directory to save profile traces
        upload_callback: Optional callback to trigger S3 upload after profiling
        profile_start_step: Session step to start profiling (default: 50)
        profile_end_step: Session step to stop profiling
        warmup_steps: Number of warmup steps for profiler schedule (default: 1)
        active_steps: Number of active profiling steps (default: 5)
        trainer: Trainer instance (required for self-removal after profiling)
        record_shapes: Record tensor shapes in profiler (default: False)
        with_stack: Record Python stack traces (default: True)
        profile_memory: Record memory allocation/deallocation (default: False)
    """

    def __init__(
        self,
        profile_dir,
        upload_callback=None,
        profile_start_step=50,
        profile_end_step=55,
        warmup_steps=1,
        active_steps=5,
        trainer=None,
        record_shapes=False,
        with_stack=True,
        profile_memory=False,
        with_flops=True,
        upload_wandb=True,
        global_rank=0,
    ):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.upload_callback = upload_callback
        self.profile_start_step = profile_start_step
        self.profile_end_step = profile_end_step
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.trainer = trainer
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.with_flops = with_flops
        self.upload_wandb = upload_wandb
        self.global_rank = global_rank
        self.upload_triggered = False
        self.starting_global_step = None
        self.session_step = 0
        self.prof = None
        self.profiling_active = False
        self.profiling_complete = False
        self.removed_from_trainer = False
        self.trace_ready = False

    def _on_trace_ready(self, prof):
        """Write a Chrome trace plus compact CPU/CUDA operator summaries."""
        torch.profiler.tensorboard_trace_handler(str(self.profile_dir))(prof)
        events = prof.key_averages()
        cuda_table = events.table(
            sort_by="self_cuda_time_total",
            row_limit=50,
        )
        cpu_table = events.table(
            sort_by="self_cpu_time_total",
            row_limit=50,
        )
        (self.profile_dir / "top_cuda_ops.txt").write_text(cuda_table)
        (self.profile_dir / "top_cpu_ops.txt").write_text(cpu_table)

        total_flops = sum(float(getattr(event, "flops", 0) or 0) for event in events)
        total_device_us = sum(
            float(getattr(event, "self_device_time_total", 0) or 0)
            for event in events
        )
        summary = {
            "global_rank": self.global_rank,
            "profile_start_step": self.profile_start_step,
            "warmup_steps": self.warmup_steps,
            "active_steps": self.active_steps,
            "total_profiled_flops": total_flops,
            "total_self_device_time_s": total_device_us / 1e6,
            "profiled_tflops_per_s": (
                total_flops / total_device_us / 1e6
                if total_device_us > 0
                else 0.0
            ),
        }
        (self.profile_dir / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        self.trace_ready = True

    def _upload_wandb_artifact(self, state):
        if not self.upload_wandb or self.global_rank != 0 or not self.trace_ready:
            return
        try:
            import wandb

            if wandb.run is None:
                logging.warning("Profiler trace exists but no active W&B run was found")
                return
            artifact = wandb.Artifact(
                name=f"torch-profile-{wandb.run.id}-step-{state.global_step}",
                type="torch-profile",
                metadata={
                    "global_rank": self.global_rank,
                    "start_step": self.profile_start_step,
                    "warmup_steps": self.warmup_steps,
                    "active_steps": self.active_steps,
                },
            )
            artifact.add_dir(str(self.profile_dir))
            wandb.log_artifact(artifact)
            logging.info("Queued torch.profiler trace as a W&B artifact")
        except Exception as error:
            logging.warning(f"Failed to upload profiler artifact to W&B: {error}")

    def on_step_begin(self, args, state, control, **kwargs):
        # Remove callback after upload triggered to eliminate all overhead
        if self.profiling_complete and self.upload_triggered and not self.removed_from_trainer:
            if self.trainer is not None and hasattr(self.trainer, "callback_handler"):
                try:
                    self.trainer.callback_handler.callbacks.remove(self)
                    self.removed_from_trainer = True
                    logging.info(
                        f"Removed ProfCallback from trainer at global step {state.global_step}"
                    )
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Failed to remove ProfCallback: {e}")
            return

        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Record starting global step on first call
        if self.starting_global_step is None:
            self.starting_global_step = state.global_step

        # Calculate session step
        self.session_step = state.global_step - self.starting_global_step

        # Start profiler when we reach the profiling window
        if self.session_step == self.profile_start_step and self.prof is None:
            logging.info(
                f"Starting profiler at global step {state.global_step} (session step {self.session_step})"
            )
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    skip_first=0,
                    wait=0,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=1,
                ),
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                on_trace_ready=self._on_trace_ready,
            )
            self.prof.__enter__()
            self.profiling_active = True
            if self.trainer is not None:
                self.trainer._torch_profiler_active = True

    def on_step_end(self, args, state, control, **kwargs):
        # Early return if profiling already complete
        if self.profiling_complete:
            return

        # Recalculate session_step to ensure accuracy
        if self.starting_global_step is not None:
            self.session_step = state.global_step - self.starting_global_step

        # Step profiler if active
        if self.profiling_active and self.prof is not None:
            self.prof.step()

        # Stop profiler when we reach the end of profiling window
        if self.session_step == self.profile_end_step and self.prof is not None:
            self.prof.__exit__(None, None, None)
            self.profiling_active = False
            if self.trainer is not None:
                self.trainer._torch_profiler_active = False

            # Explicitly release profiler resources to minimize CUPTI overhead
            # Combined with TEARDOWN_CUPTI=1 env var for full cleanup
            del self.prof
            self.prof = None

            # Force CUDA synchronization to ensure profiler cleanup completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.profiling_complete = True
            logging.info(
                f"Profiler stopped and resources released at global step {state.global_step} "
                f"(session step {self.session_step})"
            )

            # Trigger upload if callback provided
            if self.upload_callback:
                logging.info(f"Triggering upload at global step {state.global_step}...")
                self.upload_callback()
            self._upload_wandb_artifact(state)

            # Mark as ready for callback removal
            self.upload_triggered = True


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class TimedDataLoader(DataLoader):
    """DataLoader that measures blocking time in ``next(iterator)``.

    The measured interval is the part of data loading that was not hidden by
    worker prefetching, which is the useful quantity for diagnosing GPU idle
    time. The timer runs in the main process, so workers never perform logging
    or touch CUDA state.
    """

    def __init__(self, *args, timing_sink=None, **kwargs):
        self.timing_sink = timing_sink
        super().__init__(*args, **kwargs)

    def __iter__(self):
        iterator = super().__iter__()
        while True:
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            if self.timing_sink is not None:
                self.timing_sink.record_data_wait(time.perf_counter() - start)
            yield batch

class BaseTrainer(transformers.Trainer):

    def __init__(self, **kwargs):
        # Increase the cache size limit for torch._dynamo to
        # accommodate videos with different numbers of frames.
        torch._dynamo.config.cache_size_limit = 1000
        # PyTorch 2.8 Inductor can miss its random-op rewrite when a ZeRO-3
        # DeepCompile fake trace turns the latent spatial dimensions into
        # SymInts. In that case lowering aten.randn raises instead of compiling
        # the graph. Keep random kernels as ATen extern calls while compiling
        # the rest of the model; these kernels are tiny relative to the DiT.
        fallback_random = os.environ.get("TORCHINDUCTOR_FALLBACK_RANDOM")
        if fallback_random is not None:
            from torch._inductor import config as inductor_config

            inductor_config.fallback_random = fallback_random.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        # Complex-layout pointwise kernels normally benchmark two Triton launch
        # configurations on their first invocation.  If a kernel mutates a
        # large activation, the benchmark path clones that activation to
        # preserve its value.  The Wan MLP bias-add output is about 420 MiB at
        # the training shape, so the temporary clone can OOM an otherwise
        # viable compiled forward.  A single deterministic pointwise config
        # avoids that one-time allocation while leaving GEMM and reduction
        # code generation enabled.
        autotune_pointwise = os.environ.get("TORCHINDUCTOR_AUTOTUNE_POINTWISE")
        if autotune_pointwise is not None:
            from torch._inductor import config as inductor_config

            inductor_config.triton.autotune_pointwise = (
                autotune_pointwise.lower() in {"1", "true", "yes", "on"}
            )

        self.compute_dtype = kwargs.pop("compute_dtype")
        self.output_dir = kwargs.pop("output_dir")
        self.timer = ContextTimer(self)

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.node_rank = int(os.environ.get("NODE_RANK", "0"))

        # Get distributed info
        self.current_step = 0

        # Profiling (legacy per-step profiling)
        self.enable_profiling = kwargs.pop("enable_profiling", False)
        self.profiling_steps = kwargs.pop("profiling_steps", 5)
        self.performance_enabled = kwargs.pop("performance_enabled", True)
        self.performance_log_steps = kwargs.pop("performance_log_steps", 10)
        self.performance_norm_steps = kwargs.pop("performance_norm_steps", 50)
        self.performance_peak_tflops = kwargs.pop("performance_peak_tflops", 312.0)
        self.performance_flop_factor = kwargs.pop("performance_flop_factor", 6.0)
        self.performance_checkpoint_flop_factor = kwargs.pop(
            "performance_checkpoint_flop_factor", 2.0
        )
        self.performance_tokens_per_sample = kwargs.pop("performance_tokens_per_sample", 0)
        # Pop new ProfCallback config options (handled in create_trainer, not here)
        kwargs.pop("enable_prof_callback", None)
        kwargs.pop("profile_start_step", None)
        kwargs.pop("profile_warmup_steps", None)
        kwargs.pop("profile_active_steps", None)
        kwargs.pop("profile_record_shapes", None)
        kwargs.pop("profile_with_stack", None)
        kwargs.pop("profile_memory", None)
        kwargs.pop("profile_with_flops", None)
        kwargs.pop("profile_ranks", None)
        kwargs.pop("profile_upload_wandb", None)
        kwargs.pop("msc_profile_url", None)
        kwargs.pop("profile_delete_after_upload", None)
        if self.enable_profiling:
            # Setup profiling directories
            self.profile_dir = Path(self.output_dir) / "profiling"
            self.memory_profile_dir = self.profile_dir / "memory"
            self.torch_profile_dir = self.profile_dir / "torch"

            self.memory_profile_dir.mkdir(exist_ok=True, parents=True)
            self.torch_profile_dir.mkdir(exist_ok=True, parents=True)

            # Start recording the memory history.
            torch.cuda.memory._record_memory_history(max_entries=100000)

        super().__init__(**kwargs)

        accelerator_config = getattr(self.args, "accelerator_config", None)
        if isinstance(accelerator_config, dict):
            accelerator_non_blocking = accelerator_config.get("non_blocking", False)
        else:
            accelerator_non_blocking = getattr(
                accelerator_config, "non_blocking", False
            )
        self.model._dreamzero_non_blocking = bool(
            accelerator_non_blocking
        )

        self.loss_queues = {}
        self.loss_queue_size = 10
        self._eval_component_sums = {}
        self._eval_component_counts = {}
        self._nvimgcodec_processor = None

        self._performance_timing_sums = {}
        self._performance_data_wait_s = 0.0
        self._performance_data_wait_count = 0
        self._performance_pending_data_wait_s = 0.0
        self._performance_pending_data_wait_count = 0
        self._performance_micro_steps = 0
        self._performance_update_start = None
        self._performance_capture_cuda = False
        self._performance_forward_events = []
        self._performance_input_events = []
        self._performance_training_events = []
        self._performance_optimizer_events = []
        self._performance_optimizer_event_start = None
        self._pending_performance_metrics = None
        self._torch_profiler_active = False
        self._performance_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self._performance_tokens_per_sample = self._infer_performance_tokens()
        self._performance_input_frames = 0
        self._performance_latent_frames = 0
        self._performance_token_sum = 0
        self._performance_token_count = 0
        self._performance_input_frame_sum = 0
        self._performance_latent_frame_sum = 0
        self._performance_checkpointing = False
        if self.performance_enabled:
            self.add_callback(
                PerformanceMetricsCallback(
                    self,
                    log_steps=self.performance_log_steps,
                    norm_steps=self.performance_norm_steps,
                )
            )

    def log(self, logs, *args, **kwargs):
        """Merge synchronized telemetry and apply stable W&B namespaces."""
        pending_metrics = getattr(self, "_pending_performance_metrics", None)
        if pending_metrics:
            logs = dict(logs)
            logs.update(pending_metrics)
            self._pending_performance_metrics = None
        return super().log(namespace_wandb_logs(logs), *args, **kwargs)

    def record_timing(self, key, seconds):
        """Accumulate a timer value until the next optimizer-step log."""
        if self.performance_enabled:
            self._performance_timing_sums[key] = (
                self._performance_timing_sums.get(key, 0.0) + float(seconds)
            )

    def record_data_wait(self, seconds):
        if self.performance_enabled:
            # HF Trainer calls next(dataloader) before on_step_begin. Keep
            # that interval pending until the next performance window starts;
            # otherwise on_step_begin would clear the just-measured wait.
            self._performance_pending_data_wait_s += float(seconds)
            self._performance_pending_data_wait_count += 1

    def _start_performance_window(self, global_step=0):
        self._performance_update_start = time.perf_counter()
        self._performance_timing_sums.clear()
        self._performance_data_wait_s = self._performance_pending_data_wait_s
        self._performance_data_wait_count = self._performance_pending_data_wait_count
        self._performance_pending_data_wait_s = 0.0
        self._performance_pending_data_wait_count = 0
        self._performance_micro_steps = 0
        self._performance_token_sum = 0
        self._performance_token_count = 0
        self._performance_input_frame_sum = 0
        self._performance_latent_frame_sum = 0
        self._performance_capture_cuda = (
            self.performance_enabled
            and torch.cuda.is_available()
            and (global_step + 1) % max(1, int(self.performance_log_steps)) == 0
        )
        self._performance_forward_events.clear()
        self._performance_input_events.clear()
        self._performance_training_events.clear()
        self._performance_optimizer_events.clear()
        self._performance_optimizer_event_start = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _new_cuda_event():
        return torch.cuda.Event(enable_timing=True)

    def _record_optimizer_event_start(self):
        if self._performance_capture_cuda:
            self._performance_optimizer_event_start = self._new_cuda_event()
            self._performance_optimizer_event_start.record()

    def _record_optimizer_event_end(self):
        if self._performance_optimizer_event_start is not None:
            end = self._new_cuda_event()
            end.record()
            self._performance_optimizer_events.append((self._performance_optimizer_event_start, end))
            self._performance_optimizer_event_start = None

    @staticmethod
    def _elapsed_cuda_events(events):
        elapsed = 0.0
        for start, end in events:
            end.synchronize()
            elapsed += start.elapsed_time(end) / 1000.0
        return elapsed

    def _record_micro_step(self):
        self._performance_micro_steps += 1

    @staticmethod
    def _unwrap_model(model):
        if model is None:
            return None
        return getattr(model, "module", model)

    def _infer_performance_tokens(self):
        if self.performance_tokens_per_sample:
            return int(self.performance_tokens_per_sample)

        model = self._unwrap_model(self.model)
        action_head = getattr(model, "action_head", None)
        dit = getattr(action_head, "model", None)
        config = getattr(action_head, "config", None)
        frame_seqlen = getattr(dit, "frame_seqlen", None)
        input_frames = getattr(config, "num_frames", None)
        action_horizon = getattr(config, "action_horizon", 0)
        if frame_seqlen is None or input_frames is None:
            return 0

        # Wan's VAE uses temporal compression of four for this 14B path.
        latent_frames = (int(input_frames) + 3) // 4
        return int(frame_seqlen) * latent_frames + int(action_horizon) + latent_frames

    def _update_performance_tokens_from_inputs(self, inputs):
        """Infer the sequence actually processed by the Wan blocks.

        A config-derived estimate cannot represent variable EgoVLA chunk sizes.
        The Wan teacher-forcing path also concatenates a clean video sequence
        with the noisy video/action/state sequence before all 40 transformer
        blocks. Reading the live batch avoids both sources of MFU error.
        """
        if self.performance_tokens_per_sample:
            return
        if not isinstance(inputs, dict):
            return

        images = inputs.get("images")
        if not torch.is_tensor(images) or images.ndim < 2:
            return
        model = self._unwrap_model(self.model)
        action_head = getattr(model, "action_head", None)
        dit = getattr(action_head, "model", None)
        frame_seqlen = getattr(dit, "frame_seqlen", None)
        if frame_seqlen is None:
            return

        input_frames = int(images.shape[1])
        latent_frames = (input_frames + 3) // 4
        video_tokens = int(frame_seqlen) * latent_frames

        actions = inputs.get("action")
        states = inputs.get("state")
        action_tokens = (
            int(actions.shape[1])
            if torch.is_tensor(actions) and actions.ndim >= 2
            else 0
        )
        state_tokens = (
            int(states.shape[1])
            if torch.is_tensor(states) and states.ndim >= 2
            else 0
        )

        # CausalWanModel.forward() receives clean_x during training and runs
        # both clean and noisy video tokens through every transformer block.
        uses_clean_teacher_forcing = (
            dit is not None and dit.__class__.__name__ == "CausalWanModel"
        )
        model_tokens = video_tokens + action_tokens + state_tokens
        if uses_clean_teacher_forcing:
            model_tokens += video_tokens

        self._performance_tokens_per_sample = model_tokens
        self._performance_input_frames = input_frames
        self._performance_latent_frames = latent_frames
        self._performance_checkpointing = bool(
            getattr(dit, "gradient_checkpointing", False)
        )
        if self._performance_update_start is not None:
            self._performance_token_sum += model_tokens
            self._performance_token_count += 1
            self._performance_input_frame_sum += input_frames
            self._performance_latent_frame_sum += latent_frames

    def _distributed_max(self, value):
        # Performance telemetry must never introduce a second collective
        # schedule alongside DeepSpeed's gradient collectives. In particular,
        # a slow rank can otherwise reach the next metric reduction at a
        # different time and make the training process look hung. W&B is
        # logged on rank 0, whose local values are representative here.
        return float(value)

    def _read_grad_norm(self, model=None, live=False):
        model = model or self.model
        if live:
            # ZeRO-2 keeps gradients partitioned and may not expose a valid
            # cached engine norm until optimizer.step().  Its live norm path
            # performs the required DP reduction and is only sampled every
            # ``performance_norm_steps`` steps.
            optimizer = getattr(model, "optimizer", None)
            scaled_global_norm = getattr(optimizer, "scaled_global_norm", None)
            if scaled_global_norm is not None:
                try:
                    value = scaled_global_norm()
                    loss_scale = float(getattr(optimizer, "loss_scale", 1.0) or 1.0)
                    return float(value.item() if hasattr(value, "item") else value) / loss_scale
                except (AttributeError, RuntimeError, TypeError, AssertionError):
                    pass
        if hasattr(model, "get_global_grad_norm"):
            try:
                value = model.get_global_grad_norm()
                if value is not None:
                    return float(value.item() if hasattr(value, "item") else value)
            except (AttributeError, RuntimeError, TypeError):
                pass

        squared = 0.0
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().float().pow(2).sum().item())
        return squared**0.5

    def _read_weight_norm(self):
        squared = None
        with torch.no_grad():
            for parameter in self.model.parameters():
                if not parameter.requires_grad:
                    continue
                value = parameter.detach().float().pow(2).sum()
                squared = value if squared is None else squared + value
        return None if squared is None else float(squared.sqrt().item())

    def _finish_performance_window(self, global_step, grad_norm=None, collect_weight_norm=False):
        if not self.performance_enabled or self._performance_update_start is None:
            return {}

        wall_time = time.perf_counter() - self._performance_update_start
        train_step_time = self._performance_timing_sums.get("training_step", 0.0)
        forward_time = self._performance_timing_sums.get("model_forward", 0.0)
        input_prepare_time = self._performance_timing_sums.get("input_prepare", 0.0)
        data_wait_time = self._performance_data_wait_s
        # The blocking next() calls happen immediately before the callback
        # window, so add them back to obtain end-to-end step time.
        wall_time += data_wait_time
        gpu_training_time = (
            self._elapsed_cuda_events(self._performance_training_events)
            if self._performance_capture_cuda
            else 0.0
        )
        gpu_forward_time = (
            self._elapsed_cuda_events(self._performance_forward_events)
            if self._performance_capture_cuda
            else 0.0
        )
        gpu_input_prepare_time = (
            self._elapsed_cuda_events(self._performance_input_events)
            if self._performance_capture_cuda
            else 0.0
        )
        gpu_optimizer_time = (
            self._elapsed_cuda_events(self._performance_optimizer_events)
            if self._performance_capture_cuda
            else 0.0
        )

        # The slowest rank determines the distributed step time.
        wall_time = self._distributed_max(wall_time)
        train_step_time = self._distributed_max(train_step_time)
        forward_time = self._distributed_max(forward_time)
        input_prepare_time = self._distributed_max(input_prepare_time)
        data_wait_time = self._distributed_max(data_wait_time)
        gpu_training_time = self._distributed_max(gpu_training_time)
        gpu_forward_time = self._distributed_max(gpu_forward_time)
        gpu_input_prepare_time = self._distributed_max(gpu_input_prepare_time)
        gpu_optimizer_time = self._distributed_max(gpu_optimizer_time)

        global_batch = (
            self.args.per_device_train_batch_size
            * max(1, self.world_size)
            * max(1, self.args.gradient_accumulation_steps)
        )
        max_allocated_gb = self._distributed_max(
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        max_reserved_gb = self._distributed_max(
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        )
        metrics = {
            "time/step_s": wall_time,
            "time/training_step_s": train_step_time,
            "time/model_forward_s": forward_time,
            "time/input_prepare_host_s": input_prepare_time,
            # CUDA event elapsed time is a stream timestamp span. It can include
            # idle gaps caused by CPU offload or a slow peer, so it must not be
            # interpreted as summed kernel-busy time.
            "time/cuda_training_span_s": gpu_training_time,
            "time/cuda_forward_span_s": gpu_forward_time,
            "time/cuda_input_prepare_span_s": gpu_input_prepare_time,
            # With Accelerate+DeepSpeed, engine.step() runs inside
            # training_step(). This outer callback interval is intentionally
            # named as a wrapper measurement rather than optimizer time.
            "time/cuda_optimizer_wrapper_span_s": gpu_optimizer_time,
            "time/data_wait_s": data_wait_time,
            "time/data_wait_fraction": data_wait_time / max(wall_time, 1e-6),
            "time/trainer_outer_step_s": max(
                wall_time - train_step_time - data_wait_time, 0.0
            ),
            "perf/micro_steps": self._performance_micro_steps,
            "perf/samples_per_s_global": global_batch / max(wall_time, 1e-6),
            "perf/samples_per_s_per_gpu": (
                global_batch / max(1, self.world_size) / max(wall_time, 1e-6)
            ),
            "perf/global_batch_size": global_batch,
            "perf/telemetry_global_rank": self.global_rank,
            "perf/torch_profiler_active": int(self._torch_profiler_active),
            "perf/gpu_max_memory_allocated_gb": max_allocated_gb,
            "perf/gpu_max_memory_reserved_gb": max_reserved_gb,
        }

        if grad_norm is not None:
            metrics["grad_norm/global_l2"] = float(grad_norm)
        if collect_weight_norm and self.global_rank == 0:
            weight_norm = self._read_weight_norm()
            if weight_norm is not None:
                metrics["weight_norm/global_l2"] = weight_norm
                if grad_norm is not None:
                    metrics["grad_norm/to_weight_ratio"] = float(grad_norm) / max(
                        weight_norm, 1e-12
                    )

        model_tokens_per_sample = (
            self._performance_token_sum / self._performance_token_count
            if self._performance_token_count > 0
            else self._performance_tokens_per_sample
        )
        input_frames_per_sample = (
            self._performance_input_frame_sum / self._performance_token_count
            if self._performance_token_count > 0
            else self._performance_input_frames
        )
        latent_frames_per_sample = (
            self._performance_latent_frame_sum / self._performance_token_count
            if self._performance_token_count > 0
            else self._performance_latent_frames
        )

        if model_tokens_per_sample and self.performance_peak_tflops > 0:
            model_flops = (
                self.performance_flop_factor
                * self._performance_trainable_params
                * model_tokens_per_sample
                * global_batch
            )
            peak_flops = self.performance_peak_tflops * 1e12 * max(1, self.world_size)
            metrics["perf/mfu_estimate"] = model_flops / max(
                wall_time * peak_flops, 1e-6
            )
            if gpu_training_time > 0:
                metrics["perf/mfu_cuda_span_estimate"] = model_flops / max(
                    gpu_training_time * peak_flops, 1e-6
                )
            if self._performance_checkpointing:
                hardware_flops = (
                    self.performance_flop_factor
                    + self.performance_checkpoint_flop_factor
                ) * self._performance_trainable_params * model_tokens_per_sample * global_batch
                metrics["perf/hfu_checkpoint_estimate"] = hardware_flops / max(
                    wall_time * peak_flops, 1e-6
                )
            metrics["perf/model_tokens_per_sample"] = model_tokens_per_sample
            metrics["perf/input_frames_per_sample"] = input_frames_per_sample
            metrics["perf/latent_frames_per_sample"] = latent_frames_per_sample
            metrics["perf/mfu_trainable_params_b"] = self._performance_trainable_params / 1e9
            metrics["perf/mfu_peak_tflops"] = self.performance_peak_tflops

        self._performance_update_start = None
        return metrics

    def _prepare_inputs(self, inputs):
        prepare_start = time.perf_counter()
        cuda_prepare_start = None
        if self._performance_capture_cuda:
            cuda_prepare_start = self._new_cuda_event()
            cuda_prepare_start.record()
        if isinstance(inputs, dict) and NVIMGCODEC_RAW_KEY in inputs:
            if self._nvimgcodec_processor is None:
                dataset = self.train_dataset
                if not getattr(dataset, "defer_media_decode", False):
                    raise RuntimeError("Received deferred media from a non-deferred dataset")
                self._nvimgcodec_processor = NvImageCodecBatchProcessor(
                    transforms=dataset.transforms,
                    collator=self.data_collator,
                    device=torch.device(f"cuda:{self.local_rank}"),
                )
            inputs = self._nvimgcodec_processor(inputs[NVIMGCODEC_RAW_KEY])
        prepared = super()._prepare_inputs(inputs)
        self._update_performance_tokens_from_inputs(prepared)
        if cuda_prepare_start is not None:
            cuda_prepare_end = self._new_cuda_event()
            cuda_prepare_end.record()
            self._performance_input_events.append(
                (cuda_prepare_start, cuda_prepare_end)
            )
        self.record_timing("input_prepare", time.perf_counter() - prepare_start)
        return prepared

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        enable_profile = self.enable_profiling and self.current_step % self.profiling_steps == 0
        if enable_profile:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_stack=True,
            )
        else:
            profile_context = contextlib.nullcontext()

        start_time = time.time()

        cuda_training_start = None
        if self._performance_capture_cuda:
            cuda_training_start = self._new_cuda_event()
            cuda_training_start.record()

        with self.timer.with_label("training_step"), profile_context as prof:
            output = super().training_step(model, inputs)

        if cuda_training_start is not None:
            cuda_training_end = self._new_cuda_event()
            cuda_training_end.record()
            self._performance_training_events.append((cuda_training_start, cuda_training_end))

        time_taken = time.time() - start_time
        self._record_micro_step()
        if os.environ.get("DREAMZERO_VERBOSE_STEP_TIMING", "0") == "1":
            print(
                f"Rank {self.global_rank} time taken for training_step {self.current_step}: {time_taken:.2f} seconds"
            )

        if enable_profile:
            trace_path = f"{self.torch_profile_dir}/trace_rank_{self.global_rank}_step_{self.current_step}.json.gz"
            print(f"Rank {self.global_rank} exporting torch profile to {trace_path}")
            prof.export_chrome_trace(trace_path)

            snapshot_path = f"{self.memory_profile_dir}/memory_snapshot_rank_{self.global_rank}_step_{self.current_step}.pickle"
            print(f"Rank {self.global_rank} dumping memory snapshot to {snapshot_path}")
            torch.cuda.memory._dump_snapshot(snapshot_path)

        self.current_step += 1
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cuda_forward_start = None
        if self._performance_capture_cuda:
            cuda_forward_start = self._new_cuda_event()
            cuda_forward_start.record()
        with self.timer.with_label("model_forward"):
            outputs = model(inputs)
        if cuda_forward_start is not None:
            cuda_forward_end = self._new_cuda_event()
            cuda_forward_end.record()
            self._performance_forward_events.append((cuda_forward_start, cuda_forward_end))

        # Track component losses without forcing a GPU synchronization on every
        # micro-batch. Training values are reduced to Python only at the logging
        # interval; validation values are aggregated by evaluation_loop().
        training_component_logs = {}
        is_training = bool(getattr(model, "training", False))
        batch_size = 1
        images = inputs.get("images") if isinstance(inputs, dict) else None
        if torch.is_tensor(images) and images.ndim > 0:
            batch_size = int(images.shape[0])
        for key, value in outputs.items():
            if key.endswith("_loss") and key != "loss":
                component = key[: -len("_loss")]
                value_tensor = (
                    value.detach().float()
                    if torch.is_tensor(value)
                    else torch.tensor(
                        float(value), device=outputs["loss"].device
                    )
                )
                if is_training:
                    queue = self.loss_queues.setdefault(key, [])
                    queue.append(value_tensor)
                    if len(queue) > self.loss_queue_size:
                        queue.pop(0)
                    if self.current_step % self.loss_queue_size == 0:
                        training_component_logs[
                            f"train_loss/{component}"
                        ] = torch.stack(queue).mean().item()
                else:
                    weighted = value_tensor * batch_size
                    previous = self._eval_component_sums.get(key)
                    self._eval_component_sums[key] = (
                        weighted if previous is None else previous + weighted
                    )
                    self._eval_component_counts[key] = (
                        self._eval_component_counts.get(key, 0) + batch_size
                    )

        if training_component_logs:
            self.log(training_component_logs)

        loss = outputs["loss"]

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self._eval_component_sums = {}
        self._eval_component_counts = {}
        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        for key in sorted(self._eval_component_sums):
            total = self._eval_component_sums[key]
            count = torch.tensor(
                float(self._eval_component_counts[key]),
                device=total.device,
                dtype=total.dtype,
            )
            pair = torch.stack((total, count))
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    pair, op=torch.distributed.ReduceOp.SUM
                )
            component = key[: -len("_loss")]
            output.metrics[f"{metric_key_prefix}_{component}_loss"] = (
                pair[0] / pair[1].clamp_min(1)
            ).item()

        return output

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            # DeepSpeed CPU Adam (ZeRO offload) expects 'bias_correction' in each param group.
            # HuggingFace Trainer's AdamW does not set it, causing KeyError in cpu_adam.step().
            if getattr(self.args, "deepspeed", None):
                for group in self.optimizer.param_groups:
                    group.setdefault("bias_correction", True)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):

        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.base_cfg.save_lora_only:
            # Save only the trainable parameters
            train_key = [k for k, v in self.model.named_parameters() if v.requires_grad]
            lora_state_dict = {k: v for k, v in self.model.state_dict().items() if k in train_key}
            state_dict = lora_state_dict

        if self.args.should_save:
            ret = self.model.save_pretrained(output_dir, state_dict=state_dict)

            # can separately save the VLM model for downstream evalualtion
            if self.base_cfg.save_llm:
                llm_output_dir = os.path.join(output_dir, "llm")
                self.model.backbone.model.save_pretrained(llm_output_dir)

            if self.base_cfg.save_value_model:
                assert hasattr(
                    self.model.action_head, "value_model"
                ), f"Value model not found in action head: {type(self.model.action_head)}"
                value_model_output_dir = os.path.join(output_dir, "value_model")
                self.model.action_head.value_model.save_pretrained(value_model_output_dir)

            return ret

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if getattr(train_dataset, "defer_media_decode", False):
            self.args.ignore_data_skip = True
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": raw_samples_collate,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            if self.args.dataloader_num_workers > 0:
                dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers
                if self.args.dataloader_prefetch_factor is not None:
                    dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            return TimedDataLoader(train_dataset, timing_sink=self, **dataloader_params)
        if not isinstance(train_dataset, (ShardedLeRobotMixtureDataset)):
            return super().get_train_dataloader()

        # During resume, don't skip the data
        self.args.ignore_data_skip = True
        curr_global_step = self.state.global_step
        print(f"Current global step: {curr_global_step}")
        if curr_global_step > 0:
            new_seed = train_dataset.seed + curr_global_step
            train_dataset.reset_seed(new_seed)
            print(
                f"Resetting seed to {new_seed}. Please note that this will make the experiment non-reproducible."
            )

        print("Creating custom train dataloader")
        # Handle the case where the dataset is an IterableDataset
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        # persistent_workers is only valid when num_workers > 0 (PyTorch raises otherwise)
        if self.args.dataloader_num_workers > 0:
            dataloader_params["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return TimedDataLoader(train_dataset, timing_sink=self, **dataloader_params)


class BaseExperiment(ABC):
    def __init__(self, cfg: DictConfig):
        # assert cfg.save_steps == 500, "save_steps must be 500 for standarized evaluation"
        assert cfg.max_steps > 0, "max_steps must be > 0 for standarized evaluation"
        assert cfg.save_total_limit >= 5, "save_total_limit must be >= 5 for standarized evaluation"

        if cfg.load_from_yaml is not None:
            # Override the default config with the loaded config.
            loaded_cfg = OmegaConf.load(cfg.load_from_yaml)
            cfg = loaded_cfg  # overwrite

        # Check if evaluation transforms are valid.
        assert cfg.transforms is not None, "Evaluation transforms are not provided."
        for tag, transform_cfg in cfg.transforms.items():
            try:
                # Check if the tag is a valid EmbodimentTag
                _ = EmbodimentTag(tag)
                # Check if the transform is a valid ComposedModalityTransform
                transform = instantiate(transform_cfg)
                assert isinstance(transform, ComposedModalityTransform), f"{transform=}"
            except Exception as e:
                raise ValueError(f"Evaluation transform {tag} is invalid: {e}")

        # Instantiate the training arguments.
        cfg.training_args.output_dir = cfg.training_args.output_dir.rstrip("/")
        cfg.training_args.run_name = cfg.training_args.output_dir.split("/")[-1]
        print(f"Run name: {cfg.training_args.run_name}")
        training_args = instantiate(cfg.training_args, _convert_="all")
        set_seed(training_args.seed)

        # Set the environment variables for wandb.
        if "WANDB_PROJECT" not in os.environ:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project
        if "WANDB_RUN_ID" not in os.environ:
            runtime_id = os.environ.get("RUNTIME_ID", None)
            """If a RUNTIME_ID is available in the environment, we use it as the wandb id,
            which will allow to display the evaluation results and the training results
            in the same wandb run. Otherwise, we create a new run."""
            if runtime_id:
                os.environ["WANDB_RUN_ID"] = runtime_id
        os.environ["WANDB_DIR"] = training_args.output_dir

        # Create the experiment config directory.
        output_dir = Path(training_args.output_dir)
        exp_cfg_dir = output_dir / "experiment_cfg"
        exp_cfg_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        wandb_config_file = output_dir / "wandb_config.json"
        with open(wandb_config_file, "w") as f:
            json.dump(
                {
                    "project": os.environ.get("WANDB_PROJECT", ""),
                    "run_id": os.environ.get("WANDB_RUN_ID", ""),
                },
                f,
            )

        # Check if we are resuming training.
        resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
        if not continue_training:
            print(f"Models is ready under {training_args.output_dir}. Skip training.")
            exit(0)
        if resume_path:
            print(f"Resuming training from {resume_path}")
            resume_from_checkpoint = True
        else:
            # First time training.
            resume_from_checkpoint = False

        # Instantiate the model.
        model = self.create_model(cfg, training_args)

        if hasattr(model.action_head, "max_steps"):
            model.action_head.max_steps = cfg.max_steps

        # Make sure model_dtype and training_args dtype are compatible.
        compute_dtype = dtype_from_string(model.config.model_dtype)

        # Create the train dataset.
        # Dump the metadata; necessary for policy to normalize the input and unnormalize the output
        train_dataset = self.create_train_dataset(cfg, model)
        print("Using dataset:")
        print(train_dataset)
        assert (
            train_dataset.merged_metadata is not None
        ), "You must set metadata_config.merge=true in order to save the metadata."

        metadata_save_path = exp_cfg_dir / "metadata.json"
        U.json_dump(
            {k: v.model_dump(mode="json") for k, v in train_dataset.merged_metadata.items()},
            metadata_save_path,
            indent=4,
        )
        print("Successfully dumped metadata")

        val_dataset = self.create_val_dataset(cfg, model)
        data_collator = self.create_data_collator(cfg, model)
        trainer = self.create_trainer(
            cfg=cfg,
            exp_cfg_dir=exp_cfg_dir,
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
            compute_dtype=compute_dtype,
        )
        self.cfg = cfg
        self.exp_cfg_dir = exp_cfg_dir
        self.training_args = training_args
        self.resume_from_checkpoint = resume_from_checkpoint
        self.train_dataset = train_dataset
        self.trainer = trainer

    def create_model(self, cfg, training_args):
        model = instantiate(cfg.model)

        if cfg.pretrained_model_path is not None:
            mprint(f"Loading pretrained weights from: {cfg.pretrained_model_path}")
            import json, gc
            from safetensors.torch import load_file

            ckpt_dir = cfg.pretrained_model_path
            target_state_dict = model.state_dict()
            safetensors_index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
            safetensors_path = os.path.join(ckpt_dir, "model.safetensors")

            if os.path.exists(safetensors_index_path):
                with open(safetensors_index_path, 'r') as f:
                    index = json.load(f)
                for shard_file in sorted(set(index["weight_map"].values())):
                    shard_path = os.path.join(ckpt_dir, shard_file)
                    mprint(f"Loading shard: {shard_path}")
                    shard_state_dict = load_file(shard_path)
                    shard_state_dict = expand_action_adapter_state_dict(
                        shard_state_dict, target_state_dict
                    )
                    model.load_state_dict(shard_state_dict, strict=False)
                    del shard_state_dict
                    gc.collect()
            elif os.path.exists(safetensors_path):
                state_dict = load_file(safetensors_path)
                state_dict = expand_action_adapter_state_dict(state_dict, target_state_dict)
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError(
                    f"No weights found at '{ckpt_dir}'. "
                    "Expected 'model.safetensors' or 'model.safetensors.index.json'."
                )

            if (hasattr(model, 'action_head')
                    and hasattr(model.action_head, 'inject_lora_after_loading')
                    and model.action_head.config.defer_lora_injection):
                model.action_head.inject_lora_after_loading()

            mprint("Successfully loaded pretrained weights")

        model.config.resume_path = model.config._name_or_path = training_args.output_dir
        mprint(f"{model}\n")
        return model

    def create_train_dataset(self, cfg, model):
        assert torch.distributed.is_initialized()
        train_dataset = instantiate(cfg.train_dataset)
        return train_dataset

    def create_val_dataset(self, cfg, model):
        val_dataset_cfg = cfg.get("val_dataset")
        if val_dataset_cfg is None:
            return None
        return instantiate(val_dataset_cfg)

    def create_data_collator(self, cfg, model):
        return instantiate(cfg.data_collator)

    def create_trainer(
        self,
        cfg,
        exp_cfg_dir,
        model,
        training_args,
        train_dataset,
        val_dataset,
        data_collator,
        compute_dtype,
    ):
        # Set the gradient accumulation steps.
        if cfg.global_batch_size is not None:
            global_bs = cfg.global_batch_size
            bs = training_args.per_device_train_batch_size
            grad_acc = compute_grad_accum_to_match_global_bs(global_bs, bs)
            training_args.gradient_accumulation_steps = grad_acc
            print(
                f"Set global batch size to {global_bs}, set gradient accumulation steps to {grad_acc}"
            )
        elif cfg.raise_error_if_global_batch_size_not_set:
            raise ValueError(
                "global_batch_size is not set. To ensure the scripts can be reproduced regardless of the number of nodes used, please set this."
            )
        else:
            warnings.warn(
                "global_batch_size is not set. This is fine for debugging, but please set this for real experiments."
            )

        # Accelerate compiles the complete VLA when TrainingArguments.torch_compile
        # is true. DreamZero also supports a lower-memory targeted mode that
        # compiles Wan and/or the frozen encoders after DeepSpeed preparation.
        # Clear all three TrainingArguments switches because Transformers treats
        # a non-null backend or mode as an implicit request to compile.
        compile_requested = os.environ.get("TORCH_COMPILE", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        compile_scope = os.environ.get("TORCH_COMPILE_SCOPE", "wan_frozen").lower()
        targeted_compile = compile_requested and compile_scope not in {"all", "none"}
        if targeted_compile:
            training_args.torch_compile = False
            training_args.torch_compile_backend = None
            training_args.torch_compile_mode = None
            print(
                f"Using targeted compile scope={compile_scope}; disabling whole-model Accelerate compile",
                flush=True,
            )

        # Instantiate the partial trainer.
        trainer_partial = instantiate(
            cfg.trainer,
            model=model,
            output_dir=training_args.output_dir,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_dtype=compute_dtype,
        )

        # Fully instantiate the trainer with dataclasses instances.
        trainer = trainer_partial(data_collator=data_collator, args=training_args)
        trainer.base_cfg = cfg
        if "wandb" in getattr(training_args, "report_to", []):
            # The stock callback rewrites every training key as ``train/<key>``.
            # Replace it before training starts so W&B keeps our semantic
            # top-level namespaces (perf, time, train_loss, ...).
            for callback in list(trainer.callback_handler.callbacks):
                if isinstance(callback, WandbCallback):
                    trainer.callback_handler.callbacks.remove(callback)
            trainer.add_callback(NamespaceWandbCallback())
        if targeted_compile:
            trainer.add_callback(TargetedCompileCallback(trainer, compile_scope))

        def safe_len(value):
            try:
                return len(value)
            except (TypeError, ValueError):
                return None

        train_dl_len = safe_len(trainer.get_train_dataloader())
        eval_dl_len = safe_len(trainer.get_eval_dataloader()) if val_dataset is not None else None

        # Save the total training steps in the config.
        with open_dict(cfg):
            cfg.total_training_steps = (
                train_dl_len * cfg.training_args.num_train_epochs
                if train_dl_len is not None
                else cfg.max_steps
            )

        # Save config.
        OmegaConf.save(cfg, exp_cfg_dir / "conf.yaml", resolve=True)

        run_name = cfg.training_args.get("run_name", None)
        ckpt_format_callback = CheckpointFormatCallback(run_name=run_name, exp_cfg_dir=exp_cfg_dir)
        trainer.add_callback(ckpt_format_callback)

        loss_log_path = str(Path(training_args.output_dir) / "loss_log.jsonl")
        trainer.add_callback(LossLoggerCallback(output_path=loss_log_path))


        # Add profiling callback (local profiling only, no S3 upload)
        # Local: {output_dir}/profiling/rank_{id}/*.pt.trace.json
        if cfg.trainer.get("enable_prof_callback", False):
            output_dir = Path(training_args.output_dir)
            global_rank = int(os.environ.get("RANK", "0"))
            profile_ranks = {
                int(rank) for rank in cfg.trainer.get("profile_ranks", [0])
            }

            # Get profiling configuration from trainer config
            profile_start_step = cfg.trainer.get("profile_start_step", 50)
            profile_warmup_steps = cfg.trainer.get("profile_warmup_steps", 1)
            profile_active_steps = cfg.trainer.get("profile_active_steps", 5)
            profile_record_shapes = cfg.trainer.get("profile_record_shapes", False)
            profile_with_stack = cfg.trainer.get(
                "profile_with_stack", False
            )  # Default False to match omni (stack traces add significant file size)
            profile_memory = cfg.trainer.get("profile_memory", False)
            profile_with_flops = cfg.trainer.get("profile_with_flops", True)
            profile_upload_wandb = cfg.trainer.get(
                "profile_upload_wandb", True
            )

            # on_step_end sees the incremented global step. This endpoint gives
            # the profiler exactly warmup+active calls to prof.step().
            profile_end_step = (
                profile_start_step
                + profile_warmup_steps
                + profile_active_steps
            )

            configured_profile_dir = cfg.get("profile_dir")
            profile_root = (
                Path(configured_profile_dir)
                if configured_profile_dir
                else output_dir / "profiling"
            )
            profile_dir = profile_root / f"rank_{global_rank}"

            if global_rank in profile_ranks:
                profile_dir.mkdir(parents=True, exist_ok=True)
                mprint(
                    f"Profiling enabled on ranks {sorted(profile_ranks)}: "
                    f"steps {profile_start_step}-{profile_end_step}, "
                    f"saving to {profile_root}"
                )
                trainer.add_callback(
                    ProfCallback(
                        profile_dir=profile_dir,
                        upload_callback=None,
                        profile_start_step=profile_start_step,
                        profile_end_step=profile_end_step,
                        warmup_steps=profile_warmup_steps,
                        active_steps=profile_active_steps,
                        trainer=trainer,
                        record_shapes=profile_record_shapes,
                        with_stack=profile_with_stack,
                        profile_memory=profile_memory,
                        with_flops=profile_with_flops,
                        upload_wandb=profile_upload_wandb,
                        global_rank=global_rank,
                    )
                )

        mprint(
            f"train dataloader length: {train_dl_len or 'streaming'}\n"
            f"eval dataloader length: {eval_dl_len or ('streaming' if val_dataset is not None else 'disabled')}\n"
            f"train dataset length: {safe_len(trainer.train_dataset) or 'streaming'}\n"
            f"GPU memory before training: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024} GB",
            flush=True,
        )
        return trainer

    def train(self):
        # Start training.
        self.trainer.train(resume_from_checkpoint=self.resume_from_checkpoint)
        self.trainer.save_state()
        safe_save_model_for_hf_trainer(
            trainer=self.trainer, output_dir=self.training_args.output_dir
        )
