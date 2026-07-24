"""Rank-local nvImageCodec preprocessing for deferred WDS media decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, Sequence

import torch


NVIMGCODEC_RAW_KEY = "__nvimgcodec_raw_samples__"


def raw_samples_collate(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep WDS samples uncollated until they reach the rank's main process."""
    return {NVIMGCODEC_RAW_KEY: features}


def _pin_cpu_tensors(value: Any) -> Any:
    """Pin tensors created by the main-process collator for async host copies."""
    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and torch.cuda.is_available():
            return value.pin_memory()
        return value
    if isinstance(value, Mapping):
        return {key: _pin_cpu_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin_cpu_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pin_cpu_tensors(item) for item in value)
    return value


class NvImageCodecBatchProcessor:
    """Decode and transform a batch in the rank process on its assigned GPU."""

    def __init__(
        self,
        transforms: Callable[[dict[str, Any]], dict[str, Any]],
        collator: Callable[[list[dict[str, Any]]], dict[str, Any]],
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"nvImageCodec requires an indexed CUDA device, got {device}")

        try:
            from nvidia import nvimgcodec
        except ImportError as exc:
            raise ImportError(
                "nvimgcodec_decode=true requires nvidia-nvimgcodec-cu12"
            ) from exc

        torch.cuda.set_device(device)
        self.device = device
        self.transforms = transforms
        self.collator = collator
        self.nvimgcodec = nvimgcodec
        self.decoder = nvimgcodec.Decoder(device_id=device.index)
        self.decode_params = nvimgcodec.DecodeParams(
            sample_format=nvimgcodec.SampleFormat.I_RGB
        )

    def _decode_features(self, features: Sequence[dict[str, Any]]) -> None:
        requests: list[tuple[int, str, int]] = []
        streams = []
        for feature_index, feature in enumerate(features):
            for key in ("video.head", "video.chest"):
                values = feature.get(key)
                if values is None:
                    continue
                for value_index, value in enumerate(values):
                    if value is not None:
                        requests.append((feature_index, key, value_index))
                        streams.append(self.nvimgcodec.CodeStream(value))

        if not streams:
            raise ValueError("Deferred WDS sample contains no JPEG frames")

        decoded = self.decoder.decode(streams, params=self.decode_params)
        tensors: dict[tuple[int, str, int], torch.Tensor] = {}
        for request, image in zip(requests, decoded):
            if image is None:
                raise ValueError(f"nvImageCodec failed to decode frame {request}")
            tensors[request] = torch.utils.dlpack.from_dlpack(image.to_dlpack())

        for feature_index, feature in enumerate(features):
            fallback = next(
                (
                    tensor
                    for (idx, _key, _value_index), tensor in tensors.items()
                    if idx == feature_index
                ),
                None,
            )
            if fallback is None:
                raise ValueError(f"No decoded frame available for sample {feature_index}")

            for key in ("video.head", "video.chest"):
                values = feature.get(key)
                if values is None:
                    continue
                feature[key] = torch.stack(
                    [
                        tensors.get(
                            (feature_index, key, value_index),
                            torch.zeros_like(fallback),
                        )
                        for value_index, _value in enumerate(values)
                    ]
                )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [dict(feature) for feature in features]
        self._decode_features(features)
        transformed = [self.transforms(feature) for feature in features]
        return _pin_cpu_tensors(self.collator(transformed))
