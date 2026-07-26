import os
from unittest.mock import patch

import torch
from torch import nn

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
)
from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE


class _RecordingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input = None

    def encode(self, videos, **_kwargs):
        self.last_input = videos.clone()
        return videos[:, :1].repeat(1, 16, 1, 1, 1)


class _RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def encode(self, videos, _scale):
        self.batch_sizes.append(videos.shape[0])
        return videos * 2


def test_joint_vae_encode_keeps_first_frame_and_zero_pads_the_rest():
    head = WANPolicyHead.__new__(WANPolicyHead)
    nn.Module.__init__(head)
    head.vae = _RecordingVAE()
    head._vae_device_ready = True

    videos = torch.arange(2 * 3 * 5 * 2 * 2, dtype=torch.float32).reshape(
        2, 3, 5, 2, 2
    )
    latents, condition, new_image = head.encode_video_with_condition(
        videos,
        tiled=False,
    )

    recorded = head.vae.last_input
    assert recorded is not None
    assert recorded.shape == (4, 3, 5, 2, 2)
    torch.testing.assert_close(recorded[:2], videos)
    torch.testing.assert_close(recorded[2:, :, :1], videos[:, :, :1])
    torch.testing.assert_close(
        recorded[2:, :, 1:],
        torch.zeros_like(videos[:, :, 1:]),
    )

    expected_latents = videos[:, :1].repeat(1, 16, 1, 1, 1)
    expected_condition_latents = recorded[2:, :1].repeat(1, 16, 1, 1, 1)
    torch.testing.assert_close(latents, expected_latents)
    torch.testing.assert_close(new_image, expected_condition_latents[:, :, :1])
    assert condition.shape == (2, 20, 5, 2, 2)
    torch.testing.assert_close(
        condition[:, :4, :1],
        torch.ones_like(condition[:, :4, :1]),
    )
    torch.testing.assert_close(
        condition[:, :4, 1:],
        torch.zeros_like(condition[:, :4, 1:]),
    )
    torch.testing.assert_close(condition[:, 4:], expected_condition_latents)


def test_non_tiled_vae_encode_uses_one_real_batch_call():
    vae = WanVideoVAE.__new__(WanVideoVAE)
    nn.Module.__init__(vae)
    vae.model = _RecordingEncoder()
    vae.scale = None

    videos = torch.randn(4, 3, 5, 2, 2)
    encoded = vae.encode(videos, tiled=False)

    assert vae.model.batch_sizes == [4]
    torch.testing.assert_close(encoded, videos * 2)


def test_clip_is_detached_only_for_explicit_clip_compile_scope():
    clip_target = WANPolicyHead.__new__(WANPolicyHead)
    nn.Module.__init__(clip_target)
    clip_target.image_encoder = nn.Linear(3, 2)
    with patch.dict(
        os.environ,
        {
            "TORCH_COMPILE": "true",
            "TORCH_COMPILE_SCOPE": "wan_blocks_vae_clip",
        },
    ):
        clip_target._detach_frozen_image_encoder_for_compile()

    assert "image_encoder" not in clip_target._modules
    assert clip_target._image_encoder_detached
    assert not clip_target._image_encoder_device_ready
    clip_target._ensure_image_encoder_on_device(torch.zeros(1))
    assert clip_target._image_encoder_device_ready
    assert next(clip_target.image_encoder.parameters()).dtype == torch.bfloat16

    vae_only = WANPolicyHead.__new__(WANPolicyHead)
    nn.Module.__init__(vae_only)
    vae_only.image_encoder = nn.Linear(3, 2)
    with patch.dict(
        os.environ,
        {"TORCH_COMPILE": "true", "TORCH_COMPILE_SCOPE": "vae"},
    ):
        vae_only._detach_frozen_image_encoder_for_compile()

    assert "image_encoder" in vae_only._modules
    assert not vae_only._image_encoder_detached
