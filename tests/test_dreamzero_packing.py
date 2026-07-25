from __future__ import annotations

import numpy as np
import torch

from groot.vla.data.dataset.dreamzero_packing import (
    PACKED_FEATURES_KEY,
    infer_chunk_count,
    pack_transformed_samples,
)
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
    collate_packed,
)
from groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk import (
    CausalWanModel,
)


ACTION_HORIZON = 24


def _sample(num_chunks: int, marker: int = 0) -> dict:
    action_steps = num_chunks * ACTION_HORIZON
    video_frames = num_chunks * 8 + 1
    return {
        "images": np.full(
            (video_frames, 2, 2, 3),
            marker,
            dtype=np.uint8,
        ),
        "state": np.zeros((num_chunks, 64), dtype=np.float32),
        "state_mask": np.ones((num_chunks, 64), dtype=bool),
        "action": np.zeros((action_steps, 48), dtype=np.float32),
        "action_mask": np.ones((action_steps, 48), dtype=bool),
        "lapa_action": np.zeros((action_steps, 48), dtype=np.float32),
        "lapa_action_mask": np.zeros((action_steps, 48), dtype=bool),
        "has_real_action": np.ones((), dtype=bool),
        "has_lapa_action": np.zeros((), dtype=bool),
        "is_cotrain_instance": np.zeros((), dtype=bool),
        "segmentation_target": np.zeros((2,), dtype=np.float32),
        "segmentation_target_mask": np.zeros((1,), dtype=np.float32),
        "embodiment_id": np.zeros((), dtype=np.int64),
        "text": f"task {marker}",
        "text_negative": "negative",
    }


class _Tokenizer:
    def __call__(self, values, return_mask=False, **_kwargs):
        batch = len(values) if isinstance(values, list) else 1
        ids = torch.arange(8).repeat(batch, 1)
        mask = torch.ones_like(ids)
        return (ids, mask) if return_mask else ids


def _composition(pack: dict) -> tuple[int, ...]:
    return tuple(
        infer_chunk_count(sample, ACTION_HORIZON)
        for sample in pack[PACKED_FEATURES_KEY]
    )


def test_packer_prefers_full_compositions_and_drops_nothing():
    source = [
        _sample(1, 1),
        _sample(3, 2),
        _sample(2, 3),
        _sample(2, 4),
        _sample(4, 5),
        _sample(1, 6),
        _sample(1, 7),
    ]
    packs = list(
        pack_transformed_samples(
            source,
            chunk_capacity=4,
            max_segments=2,
            action_horizon=ACTION_HORIZON,
            pending_limit=2,
        )
    )

    assert [_composition(pack) for pack in packs] == [
        (1, 3),
        (2, 2),
        (4,),
        (1, 1),
    ]
    packed_markers = [
        int(sample["images"][0, 0, 0, 0])
        for pack in packs
        for sample in pack[PACKED_FEATURES_KEY]
    ]
    assert sorted(packed_markers) == list(range(1, 8))


def test_collate_packed_has_fixed_two_slot_shapes_and_masks_padding():
    pack = next(
        pack_transformed_samples(
            [_sample(1, 1), _sample(3, 2)],
            chunk_capacity=4,
            max_segments=2,
            action_horizon=ACTION_HORIZON,
        )
    )
    batch = collate_packed(
        pack,
        _Tokenizer(),
        num_views=1,
        embodiment_tag_mapping={},
        action_horizon=ACTION_HORIZON,
        video_frames_per_chunk=8,
    )

    assert tuple(batch["images"].shape) == (1, 2, 33, 2, 2, 3)
    assert tuple(batch["state"].shape) == (1, 2, 4, 64)
    assert tuple(batch["action"].shape) == (1, 2, 96, 48)
    assert batch["packed_num_chunks"].tolist() == [[1, 3]]
    assert batch["packed_segment_valid"].tolist() == [[True, True]]
    assert not batch["action_mask"][0, 0, 24:].any()
    assert not batch["state_mask"][0, 0, 1:].any()
    assert not batch["images"][0, 0, 9:].any()


def test_single_four_chunk_pack_keeps_dummy_slot_zero():
    pack = next(
        pack_transformed_samples(
            [_sample(4, 9)],
            chunk_capacity=4,
            max_segments=2,
            action_horizon=ACTION_HORIZON,
        )
    )
    batch = collate_packed(
        pack,
        _Tokenizer(),
        num_views=1,
        embodiment_tag_mapping={},
        action_horizon=ACTION_HORIZON,
        video_frames_per_chunk=8,
    )

    assert batch["packed_num_chunks"].tolist() == [[4, 0]]
    assert batch["packed_segment_valid"].tolist() == [[True, False]]
    assert not batch["images"][0, 1].any()
    assert not batch["action_mask"][0, 1].any()


def test_packed_layout_resets_positions_and_has_fixed_physical_length():
    model = CausalWanModel(
        model_type="i2v",
        frame_seqlen=4,
        in_dim=36,
        dim=24,
        ffn_dim=48,
        out_dim=16,
        num_heads=4,
        num_layers=1,
        max_chunk_size=4,
        num_frame_per_block=2,
        action_dim=48,
        max_state_dim=64,
        hidden_size=8,
        num_action_per_block=24,
        num_state_per_block=1,
    )
    grid_size = torch.tensor([10, 2, 2], dtype=torch.long)
    layout_31 = model._build_packed_training_layout(
        grid_size=grid_size,
        packed_num_chunks=torch.tensor([[3, 1]]),
        packed_chunk_capacity=4,
        packed_max_segments=2,
    )
    layout_40 = model._build_packed_training_layout(
        grid_size=grid_size,
        packed_num_chunks=torch.tensor([[4, 0]]),
        packed_chunk_capacity=4,
        packed_max_segments=2,
    )

    # Fixed physical shape: clean/noisy video plus 96 action and 4 state.
    assert layout_31["total_length"] == 2 * 10 * 4 + 96 + 4
    assert layout_40["total_length"] == layout_31["total_length"]
    assert layout_31["rope_freqs"].shape[0] == layout_31["total_length"]
    assert len(layout_31["segment_token_indices"]) == 2
    assert layout_40["segment_token_indices"][1].numel() == 0

    first_segment = layout_31["segments"][0]
    second_segment = layout_31["segments"][1]
    assert first_segment["latent_frames"] == 7
    assert second_segment["latent_frames"] == 3
    assert first_segment["clean_range"] == (0, 28)
    assert second_segment["clean_range"] == (28, 40)
