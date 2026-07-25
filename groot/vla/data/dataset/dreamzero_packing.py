from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from typing import Any


PACKED_FEATURES_KEY = "__dreamzero_packed_features__"
PACK_CHUNK_CAPACITY_KEY = "__dreamzero_pack_chunk_capacity__"
PACK_MAX_SEGMENTS_KEY = "__dreamzero_pack_max_segments__"


def infer_chunk_count(sample: dict[str, Any], action_horizon: int) -> int:
    """Infer the number of DreamZero chunks from a transformed sample."""
    if action_horizon <= 0:
        raise ValueError(f"action_horizon must be positive, got {action_horizon}")
    if "action" not in sample:
        raise KeyError("Packed DreamZero training requires an action tensor")

    action_length = int(sample["action"].shape[0])
    chunk_count, remainder = divmod(action_length, action_horizon)
    if remainder != 0 or chunk_count <= 0:
        raise ValueError(
            "Action length must contain a positive whole number of chunks: "
            f"action_length={action_length}, action_horizon={action_horizon}"
        )
    return chunk_count


def filter_fixed_chunk_samples(
    samples: Iterable[dict[str, Any]],
    *,
    required_chunk_count: int,
    action_horizon: int = 24,
) -> Iterator[dict[str, Any]]:
    """Yield only samples that exactly fill the requested chunk budget."""
    if required_chunk_count <= 0:
        raise ValueError(
            "required_chunk_count must be positive, got "
            f"{required_chunk_count}"
        )

    for sample in samples:
        if infer_chunk_count(sample, action_horizon) == required_chunk_count:
            yield sample


def make_packed_feature(
    samples: list[dict[str, Any]],
    *,
    chunk_capacity: int,
    max_segments: int,
) -> dict[str, Any]:
    if not samples or len(samples) > max_segments:
        raise ValueError(
            f"Expected 1..{max_segments} samples in a pack, got {len(samples)}"
        )
    return {
        PACKED_FEATURES_KEY: samples,
        PACK_CHUNK_CAPACITY_KEY: int(chunk_capacity),
        PACK_MAX_SEGMENTS_KEY: int(max_segments),
    }


def pack_transformed_samples(
    samples: Iterable[dict[str, Any]],
    *,
    chunk_capacity: int,
    max_segments: int = 2,
    action_horizon: int = 24,
    pending_limit: int = 8,
) -> Iterator[dict[str, Any]]:
    """Pack transformed samples without dropping any sample.

    Full packs are preferred. With the current two-segment policy this produces
    the primary 4, 3+1, and 2+2 compositions for a four-chunk capacity. A
    bounded pending queue prevents decoded videos from accumulating forever;
    when the source length distribution cannot form a full pack, the oldest
    samples are emitted in the fullest legal fallback pack and the collator
    pads the unused chunk capacity.
    """
    if chunk_capacity <= 0:
        raise ValueError(f"chunk_capacity must be positive, got {chunk_capacity}")
    if max_segments != 2:
        raise ValueError(
            "The initial DreamZero packed implementation requires "
            f"max_segments=2, got {max_segments}"
        )
    if pending_limit < max_segments:
        raise ValueError(
            f"pending_limit must be at least {max_segments}, got {pending_limit}"
        )

    pending: deque[tuple[int, dict[str, Any]]] = deque()

    def pop_full_partner(chunk_count: int) -> dict[str, Any] | None:
        complement = chunk_capacity - chunk_count
        for index, (candidate_chunks, candidate) in enumerate(pending):
            if candidate_chunks == complement:
                del pending[index]
                return candidate
        return None

    def pop_oldest_fallback() -> list[dict[str, Any]]:
        first_chunks, first = pending.popleft()
        best_index: int | None = None
        best_total = first_chunks
        for index, (candidate_chunks, _) in enumerate(pending):
            total = first_chunks + candidate_chunks
            if total <= chunk_capacity and total > best_total:
                best_index = index
                best_total = total
        if best_index is None:
            return [first]
        _, second = pending[best_index]
        del pending[best_index]
        return [first, second]

    for sample in samples:
        chunk_count = infer_chunk_count(sample, action_horizon)
        if chunk_count > chunk_capacity:
            raise ValueError(
                f"Sample has {chunk_count} chunks, exceeding pack capacity "
                f"{chunk_capacity}"
            )

        if chunk_count == chunk_capacity:
            yield make_packed_feature(
                [sample],
                chunk_capacity=chunk_capacity,
                max_segments=max_segments,
            )
            continue

        partner = pop_full_partner(chunk_count)
        if partner is not None:
            yield make_packed_feature(
                [partner, sample],
                chunk_capacity=chunk_capacity,
                max_segments=max_segments,
            )
            continue

        pending.append((chunk_count, sample))
        while len(pending) > pending_limit:
            yield make_packed_feature(
                pop_oldest_fallback(),
                chunk_capacity=chunk_capacity,
                max_segments=max_segments,
            )

    while pending:
        yield make_packed_feature(
            pop_oldest_fallback(),
            chunk_capacity=chunk_capacity,
            max_segments=max_segments,
        )
