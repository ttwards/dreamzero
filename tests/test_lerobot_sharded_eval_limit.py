from types import MethodType, SimpleNamespace

import groot.vla.data.dataset.lerobot_sharded as lerobot_sharded
from groot.vla.data.dataset.lerobot_sharded import (
    ShardedLeRobotMixtureDataset,
    _group_trajectories_into_shards,
)


def _finite_dataset(max_samples_per_rank):
    dataset = object.__new__(ShardedLeRobotMixtureDataset)
    dataset.fixed_chunk_count = None
    dataset.pack_chunk_capacity = None
    dataset.max_samples_per_rank = max_samples_per_rank
    dataset._iter_transformed_samples = MethodType(
        lambda _self: iter(range(20)),
        dataset,
    )
    return dataset


def test_validation_limit_is_split_evenly_across_workers():
    original_get_worker_info = lerobot_sharded.get_worker_info
    try:
        worker_outputs = []
        for worker_id in range(4):
            lerobot_sharded.get_worker_info = lambda worker_id=worker_id: SimpleNamespace(
                id=worker_id,
                num_workers=4,
            )
            worker_outputs.append(list(_finite_dataset(5)))
    finally:
        lerobot_sharded.get_worker_info = original_get_worker_info

    assert [len(output) for output in worker_outputs] == [2, 1, 1, 1]
    assert sum(map(len, worker_outputs)) == 5


def test_unlimited_training_iterator_is_unchanged():
    assert list(iter(_finite_dataset(None))) == list(range(20))


def test_trajectory_shards_handle_exact_cutoffs_without_empty_shards():
    shards, lengths = _group_trajectories_into_shards(
        [0, 1],
        {0: range(1000), 1: range(1000)},
        num_steps_per_shard=1000,
    )

    assert shards == [[0], [1]]
    assert lengths.tolist() == [1000, 1000]


def test_trajectory_shards_allow_one_episode_to_cross_multiple_cutoffs():
    shards, lengths = _group_trajectories_into_shards(
        [0, 1, 2],
        {0: range(2500), 1: range(100), 2: range(100)},
        num_steps_per_shard=1000,
    )

    assert shards == [[0], [1, 2]]
    assert lengths.tolist() == [2500, 200]
    assert all(shards)
