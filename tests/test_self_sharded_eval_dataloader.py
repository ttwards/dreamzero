from types import SimpleNamespace

from torch.utils.data import DataLoader
from transformers import Trainer

from groot.vla.data.dataset.lerobot_sharded import ShardedLeRobotMixtureDataset
from groot.vla.experiment.base import BaseTrainer


def _trainer_with_eval_dataset(eval_dataset):
    trainer = object.__new__(BaseTrainer)
    trainer.eval_dataset = eval_dataset
    trainer.data_collator = lambda samples: samples
    trainer.args = SimpleNamespace(
        eval_batch_size=1,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=None,
    )
    return trainer


def test_self_sharded_eval_dataset_bypasses_trainer_dispatch(monkeypatch):
    eval_dataset = object.__new__(ShardedLeRobotMixtureDataset)
    eval_dataset.max_samples_per_rank = 4
    trainer = _trainer_with_eval_dataset(eval_dataset)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("self-sharded eval loader must bypass Trainer dispatch")

    monkeypatch.setattr(Trainer, "get_eval_dataloader", fail_if_called)

    dataloader = trainer.get_eval_dataloader()

    assert type(dataloader) is DataLoader
    assert dataloader.dataset is eval_dataset
    assert dataloader.batch_size == 1
    assert len(dataloader) == 4


def test_other_eval_datasets_keep_trainer_behavior(monkeypatch):
    eval_dataset = [1, 2, 3]
    trainer = _trainer_with_eval_dataset(eval_dataset)
    sentinel = object()

    monkeypatch.setattr(
        Trainer,
        "get_eval_dataloader",
        lambda _self, dataset=None: sentinel if dataset is eval_dataset else None,
    )

    assert trainer.get_eval_dataloader() is sentinel
