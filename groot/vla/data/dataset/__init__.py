from typing import TYPE_CHECKING

from .egovla_wds import EgoVLAWdsDataset

if TYPE_CHECKING:
    from .lerobot import ModalityConfig

__all__ = [
    "EgoVLAWdsDataset",
    "ModalityConfig",
]


def __getattr__(name: str):
    """Avoid importing LeRobot's optional video stack for native WDS users."""
    if name == "ModalityConfig":
        from .lerobot import ModalityConfig

        return ModalityConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
