"""Feature Encoder and Action Space (Phase 5)."""

from .action_space import (
    ACTION_SPACE_SIZE,
    FEATURE_VERSION,
    action_from_id,
    action_to_id,
    canonical_action,
    legal_mask,
)

__all__ = [
    "ACTION_SPACE_SIZE",
    "FEATURE_DIM",
    "FEATURE_VERSION",
    "ObservationEncoder",
    "action_from_id",
    "action_to_id",
    "canonical_action",
    "encode_observation",
    "legal_mask",
]


def __getattr__(name):
    # Lazy import of the torch-dependent encoder, so the dataset/parser path
    # (which only needs FEATURE_VERSION) does not pull in torch.
    if name in ("FEATURE_DIM", "ObservationEncoder", "encode_observation"):
        from . import encoder

        return getattr(encoder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
