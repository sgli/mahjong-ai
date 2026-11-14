"""Versioned feature/action choices shared by PPO training and evaluation."""

from __future__ import annotations

from .action_space import ACTION_SPACE_SIZE, action_to_id, legal_mask
from .action_space_v2 import ACTION_SPACE_SIZE_V2, action_to_id_v2, legal_mask_v2
from .encoder import FEATURE_DIM, ObservationEncoder
from .encoder_v2 import FEATURE_DIM_V2, ObservationEncoderV2
from .encoder_v3 import FEATURE_DIM_V3, ObservationEncoderV3


def resolve_policy_schema(feature_version: str, action_version: str):
    """Return (encoder, feature_dim, action_size), rejecting mixed versions."""
    if (feature_version, action_version) == ("feature-v1", "action-v1"):
        return ObservationEncoder(), FEATURE_DIM, ACTION_SPACE_SIZE
    if (feature_version, action_version) == ("feature-v2", "action-v2"):
        return ObservationEncoderV2(), FEATURE_DIM_V2, ACTION_SPACE_SIZE_V2
    if (feature_version, action_version) == ("feature-v3", "action-v2"):
        return ObservationEncoderV3(), FEATURE_DIM_V3, ACTION_SPACE_SIZE_V2
    raise ValueError(f"unsupported policy schema: {feature_version}/{action_version}")


def policy_action_ops(encoder, model):
    """Return (action_to_id, make_mask, allow_kyuushu) for a matched policy."""
    feature_version = getattr(encoder, "feature_version", "feature-v1")
    action_version = getattr(model, "action_schema_version", getattr(encoder, "action_schema_version", "action-v1"))
    _, feature_dim, action_size = resolve_policy_schema(feature_version, action_version)
    if getattr(model, "feature_dim", feature_dim) != feature_dim or getattr(model, "action_size", action_size) != action_size:
        raise ValueError("model dimensions do not match its declared policy schema")
    if action_version == "action-v2":
        allow_abort = bool(getattr(model, "allow_kyuushu", False))
        return action_to_id_v2, lambda actions: legal_mask_v2(actions, allow_kyuushu=allow_abort), allow_abort
    return action_to_id, legal_mask, False
