"""Trajectory / episode schema for PPO (Phase 10.2 §11).

Fixes the data semantics of a rollout step and episode metadata, so rollouts no
longer rely on loose Python dicts.  Phase 10.2 does not require performance
optimisation here — only semantic clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrajectoryStep:
    """One transition recorded for one seat."""

    features: object  # [FEATURE_DIM] tensor (encoder output)
    action_id: int
    legal_mask: object  # [ACTION_SPACE_SIZE] bool tensor
    old_log_prob: float
    value: float
    reward: float
    done: bool


@dataclass
class EpisodeMetadata:
    """Provenance for one episode (game)."""

    episode_id: str
    seat: int
    step: int
    seed: int
    policy_version: str
    feature_version: str
    action_schema_version: str
    environment_version: str
    reward_version: str


@dataclass
class TrajectoryBuffer:
    """Per-seat trajectory buffers (replaces the loose dict in ``collect_trajectory``)."""

    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: EpisodeMetadata | None = None

    def append(self, step: TrajectoryStep) -> None:
        self.steps.append(step)


__all__ = ["EpisodeMetadata", "TrajectoryBuffer", "TrajectoryStep"]
