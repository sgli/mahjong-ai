"""Evaluation: opponent pool + self-play + benchmark (Phase 8/9)."""

from .benchmark import PromotionResult, compute_metrics, compute_opponents_metrics, compute_rotated_metrics, run_promotion, rules_test, smoke_test
from .diagnosis import entropy_exploration_metrics, value_quality_metrics
from .opponent import Opponent, PolicyOpponent, RandomOpponent, RuleOpponent, load_policy
from .pool import OpponentPool, default_pool
from .retention import retention_metrics
from .selfplay import GameResult, run_game, run_round, summarize

__all__ = [
    "GameResult",
    "Opponent",
    "OpponentPool",
    "PolicyOpponent",
    "PromotionResult",
    "RandomOpponent",
    "RuleOpponent",
    "compute_metrics",
    "compute_opponents_metrics",
    "compute_rotated_metrics",
    "default_pool",
    "entropy_exploration_metrics",
    "load_policy",
    "retention_metrics",
    "run_game",
    "run_promotion",
    "run_round",
    "rules_test",
    "smoke_test",
    "summarize",
    "value_quality_metrics",
]
