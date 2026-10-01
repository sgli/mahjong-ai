"""Evaluation: opponent pool + self-play + benchmark (Phase 8/9)."""

from .benchmark import PromotionResult, compute_metrics, compute_rotated_metrics, run_promotion, rules_test, smoke_test
from .opponent import Opponent, PolicyOpponent, RandomOpponent, load_policy
from .pool import OpponentPool, default_pool
from .selfplay import GameResult, run_game, run_round, summarize

__all__ = [
    "GameResult",
    "Opponent",
    "OpponentPool",
    "PolicyOpponent",
    "PromotionResult",
    "RandomOpponent",
    "compute_metrics",
    "compute_rotated_metrics",
    "default_pool",
    "load_policy",
    "run_game",
    "run_promotion",
    "run_round",
    "rules_test",
    "smoke_test",
    "summarize",
]
