"""Training opponent pool with config-driven proportional sampling (Phase 10.3 §12/§13).

Opponent probabilities come **from config** (never hard-coded); sampling is
seeded (same seed ⇒ same sequence).  Historical checkpoint registration is a
thin helper that scans a directory for milestone checkpoints.
"""

from __future__ import annotations

import random
from pathlib import Path


class TrainingOpponentPool:
    """Weighted opponent sampler keyed by opponent id.

    ``opponents``: ``{id: checkpoint_or_factory}`` (checkpoint path or any label).
    ``probabilities``: ``{id: float}`` (must come from config).
    """

    def __init__(self, opponents: dict[str, object], probabilities: dict[str, float], *, seed: int):
        ids = list(probabilities)
        if not ids:
            raise ValueError("opponent pool probabilities must not be empty")
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("opponent pool probabilities must sum to > 0")
        # 归一化 + 只保留 probabilities 中声明的 id
        self._ids = [i for i in ids if probabilities.get(i, 0.0) > 0]
        self._weights = [probabilities[i] for i in self._ids]
        self._opponents = opponents
        self._rng = random.Random(seed)
        self.seed = seed

    def sample(self, n: int = 1) -> list[str]:
        """Sample ``n`` opponent ids with replacement (seeded, reproducible)."""
        return self._rng.choices(self._ids, weights=self._weights, k=n)

    def sample_one(self) -> str:
        return self.sample(1)[0]

    def ids(self) -> list[str]:
        return list(self._ids)

    def opponent(self, opponent_id: str):
        return self._opponents.get(opponent_id)


def register_historical_checkpoints(
    pool_dir: str | Path,
    *,
    prefix: str = "historical",
    pattern: str = "*.pt",
) -> dict[str, str]:
    """Register milestone checkpoints under ``pool_dir`` as historical opponents.

    Returns ``{opponent_id: checkpoint_path}`` for every matching checkpoint
    (sorted by name); do not overwrite an existing ``latest`` — the caller keeps
    ``current-ppo`` separately.
    """
    d = Path(pool_dir)
    if not d.is_dir():
        return {}
    out: dict[str, str] = {}
    for p in sorted(d.glob(pattern)):
        if p.name in ("latest.pt", "best.pt"):
            continue  # 里程碑只注册 epoch_*.pt / 显式 checkpoint
        out[f"{prefix}:{p.stem}"] = str(p)
    return out


__all__ = ["TrainingOpponentPool", "register_historical_checkpoints"]
