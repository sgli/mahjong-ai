"""Opponent pool + historical checkpoint registration (Phase 10.3 §12/§13/§29)."""

from __future__ import annotations

from pathlib import Path

from mahjong.training import TrainingOpponentPool, register_historical_checkpoints


def test_sampling_reproducible_same_seed():
    probs = {"random-v1": 0.1, "rule-v1": 0.2, "bc-v2.1": 0.3, "ppo-v1": 0.2, "historical": 0.2}
    a = TrainingOpponentPool({}, probs, seed=42)
    b = TrainingOpponentPool({}, probs, seed=42)
    assert a.sample(20) == b.sample(20)


def test_sampling_probabilities_match_config_statistically():
    probs = {"random-v1": 0.1, "rule-v1": 0.2, "bc-v2.1": 0.7}
    pool = TrainingOpponentPool({}, probs, seed=7)
    n = 10000
    samples = pool.sample(n)
    from collections import Counter

    counts = Counter(samples)
    for k, p in probs.items():
        assert abs(counts[k] / n - p) < 0.03, f"{k}: {counts[k]/n} vs {p}"


def test_register_historical_checkpoints(tmp_path):
    for name in ("epoch_20k.pt", "epoch_50k.pt", "latest.pt", "best.pt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    reg = register_historical_checkpoints(tmp_path)
    assert "historical:epoch_20k" in reg
    assert "historical:epoch_50k" in reg
    assert all("latest" not in k and "best" not in k for k in reg)
