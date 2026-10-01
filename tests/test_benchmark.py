"""Tests for benchmark metrics + promotion flow (Phase 9)."""

import random

from mahjong.evaluation import (
    GameResult,
    RandomOpponent,
    compute_metrics,
    run_promotion,
    smoke_test,
)
from mahjong.evaluation.selfplay import run_round


def _stats(win=0, dealt=0, riichi=0, call=0, win_points=0):
    return {
        "win_count": win,
        "tsumo_count": 0,
        "ron_count": 0,
        "dealt_in_count": dealt,
        "riichi_count": riichi,
        "call_count": call,
        "win_points": win_points,
    }


def _result(seed, rank, score, stats):
    return GameResult(
        seed=seed,
        opponents={0: "c@v1", 1: "r@v1", 2: "r@v1", 3: "r@v1"},
        final_scores=[score, 25000, 25000, 25000],
        final_ranks=[rank, 0, 1, 2],
        steps=100,
        illegal_actions=0,
        stats=stats,
    )


def test_compute_metrics_hand_computed():
    results = [
        _result(0, 1, 26000, [_stats(win=1, riichi=1, win_points=8000), _stats(), _stats(), _stats()]),
        _result(1, 2, 24000, [_stats(dealt=1, call=1), _stats(), _stats(), _stats()]),
    ]
    m = compute_metrics(results, seat=0)
    assert m["mean_rank"] == 1.5
    assert m["rank_distribution"] == [0, 1, 1, 0]
    assert m["win_rate"] == 0.5
    assert m["deal_in_rate"] == 0.5
    assert m["riichi_rate"] == 0.5
    assert m["call_rate"] == 0.5
    assert m["avg_win_points"] == 8000.0
    assert m["mean_score_change"] == 0.0  # (26000+24000)/2 - 25000


def test_smoke_test_passes_with_random():
    assert smoke_test(RandomOpponent(), seed=0) is True


def test_promotion_flow_runs(tmp_path):
    candidate = RandomOpponent(opponent_id="cand", model_version="v1")
    opponents = [RandomOpponent(opponent_id=f"r{s}", model_version="v1") for s in range(3)]
    baseline = RandomOpponent(opponent_id="base", model_version="v1")
    result = run_promotion(
        candidate, opponents, baseline=baseline, games=2, seed=0,
        candidate_seat=0, rules_games=1, output_dir=tmp_path,
    )
    assert result.smoke_passed is True
    assert result.rules_passed is True
    assert result.illegal_rate == 0.0
    assert set(result.metrics) == {
        "mean_rank", "rank_distribution", "win_rate", "deal_in_rate",
        "riichi_rate", "call_rate", "avg_win_points", "mean_score_change",
    }
    # opponents recorded with id@version
    assert result.opponents[0] == "cand@v1"
    assert result.promoted is not None
    # result JSON written (no overwrite: unique dir)
    out_dirs = list(tmp_path.glob("cand-*"))
    assert len(out_dirs) == 1
    assert (out_dirs[0] / "result.json").exists()


def test_no_overwrite_history(tmp_path):
    candidate = RandomOpponent(opponent_id="cand", model_version="v1")
    opponents = [RandomOpponent() for _ in range(3)]
    run_promotion(candidate, opponents, baseline=None, games=1, seed=0, output_dir=tmp_path)
    run_promotion(candidate, opponents, baseline=None, games=1, seed=0, output_dir=tmp_path)
    assert len(list(tmp_path.glob("cand-*"))) == 2


def test_same_seed_reproducible_metrics():
    opponents = [RandomOpponent(opponent_id=f"r{s}", model_version="v1") for s in range(4)]
    r1 = run_round(opponents, games=2, seed=7)
    r2 = run_round(opponents, games=2, seed=7)
    m1 = compute_metrics(r1, 0)
    m2 = compute_metrics(r2, 0)
    assert m1 == m2
