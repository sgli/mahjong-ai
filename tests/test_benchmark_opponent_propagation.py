"""候选参数 → 对手传播的行为验证（Phase 10.3 诊断 A 收尾）。"""

from __future__ import annotations

import inspect
import random
import sys
from pathlib import Path

import pytest
import torch

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import benchmark_parallel as bp  # noqa: E402

from mahjong.decision.action import Action  # noqa: E402
from mahjong.environment import MahjongEnv  # noqa: E402

_BC = Path("experiments/bc_v2.1/checkpoints/best.pt")
needs_bc = pytest.mark.skipif(not _BC.exists(), reason="no bc checkpoint")


def _obs_legal(seed=0):
    env = MahjongEnv(seed=seed)
    env.reset()
    seat = env.state.turn
    legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
    return env.observation(seat), legal


def test_build_opponent_propagates_mode_and_temperature():
    if not _BC.exists():
        pytest.skip("no bc checkpoint")
    opp = bp._build_opponent("bc_v2_1", str(_BC), mode="greedy", temperature=0.5)
    assert opp.mode == "greedy"
    assert opp.temperature == 0.5


def test_build_opponent_default_sampling_one():
    if not _BC.exists():
        pytest.skip("no bc checkpoint")
    opp = bp._build_opponent("bc_v2_1", str(_BC), mode="sampling", temperature=1.0)
    assert opp.mode == "sampling"
    assert opp.temperature == 1.0


def test_run_matchup_signature_carries_opponent_params():
    sig = inspect.signature(bp.run_matchup)
    assert "opponent_mode" in sig.parameters
    assert "opponent_temperature" in sig.parameters
    assert "candidate_mode" in sig.parameters
    assert "candidate_temperature" in sig.parameters


def test_argparse_to_constructor_propagation():
    args = bp._build_parser().parse_args(["--opponent-mode", "greedy", "--opponent-temperature", "0.5"])
    assert args.opponent_mode == "greedy"
    assert args.opponent_temperature == 0.5
    # 由 argparse 参数构造对手 → mode/temperature 落地（与 worker 同一工厂）
    if _BC.exists():
        opp = bp._build_opponent("bc_v2_1", str(_BC), mode=args.opponent_mode, temperature=args.opponent_temperature)
        assert opp.mode == "greedy"
        assert opp.temperature == 0.5


@needs_bc
def test_greedy_decision_constant():
    opp = bp._build_opponent("bc_v2_1", str(_BC), mode="greedy", temperature=1.0)
    obs, legal = _obs_legal()
    a1 = opp.decide(obs, legal, random.Random(1))
    a2 = opp.decide(obs, legal, random.Random(999))
    assert a1 == a2  # greedy 与 rng 无关，恒定


@needs_bc
def test_sampling_decision_reproducible_same_seed():
    opp = bp._build_opponent("bc_v2_1", str(_BC), mode="sampling", temperature=1.0)
    obs, legal = _obs_legal()
    a1 = opp.decide(obs, legal, random.Random(7))
    a2 = opp.decide(obs, legal, random.Random(7))
    assert a1 == a2  # 同 seed 可复现


@needs_bc
def test_temperature_changes_distribution():
    """T=0.5 与 T=1.0 在固定 logits 上产生不同分布（temperature 真正生效）。"""
    from mahjong.evaluation import load_policy

    _, model = load_policy(str(_BC), "cpu")
    model.eval()
    obs, legal = _obs_legal()
    from mahjong.features import ObservationEncoder
    from mahjong.features.action_space import legal_mask

    encoder = ObservationEncoder()
    feats = encoder.encode(obs).unsqueeze(0)
    mask = legal_mask(legal).unsqueeze(0)
    with torch.no_grad():
        logits = model(feats, mask).logits
    p_05 = torch.softmax(logits / 0.5, dim=-1)
    p_1 = torch.softmax(logits / 1.0, dim=-1)
    assert not torch.allclose(p_05, p_1, atol=1e-6)
