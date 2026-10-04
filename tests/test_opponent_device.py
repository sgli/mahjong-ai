"""Opponent decide 的 generator 设备一致性（CUDA bug 回归）。"""

from __future__ import annotations

import random

import pytest
import torch

from mahjong.decision.action import Action
from mahjong.environment import MahjongEnv
from mahjong.evaluation import PolicyOpponent

_BC = "experiments/bc_v2.1/checkpoints/best.pt"


def _obs_legal(seed=0):
    env = MahjongEnv(seed=seed)
    env.reset()
    seat = env.state.turn
    legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
    return env.observation(seat), legal


def test_cpu_same_seed_same_result():
    if not __import__("pathlib").Path(_BC).exists():
        pytest.skip("no bc checkpoint")
    opp = PolicyOpponent(opponent_id="x", type="bc", model_version="bc-v2.1", checkpoint=_BC, device="cpu", mode="sampling")
    obs, legal = _obs_legal()
    a1 = opp.decide(obs, legal, random.Random(7))
    a2 = opp.decide(obs, legal, random.Random(7))
    assert a1 == a2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_cuda_decide_no_crash():
    if not __import__("pathlib").Path(_BC).exists():
        pytest.skip("no bc checkpoint")
    opp = PolicyOpponent(opponent_id="x", type="bc", model_version="bc-v2.1", checkpoint=_BC, device="cuda", mode="sampling")
    obs, legal = _obs_legal()
    a = opp.decide(obs, legal, random.Random(7))  # 不应抛 generator 设备错误
    assert a in legal
