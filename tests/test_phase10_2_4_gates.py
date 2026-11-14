from types import SimpleNamespace
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_paired_ppo import mirror_tolerance, _paired_analysis
from mahjong.decision.action import Action, ActionType
from mahjong.training.opponent_pool import collect_trajectory_with_opponents


def test_feature_v3_does_not_mask_opponent_abort():
    abort = Action(ActionType.KYUUSHU_KYUHAI)
    discard = Action(ActionType.DISCARD, tile='1m')
    class Env:
        state = SimpleNamespace(turn=1, final_ranks=[0, 1, 2, 3])
        reward_config = SimpleNamespace(placement_bonus=(2, 1, -1, -2))
        def reset(self): self.ended = False
        def done(self): return self.ended
        def legal_actions(self, seat): return [abort, discard]
        def observation(self, seat): return None
        def step(self, action):
            assert action == abort
            self.ended = True
            return {}, True
    class Opponent:
        def decide(self, obs, legal, rng):
            assert abort in legal
            return abort
    enc = SimpleNamespace(feature_version='feature-v3', action_schema_version='action-v2')
    model = SimpleNamespace(action_schema_version='action-v2', allow_kyuushu=False)
    _, steps, illegal = collect_trajectory_with_opponents(
        Env(), model, enc, 'cpu', learner_seat=0, opponents={1: Opponent()}, max_steps=10)
    assert (steps, illegal) == (1, 0)


def test_sample_size_calibration_and_legacy_gate():
    assert mirror_tolerance(10000, 'fixed-v1') == .02
    assert .0287 < mirror_tolerance(10000, 'normal99-v2') < .0289
    assert mirror_tolerance(20000, 'normal99-v2') < mirror_tolerance(10000, 'normal99-v2')
    with pytest.raises(ValueError): mirror_tolerance(0, 'normal99-v2')


def test_invalid_control_is_not_reported_as_valid_model_failure():
    rows = [dict(seed=i, seat=i % 4, game_index=i, final_ranks=[0, 0, 0, 0],
                 illegal_actions=0, completed=True, error=None) for i in range(100)]
    result = _paired_analysis(rows, rows, bootstrap_seed=1, replicates=10,
                              mirror_sanity_mode='normal99-v2')
    assert not result['valid_run']
    assert result['decision'] == 'INVALID'
