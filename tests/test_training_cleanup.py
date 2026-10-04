"""Tests for training/model cleanup (pending flush, streaming batches, unified Policy)."""

import torch

from mahjong.decision.action import Action, ActionType
from mahjong.decision.observation import PlayerObservation
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder
from mahjong.model import Policy
from mahjong.training import flush_terminal_rewards


class _FakeEnv:
    class _State:
        final_ranks = [0, 1, 2, 3]

    class _RC:
        placement_bonus = (2.0, 1.0, -1.0, -2.0)

    state = _State()
    reward_config = _RC()


def _obs():
    return PlayerObservation(
        seat=0, bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0,
        scores=(25000, 25000, 25000, 25000), riichi=(False, False, False, False),
        hand=("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"),
        melds=(), opponents_melds=((), (), (), ()), discards=((), (), (), ()),
        dora_markers=("2m",), turn=0,
    )


# -- item 1: pending reward flush ---------------------------------------------
def test_flush_terminal_rewards_flushes_pending_and_placement():
    from mahjong.training import TrajectoryBuffer, TrajectoryStep

    traj = {s: TrajectoryBuffer() for s in range(4)}
    for s in range(4):
        traj[s].append(TrajectoryStep(features=None, action_id=0, legal_mask=None, old_log_prob=0.0, value=0.0, reward=0.0, done=False))
    pending = {0: 0.0, 1: 2.5, 2: 0.0, 3: -1.0}
    flush_terminal_rewards(_FakeEnv(), traj, pending)
    # pending added to the seat's last transition, then placement bonus
    assert traj[0].steps[-1].reward == 2.0  # 0 + rank0 bonus
    assert traj[1].steps[-1].reward == 2.5 + 1.0  # pending + rank1 bonus
    assert traj[2].steps[-1].reward == -1.0  # rank2 bonus
    assert traj[3].steps[-1].reward == -1.0 + -2.0  # pending + rank3 bonus
    assert all(pending[s] == 0.0 for s in range(4))


# -- item 2: streaming iter_batches -------------------------------------------
def test_iter_batches_streaming_order(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from mahjong.training import bc

    table = pa.table({"i": pa.array(range(5), type=pa.int64())})
    path = tmp_path / "d.parquet"
    pq.write_table(table, path)

    def fake_row_to_tensors(row, encoder):
        return (torch.tensor([float(row["i"])]), int(row["i"]), torch.tensor([True]))

    monkeypatch.setattr(bc, "row_to_tensors", fake_row_to_tensors)
    batches = list(bc.iter_batches([path], None, batch_size=2, device=torch.device("cpu")))
    assert len(batches) == 3  # 5 rows / batch_size 2 -> 2+2+1
    feats = torch.cat([b[0] for b in batches], 0)
    acts = torch.cat([b[1] for b in batches], 0)
    assert feats.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert acts.tolist() == [0, 1, 2, 3, 4]


# -- item 3: unified Policy interface -----------------------------------------
def test_policy_interface_unified():
    torch.manual_seed(0)
    policy = Policy(ObservationEncoder(), FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    obs = _obs()
    legal = [Action(ActionType.DISCARD, tile=t) for t in ("1m", "2m", "3m")]

    out = policy.forward(obs, legal)
    assert out.logits is not None and out.value is not None
    assert out.logits.shape == (1, ACTION_SPACE_SIZE)
    assert out.value.shape == (1,)

    action_id, logp = policy.sample_action(obs, legal)
    assert 0 <= action_id < ACTION_SPACE_SIZE
    assert logp <= 0.0

    log_prob, value, entropy = policy.evaluate_actions(obs, legal, torch.tensor([action_id]))
    assert log_prob.shape == ()
    assert value.shape == ()
    assert entropy.shape == ()
    assert entropy.item() >= 0.0
