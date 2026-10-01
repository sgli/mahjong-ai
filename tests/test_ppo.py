"""Tests for the PPO value head, GAE, clip objective and trajectory collection."""

import torch

from mahjong.decision import Action, ActionType, PlayerObservation
from mahjong.environment import MahjongEnv
from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM, ObservationEncoder, legal_mask
from mahjong.model import MLPPolicy
from mahjong.training import attribute_step_rewards, clipped_surrogate, collect_trajectory, compute_gae, ppo_loss


def _obs():
    return PlayerObservation(
        seat=0, bakaze="E", kyoku=1, honba=0, kyotaku=0, oya=0,
        scores=(25000, 25000, 25000, 25000), riichi=(False, False, False, False),
        hand=("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"),
        melds=(), opponents_melds=((), (), (), ()), discards=((), (), (), ()),
        dora_markers=("2m",), turn=0,
    )


# -- value head / policy interface ---------------------------------------------
def test_value_head_output_shape():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    features = torch.randn(4, FEATURE_DIM)
    mask = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :5] = True
    out = model(features, mask)
    assert out.value is not None
    assert out.value.shape == (4,)


def test_sample_action_legal_and_log_prob():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    features = torch.randn(4, FEATURE_DIM)
    mask = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :3] = True
    action_ids, logp = model.sample(features, mask)
    assert mask.gather(1, action_ids.unsqueeze(-1)).all()
    assert logp.shape == (4,)
    assert (logp <= 0.0).all()


def test_evaluate_actions_returns_triplet():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    features = torch.randn(4, FEATURE_DIM)
    mask = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :3] = True
    action_ids = torch.tensor([0, 1, 2, 0])
    log_probs, values, entropy = model.evaluate_actions(features, mask, action_ids)
    assert log_probs.shape == (4,)
    assert values.shape == (4,)
    assert entropy.shape == (4,)
    assert (entropy >= 0.0).all()


# -- GAE / clip math -----------------------------------------------------------
def test_compute_gae_hand_computed():
    rewards = [1.0, 0.0, 1.0]
    values = [0.5, 0.5, 0.5]
    dones = [False, False, True]
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
    assert torch.allclose(torch.tensor(advantages), torch.tensor([1.31, 0.4, 0.5]), atol=1e-6)
    assert torch.allclose(torch.tensor(returns), torch.tensor([1.81, 0.9, 1.0]), atol=1e-6)


def test_clipped_surrogate_hand_computed():
    # ratio == 1 -> no clipping
    s = clipped_surrogate(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), torch.tensor([1.0, -1.0]), 0.2)
    assert torch.allclose(s, torch.tensor([1.0, -1.0]))

    # positive advantage + large ratio -> clipped to 1+eps
    s2 = clipped_surrogate(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), 0.2)
    assert torch.allclose(s2, torch.tensor([1.2]), atol=1e-6)

    # negative advantage + large ratio -> not clipped (min takes surr1)
    s3 = clipped_surrogate(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([-1.0]), 0.2)
    assert torch.allclose(s3, torch.tensor([-torch.exp(torch.tensor(2.0))]), atol=1e-3)


def test_ppo_loss_runs_and_has_metrics():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    features = torch.randn(8, FEATURE_DIM)
    mask = torch.zeros(8, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :5] = True
    action_ids = torch.randint(0, 5, (8,))
    old_logp = torch.zeros(8)
    advantages = torch.randn(8)
    returns = torch.randn(8)
    loss, metrics = ppo_loss(
        model, features, mask, action_ids, old_logp, advantages, returns,
        clip_eps=0.2, vf_coef=0.5, ent_coef=0.01,
    )
    assert torch.isfinite(loss)
    for k in ("policy_loss", "value_loss", "entropy"):
        assert k in metrics


# -- trajectory collection -----------------------------------------------------
def _empty_traj():
    return {
        s: {"features": [], "action_ids": [], "log_probs": [], "values": [], "rewards": [], "masks": [], "dones": []}
        for s in range(4)
    }


def test_attribute_step_rewards_ron():
    traj = _empty_traj()
    # seat 0 previously discarded (its last reward is 0), seat 1 rons
    for s in range(4):
        traj[s]["rewards"].append(0.0)
    pending = {s: 0.0 for s in range(4)}
    # env.step returns the settlement deltas; acting seat is the ron winner (1)
    r = attribute_step_rewards({0: -2.6, 1: 3.6, 2: 0.0, 3: 0.0}, 1, traj, pending)
    assert r == 3.6  # winner's reward is recorded on its transition
    assert traj[0]["rewards"][-1] == -2.6  # discarder's ron payment attributed retroactively
    assert traj[2]["rewards"][-1] == 0.0
    assert traj[3]["rewards"][-1] == 0.0


def test_attribute_step_rewards_tsumo():
    traj = _empty_traj()
    for s in range(4):
        traj[s]["rewards"].append(0.0)
    pending = {s: 0.0 for s in range(4)}
    # dealer tsumo: winner +3*2*scale, others -2*scale (example)
    r = attribute_step_rewards({0: 6.0, 1: -2.0, 2: -2.0, 3: -2.0}, 0, traj, pending)
    assert r == 6.0
    assert traj[1]["rewards"][-1] == -2.0
    assert traj[2]["rewards"][-1] == -2.0
    assert traj[3]["rewards"][-1] == -2.0
    # point conservation: winner reward + three losers' payments == 0
    assert r + traj[1]["rewards"][-1] + traj[2]["rewards"][-1] + traj[3]["rewards"][-1] == 0.0


def test_collect_trajectory_runs():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=5)
    traj, steps, illegal = collect_trajectory(env, model, encoder, torch.device("cpu"), max_steps=400)
    assert steps > 0
    assert illegal == 0
    assert any(len(traj[s]["features"]) > 0 for s in range(4))


def test_collect_trajectory_reward_conservation():
    """Per-seat recorded rewards must equal env.reward(seat) (all deltas attributed)."""
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    encoder = ObservationEncoder()
    env = MahjongEnv(seed=11)
    traj, steps, illegal = collect_trajectory(env, model, encoder, torch.device("cpu"), max_steps=3000)
    assert illegal == 0
    for s in range(4):
        total = sum(traj[s]["rewards"]) if traj[s]["rewards"] else 0.0
        assert abs(total - env.reward(s)) < 1e-6, f"seat {s}: {total} != {env.reward(s)}"


# -- seed reproducibility ------------------------------------------------------
def test_seed_reproducible():
    torch.manual_seed(7)
    m1 = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    f1 = torch.randn(4, FEATURE_DIM)
    msk1 = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    msk1[:, :5] = True
    torch.manual_seed(7)
    m2 = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    f2 = torch.randn(4, FEATURE_DIM)
    msk2 = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    msk2[:, :5] = True
    assert torch.equal(m1(f1, msk1).logits, m2(f2, msk2).logits)
    assert torch.equal(m1(f1, msk1).value, m2(f2, msk2).value)


def test_checkpoint_value_head_roundtrip(tmp_path):
    from mahjong.training import load_checkpoint, save_checkpoint

    torch.manual_seed(3)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32, 32))
    save_checkpoint(
        tmp_path, model.state_dict(),
        config={"model": {"hidden_sizes": [32, 32]}}, metrics={},
        dataset_version="decision-v1", feature_version="feature-v1",
        action_schema_version="action-v1", training_step=1,
    )
    loaded = load_checkpoint(tmp_path)["state_dict"]
    assert "value_head.weight" in loaded
    assert torch.equal(loaded["value_head.weight"], model.state_dict()["value_head.weight"])
