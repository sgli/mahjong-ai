"""Tests for the MLP policy."""

import torch

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training import load_checkpoint, save_checkpoint


def _model_and_batch(seed=0, hidden=(32, 32)):
    torch.manual_seed(seed)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, hidden)
    features = torch.randn(4, FEATURE_DIM)
    mask = torch.zeros(4, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :3] = True  # only first 3 actions legal
    return model, features, mask


def test_forward_output_shape():
    model, features, mask = _model_and_batch()
    out = model(features, mask)
    assert out.logits.shape == (4, ACTION_SPACE_SIZE)
    assert out.value is not None
    assert out.value.shape == (4,)


def test_mask_shields_illegal_actions():
    model, features, mask = _model_and_batch()
    probs = model.action_probs(features, mask)
    # illegal positions have ~0 probability
    assert (probs[:, ~mask[0]] < 1e-6).all()
    # legal positions sum to 1
    assert torch.allclose(probs[:, mask[0]].sum(dim=-1), torch.ones(4), atol=1e-5)


def test_sample_never_illegal():
    model, features, mask = _model_and_batch()
    sampled, logp = model.sample(features, mask)
    assert mask.gather(1, sampled.unsqueeze(-1)).all()
    assert logp.shape == (4,)


def test_same_seed_reproducible():
    m1, f1, msk1 = _model_and_batch(seed=7)
    m2, f2, msk2 = _model_and_batch(seed=7)
    assert torch.equal(m1(f1, msk1).logits, m2(f2, msk2).logits)


def test_checkpoint_save_load_preserves_weights(tmp_path):
    model, _, _ = _model_and_batch(seed=3)
    save_checkpoint(
        tmp_path,
        model.state_dict(),
        config={"model": {"hidden_sizes": [32, 32]}},
        metrics={"train": {"loss": 0.5}},
        dataset_version="decision-v1",
        feature_version="feature-v1",
        action_schema_version="action-v1",
        git_commit_hash="abc123",
        training_step=1,
    )
    loaded = load_checkpoint(tmp_path)
    assert set(loaded["state_dict"].keys()) == set(model.state_dict().keys())
    for key in model.state_dict():
        assert torch.equal(loaded["state_dict"][key], model.state_dict()[key])
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "git_commit.txt").read_text() == "abc123"
    assert (tmp_path / "feature_version.txt").read_text() == "feature-v1"
