"""Tests for the BC training loop and metrics."""

import torch
import torch.nn.functional as F

from mahjong.features import ACTION_SPACE_SIZE, FEATURE_DIM
from mahjong.model import MLPPolicy
from mahjong.training import compute_metrics, save_checkpoint


def test_compute_metrics_sane():
    torch.manual_seed(0)
    batch = 8
    logits = torch.randn(batch, ACTION_SPACE_SIZE)
    masks = torch.zeros(batch, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :5] = True
    action_ids = torch.randint(0, 5, (batch,))
    mets = compute_metrics(logits, action_ids, masks)
    for key in ("loss", "accuracy", "top3", "top5", "illegal_prob", "illegal_rate"):
        assert key in mets
    assert 0.0 <= mets["accuracy"] <= 1.0
    assert mets["illegal_rate"] == 0.0  # masked argmax is always legal
    assert mets["illegal_prob"] < 1e-6


def test_bc_few_steps_loss_decreases():
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    features = torch.randn(8, FEATURE_DIM)
    masks = torch.zeros(8, ACTION_SPACE_SIZE, dtype=torch.bool)
    masks[:, :5] = True
    action_ids = torch.randint(0, 5, (8,))

    losses = []
    for _ in range(40):
        out = model(features, masks)
        loss = F.cross_entropy(out.logits, action_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (32,))
    save_checkpoint(
        tmp_path / "ckpt",
        model.state_dict(),
        config={"model": {"hidden_sizes": [32]}},
        metrics={"train": {"loss": 0.1, "accuracy": 0.9}},
        dataset_version="decision-v1",
        feature_version="feature-v1",
        action_schema_version="action-v1",
        training_step=10,
    )
    assert (tmp_path / "ckpt" / "model.pt").exists()
    assert (tmp_path / "ckpt" / "config.yaml").exists()
    assert (tmp_path / "ckpt" / "metrics.json").exists()
    assert (tmp_path / "ckpt" / "dataset_version.txt").read_text() == "decision-v1"
    assert (tmp_path / "ckpt" / "action_schema_version.txt").read_text() == "action-v1"
