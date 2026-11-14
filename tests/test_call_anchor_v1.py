from copy import deepcopy

import torch

from mahjong.features.action_space_v2 import action_from_id_v2
from mahjong.model.policy import MLPPolicy
from mahjong.training.ppo import ppo_loss


CALL = next(i for i in range(325) if action_from_id_v2(i).type.value == 'pon')
PASS = next(i for i in range(325) if action_from_id_v2(i).type.value == 'pass')


def fixture():
    torch.manual_seed(7)
    bc = MLPPolicy(4, 325, (8,))
    model = deepcopy(bc)
    with torch.no_grad():
        model.action_head.bias[CALL] += 2.0
    x = torch.randn(3, 4)
    masks = torch.zeros(3, 325, dtype=torch.bool)
    masks[:, PASS] = True
    masks[0, CALL] = masks[1, CALL] = True
    action = torch.full((3,), PASS)
    old = model.log_probs(x, masks).gather(1, action[:, None]).squeeze(1).detach()
    return model, bc, x, masks, action, old


def loss(model, bc, x, masks, action, old, beta, warmup=False):
    return ppo_loss(model, x, masks, action, old, torch.zeros(3), torch.zeros(3),
                    clip_eps=.2, vf_coef=0., ent_coef=0., bc_model=bc,
                    call_anchor_beta=beta, decision_only=True, warmup=warmup)


def test_call_anchor_penalizes_increased_call_mass_and_pushes_it_down():
    model, bc, x, masks, action, old = fixture()
    base, _ = loss(model, bc, x, masks, action, old, 0.)
    anchored, metrics = loss(model, bc, x, masks, action, old, 1.)
    assert anchored > base
    assert metrics['call_anchor_opportunities'] == 2
    assert metrics['call_mass_theta'] > metrics['call_mass_bc']
    anchored.backward()
    assert model.action_head.bias.grad[CALL] > 0


def test_call_anchor_zero_or_warmup_preserves_old_objective():
    model, bc, x, masks, action, old = fixture()
    base, _ = loss(model, bc, x, masks, action, old, 0.)
    same, _ = loss(model, bc, x, masks, action, old, 0.)
    assert torch.equal(base, same)
    warm_base, _ = loss(model, bc, x, masks, action, old, 0., warmup=True)
    warm_anchor, _ = loss(model, bc, x, masks, action, old, 1., warmup=True)
    assert torch.equal(warm_base, warm_anchor)


def test_no_call_opportunity_has_no_extra_loss():
    model, bc, x, masks, action, old = fixture()
    masks[:, CALL] = False
    old = model.log_probs(x, masks).gather(1, action[:, None]).squeeze(1).detach()
    base, _ = loss(model, bc, x, masks, action, old, 0.)
    anchored, metrics = loss(model, bc, x, masks, action, old, 1.)
    assert torch.equal(base, anchored)
    assert metrics['call_anchor_opportunities'] == 0
