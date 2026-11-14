"""Behavioral regressions for combined PPO controls."""
import torch
from mahjong.model.policy import MLPPolicy
from mahjong.training.ppo import ppo_loss, set_value_warmup_trainable


def inputs():
    torch.manual_seed(42)
    model = MLPPolicy(4, 3, (8,), value_separate_trunk=True)
    bc = MLPPolicy(4, 3, (8,))
    x = torch.randn(4, 4)
    masks = torch.ones(4, 3, dtype=torch.bool)
    a = torch.zeros(4, dtype=torch.long)
    old = model.log_probs(x, masks)[:, 0].detach()
    return model, bc, x, masks, a, old


def test_warmup_with_nonzero_anchor_preserves_actor_and_trains_value_trunk():
    m, bc, x, mask, a, old = inputs()
    set_value_warmup_trainable(m)
    before = {n: p.detach().clone() for n,p in m.named_parameters()}
    loss, _ = ppo_loss(m, x, mask, a, old, torch.ones(4), torch.ones(4)*3,
                       clip_eps=.2, vf_coef=.5, ent_coef=.01,
                       bc_model=bc, bc_anchor_beta=1., warmup=True)
    loss.backward()
    opt = torch.optim.Adam(m.parameters(), lr=.01); opt.step()
    for n,p in m.named_parameters():
        if not n.startswith('value_'):
            assert p.grad is None
            assert torch.equal(p, before[n])
    assert any(not torch.equal(p, before[n]) for n,p in m.named_parameters() if n.startswith('value_trunk.'))


def test_warmup_loss_excludes_anchor_even_without_freezing_parameters():
    m, bc, x, mask, a, old = inputs()
    kwargs = dict(clip_eps=.2, vf_coef=.5, ent_coef=.01, warmup=True)
    base,_ = ppo_loss(m,x,mask,a,old,torch.ones(4),torch.ones(4),**kwargs)
    anchored,_ = ppo_loss(m,x,mask,a,old,torch.ones(4),torch.ones(4),bc_model=bc,bc_anchor_beta=1.,**kwargs)
    assert torch.equal(base, anchored)


def test_forced_rows_do_not_dilute_decision_policy_gradient():
    m, _, x, mask, a, old = inputs()
    kwargs = dict(clip_eps=.2, vf_coef=0., ent_coef=.01, decision_only=True)
    loss,_ = ppo_loss(m,x,mask,a,old,torch.ones(4),torch.zeros(4),**kwargs)
    grad = torch.autograd.grad(loss, m.action_head.weight)[0]
    forced = torch.zeros_like(mask); forced[:,0] = True
    loss2,_ = ppo_loss(m,torch.cat([x,x]),torch.cat([mask,forced]),torch.cat([a,a]),
                       torch.cat([old,torch.zeros(4)]),torch.ones(8),torch.zeros(8),**kwargs)
    grad2 = torch.autograd.grad(loss2,m.action_head.weight)[0]
    assert torch.allclose(grad,grad2,atol=1e-7)


def test_all_forced_batch_finite_with_no_actor_gradient():
    m,bc,x,mask,a,old = inputs()
    mask[:,1:] = False
    loss,_=ppo_loss(m,x,mask,a,torch.zeros(4),torch.ones(4),torch.ones(4),
                    clip_eps=.2,vf_coef=.5,ent_coef=.01,decision_only=True,bc_model=bc,bc_anchor_beta=.1)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.equal(m.action_head.weight.grad,torch.zeros_like(m.action_head.weight.grad))


def test_training_kl_limit_stops_further_updates():
    from mahjong.features import FEATURE_DIM, ACTION_SPACE_SIZE, ObservationEncoder
    from mahjong.training.ppo import train_ppo
    torch.manual_seed(5)
    m = MLPPolicy(FEATURE_DIM, ACTION_SPACE_SIZE, (16,))
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    history = train_ppo(m, ObservationEncoder(), opt, num_envs=1, env_seed=42,
                        epochs=1, steps_per_epoch=100, batch_size=32,
                        update_epochs=4, gamma=.99, lam=.95, clip_eps=.2,
                        vf_coef=.5, ent_coef=.01, device=torch.device('cpu'),
                        max_episode_steps=160, target_kl=1e-6,
                        decision_only=True, max_grad_norm=.5)
    assert history[0]['target_kl_stopped']
    assert 0 < history[0]['optimizer_updates'] < 20
