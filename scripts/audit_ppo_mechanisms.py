"""Bounded real-trajectory audit; descriptive evidence, never a strength gate."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from mahjong.environment import MahjongEnv
from mahjong.evaluation import load_policy
from mahjong.features.action_space_v2 import action_from_id_v2
from mahjong.training.opponent_pool import build_opponent, collect_trajectory_with_opponents
from mahjong.training.ppo import compute_gae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths = {'bc': 'experiments/bc_v4/checkpoints_20w/best.pt',
             'ppo': 'experiments/ppo_v4_full_critic/seed42_100k/best.pt'}
    loaded = {k: load_policy(v, device) for k, v in paths.items()}
    opponents = {s: build_opponent('bc-v4', bc_checkpoint=paths['bc'], checkpoint_map={'bc-v4': paths['bc']},
                                 device=device, mode='greedy') for s in range(4)}
    groups = {}
    for aid in range(325):
        groups.setdefault(action_from_id_v2(aid).type.value, []).append(aid)
    result = {'protocol': 'seeds910000+, alternating BC/PPO sampling learner vs greedy BC-v4; not strength evidence',
              'games': [], 'groups': {}}
    all_x, all_mask, all_a, all_adv, sources = [], [], [], [], []
    max_error = 0.0
    for g in range(args.games):
        key = 'bc' if g % 2 == 0 else 'ppo'
        encoder, model = loaded[key]
        torch.manual_seed(910000 + g)
        env = MahjongEnv(seed=910000 + g)
        seat = (g // 2) % 4
        traj, steps, illegal = collect_trajectory_with_opponents(
            env, model, encoder, device, learner_seat=seat,
            opponents={s: opponents[s] for s in range(4) if s != seat},
            max_steps=20000, seed=910000 + g)
        assert env.done() and not illegal
        for s, buf in traj.items():
            expected = (env.state.scores[s] - 25000) * .001 + env.reward_config.placement_bonus[env.state.final_ranks[s]]
            max_error = max(max_error, abs(sum(t.reward for t in buf.steps) - expected))
        ts = traj[seat].steps
        rewards, values, dones = ([getattr(t, field) for t in ts] for field in ('reward', 'value', 'done'))
        adv, ret = compute_gae(rewards, values, dones, .99, .95)
        _, mc = compute_gae(rewards, values, dones, 1., 1.)
        def ev(target):
            y, v = torch.tensor(target), torch.tensor(values)
            return float(1 - (y - v).var() / y.var()) if y.var() > 0 else None
        result['games'].append({'learner': key, 'seed': 910000+g, 'seat': seat,
                                'steps': steps, 'learner_steps': len(ts),
                                'rank': env.state.final_ranks[seat],
                                'gae_explained_variance': ev(ret), 'mc_explained_variance': ev(mc)})
        all_x.extend(t.features for t in ts)
        all_mask.extend(t.legal_mask for t in ts)
        all_a.extend(t.action_id for t in ts)
        all_adv.extend(adv)
        sources.extend([key] * len(ts))
    x, mask = torch.stack(all_x), torch.stack(all_mask)
    aids, advantages = torch.tensor(all_a), torch.tensor(all_adv)
    probs = {}
    for key, (_, model) in loaded.items():
        with torch.no_grad():
            probs[key] = torch.cat([model.action_probs(x[i:i+256].to(device), mask[i:i+256].to(device)).cpu()
                                    for i in range(0, len(x), 256)])
    for name, ids in groups.items():
        opportunity = mask[:, ids].any(-1)
        selected = torch.isin(aids, torch.tensor(ids))
        if not opportunity.any():
            continue
        result['groups'][name] = {
            'opportunities': int(opportunity.sum()), 'selected': int(selected.sum()),
            'raw_advantage_mean_selected': float(advantages[selected].mean()) if selected.any() else None,
            **{k+'_probability_on_opportunity': float(p[opportunity][:, ids].sum(-1).mean()) for k,p in probs.items()},
            **{k+'_greedy_fraction_on_opportunity': float(torch.isin(p[opportunity].argmax(-1), torch.tensor(ids)).float().mean()) for k,p in probs.items()}}
    result['forced_action_fraction'] = float((mask.sum(-1) == 1).float().mean())
    # Local gradient attribution on BC-collected choice states only. This is
    # a directional derivative, not a simulated Adam step or causal strength test.
    ix = torch.tensor([i for i,s in enumerate(sources) if s == 'bc' and mask[i].sum() > 1][:512])
    bc = loaded['bc'][1]
    bc.train()  # CUDA GRU backward requires training forward; no dropout here.
    xx, mm, aa = x[ix].to(device), mask[ix].to(device), aids[ix].to(device)
    adv = advantages[ix].to(device)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    lp, _, entropy = bc.evaluate_actions(xx, mm, aa)
    policy_objective = -(torch.exp(lp-lp.detach()) * adv).mean()
    entropy_objective = -.01 * entropy.mean()
    calls = groups['chi'] + groups['pon']
    eligible = mm[:,calls].any(-1)
    call_mass = bc.action_probs(xx,mm)[eligible][:,calls].sum(-1).mean()
    params = [p for name,p in bc.named_parameters() if not name.startswith('value_')]
    def grads(loss):
        return torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    call_grad = grads(call_mass)
    result['local_gradient_diagnostic'] = {'samples': len(ix), 'call_opportunities': int(eligible.sum()),
                                          'meaning': 'negative gradient dot call-probability gradient; positive means increasing calls locally; not Adam effect'}
    for name, objective in [('policy',policy_objective),('entropy_coef_0.01',entropy_objective)]:
        gs = grads(objective)
        norm = sum(float(g.square().sum()) for g in gs if g is not None) ** .5
        direction = -sum(float(g.mul(c).sum()) for g,c in zip(gs,call_grad) if g is not None and c is not None)
        result['local_gradient_diagnostic'][name] = {'gradient_norm': norm, 'call_probability_direction': direction}
    result['max_reward_ledger_error'] = max_error
    result['samples'] = len(x)
    result['elapsed_seconds'] = time.perf_counter()-started
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
