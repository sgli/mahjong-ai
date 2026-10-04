"""PPO training (Phase 7).

Self-play trajectory collection (all four seats use the current policy), GAE
advantage estimation, and clipped PPO updates with a value head.  CPU small
scale: run one or a few MahjongEnv workers in-process (no multiprocessing).

Reward per seat is its own score delta (``env.step`` returns per-seat deltas at
kyoku settlement) plus the placement bonus at game end (added to each seat's
last transition).  This is the default scheme; it is configurable via the env's
``RewardConfig``.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from ..decision.action import Action
from ..environment import MahjongEnv
from ..features.action_space import action_to_id, legal_mask
from ..features.encoder import ObservationEncoder
from ..model.policy import MLPPolicy, remap_bc_state_dict
from .ppo_buffer import EpisodeMetadata, TrajectoryBuffer, TrajectoryStep

log = logging.getLogger("training.ppo")


def init_ppo_from_bc(model: MLPPolicy, bc_state_dict: dict, *, reinit_value_head: bool = True) -> MLPPolicy:
    """Initialise a PPO model from a BC checkpoint (§4).

    BC trunk + BC action head are kept exactly (policy unchanged); the value
    head is re-initialised (BC has no trained value head).  Returns ``model``.
    """
    sd = remap_bc_state_dict(dict(bc_state_dict))
    # 保留 trunk + action head，剔除 value head（重新初始化）
    filtered = {k: v for k, v in sd.items() if not k.startswith("value_head.")}
    model.load_state_dict(filtered, strict=False)
    if reinit_value_head:
        for name, param in model.named_parameters():
            if name.startswith("value_head."):
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
    return model


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Return (advantages, returns) for one contiguous episode."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def clipped_surrogate(log_probs, old_log_probs, advantages, clip_eps: float) -> torch.Tensor:
    """PPO clipped surrogate (per-sample); ``min(r*A, clip(r)*A)``."""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return torch.min(surr1, surr2)


def approx_kl(old_log_probs, log_probs) -> torch.Tensor:
    """Fixed KL approximation: ``mean(old_log_prob - new_log_prob)`` (§14)."""
    return (old_log_probs - log_probs).mean()


def clip_fraction(old_log_probs, log_probs, clip_eps: float) -> torch.Tensor:
    """Fixed clip fraction: ``fraction(|ratio - 1| > clip_eps)`` (§14)."""
    ratio = torch.exp(log_probs - old_log_probs)
    return ((ratio - 1.0).abs() > clip_eps).float().mean()


def ppo_loss(
    model: MLPPolicy,
    features,
    masks,
    action_ids,
    old_log_probs,
    advantages,
    returns,
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    """Clipped surrogate objective + value loss + entropy bonus.

    Also reports PPO diagnostics (§14): ``approx_kl = mean(old - new)`` and
    ``clip_fraction = fraction(|ratio - 1| > clip_eps)``.
    """
    log_probs, values, entropy = model.evaluate_actions(features, masks, action_ids)
    ratio = torch.exp(log_probs - old_log_probs)
    policy_loss = -clipped_surrogate(log_probs, old_log_probs, advantages, clip_eps).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropy.mean()
    loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    metrics = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
        "mean_ratio": float(ratio.mean().item()),
        "approx_kl": float(approx_kl(old_log_probs, log_probs).item()),
        "clip_fraction": float(clip_fraction(old_log_probs, log_probs, clip_eps).item()),
    }
    return loss, metrics


def attribute_step_rewards(rewards: dict, acting_seat: int, traj: dict, pending: dict) -> float:
    """Distribute per-seat settlement deltas across each seat's own trajectory.

    ``env.step`` returns one delta per seat (score change).  The acting seat's
    delta belongs to the transition being recorded right now; every other
    seat's delta belongs to *its* most recent transition (e.g. a discarder's
    ron payment), so it is added retroactively.  If a seat has no transition
    yet, its delta is held in ``pending`` until it acts.
    """
    reward = rewards.get(acting_seat, 0.0) + pending.pop(acting_seat, 0.0)
    for s, r in rewards.items():
        if s != acting_seat and r != 0.0:
            buf = traj[s]
            if buf.steps:
                buf.steps[-1].reward += r
            else:
                pending[s] = pending.get(s, 0.0) + r
    return reward


def flush_terminal_rewards(env: MahjongEnv, traj: dict, pending: dict) -> None:
    """At game end, flush residual pending rewards + placement bonuses to each seat's last transition."""
    for s in range(4):
        buf = traj[s]
        if pending.get(s, 0.0):
            if buf.steps:
                buf.steps[-1].reward += pending[s]
            pending[s] = 0.0
        if buf.steps:
            rank = env.state.final_ranks[s]
            buf.steps[-1].reward += env.reward_config.placement_bonus[rank]


def collect_trajectory(env: MahjongEnv, model: MLPPolicy, encoder: ObservationEncoder, device, *, max_steps: int):
    """Run one game (self-play), returning per-seat ``TrajectoryBuffer`` + step count (§11)."""
    env.reset()
    traj: dict[int, TrajectoryBuffer] = {s: TrajectoryBuffer() for s in range(4)}
    pending = {s: 0.0 for s in range(4)}
    illegal = 0
    steps = 0
    while not env.done() and steps < max_steps:
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action)]
        obs = env.observation(seat)
        features = encoder.encode(obs).to(device)
        mask = legal_mask(legal).to(device)

        with torch.no_grad():
            out = model(features.unsqueeze(0), mask.unsqueeze(0))
            probs = torch.softmax(out.logits, dim=-1)
            action_id = int(torch.multinomial(probs, 1).squeeze(-1).item())
            log_prob = float(torch.log(probs[0, action_id] + 1e-12).item())
            value = float(out.value[0].item())

        if not mask[action_id].item():
            illegal += 1  # should never happen (masked sampling)

        # map the sampled id back to the actual legal Action (restores the
        # context-dependent ``target`` seat for ron/pon/chi/daiminkan).
        action = next(a for a in legal if action_to_id(a) == action_id)
        rewards, done = env.step(action)
        reward = attribute_step_rewards(rewards, seat, traj, pending)

        traj[seat].append(
            TrajectoryStep(
                features=features.cpu(),
                action_id=action_id,
                legal_mask=mask.cpu(),
                old_log_prob=log_prob,
                value=value,
                reward=reward,
                done=bool(done),
            )
        )
        steps += 1

    if env.done():
        flush_terminal_rewards(env, traj, pending)
    return traj, steps, illegal


def _stack(items, device):
    return torch.stack([torch.as_tensor(x) for x in items]).to(device)


def train_ppo(
    model: MLPPolicy,
    encoder: ObservationEncoder,
    optimizer,
    *,
    num_envs: int,
    env_seed: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    update_epochs: int,
    gamma: float,
    lam: float,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    device,
    max_episode_steps: int = 5000,
    log_interval_epochs: int = 1,
    start_epoch: int = 0,
    global_step: int = 0,
    on_epoch_end=None,
) -> list[dict]:
    """Run PPO training and return per-epoch metrics history.

    ``start_epoch`` / ``global_step`` support resume: the loop runs
    ``range(start_epoch + 1, epochs + 1)`` and ``global_step`` is cumulative.
    """
    envs = [MahjongEnv(seed=env_seed + i) for i in range(num_envs)]
    history = []

    for epoch in range(start_epoch + 1, epochs + 1):
        buffer = {"features": [], "action_ids": [], "log_probs": [], "values": [], "rewards": [], "masks": [], "dones": []}
        total_steps = 0
        illegal_total = 0
        episode_rewards: list[float] = []
        returns_sum: list[float] = []

        # §17：steps_per_epoch = 每 epoch 的 env step 上限。跑完当前 episode 再停，
        # 因此实际 steps 会略超出（∈ [N, N + max_episode_steps)），并记录 target_steps。
        while total_steps < steps_per_epoch:
            for env in envs:
                traj, n_steps, illegal = collect_trajectory(env, model, encoder, device, max_steps=max_episode_steps)
                total_steps += n_steps
                illegal_total += illegal
                for s in range(4):
                    buf = traj[s]
                    if not buf.steps:
                        continue
                    rewards = [st.reward for st in buf.steps]
                    values = [st.value for st in buf.steps]
                    dones = [st.done for st in buf.steps]
                    adv, ret = compute_gae(rewards, values, dones, gamma, lam)
                    buffer["features"].extend(st.features for st in buf.steps)
                    buffer["action_ids"].extend(st.action_id for st in buf.steps)
                    buffer["log_probs"].extend(st.old_log_prob for st in buf.steps)
                    buffer["values"].extend(values)
                    buffer["rewards"].extend(rewards)
                    buffer["masks"].extend(st.legal_mask for st in buf.steps)
                    buffer["dones"].extend(dones)
                    buffer.setdefault("advantages", []).extend(adv)
                    buffer.setdefault("returns", []).extend(ret)
                    episode_rewards.append(sum(rewards))
                    returns_sum.append(sum(ret))
                if total_steps >= steps_per_epoch:
                    break

        # §15：rollout-wide advantage normalization（GAE 完成后对整 buffer 归一化一次）
        adv_tensor = torch.tensor(buffer["advantages"], dtype=torch.float32)
        advantage_mean = float(adv_tensor.mean().item())
        advantage_std = float(adv_tensor.std().item())
        buffer["advantages"] = ((adv_tensor - advantage_mean) / (advantage_std + 1e-8)).tolist()
        episode_count = len(episode_rewards)
        episode_length = total_steps / max(episode_count, 1)

        # PPO updates
        n = len(buffer["features"])
        order = torch.randperm(n)
        metrics_accum = {}
        updates = 0
        for _ in range(update_epochs):
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if not len(idx):
                    continue
                features = torch.stack([buffer["features"][i] for i in idx]).to(device)
                masks = torch.stack([buffer["masks"][i] for i in idx]).to(device)
                action_ids = torch.tensor([buffer["action_ids"][i] for i in idx], dtype=torch.long, device=device)
                old_logp = torch.tensor([buffer["log_probs"][i] for i in idx], dtype=torch.float32, device=device)
                advantages = torch.tensor([buffer["advantages"][i] for i in idx], dtype=torch.float32, device=device)
                returns = torch.tensor([buffer["returns"][i] for i in idx], dtype=torch.float32, device=device)

                loss, mets = ppo_loss(
                    model, features, masks, action_ids, old_logp, advantages, returns,
                    clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                for k, v in mets.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                updates += 1

        global_step += total_steps
        epoch_metrics = {k: v / max(updates, 1) for k, v in metrics_accum.items()}
        epoch_metrics.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "mean_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
                "mean_return": sum(returns_sum) / max(len(returns_sum), 1),
                "steps": total_steps,
                "target_steps": steps_per_epoch,
                "illegal_rate": illegal_total / max(total_steps, 1),
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "episode_count": episode_count,
                "episode_length": episode_length,
            }
        )
        history.append(epoch_metrics)
        if on_epoch_end is not None:
            on_epoch_end(epoch, epoch_metrics)
        if epoch % log_interval_epochs == 0:
            log.info(
                "epoch %d policy_loss=%.4f value_loss=%.4f entropy=%.4f mean_reward=%.4f illegal_rate=%.4f",
                epoch,
                epoch_metrics["policy_loss"],
                epoch_metrics["value_loss"],
                epoch_metrics["entropy"],
                epoch_metrics["mean_reward"],
                epoch_metrics["illegal_rate"],
            )
    return history


__all__ = [
    "approx_kl",
    "attribute_step_rewards",
    "clip_fraction",
    "clipped_surrogate",
    "collect_trajectory",
    "compute_gae",
    "flush_terminal_rewards",
    "init_ppo_from_bc",
    "ppo_loss",
    "train_ppo",
]
