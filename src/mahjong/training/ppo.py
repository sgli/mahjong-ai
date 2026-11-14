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

from ..decision.action import Action, ActionType
from ..environment import MahjongEnv
from ..features.encoder import ObservationEncoder
from ..features.policy_schema import policy_action_ops
from ..model.policy import MLPPolicy, remap_bc_state_dict
from .attribution import attribute_game
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
    # H1：value 独立 trunk 也从 BC trunk 初始化（warm start）
    if getattr(model, "value_separate_trunk", False):
        for k, v in list(filtered.items()):
            if k.startswith("trunk."):
                filtered[f"value_trunk.{k[len('trunk.'):]}"] = v
    if getattr(model, "value_separate_encoder", False):
        for k, v in list(filtered.items()):
            if k.startswith("tile_embedding.") or k.startswith("river_encoder."):
                filtered[f"value_{k}"] = v
    model.load_state_dict(filtered, strict=False)
    if reinit_value_head:
        for name, param in model.named_parameters():
            if name.startswith("value_head."):
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
    return model


def compute_gae(rewards, values, dones, gamma: float, lam: float, *, last_value: float = 0.0):
    """Return (advantages, returns), bootstrapping nonterminal rollout cuts."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = float(last_value)
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
    bc_model: MLPPolicy | None = None,
    bc_anchor_beta: float = 0.0,
    call_anchor_beta: float = 0.0,
    warmup: bool = False,
    decision_only: bool = False,
):
    """Clipped surrogate objective + value loss + entropy bonus (+ optional BC anchor).

    ``bc_anchor_beta > 0`` 时加入 ``beta * KL(π_θ || π_BC)``（前向 KL：以 θ 分布为权重，
    惩罚 θ 在 BC 低概率处的质量）；同时报告 ``kl_bc_to_theta`` 作为诊断（不参与 loss）。
    ``β=0`` 或 ``bc_model=None`` 时与现有 PPO **完全等价**（不额外 forward）。

    ``warmup=True``（E-C1 预热期）：**只做 value 回归**（policy/entropy/BC-anchor 项不参与），
    但仍报告 policy 侧诊断（不产生梯度）。
    """
    log_probs, values, entropy = model.evaluate_actions(features, masks, action_ids)
    ratio = torch.exp(log_probs - old_log_probs)
    choice = masks.sum(-1) > 1 if decision_only else torch.ones_like(action_ids, dtype=torch.bool)
    def actor_mean(t):
        return t[choice].mean() if choice.any() else t.sum() * 0.0
    policy_loss = -actor_mean(clipped_surrogate(log_probs, old_log_probs, advantages, clip_eps))
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -actor_mean(entropy)
    if warmup:
        loss = vf_coef * value_loss
    else:
        loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    metrics = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.mean().item()),
        "mean_ratio": float(ratio.mean().item()),
        "ratio_std": float(ratio.std().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
        "approx_kl": float(approx_kl(old_log_probs, log_probs).item()),
        "clip_fraction": float(clip_fraction(old_log_probs, log_probs, clip_eps).item()),
        "clip_positive_fraction": float(((ratio > 1.0 + clip_eps) & (advantages > 0)).float().mean().item()),
        "clip_negative_fraction": float(((ratio < 1.0 - clip_eps) & (advantages < 0)).float().mean().item()),
    }
    if (bc_anchor_beta > 0.0 or call_anchor_beta > 0.0) and bc_model is not None:
        theta_logp = model.log_probs(features, masks)  # [B, 270]
        with torch.no_grad():
            bc_logp = bc_model.log_probs(features, masks)
        theta_probs = torch.exp(theta_logp)
        safe_theta = torch.where(torch.isfinite(theta_logp), theta_logp, torch.zeros_like(theta_logp))
        safe_bc = torch.where(torch.isfinite(bc_logp), bc_logp, torch.zeros_like(bc_logp))
        kl_theta_bc = (theta_probs * (safe_theta - safe_bc)).sum(-1).mean()
        bc_probs = torch.exp(bc_logp)
        kl_bc_theta = (bc_probs * (safe_bc - safe_theta)).sum(-1).mean()
        if not warmup:
            loss = loss + bc_anchor_beta * actor_mean((theta_probs * (safe_theta - safe_bc)).sum(-1))
        metrics["kl_to_bc"] = float(kl_theta_bc.item())
        metrics["kl_bc_to_theta"] = float(kl_bc_theta.item())
        if call_anchor_beta > 0.0:
            from ..features.action_space_v2 import action_from_id_v2
            if model.action_size != 325:
                raise ValueError('call anchor requires action-v2 size 325')
            ids = [i for i in range(325) if action_from_id_v2(i).type.value in ('chi', 'pon')]
            opportunity = masks[:, ids].any(-1) & choice
            if opportunity.any():
                p = theta_probs[opportunity][:, ids].sum(-1).clamp(1e-6, 1 - 1e-6)
                q = bc_probs[opportunity][:, ids].sum(-1).clamp(1e-6, 1 - 1e-6)
                call_kl = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
                if not warmup:
                    loss = loss + call_anchor_beta * call_kl.mean()
                metrics['call_anchor_kl'] = float(call_kl.mean().item())
                metrics['call_mass_theta'] = float(p.mean().item())
                metrics['call_mass_bc'] = float(q.mean().item())
                metrics['call_anchor_opportunities'] = int(opportunity.sum().item())
            else:
                metrics['call_anchor_kl'] = 0.0
                metrics['call_anchor_opportunities'] = 0
    return loss, metrics


def set_value_warmup_trainable(model, freeze_trunk=True):
    """Train all separate critic modules while keeping the entire actor frozen.

    Shared modules retain the legacy explicit freeze_trunk behavior.
    """
    for name, p in model.named_parameters():
        p.requires_grad_(name.startswith('value_') if freeze_trunk
                         else not name.startswith('action_head.'))


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


def collect_trajectory(
    env: MahjongEnv, model: MLPPolicy, encoder: ObservationEncoder, device, *,
    max_steps: int, rng_seed: int = 0, env_index: int = 0, rng_version: str = "rng-v1",
):
    """Run one game (self-play), returning per-seat ``TrajectoryBuffer`` + step count (§11).

    ``rng_version="rng-v2"`` 时使用 per-env ``torch.Generator``（seed = rng_seed + 100000*env_index），
    与 ``collect_batch_trajectory`` 完全一致，使 sampling 逐位对齐（§8）。
    """
    to_id, make_mask, allow_abort = policy_action_ops(encoder, model)
    env.reset()
    traj: dict[int, TrajectoryBuffer] = {s: TrajectoryBuffer() for s in range(4)}
    pending = {s: 0.0 for s in range(4)}
    rng = torch.Generator(device=device).manual_seed(rng_seed + 100_000 * env_index) if rng_version == "rng-v2" else None
    illegal = 0
    steps = 0
    while not env.done() and steps < max_steps:
        seat = env.state.turn
        legal = [a for a in env.legal_actions(seat) if isinstance(a, Action) and (a.type is not ActionType.KYUUSHU_KYUHAI or allow_abort)]
        obs = env.observation(seat)
        features = encoder.encode(obs).to(device)
        mask = make_mask(legal).to(device)

        with torch.no_grad():
            out = model(features.unsqueeze(0), mask.unsqueeze(0))
            probs = torch.softmax(out.logits, dim=-1)
            if rng is not None:
                action_id = int(torch.multinomial(probs, 1, generator=rng).squeeze(-1).item())
            else:
                action_id = int(torch.multinomial(probs, 1).squeeze(-1).item())
            log_prob = float(torch.log(probs[0, action_id] + 1e-12).item())
            value = float(out.value[0].item())

        if not mask[action_id].item():
            illegal += 1  # should never happen (masked sampling)

        # map the sampled id back to the actual legal Action (restores the
        # context-dependent ``target`` seat for ron/pon/chi/daiminkan).
        action = next(a for a in legal if to_id(a) == action_id)
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


def _ingest_trajectory(traj, buffer, episode_rewards, returns_sum, gamma, lam, reward_clip: float = 0.0, bootstrap_value: float = 0.0) -> int:
    """把一条 game 的 per-seat TrajectoryBuffer 计入 epoch buffer；返回该 game 的 env steps。

    ``reward_clip > 0`` 时把 per-step reward winsorize 到 [−c, +c]（在 GAE 之前），
    只做尺度处理，不改 reward 定义。
    """
    n_steps = 0
    for s in range(4):
        buf = traj[s]
        if not buf.steps:
            continue
        n_steps += len(buf.steps)
        rewards = [st.reward for st in buf.steps]
        if reward_clip > 0.0:
            rewards = [max(-reward_clip, min(reward_clip, r)) for r in rewards]
        values = [st.value for st in buf.steps]
        dones = [st.done for st in buf.steps]
        adv, ret = compute_gae(rewards, values, dones, gamma, lam, last_value=bootstrap_value)
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
    return n_steps


def _merge_attribution(per_opp_accum, per_seat_accum, per_opp, per_seat):
    for k, v in per_opp.items():
        d = per_opp_accum.setdefault(k, {"episode_reward": 0.0, "return": 0.0, "win": 0, "rank_sum": 0, "episodes": 0})
        for f in ("episode_reward", "return", "win", "rank_sum", "episodes"):
            d[f] += v[f]
    for s, stats in per_seat.items():
        d = per_seat_accum.setdefault(s, {})
        for metric in ("reward", "value", "advantage", "return"):
            st = stats[metric]
            acc = d.setdefault(metric, {"sum": 0.0, "sum_sq": 0.0, "min": float("inf"), "max": float("-inf"), "n": 0})
            mean = st["mean"]
            acc["sum"] += mean * st["n"]
            acc["sum_sq"] += (st["std"] ** 2 + mean ** 2) * st["n"]
            acc["min"] = min(acc["min"], st["min"])
            acc["max"] = max(acc["max"], st["max"])
            acc["n"] += st["n"]


def opponent_pool_game_index(epoch: int, local_game_index: int) -> int:
    """Return a deterministic, non-overlapping index for each epoch/game pair."""
    if epoch < 0 or local_game_index < 0 or local_game_index >= (1 << 32):
        raise ValueError("epoch and local_game_index must be non-negative; local_game_index must fit uint32")
    return (epoch << 32) | local_game_index


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
    diagnose: bool = False,
    bc_model: MLPPolicy | None = None,
    bc_anchor_beta: float = 0.0,
    call_anchor_beta: float = 0.0,
    rollout_version: str = "rollout-v1",
    attribution: bool = False,
    opponent_pool=None,
    opponent_bc_checkpoint: str | None = None,
    opponent_checkpoint_map: dict | None = None,
    current_ppo: tuple | None = None,
    opponent_mode: str = "sampling",
    opponent_temperature: float = 1.0,
    opponent_pool_seed: int = 0,
    rng_version: str = "rng-v1",
    warmup_steps: int = 0,
    warmup_freeze_trunk: bool = True,
    reward_clip: float = 0.0,
    reward_score_scale: float | None = None,
    reward_placement_scale: list | None = None,
    step_budget_unit: str = "environment_steps",
    warmup_epochs: int = 0,
    decision_only: bool = False,
    max_grad_norm: float = 0.0,
    target_kl: float = 0.0,
) -> list[dict]:
    """Run PPO training and return per-epoch metrics history.

    ``start_epoch`` / ``global_step`` support resume: the loop runs
    ``range(start_epoch + 1, epochs + 1)`` and ``global_step`` is cumulative.
    """
    if step_budget_unit not in {"environment_steps", "learner_transitions"}:
        raise ValueError(f"unknown step_budget_unit {step_budget_unit!r}")
    if min(warmup_epochs, max_grad_norm, target_kl) < 0:
        raise ValueError('warmup_epochs/max_grad_norm/target_kl must be nonnegative')
    policy_action_ops(encoder, model)
    if call_anchor_beta < 0 or (call_anchor_beta > 0 and bc_model is None):
        raise ValueError('call anchor requires a nonnegative beta and a frozen BC model')
    if (bc_anchor_beta > 0 or call_anchor_beta > 0) and bc_model is not None:
        if bc_model.feature_dim != model.feature_dim or bc_model.action_size != model.action_size:
            raise ValueError("BC anchor schema must match the PPO policy schema")
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if step_budget_unit == "learner_transitions" and opponent_pool is None:
        raise ValueError("learner_transitions budget currently requires opponent_pool rollouts")
    envs = [MahjongEnv(seed=env_seed + i) for i in range(num_envs)]
    # Opponent policies are immutable during a training run.  Resolve and load
    # each configured opponent once; rebuilding PolicyOpponent inside every
    # game reloaded its checkpoint (and model) repeatedly, dominating rollout
    # time for large learner-transition budgets.
    opponent_instances = None
    if opponent_pool is not None:
        from .opponent_pool import build_opponent

        opponent_instances = {
            opponent_id: build_opponent(
                opponent_id,
                bc_checkpoint=opponent_bc_checkpoint,
                checkpoint_map=opponent_checkpoint_map,
                current_ppo=current_ppo,
                device=device,
                mode=opponent_mode,
                temperature=opponent_temperature,
            )
            for opponent_id in opponent_pool.ids()
        }
    if reward_score_scale is not None or reward_placement_scale is not None:
        for env in envs:
            if reward_score_scale is not None:
                env.reward_config.score_delta_scale = reward_score_scale
            if reward_placement_scale is not None:
                env.reward_config.placement_bonus = tuple(reward_placement_scale)
    history = []

    for epoch in range(start_epoch + 1, epochs + 1):
        buffer = {"features": [], "action_ids": [], "log_probs": [], "values": [], "rewards": [], "masks": [], "dones": []}
        total_steps = 0
        illegal_total = 0
        episode_rewards: list[float] = []
        returns_sum: list[float] = []
        per_opp_accum: dict[str, dict] = {}
        per_seat_accum: dict[int, dict] = {}
        attribution_count = 0
        complete_games = 0
        budget_truncated_games = 0

        # environment_steps preserves the legacy whole-game lower-bound target;
        # learner_transitions stops at an exact learner-buffer sample count.
        if opponent_pool is not None:
            from .opponent_pool import collect_trajectory_with_opponents, sample_seats

            game_index = 0
            while (
                len(buffer["features"]) < steps_per_epoch
                if step_budget_unit == "learner_transitions"
                else total_steps < steps_per_epoch
            ):
                # Epoch is part of the stateless seed namespace so pool training
                # never replays the same game/opponent seeds after resetting
                # the local episode counter.
                pool_game_index = opponent_pool_game_index(epoch, game_index)
                learner_seat, seat_types = sample_seats(opponent_pool, pool_game_index, learner_type="self")
                opponents = {
                    s: opponent_instances[seat_types[s]]
                    for s in range(4) if s != learner_seat
                }
                env = MahjongEnv(seed=env_seed + pool_game_index)
                if reward_score_scale is not None:
                    env.reward_config.score_delta_scale = reward_score_scale
                if reward_placement_scale is not None:
                    env.reward_config.placement_bonus = tuple(reward_placement_scale)
                collect_kwargs = {}
                if step_budget_unit == "learner_transitions":
                    collect_kwargs["max_learner_steps"] = steps_per_epoch - len(buffer["features"])
                collected = collect_trajectory_with_opponents(
                    env, model, encoder, device, learner_seat=learner_seat, opponents=opponents,
                    max_steps=max_episode_steps, greedy=False, seed=opponent_pool_seed + pool_game_index,
                    **collect_kwargs,
                )
                if step_budget_unit == "learner_transitions":
                    traj, n_steps, illegal, bootstrap_value, budget_truncated = collected
                    budget_truncated_games += int(budget_truncated)
                    complete_games += int(not budget_truncated)
                else:
                    traj, n_steps, illegal = collected
                    bootstrap_value = 0.0
                    complete_games += int(env.done())
                total_steps += n_steps
                illegal_total += illegal
                # 只 ingest learner 轨迹（对手轨迹不参与梯度）
                learner_only = {learner_seat: traj[learner_seat]}
                for s in range(4):
                    learner_only.setdefault(s, TrajectoryBuffer())
                _ingest_trajectory(
                    learner_only, buffer, episode_rewards, returns_sum, gamma, lam,
                    reward_clip, bootstrap_value=bootstrap_value,
                )
                if attribution and env.done():
                    po, ps = attribute_game(traj, seat_types, env.state.final_ranks, gamma=gamma, lam=lam)
                    _merge_attribution(per_opp_accum, per_seat_accum, po, ps)
                    attribution_count += 1
                game_index += 1
        else:
            use_optimized = rollout_version in ("rollout-v2", "optimized")
            while total_steps < steps_per_epoch:
                if use_optimized:
                    from .parallel_rollout import collect_batch_trajectory

                    trajs, dones, batch_steps, illegal = collect_batch_trajectory(
                        envs, model, encoder, device, max_steps=max_episode_steps, greedy=False, base_seed=env_seed
                    )
                    illegal_total += illegal
                    for i, traj in enumerate(trajs):
                        total_steps += _ingest_trajectory(traj, buffer, episode_rewards, returns_sum, gamma, lam, reward_clip)
                        if attribution and envs[i].done():
                            po, ps = attribute_game(traj, ["self"] * 4, envs[i].state.final_ranks, gamma=gamma, lam=lam)
                            _merge_attribution(per_opp_accum, per_seat_accum, po, ps)
                            attribution_count += 1
                else:
                    for env_index, env in enumerate(envs):
                        traj, n_steps, illegal = collect_trajectory(
                            env, model, encoder, device, max_steps=max_episode_steps,
                            rng_seed=env_seed, env_index=env_index, rng_version=rng_version,
                        )
                        total_steps += n_steps
                        illegal_total += illegal
                        _ingest_trajectory(traj, buffer, episode_rewards, returns_sum, gamma, lam, reward_clip)
                        if attribution and env.done():
                            po, ps = attribute_game(traj, ["self"] * 4, env.state.final_ranks, gamma=gamma, lam=lam)
                            _merge_attribution(per_opp_accum, per_seat_accum, po, ps)
                            attribution_count += 1
                        if total_steps >= steps_per_epoch:
                            break

        # §15：rollout-wide advantage normalization（GAE 完成后对整 buffer 归一化一次）
        adv_tensor = torch.tensor(buffer["advantages"], dtype=torch.float32)
        choice = torch.stack(buffer['masks']).sum(-1) > 1
        norm_adv = adv_tensor[choice] if decision_only and choice.any() else adv_tensor
        advantage_mean = float(norm_adv.mean().item())
        advantage_std = float(norm_adv.std(unbiased=not decision_only and len(norm_adv) > 1).item())
        values_before = torch.tensor(buffer['values'])
        targets = torch.tensor(buffer['returns'])
        target_var = targets.var(unbiased=False)
        value_ev_before = float(1 - (targets-values_before).var(unbiased=False)/target_var) if target_var > 0 else None
        buffer["advantages"] = ((adv_tensor - advantage_mean) / (advantage_std + 1e-8)).tolist()
        episode_count = len(episode_rewards)
        episode_length = total_steps / max(episode_count, 1)

        # E-C1 预热：global_step < warmup_steps 时只训 value（冻结 actor）
        in_warmup = epoch <= warmup_epochs or (warmup_steps > 0 and global_step < warmup_steps)
        warmup_ev_sum = 0.0
        warmup_ev_n = 0
        if in_warmup:
            set_value_warmup_trainable(model, warmup_freeze_trunk)

        # PPO updates
        n = len(buffer["features"])
        metrics_accum = {}
        updates = 0
        update_epoch_stats: list[dict] = []
        kl_stopped = False
        for _ue in range(update_epochs):
            order = torch.randperm(n)
            ue_accum: dict[str, float] = {}
            ue_updates = 0
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

                if target_kl > 0 and not in_warmup:
                    with torch.no_grad():
                        new_logp, _, _ = model.evaluate_actions(features, masks, action_ids)
                        delta = new_logp - old_logp
                        eligible = masks.sum(-1) > 1 if decision_only else torch.ones_like(action_ids, dtype=torch.bool)
                        k3 = ((delta.exp() - 1) - delta)[eligible]
                        if k3.numel() and float(k3.mean()) > target_kl:
                            kl_stopped = True
                            break

                loss, mets = ppo_loss(
                    model, features, masks, action_ids, old_logp, advantages, returns,
                    clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef,
                    bc_model=bc_model, bc_anchor_beta=bc_anchor_beta, call_anchor_beta=call_anchor_beta,
                    warmup=in_warmup,
                    decision_only=decision_only,
                )
                optimizer.zero_grad()
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite PPO loss')
                loss.backward()
                if max_grad_norm > 0:
                    # Separate clipping prevents large critic gradients from
                    # shrinking an already isolated actor's update.
                    for is_value in (False, True):
                        params = [p for name, p in model.named_parameters()
                                  if name.startswith('value_') == is_value and p.grad is not None]
                        if params:
                            torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                if in_warmup:
                    from ..evaluation import value_quality_metrics

                    with torch.no_grad():
                        _, vals, _ = model.evaluate_actions(features, masks, action_ids)
                    vq = value_quality_metrics(vals, returns)
                    warmup_ev_sum += vq["explained_variance"] * len(idx)
                    warmup_ev_n += len(idx)
                for k, v in mets.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v
                if diagnose:
                    for k, v in mets.items():
                        ue_accum[k] = ue_accum.get(k, 0.0) + v
                    ue_updates += 1
                updates += 1
            if diagnose and ue_updates:
                update_epoch_stats.append({f"update_epoch_{_ue + 1}_{k}": v / ue_updates for k, v in ue_accum.items()})
            if kl_stopped:
                break

        if in_warmup:
            # 恢复所有参数可训练（预热结束或下一 epoch 继续预热前由上层再冻结）
            for p in model.parameters():
                p.requires_grad_(True)
        if updates == 0:
            raise RuntimeError('PPO epoch performed no optimizer updates; check rollout log probabilities and KL limit')

        global_step += total_steps
        epoch_metrics = {k: v / max(updates, 1) for k, v in metrics_accum.items()}
        epoch_metrics.update({'in_warmup': in_warmup, 'optimizer_updates': updates,
                              'target_kl_stopped': kl_stopped, 'decision_only': decision_only,
                              'choice_fraction': float(choice.float().mean()),
                              'value_explained_variance_before': value_ev_before,
                              'warmup_epochs': warmup_epochs})
        if in_warmup:
            epoch_metrics["warmup_ev"] = warmup_ev_sum / max(warmup_ev_n, 1)
            epoch_metrics["warmup_steps"] = warmup_steps
            epoch_metrics["warmup_freeze_trunk"] = warmup_freeze_trunk
            epoch_metrics["warmup_actual_steps"] = global_step
            # 预热刚结束（本 epoch 跨过 warmup_steps）时记录 final EV
            if epoch >= warmup_epochs and global_step >= warmup_steps:
                epoch_metrics["warmup_final_ev"] = epoch_metrics["warmup_ev"]
        if diagnose:
            for ue in update_epoch_stats:
                epoch_metrics.update(ue)
            # §4 分布统计（reward/value/advantage/return，按 epoch）
            for name in ("rewards", "values", "advantages", "returns"):
                t = torch.tensor(buffer[name], dtype=torch.float32)
                epoch_metrics[f"{name}_mean"] = float(t.mean().item())
                epoch_metrics[f"{name}_std"] = float(t.std().item())
                epoch_metrics[f"{name}_min"] = float(t.min().item())
                epoch_metrics[f"{name}_max"] = float(t.max().item())
        epoch_metrics.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "mean_reward": sum(episode_rewards) / max(len(episode_rewards), 1),
                "mean_return": sum(returns_sum) / max(len(returns_sum), 1),
                "steps": total_steps,
                "target_steps": steps_per_epoch,
                "step_budget_unit": step_budget_unit,
                "environment_steps": total_steps,
                "learner_transitions": len(buffer["features"]),
                "rollout_buffer_samples": len(buffer["features"]),
                "complete_games": complete_games,
                "budget_truncated_games": budget_truncated_games,
                "rollout_version": rollout_version,
                "rng_version": rng_version,
                "reward_clip": reward_clip,
                "reward_score_scale": reward_score_scale,
                "reward_placement_scale": reward_placement_scale,
                "illegal_rate": illegal_total / max(total_steps, 1),
                "advantage_mean": advantage_mean,
                "advantage_std": advantage_std,
                "episode_count": episode_count,
                "episode_length": episode_length,
            }
        )
        if attribution and attribution_count:
            epoch_metrics["per_opponent"] = {
                k: {
                    "episode_reward": v["episode_reward"],
                    "return": v["return"],
                    "win": v["win"],
                    "rank_sum": v["rank_sum"],
                    "episodes": v["episodes"],
                }
                for k, v in per_opp_accum.items()
            }
            epoch_metrics["per_seat"] = {
                s: {
                    m: {
                        "mean": (a["sum"] / a["n"]) if a["n"] else 0.0,
                        "std": ((a["sum_sq"] / a["n"]) - (a["sum"] / a["n"]) ** 2) ** 0.5 if a["n"] else 0.0,
                        "min": a["min"],
                        "max": a["max"],
                        "n": a["n"],
                    }
                    for m, a in stats.items()
                }
                for s, stats in per_seat_accum.items()
            }
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
    "opponent_pool_game_index",
    "ppo_loss",
    "train_ppo",
]
