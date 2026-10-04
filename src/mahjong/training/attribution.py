"""Per-opponent / per-seat attribution（Phase 10.3 §4/§11）。"""

from __future__ import annotations


def _dist_stats(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean": mean, "std": var ** 0.5, "min": min(values), "max": max(values), "n": n}


def attribute_game(traj, seat_opponent_types, final_ranks, *, gamma: float, lam: float) -> tuple[dict, dict]:
    """Attribution for one game.

    ``traj``: dict seat -> TrajectoryBuffer（同一 game）。
    ``seat_opponent_types``: list of 4 opponent-type strings（seat 顺序）。
    ``final_ranks``: list of 4 ranks（或 None，缺省按 0 处理 win/rank）。

    Returns ``(per_opponent, per_seat)``：
    - per_opponent[type] = {episode_reward, return, win, rank_sum, episodes}
    - per_seat[seat] = {reward:{mean/std/min/max/n}, value:{...}, advantage:{...}, return:{...}}
    """
    from .ppo import compute_gae  # 延迟导入，避免循环依赖

    per_opponent: dict[str, dict] = {}
    per_seat: dict[int, dict] = {}
    for s in range(4):
        buf = traj[s]
        if not buf.steps:
            continue
        rewards = [st.reward for st in buf.steps]
        values = [st.value for st in buf.steps]
        dones = [st.done for st in buf.steps]
        adv, ret = compute_gae(rewards, values, dones, gamma, lam)

        otype = seat_opponent_types[s] if s < len(seat_opponent_types) else "unknown"
        o = per_opponent.setdefault(otype, {"episode_reward": 0.0, "return": 0.0, "win": 0, "rank_sum": 0, "episodes": 0})
        o["episode_reward"] += sum(rewards)
        o["return"] += sum(ret)
        o["episodes"] += 1
        if final_ranks is not None and s < len(final_ranks):
            o["rank_sum"] += final_ranks[s]
            if final_ranks[s] == 0:
                o["win"] += 1

        per_seat[s] = {
            "reward": _dist_stats(rewards),
            "value": _dist_stats(values),
            "advantage": _dist_stats(adv),
            "return": _dist_stats(ret),
        }
    return per_opponent, per_seat


__all__ = ["attribute_game"]
