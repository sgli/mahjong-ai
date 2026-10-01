#!/usr/bin/env python
"""Rule validation: replay real tenhou/majsoul games and reproduce scoring.

For every ``hora`` and ``ryukyoku`` event, reconstruct the winning context
(hand / melds / win tile / dora / ura / honba / kyotaku / situational flags)
and re-score with the corresponding platform preset (TENHOU_RULES for tenhou,
MAJSOUL_RULES for majsoul), then compare against the recorded ``deltas``.

This script is read-only: it never changes the rule core.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mahjong.parser import MjaiParser  # noqa: E402
from mahjong.parser.events import (  # noqa: E402
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Hora,
    Kakan,
    Pon,
    Reach,
    Ryukyoku,
    StartKyoku,
    Tsumo,
)
from mahjong.replay import ReplayEngine  # noqa: E402
from mahjong.rules import (  # noqa: E402
    MAJSOUL_RULES,
    TENHOU_RULES,
    WinContext,
    ron_payment,
    score_hand,
    shanten,
    tsumo_payments,
)
from mahjong.rules.decompose import iter_structures  # noqa: E402
from mahjong.rules.scoring import _is_terminal_or_honor  # noqa: E402
from mahjong.rules.tiles import HONORS, normalize, rank_of  # noqa: E402
from mahjong.rules.yaku import _is_pinfu, _is_yakuhai, _wait_types  # noqa: E402

SEAT_WINDS = ("E", "S", "W", "N")
_DEAD_WALL = 14
_DEAL = 52  # mjai start_kyoku tehais 均为 13 张，庄家经首摸补到 14


def _ceil100(x: int) -> int:
    return ((x + 99) // 100) * 100


def classify_files(data_root: str | Path) -> dict[str, list[Path]]:
    root = Path(data_root)
    out: dict[str, list[Path]] = {"tenhou": [], "majsoul": []}
    if not root.is_dir():
        return out
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith("tenhou-"):
            out["tenhou"].extend(sorted(d.glob("*.mjai.json")))
        elif d.is_dir() and d.name.startswith("majsoul-"):
            out["majsoul"].extend(sorted(d.glob("*.mjai.json")))
    return out


def sample_files(files: list[Path], limit: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    return rng.sample(files, min(limit, len(files)))


@dataclass
class HoraRecord:
    file: str
    kyoku: str
    winner: int
    target: int
    is_tsumo: bool
    hand: list[str]
    melds: list
    win_tile: str
    yaku: list
    yakuman: list
    han: int
    fu: int
    dora: int = 0
    dora_markers: tuple = ()
    ura_markers: tuple = ()
    honba: int = 0
    kyotaku: int = 0
    riichi: bool = False
    ippatsu: bool = False
    menzen: bool = True
    double_riichi: bool = False
    haitei: bool = False
    houtei: bool = False
    fu_debug: list = field(default_factory=list)
    discards: list = field(default_factory=list)
    meld_kinds: list = field(default_factory=list)
    expected: list[int] = field(default_factory=list)
    actual: list[int] = field(default_factory=list)
    match: bool = True
    category: str = "match"


@dataclass
class RyukyokuRecord:
    file: str
    kyoku: str
    expected: list[int]
    actual: list[int]
    match: bool
    nagashi: list[int]
    hands: list = field(default_factory=list)
    tenpai: list = field(default_factory=list)


@dataclass
class FileReport:
    file: str
    platform: str
    replay_inconsistencies: int
    horas: list[HoraRecord] = field(default_factory=list)
    ryukyoku: list[RyukyokuRecord] = field(default_factory=list)


def _terminal_or_honor(tile: str) -> bool:
    return tile in HONORS or rank_of(tile) in (1, 9)


def fu_breakdown(hand, melds, ctx: WinContext) -> list[str]:
    """逐分量打印每个 (结构, 听牌形) 的符计算（用于 --debug-fu）。"""
    out: list[str] = []
    for st in iter_structures(hand, melds):
        for wt in _wait_types(st, ctx.win_tile):
            pinfu = _is_pinfu(st, ctx, wt)
            fu = 20
            parts = ["base20"]
            if ctx.is_menzen and not ctx.is_tsumo:
                fu += 10
                parts.append("menzen_ron+10")
            if ctx.is_tsumo and not pinfu:
                fu += 2
                parts.append("tsumo+2")
            for g in st.sets:
                if g.kind == "run":
                    continue
                terminal = _is_terminal_or_honor(g.tiles[0])
                ron_completed = not ctx.is_tsumo and normalize(ctx.win_tile) in g.tiles
                if g.kind == "triplet":
                    v = (4 if terminal else 2) if (g.open or ron_completed) else (8 if terminal else 4)
                else:
                    v = (16 if terminal else 8) if g.open else (32 if terminal else 16)
                fu += v
                parts.append(f"{g.tiles[0]}x3:{v}")
            if _is_yakuhai(st.pair, ctx):
                fu += 2
                parts.append("pair+2")
                if st.pair == ctx.seat_wind == ctx.round_wind:
                    fu += 2
                    parts.append("double_wind+2")
            if not pinfu and wt in ("kanchan", "penchan", "tanki"):
                fu += 2
                parts.append(f"wait({wt})+2")
            rounded = 20 if (pinfu and ctx.is_tsumo) else (30 if pinfu else max(((fu + 9) // 10) * 10, 30))
            out.append(f"    [wt={wt} pinfu={pinfu}] {parts} -> fu={fu} rounded={rounded}")
    return out


def expected_hora_deltas(base: int, *, is_tsumo: bool, is_dealer: bool, winner: int, target: int, oya: int, honba: int, kyotaku: int) -> list[int]:
    deltas = [0, 0, 0, 0]
    if is_tsumo:
        if is_dealer:
            each = _ceil100(base * 2) + 100 * honba
            for s in range(4):
                if s != winner:
                    deltas[s] -= each
                    deltas[winner] += each
        else:
            dealer_pay = _ceil100(base * 2) + 100 * honba
            other_pay = _ceil100(base) + 100 * honba
            deltas[oya] -= dealer_pay
            deltas[winner] += dealer_pay
            for s in range(4):
                if s != winner and s != oya:
                    deltas[s] -= other_pay
                    deltas[winner] += other_pay
        deltas[winner] += kyotaku * 1000
    else:
        pay = ron_payment(base, is_dealer) + 300 * honba
        deltas[target] -= pay
        deltas[winner] += pay
        deltas[winner] += kyotaku * 1000
    return deltas


def reconstruct_hora(events, states, i: int, live: int, last_tsumo_tile: dict, last_draw_rinchan: dict, chankan_active, double_riichi: dict):
    """Return (win_tile, hand, melds, ctx) for the hora at index i."""
    ev = events[i]
    state = states[i]
    winner = ev.actor
    target = ev.target
    is_tsumo = target == winner
    ps = state.players[winner]

    is_chankan = (not is_tsumo) and chankan_active is not None and target == chankan_active[0]
    if is_tsumo:
        win_tile = last_tsumo_tile.get(winner, "")
    elif is_chankan:
        win_tile = chankan_active[1]
    else:
        win_tile = state.players[target].discards[-1] if state.players[target].discards else ""

    hand = list(ps.hand) if is_tsumo else list(ps.hand) + [win_tile]
    melds = list(ps.melds)
    is_menzen = not any(m.kind in ("chi", "pon", "daiminkan", "kakan") for m in melds)
    ctx = WinContext(
        round_wind=state.bakaze,
        seat_wind=SEAT_WINDS[(winner - state.oya) % 4],
        win_tile=win_tile,
        is_tsumo=is_tsumo,
        is_menzen=is_menzen,
        is_riichi=ps.riichi,
        is_double_riichi=double_riichi.get(winner, False),
        is_ippatsu=ps.ippatsu,
        is_haitei=is_tsumo and live == 0,
        is_houtei=(not is_tsumo) and live == 0,
        is_rinshan=is_tsumo and last_draw_rinchan.get(winner, False),
        is_chankan=is_chankan,
    )
    return win_tile, hand, melds, ctx


def reconstruct_ryukyoku(events, states, i: int, rules, discard_called: list[bool]):
    """Return (expected_deltas, nagashi_seats) for the ryukyoku at index i."""
    ev = events[i]
    state = states[i]
    deltas = [0, 0, 0, 0]

    # 流局满贯：全幺九舍牌 且 从未被 吃/碰/大明杠（discard_called=False）才成立。
    nagashi = [
        s for s in range(4)
        if rules.nagashi_mangan
        and not discard_called[s]
        and len(state.players[s].discards) >= 17  # 荒牌流局才可能（排除流局中止的短河）
        and all(_terminal_or_honor(t) for t in state.players[s].discards)
    ]

    if nagashi:
        # 数据实证：流局满贯为定额结算（子家 +8000 / 庄家 +12000），不加本场、不取供托。
        for seat in nagashi:
            if seat == state.oya:
                pay = 4000  # 庄家流局满贯：三家各付 4000
                for s in range(4):
                    if s != seat:
                        deltas[s] -= pay
                        deltas[seat] += pay
            else:
                dealer_pay = 4000
                other_pay = 2000
                deltas[state.oya] -= dealer_pay
                deltas[seat] += dealer_pay
                for s in range(4):
                    if s != seat and s != state.oya:
                        deltas[s] -= other_pay
                        deltas[seat] += other_pay
        # 流局满贯视为「和牌」，其余三家不再做听牌/不听点棒转移
        return deltas, nagashi

    others = [0, 1, 2, 3]
    tenpai = {
        s: bool(state.players[s].hand) and shanten(state.players[s].hand, state.players[s].melds) <= 0
        for s in others
    }
    n_tenpai = sum(tenpai.values())
    n_noten = len(others) - n_tenpai
    if 1 <= n_tenpai <= len(others) - 1 and n_noten >= 1:
        total = 3000  # 听牌点棒转移不叠加本场（数据实证）
        for s in others:
            if tenpai[s]:
                deltas[s] += total // n_tenpai
            else:
                deltas[s] -= total // n_noten
    return deltas, nagashi


def validate_file(path: Path, platform: str, rules) -> FileReport:
    parser = MjaiParser()
    events = list(parser.parse_file(path))
    engine = ReplayEngine(game_id=path.stem)
    states = list(engine.replay(events))
    report = FileReport(file=str(path), platform=platform, replay_inconsistencies=len(engine.inconsistencies))

    live = 0
    last_tsumo_tile: dict[int, str] = {}
    last_draw_rinchan: dict[int, bool] = {}
    pending_rinchan = None
    chankan_active = None
    discard_called = [False] * 4

    # 数据实证：多家和(daburon) 时，本场棒与立直供托只付给放铳者下家方向最近的和了者
    # （其余和了者不重复得本场/供托）。预计算每家和的「最近和了者」。
    kyoku_starts = [j for j, e in enumerate(events) if isinstance(e, StartKyoku)]
    closest_ron_by_start = {}
    for k, start in enumerate(kyoku_starts):
        end = kyoku_starts[k + 1] if k + 1 < len(kyoku_starts) else len(events)
        rons = [(e.actor, e.target) for e in events[start:end] if isinstance(e, Hora) and e.target != e.actor]
        if len(rons) >= 2:
            target = rons[0][1]
            winners = [a for a, _ in rons]
            closest = min(winners, key=lambda a: (a - target - 1) % 4)
            closest_ron_by_start[start] = closest
    current_kyoku_start = -1

    for i, ev in enumerate(events):
        if isinstance(ev, StartKyoku):
            live = 136 - _DEAD_WALL - _DEAL
            pending_rinchan = None
            chankan_active = None
            last_tsumo_tile = {}
            last_draw_rinchan = {}
            discard_called = [False] * 4
            double_riichi = {s: False for s in range(4)}
            has_discarded = [False] * 4
            current_kyoku_start = i
        elif isinstance(ev, Tsumo):
            is_rinchan = pending_rinchan == ev.actor
            if not is_rinchan:
                live -= 1  # 杠补摸(rinchan)不递减 live 墙
            last_draw_rinchan[ev.actor] = is_rinchan
            last_tsumo_tile[ev.actor] = ev.pai
            pending_rinchan = None
            chankan_active = None
        elif isinstance(ev, Dahai):
            chankan_active = None
            has_discarded[ev.actor] = True
        elif isinstance(ev, (Ankan, Daiminkan, Kakan)):
            pending_rinchan = ev.actor
            if isinstance(ev, Kakan):
                chankan_active = (ev.actor, ev.pai)
            if isinstance(ev, Daiminkan):
                discard_called[ev.target] = True  # 大明杠也叫牌
        elif isinstance(ev, (Chi, Pon)):
            discard_called[ev.target] = True
        elif isinstance(ev, Reach):
            if not has_discarded[ev.actor]:
                double_riichi[ev.actor] = True
        elif isinstance(ev, Hora):
            win_tile, hand, melds, ctx = reconstruct_hora(
                events, states, i, live, last_tsumo_tile, last_draw_rinchan, chankan_active, double_riichi
            )
            result = score_hand(
                hand, melds, win_tile, ctx,
                dora_indicators=states[i].dora_markers,
                ura_indicators=ev.ura_markers,
                kiriage_mangan=rules.kiriage_mangan,
            )
            prev_kyotaku = states[i - 1].kyotaku if i > 0 else states[i].kyotaku
            honba_eff = states[i].honba
            kyotaku_eff = prev_kyotaku
            if ev.target != ev.actor and current_kyoku_start in closest_ron_by_start:
                if ev.actor != closest_ron_by_start[current_kyoku_start]:
                    honba_eff = 0
                    kyotaku_eff = 0
            expected = expected_hora_deltas(
                result.base_points,
                is_tsumo=ev.target == ev.actor,
                is_dealer=ev.actor == states[i].oya,
                winner=ev.actor,
                target=ev.target,
                oya=states[i].oya,
                honba=honba_eff,
                kyotaku=kyotaku_eff,
            )
            match = list(expected) == list(ev.deltas)
            category = "match" if match else "unknown"
            report.horas.append(
                HoraRecord(
                    file=path.name, kyoku=states[i].round_id, winner=ev.actor, target=ev.target,
                    is_tsumo=ev.target == ev.actor, hand=hand, melds=melds, win_tile=win_tile,
                    yaku=result.yaku, yakuman=result.yakuman, han=result.han, fu=result.fu,
                    dora=result.dora, dora_markers=states[i].dora_markers, ura_markers=ev.ura_markers,
                    honba=states[i].honba, kyotaku=prev_kyotaku,
                    riichi=states[i].players[ev.actor].riichi,
                    ippatsu=states[i].players[ev.actor].ippatsu,
                    menzen=ctx.is_menzen,
                    double_riichi=double_riichi.get(ev.actor, False),
                    haitei=ctx.is_haitei,
                    houtei=ctx.is_houtei,
                    fu_debug=fu_breakdown(hand, melds, ctx),
                    discards=list(states[i].players[ev.actor].discards),
                    meld_kinds=[(m.kind, list(m.tiles)) for m in melds],
                    expected=expected, actual=list(ev.deltas), match=match, category=category,
                )
            )
        elif isinstance(ev, Ryukyoku):
            expected, nagashi = reconstruct_ryukyoku(events, states, i, rules, discard_called)
            match = list(expected) == list(ev.deltas)
            tenpai_flags = [shanten(states[i].players[s].hand, states[i].players[s].melds) <= 0 for s in range(4)]
            report.ryukyoku.append(
                RyukyokuRecord(
                    file=path.name, kyoku=states[i].round_id, expected=expected, actual=list(ev.deltas),
                    match=match, nagashi=nagashi,
                    hands=[list(states[i].players[s].hand) for s in range(4)],
                    tenpai=tenpai_flags,
                )
            )
    return report


def summarize(reports: list[FileReport]) -> dict:
    total_horas = sum(len(r.horas) for r in reports)
    hora_matches = sum(sum(1 for h in r.horas if h.match) for r in reports)
    ryukyoku_total = sum(len(r.ryukyoku) for r in reports)
    ryukyoku_matches = sum(sum(1 for y in r.ryukyoku if y.match) for r in reports)
    inconsistencies = sum(r.replay_inconsistencies for r in reports)
    return {
        "files": len(reports),
        "kyoku": total_horas + ryukyoku_total,
        "horas": total_horas,
        "hora_matches": hora_matches,
        "ryukyoku": ryukyoku_total,
        "ryukyoku_matches": ryukyoku_matches,
        "replay_inconsistencies": inconsistencies,
        "hora_match_rate": hora_matches / max(total_horas, 1),
        "ryukyoku_match_rate": ryukyoku_matches / max(ryukyoku_total, 1),
    }


def run_platform(data_root: str, platform: str, limit: int, seed: int) -> tuple[list[FileReport], dict]:
    files = classify_files(data_root).get(platform, [])
    chosen = sample_files(files, limit, seed)
    rules = TENHOU_RULES if platform == "tenhou" else MAJSOUL_RULES
    reports = [validate_file(p, platform, rules) for p in chosen]
    return reports, summarize(reports)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate rule scoring against real replays.")
    ap.add_argument("--data-root", default=r"F:\Mahjong AI\mahjong DB")
    ap.add_argument("--platform", default="auto", choices=["auto", "tenhou", "majsoul"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-mismatches", action="store_true", help="print hand/melds/context for each mismatch")
    ap.add_argument("--debug-fu", action="store_true", help="print per-component fu breakdown for mismatches")
    args = ap.parse_args(argv)

    platforms = ["tenhou", "majsoul"] if args.platform == "auto" else [args.platform]
    for platform in platforms:
        reports, summary = run_platform(args.data_root, platform, args.limit, args.seed)
        print(f"== {platform} ==")
        print(summary)
        ryk_mismatches = [y for r in reports for y in r.ryukyoku if not y.match]
        for y in ryk_mismatches[:20]:
            print(
                f"  RYUKYOKU MISMATCH {y.file} {y.kyoku} expected={y.expected} actual={y.actual} "
                f"tenpai={y.tenpai} nagashi={y.nagashi}"
            )
            if args.show_mismatches:
                for s in range(4):
                    print(f"    seat{s} ({len(y.hands[s])} tiles): {sorted(y.hands[s])}")
        mismatches = [h for r in reports for h in r.horas if not h.match]
        for h in mismatches[:50]:
            print(
                f"  MISMATCH {h.file} {h.kyoku} seat{h.winner} tsumo={h.is_tsumo} "
                f"yaku={h.yaku} yakuman={h.yakuman} han={h.han} fu={h.fu} dora={h.dora} "
                f"dora_markers={h.dora_markers} ura={h.ura_markers} "
                f"honba={h.honba} kyotaku={h.kyotaku} riichi={h.riichi} ippatsu={h.ippatsu} "
                f"menzen={h.menzen} double_riichi={h.double_riichi} haitei={h.haitei} houtei={h.houtei} "
                f"win={h.win_tile} expected={h.expected} actual={h.actual}"
            )
            if args.show_mismatches:
                print(f"    hand={sorted(h.hand)} melds={h.meld_kinds} discards={h.discards}")
            if args.debug_fu:
                for line in h.fu_debug:
                    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
