"""Mahjong Environment (deterministic, full hanchan).

Reuses the unified rule core (``rules/``) for hand shape, calls, yaku, fu,
scoring and dora, so Environment / Replay / Decision agree on the same rules
(ENVIRONMENT_SPEC section 7).

Design / assumptions (Tenhou):

- 136 tiles; ``aka_flag`` swaps one ``5m/5p/5s`` for the red fives.
- Dead wall (14): 5 dora indicators, 5 ura indicators, 4 kan replacements.
- Riichi requires closed hand + tenpai-after-discard + score >= 1000 + >=4
  wall tiles; the -1000 stick and ``kyotaku += 1`` happen immediately.
- Ron requires a yaku (``score_hand().has_yaku``) and is blocked by discard
  furiten.  Temporary furiten (passing a ron) is not modelled.
- Multi-ron uses **atamahane** (the closest seat to the discarder wins); double
  ron is not resolved (but every eligible player still sees RON as legal).
- Chankan (robbing a kakan) is implemented as a ron-only response window.
- Exhaustive draw (荒牌流局) settles tenpai/noten (3000 + 300*honba, split).
- Kyuushu kyuuhai (九種九牌) is exposed as the special string action
  ``"kyuushu_kyuuhai"`` on a non-dealer's first draw with >= 9 distinct
  terminals/honours.  Four-wind / four-kan / four-riichi aborts are implemented
  (``RulesConfig`` flags + ``_check_*_abort``; see tests/test_environment_aborts.py).
- West round (西入): after South 4, if no player has >= 30000 points, a West
  round is played.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..decision.action import Action, ActionType
from ..decision.observation import PlayerObservation
from ..replay.state import Meld
from ..rules import (
    WinContext,
    is_agari,
    is_furiten,
    is_tenpai,
    score_hand,
    tenpai_tiles,
)
from ..rules.config import TENHOU_RULES, RulesConfig
from ..rules.legal import ankan_actions, can_riichi, kakan_actions, validate_ankan_consumed
from ..rules.tiles import HONORS, rank_of
from .state import (
    PHASE_CHANKAN,
    PHASE_DISCARD,
    PHASE_ENDED,
    PHASE_RESPONSE,
    EnvState,
    Player,
    SeatStats,
)
from .wall import Wall

SEAT_WINDS = ("E", "S", "W", "N")
KYUUSHU_ACTION = "kyuushu_kyuuhai"

#: RL environment version (Phase 10.2 §9). Distinct from ``RulesConfig.rules_id``
#: (tenhou-v1) — this string identifies the *Environment* wiring used for RL,
#: and is written into configs / checkpoints / experiments / benchmarks.
ENVIRONMENT_VERSION = "tenhou-v2-rl"


@dataclass
class RewardConfig:
    score_delta_scale: float = 0.001  # points -> thousands
    placement_bonus: tuple[float, float, float, float] = (2.0, 1.0, -1.0, -2.0)  # by rank


def _remove_one(hand: list[str], tile: str) -> None:
    hand.remove(tile)


def _without(hand: list[str], tile: str | None) -> list[str]:
    """Return a copy of ``hand`` with one occurrence of ``tile`` removed."""
    out = list(hand)
    if tile is not None and tile in out:
        out.remove(tile)
    return out


class MahjongEnv:
    def __init__(
        self,
        seed: int = 0,
        aka_flag: bool = True,
        reward_config: RewardConfig | None = None,
        rules: RulesConfig = TENHOU_RULES,
    ) -> None:
        self.seed = seed
        self.rules = rules
        self.aka_flag = aka_flag and rules.red_fives > 0
        self.reward_config = reward_config or RewardConfig()
        self.state: EnvState | None = None
        self._kyoku_counter = 0
        self._accumulated: dict[int, float] = {}

    # -- lifecycle --------------------------------------------------------------
    def reset(self, seed: int | None = None) -> EnvState:
        if seed is not None:
            self.seed = seed
        self._kyoku_counter = 0
        self._accumulated = {s: 0.0 for s in range(4)}
        state = EnvState()
        state.scores = [25000, 25000, 25000, 25000]
        state.players = [Player(i) for i in range(4)]
        state.stats = [SeatStats() for _ in range(4)]
        state.oya = 0
        state.kyotaku = 0
        self.state = state
        self._start_kyoku()
        return state

    def _start_kyoku(self) -> None:
        state = self.state
        wall = Wall(self.seed * 1_000_003 + self._kyoku_counter * 7919, self.aka_flag)
        self._kyoku_counter += 1
        state.wall = wall
        state.dora_indicators = list(wall.dora_indicators)
        state.ura_indicators = wall.ura_indicators
        # note: kyotaku carries over from a previous ryukyoku (not reset here)
        state.tenpai_flags = [False] * 4
        state.first_draw_done = [False] * 4
        state.game_over = False
        state.final_ranks = None
        state.pending_responders = []
        state.response_choices = {}
        state.discarder = None
        state.discard_tile = None

        for seat in range(4):
            p = state.players[seat]
            n = 14 if seat == state.oya else 13
            p.hand = sorted(wall.draw() for _ in range(n))
            p.melds = []
            p.discards = []
            p.riichi = False
            p.ippatsu = False
            p.discard_called = False
            p.last_drawn = None
            p.drawn_this_turn = seat == state.oya

        state.turn = state.oya
        state.phase = PHASE_DISCARD

    # -- interface --------------------------------------------------------------
    def legal_actions(self, seat: int | None = None) -> list:
        state = self.state
        if seat is None:
            seat = state.turn
        if state.phase == PHASE_DISCARD and seat == state.turn:
            return self._discard_legal(seat)
        if state.phase in (PHASE_RESPONSE, PHASE_CHANKAN) and seat == state.turn:
            return self._response_legal(seat)
        return []

    def observation(self, seat: int) -> PlayerObservation:
        s = self.state
        ps = s.players
        return PlayerObservation(
            seat=seat,
            bakaze=s.round_wind,
            kyoku=s.kyoku,
            honba=s.honba,
            kyotaku=s.kyotaku,
            oya=s.oya,
            scores=tuple(s.scores),
            riichi=tuple(p.riichi for p in ps),
            hand=tuple(sorted(ps[seat].hand)),
            melds=tuple(ps[seat].melds),
            opponents_melds=tuple(tuple(ps[i].melds) if i != seat else () for i in range(4)),
            discards=tuple(tuple(p.discards) for p in ps),
            dora_markers=tuple(s.dora_indicators),
            turn=s.turn,
        )

    def done(self) -> bool:
        return bool(self.state and self.state.game_over)

    def reward(self, seat: int) -> float:
        return self._accumulated.get(seat, 0.0)

    def step(self, action) -> tuple[dict[int, float], bool]:
        state = self.state
        seat = state.turn
        legal = self.legal_actions(seat)
        if action not in legal:
            raise ValueError(f"illegal action {action!r} for seat {seat} (phase {state.phase})")

        if state.phase == PHASE_DISCARD:
            rewards = self._step_discard(seat, action)
        elif state.phase == PHASE_RESPONSE:
            rewards = self._step_response(seat, action)
        else:
            rewards = self._step_response(seat, action, chankan=True)

        for s, r in rewards.items():
            self._accumulated[s] = self._accumulated.get(s, 0.0) + r
        return rewards, self.done()

    # -- legal actions ----------------------------------------------------------
    def _discard_legal(self, seat: int) -> list:
        state = self.state
        p = state.players[seat]
        hand = p.hand
        distinct = sorted(set(hand))

        actions: list = []
        if p.riichi:
            if p.drawn_this_turn and p.last_drawn is not None:
                actions.append(Action(ActionType.DISCARD, tile=p.last_drawn))
        else:
            actions.extend(Action(ActionType.DISCARD, tile=t) for t in distinct)
            if p.drawn_this_turn and can_riichi(hand, p.melds, p.score) and state.wall.remaining >= 4:
                for t in distinct:
                    h = list(hand)
                    _remove_one(h, t)
                    if is_tenpai(h, p.melds):
                        actions.append(Action(ActionType.RIICHI, tile=t))

        if p.drawn_this_turn:
            if self._can_tsumo(seat):
                actions.append(Action(ActionType.TSUMO))
            ankans = ankan_actions(hand)
            kakans = kakan_actions(hand, p.melds)
            if p.riichi:
                # 立直后暗杠/加杠仅当不改变听牌张集合才合法
                ankans = [a for a in ankans if self._riichi_kan_ok(seat, a)]
                kakans = [a for a in kakans if self._riichi_kan_ok(seat, a)]
            actions.extend(ankans)
            actions.extend(kakans)
            if not p.riichi and self._can_kyuushu(seat):
                actions.append(KYUUSHU_ACTION)
        return actions

    def _riichi_kan_ok(self, seat: int, action: Action) -> bool:
        """True iff an ankan/kakan leaves the riichi player's wait tiles unchanged."""
        state = self.state
        p = state.players[seat]
        hand = list(p.hand)
        old_wait = set(tenpai_tiles(_without(hand, p.last_drawn), p.melds))

        if action.kan_kind == "ankan":
            new_hand = list(hand)
            for t in action.consumed:
                _remove_one(new_hand, t)
            new_melds = p.melds + [Meld(kind="ankan", tiles=tuple(sorted(action.consumed)), from_=None, called=None)]
        else:  # kakan
            new_hand = list(hand)
            _remove_one(new_hand, action.tile)
            new_melds = []
            for m in p.melds:
                if m.kind == "pon" and sorted(m.tiles) == sorted(action.consumed):
                    new_melds.append(
                        Meld(kind="kakan", tiles=tuple(sorted(list(m.tiles) + [action.tile])), from_=m.from_, called=m.called)
                    )
                else:
                    new_melds.append(m)
        new_wait = set(tenpai_tiles(new_hand, new_melds))
        return old_wait == new_wait

    def _response_legal(self, seat: int) -> list:
        state = self.state
        p = state.players[seat]
        tile = state.discard_tile
        discarder = state.discarder

        if state.phase == PHASE_CHANKAN:
            actions = []
            if self._can_ron(seat, tile, discarder, is_chankan=True):
                actions.append(Action(ActionType.RON, target=discarder))
            actions.append(Action(ActionType.PASS))
            return actions

        actions = []
        if self._can_ron(seat, tile, discarder):
            actions.append(Action(ActionType.RON, target=discarder))
        if not p.riichi and len(p.melds) < 4:
            from ..rules.legal import chi_actions, daiminkan_actions, pon_actions

            if seat == (discarder + 1) % 4:
                actions.extend(chi_actions(p.hand, tile, discarder))
            actions.extend(pon_actions(p.hand, tile, discarder))
            actions.extend(daiminkan_actions(p.hand, tile, discarder))
        actions.append(Action(ActionType.PASS))
        return actions

    def _can_tsumo(self, seat: int) -> bool:
        p = self.state.players[seat]
        if not is_agari(p.hand, p.melds):
            return False
        return self._has_yaku(seat, is_tsumo=True, win_tile=p.last_drawn, is_haitei=self.state.wall.remaining == 0)

    def _can_ron(self, seat: int, tile: str, discarder: int, *, is_chankan: bool = False) -> bool:
        p = self.state.players[seat]
        if seat == discarder:
            return False
        if p.temporary_furiten:
            return False  # 临时振听：立直见逃后直到下次摸牌前不可荣和
        winning = list(p.hand) + [tile]
        if not is_agari(winning, p.melds):
            return False
        if is_furiten(p.hand, p.melds, p.discards):
            return False
        is_houtei = not is_chankan and self.state.wall.remaining == 0
        return self._has_yaku(seat, is_tsumo=False, win_tile=tile, is_chankan=is_chankan, is_houtei=is_houtei)

    def _seat_wind(self, seat: int) -> str:
        # Tenhou seat winds are relative to the dealer: dealer is always East.
        return SEAT_WINDS[(seat - self.state.oya) % 4]

    def _has_yaku(self, seat: int, *, is_tsumo: bool, win_tile: str | None, **flags) -> bool:
        state = self.state
        p = state.players[seat]
        hand = p.hand if is_tsumo else list(p.hand) + [win_tile]
        ctx = WinContext(
            round_wind=state.round_wind,
            seat_wind=self._seat_wind(seat),
            win_tile=win_tile or p.last_drawn,
            is_tsumo=is_tsumo,
            is_menzen=state.menzen(seat),
            is_riichi=p.riichi,
            is_ippatsu=p.ippatsu,
            **flags,
        )
        ura = state.ura_indicators if p.riichi else ()
        result = score_hand(hand, p.melds, ctx.win_tile, ctx, dora_indicators=state.dora_indicators, ura_indicators=ura)
        return result.has_yaku

    def _can_kyuushu(self, seat: int) -> bool:
        if not self.rules.nine_terminals_abort:
            return False
        state = self.state
        p = state.players[seat]
        if seat == state.oya or state.first_draw_done[seat]:
            return False
        distinct_yaochuu = {t for t in p.hand if (rank_of(t) in (1, 9) or t in HONORS)}
        return len(distinct_yaochuu) >= 9

    # -- step: discard phase ----------------------------------------------------
    def _step_discard(self, seat: int, action) -> dict:
        state = self.state
        if action == KYUUSHU_ACTION:
            return self._resolve_ryukyoku(abort=True)

        p = state.players[seat]
        if action.type is ActionType.DISCARD:
            self._discard_tile(seat, action.tile)
            return {}
        if action.type is ActionType.RIICHI:
            p.riichi = True
            p.score -= 1000
            state.scores[seat] -= 1000  # 立直棒立即结算：同步 state.scores（修复 t81 待决策 bug）
            state.kyotaku += 1
            state.stats[seat].riichi_count += 1
            self._discard_tile(seat, action.tile, close_ippatsu=False)
            p.ippatsu = True
            return {seat: -self.reward_config.score_delta_scale * 1000}
        if action.type is ActionType.TSUMO:
            return self._resolve_tsumo(seat)
        if action.type is ActionType.KAN:
            return self._apply_kan(seat, action)
        raise ValueError(f"unhandled discard action {action!r}")

    def _discard_tile(self, seat: int, tile: str, *, close_ippatsu: bool = True) -> None:
        state = self.state
        p = state.players[seat]
        if tile not in p.hand:
            raise ValueError(f"tile {tile!r} not in hand of seat {seat}")
        _remove_one(p.hand, tile)
        p.discards.append(tile)
        p.last_drawn = None
        p.drawn_this_turn = False
        if close_ippatsu and p.riichi:
            p.ippatsu = False  # the riichi player's next discard closes ippatsu

        state.discarder = seat
        state.discard_tile = tile
        state.pending_responders = [(seat + d) % 4 for d in (1, 2, 3)]
        state.response_choices = {}
        state.turn = state.pending_responders[0]
        state.phase = PHASE_RESPONSE

    def _apply_kan(self, seat: int, action: Action) -> dict:
        state = self.state
        p = state.players[seat]
        kind = action.kan_kind
        state.stats[seat].call_count += 1
        if kind == "ankan":
            if not validate_ankan_consumed(action.consumed):
                raise ValueError(f"invalid ankan consumed: {action.consumed!r}")
            for t in action.consumed:
                _remove_one(p.hand, t)
            p.melds.append(Meld(kind="ankan", tiles=tuple(sorted(action.consumed)), from_=None, called=None))
            self._cancel_ippatsu_others(seat)
            if self._check_four_kan_abort():
                return self._resolve_ryukyoku(abort=True)
            self._kan_draw_for(seat)
        elif kind == "kakan":
            _remove_one(p.hand, action.tile)
            for i, m in enumerate(p.melds):
                if m.kind == "pon" and sorted(m.tiles) == sorted(action.consumed):
                    p.melds[i] = Meld(kind="kakan", tiles=tuple(sorted(list(m.tiles) + [action.tile])), from_=m.from_, called=m.called)
                    break
            else:
                raise ValueError("kakan: no matching pon meld")
            self._cancel_ippatsu_others(seat)
            state.discarder = seat
            state.discard_tile = action.tile
            state.pending_responders = [(seat + d) % 4 for d in (1, 2, 3)]
            state.response_choices = {}
            state.turn = state.pending_responders[0]
            state.phase = PHASE_CHANKAN
        else:
            raise ValueError(f"unexpected kan kind {kind!r} in discard phase")
        return {}

    # -- step: response phase ---------------------------------------------------
    def _step_response(self, seat: int, action: Action, *, chankan: bool = False) -> dict:
        state = self.state
        state.response_choices[seat] = action
        p = state.players[seat]
        # 临时振听：立直者见逃（可荣却选择不荣）后进入临时振听
        if p.riichi and getattr(action, "type", None) is not ActionType.RON:
            if self._can_ron(seat, state.discard_tile, state.discarder, is_chankan=chankan):
                p.temporary_furiten = True
        state.pending_responders.remove(seat)
        if state.pending_responders:
            state.turn = state.pending_responders[0]
            return {}
        return self._resolve_responses(chankan=chankan)

    def _resolve_responses(self, *, chankan: bool = False) -> dict:
        state = self.state
        discarder = state.discarder
        choices = state.response_choices

        ron_seats = [s for s, a in choices.items() if getattr(a, "type", None) is ActionType.RON]
        if ron_seats:
            if self.rules.atamahane:
                winner = min(ron_seats, key=lambda s: (s - discarder) % 4)  # 头跳：最近座位胜
                return self._resolve_ron(winner, discarder, is_chankan=chankan)
            return self._resolve_multi_ron(ron_seats, discarder, is_chankan=chankan)

        call_seats = [s for s, a in choices.items() if getattr(a, "type", None) in (ActionType.PON, ActionType.KAN, ActionType.CHI)]
        if call_seats:
            caller = min(call_seats, key=lambda s: (s - discarder) % 4)
            return self._apply_call(caller, choices[caller], discarder)

        # everyone passed
        state.discarder = None
        state.discard_tile = None
        state.phase = PHASE_DISCARD
        if chankan:
            # four-kan abort is checked after the kakan's chankan window closes
            if self._check_four_kan_abort():
                return self._resolve_ryukyoku(abort=True)
            self._kan_draw_for(discarder)
        else:
            if self._check_four_riichi_abort() or self._check_four_wind_abort():
                return self._resolve_ryukyoku(abort=True)
            result = self._draw_for((discarder + 1) % 4)
            if result is not None:
                return result
        return {}

    def _apply_call(self, seat: int, action: Action, discarder: int) -> dict:
        state = self.state
        p = state.players[seat]
        tile = state.discard_tile
        state.players[discarder].discards.pop()
        state.players[discarder].discard_called = True  # 舍牌被叫（流局满贯资格）
        self._cancel_ippatsu_others(seat)
        state.stats[seat].call_count += 1

        if action.type is ActionType.CHI:
            for t in action.consumed:
                _remove_one(p.hand, t)
            p.melds.append(Meld(kind="chi", tiles=tuple(sorted(action.consumed + (tile,))), from_=discarder, called=tile))
        elif action.type is ActionType.PON:
            for t in action.consumed:
                _remove_one(p.hand, t)
            p.melds.append(Meld(kind="pon", tiles=tuple(sorted(action.consumed + (tile,))), from_=discarder, called=tile))
        elif action.type is ActionType.KAN:  # daiminkan
            for t in action.consumed:
                _remove_one(p.hand, t)
            p.melds.append(Meld(kind="daiminkan", tiles=tuple(sorted(action.consumed + (tile,))), from_=discarder, called=tile))

        p.drawn_this_turn = False
        p.last_drawn = None
        state.turn = seat
        state.phase = PHASE_DISCARD
        state.discarder = None
        state.discard_tile = None

        if action.type is ActionType.KAN:
            if self._check_four_kan_abort():
                return self._resolve_ryukyoku(abort=True)
            self._kan_draw_for(seat)
        return {}

    # -- abortive draws ---------------------------------------------------------
    def _check_four_wind_abort(self) -> bool:
        """四风连打：第一巡（无鸣牌）四家首弃为同一风牌."""
        if not self.rules.four_wind_abort:
            return False
        state = self.state
        if any(len(p.melds) != 0 or len(p.discards) != 1 for p in state.players):
            return False
        first = [p.discards[0] for p in state.players]
        return all(t in ("E", "S", "W", "N") for t in first) and len(set(first)) == 1

    def _check_four_riichi_abort(self) -> bool:
        """四家立直：四家均立直后，第四家立直弃牌无人荣和即流局."""
        return self.rules.four_riichi_abort and all(p.riichi for p in self.state.players)

    def _check_four_kan_abort(self) -> bool:
        """四杠散了：4 个杠且由 >= 2 家杠出才流局（同一家 4 杠不中止）."""
        if not self.rules.four_kan_abort:
            return False
        state = self.state
        total = 0
        owners: set[int] = set()
        for seat, p in enumerate(state.players):
            n = sum(1 for m in p.melds if m.kind in ("ankan", "daiminkan", "kakan"))
            if n:
                owners.add(seat)
                total += n
        return total >= 4 and len(owners) >= 2

    def _is_all_last(self) -> bool:
        s = self.state
        return s.round_wind == "S" and s.kyoku == 4

    def _is_top(self, seat: int) -> bool:
        s = self.state
        return all(s.scores[seat] >= s.scores[i] for i in range(4))

    # -- drawing ----------------------------------------------------------------
    def _draw_for(self, seat: int) -> dict | None:
        state = self.state
        p = state.players[seat]
        state.first_draw_done[seat] = True
        if state.wall.remaining == 0:
            # 荒牌流局：返回结算 rewards（修复：之前被调用方丢弃，导致 reward 缺失）
            return self._resolve_ryukyoku(abort=False)
        tile = state.wall.draw()
        p.hand.append(tile)
        p.hand.sort()
        p.last_drawn = tile
        p.drawn_this_turn = True
        p.temporary_furiten = False  # 临时振听在下一次摸牌后解除
        state.turn = seat
        state.phase = PHASE_DISCARD
        return None

    def _kan_draw_for(self, seat: int) -> None:
        state = self.state
        p = state.players[seat]
        tile = state.wall.kan_draw()
        p.hand.append(tile)
        p.hand.sort()
        p.last_drawn = tile
        p.drawn_this_turn = True
        p.temporary_furiten = False  # 临时振听在下一次摸牌后解除
        state.turn = seat
        state.phase = PHASE_DISCARD

    def _cancel_ippatsu_others(self, actor: int) -> None:
        for s in range(4):
            if s != actor and self.state.players[s].riichi:
                self.state.players[s].ippatsu = False

    # -- settlement -------------------------------------------------------------
    def _win_context(self, seat: int, is_tsumo: bool, win_tile: str, **flags) -> WinContext:
        state = self.state
        p = state.players[seat]
        return WinContext(
            round_wind=state.round_wind,
            seat_wind=self._seat_wind(seat),
            win_tile=win_tile,
            is_tsumo=is_tsumo,
            is_menzen=state.menzen(seat),
            is_riichi=p.riichi,
            is_ippatsu=p.ippatsu,
            **flags,
        )

    def _score_win(self, seat: int, *, is_tsumo: bool, win_tile: str, **flags):
        state = self.state
        p = state.players[seat]
        hand = p.hand if is_tsumo else list(p.hand) + [win_tile]
        ctx = self._win_context(seat, is_tsumo, win_tile, **flags)
        ura = state.ura_indicators if p.riichi else ()
        return score_hand(
            hand, p.melds, win_tile, ctx,
            dora_indicators=state.dora_indicators,
            ura_indicators=ura,
            kiriage_mangan=self.rules.kiriage_mangan,
            double_yakuman=self.rules.double_yakuman,
        )

    def _resolve_tsumo(self, seat: int) -> dict:
        state = self.state
        p = state.players[seat]
        win_tile = p.last_drawn
        result = self._score_win(seat, is_tsumo=True, win_tile=win_tile, is_haitei=state.wall.remaining == 0)
        base = result.base_points
        is_dealer = seat == state.oya
        honba_pay = 100 * state.honba

        deltas = [0, 0, 0, 0]
        if is_dealer:
            each = _ceil100(base * 2) + honba_pay
            for s in range(4):
                if s != seat:
                    deltas[s] -= each
                    deltas[seat] += each
        else:
            dealer_pay = _ceil100(base * 2) + honba_pay
            other_pay = _ceil100(base) + honba_pay
            deltas[state.oya] -= dealer_pay
            deltas[seat] += dealer_pay
            for s in range(4):
                if s != seat and s != state.oya:
                    deltas[s] -= other_pay
                    deltas[seat] += other_pay
        deltas[seat] += state.kyotaku * 1000
        state.kyotaku = 0

        stats = state.stats[seat]
        stats.win_count += 1
        stats.tsumo_count += 1
        stats.win_points += deltas[seat]

        top_before = self._is_top(seat)
        self._apply_deltas(deltas)
        if self.rules.agari_yame and self._is_all_last() and top_before:
            self._end_game()  # 和了止め：all last 且 top 和牌即终局
        else:
            self._end_kyoku(renchan=seat == state.oya)
        return {s: self.reward_config.score_delta_scale * d for s, d in enumerate(deltas)}

    def _resolve_ron(self, seat: int, discarder: int, *, is_chankan: bool = False) -> dict:
        state = self.state
        win_tile = state.discard_tile
        is_houtei = not is_chankan and state.wall.remaining == 0
        result = self._score_win(seat, is_tsumo=False, win_tile=win_tile, is_chankan=is_chankan, is_houtei=is_houtei)
        base = result.base_points
        is_dealer = seat == state.oya
        pay = _ceil100(base * (6 if is_dealer else 4)) + 300 * state.honba

        deltas = [0, 0, 0, 0]
        deltas[discarder] -= pay
        deltas[seat] += pay
        deltas[seat] += state.kyotaku * 1000
        state.kyotaku = 0

        stats = state.stats[seat]
        stats.win_count += 1
        stats.ron_count += 1
        stats.win_points += deltas[seat]
        state.stats[discarder].dealt_in_count += 1

        top_before = self._is_top(seat)
        self._apply_deltas(deltas)
        if self.rules.agari_yame and self._is_all_last() and top_before:
            self._end_game()  # 和了止め：all last 且 top 和牌即终局
        else:
            self._end_kyoku(renchan=seat == state.oya)
        return {s: self.reward_config.score_delta_scale * d for s, d in enumerate(deltas)}

    def _resolve_multi_ron(self, winners: list[int], discarder: int, *, is_chankan: bool = False) -> dict:
        """多家和牌（雀魂/无头跳）：每个可荣者各自结算，放铳者依次全额支付。

        假设（标准 WRC/无头跳规则）：
        - 每家胜者得到其完整荣和点数（本场 300×honba 计入每家）；
        - 立直供托（riichi sticks）归「距放铳者最近」的胜者一家（头跳仅对供托生效）；
        - 若庄家在和牌者中则连庄。
        """
        state = self.state
        win_tile = state.discard_tile
        ordered = sorted(winners, key=lambda s: (s - discarder) % 4)
        deltas = [0, 0, 0, 0]
        top_flags = [self._is_top(s) for s in ordered]
        kyotaku = state.kyotaku * 1000
        for i, seat in enumerate(ordered):
            is_houtei = not is_chankan and state.wall.remaining == 0
            result = self._score_win(seat, is_tsumo=False, win_tile=win_tile, is_chankan=is_chankan, is_houtei=is_houtei)
            base = result.base_points
            is_dealer = seat == state.oya
            pay = _ceil100(base * (6 if is_dealer else 4)) + 300 * state.honba
            deltas[discarder] -= pay
            deltas[seat] += pay
            win_points = pay
            if i == 0:
                deltas[seat] += kyotaku
                win_points += kyotaku
            stats = state.stats[seat]
            stats.win_count += 1
            stats.ron_count += 1
            stats.win_points += win_points
        state.stats[discarder].dealt_in_count += len(ordered)
        state.kyotaku = 0

        self._apply_deltas(deltas)
        if self.rules.agari_yame and self._is_all_last() and any(top_flags):
            self._end_game()
        else:
            self._end_kyoku(renchan=state.oya in winners)
        return {s: self.reward_config.score_delta_scale * d for s, d in enumerate(deltas)}

    def _is_nagashi(self, seat: int) -> bool:
        """流局满贯资格：所有舍牌均为幺九牌，且从未被他人 吃/碰/大明杠."""
        p = self.state.players[seat]
        if p.discard_called or not p.discards:
            return False
        return all(t in HONORS or rank_of(t) in (1, 9) for t in p.discards)

    def _resolve_nagashi(self, nagashi: list[int]) -> dict:
        """流局满贯结算（视为自摸满贯），达成者不参与普通听牌/不听结算."""
        state = self.state
        deltas = [0, 0, 0, 0]
        honba = 100 * state.honba
        for seat in nagashi:
            if seat == state.oya:
                pay = 4000 + honba
                for s in range(4):
                    if s != seat:
                        deltas[s] -= pay
                        deltas[seat] += pay
            else:
                dealer_pay = 4000 + honba
                other_pay = 2000 + honba
                deltas[state.oya] -= dealer_pay
                deltas[seat] += dealer_pay
                for s in range(4):
                    if s != seat and s != state.oya:
                        deltas[s] -= other_pay
                        deltas[seat] += other_pay

        # 供托：唯一达成者全得；多家达成时不分发（携带至下一局）——规则假设
        if len(nagashi) == 1:
            deltas[nagashi[0]] += state.kyotaku * 1000
            state.kyotaku = 0

        # 普通听牌/不听结算：排除达成者，其余照常
        others = [s for s in range(4) if s not in nagashi]
        tenpai = {s: bool(state.players[s].hand) and is_tenpai(state.players[s].hand, state.players[s].melds) for s in others}
        n_tenpai = sum(tenpai.values())
        n_noten = len(others) - n_tenpai
        if 1 <= n_tenpai <= len(others) - 1 and n_noten >= 1:
            total = 3000  # 听牌点棒转移不叠加本场（数据实证）
            for s in others:
                if tenpai[s]:
                    deltas[s] += total // n_tenpai
                else:
                    deltas[s] -= total // n_noten

        self._apply_deltas(deltas)
        renchan = state.oya in nagashi or tenpai.get(state.oya, False)
        self._end_kyoku(renchan=renchan)
        return {s: self.reward_config.score_delta_scale * d for s, d in enumerate(deltas)}

    def _resolve_ryukyoku(self, *, abort: bool) -> dict:
        state = self.state
        if abort:
            deltas = [0, 0, 0, 0]
            self._apply_deltas(deltas)
            self._end_kyoku(renchan=True)
            return {s: 0.0 for s in range(4)}

        # 流局满贯（荒牌流局）
        if self.rules.nagashi_mangan:
            nagashi = [s for s in range(4) if self._is_nagashi(s)]
            if nagashi:
                return self._resolve_nagashi(nagashi)

        # 普通听牌/不听结算
        deltas = [0, 0, 0, 0]
        tenpai = [False] * 4
        for s in range(4):
            p = state.players[s]
            tenpai[s] = bool(p.hand) and is_tenpai(p.hand, p.melds)
        n_tenpai = sum(tenpai)
        n_noten = 4 - n_tenpai
        if 1 <= n_tenpai <= 3 and n_noten >= 1:
            total = 3000  # 听牌点棒转移不叠加本场（数据实证）
            for s in range(4):
                if tenpai[s]:
                    deltas[s] += total // n_tenpai
                else:
                    deltas[s] -= total // n_noten
        # kyotaku (riichi sticks) carries to the next kyoku; do NOT reset here.
        self._apply_deltas(deltas)
        renchan = tenpai[state.oya]
        self._end_kyoku(renchan=renchan)
        return {s: self.reward_config.score_delta_scale * d for s, d in enumerate(deltas)}

    def _apply_deltas(self, deltas: list[int]) -> None:
        state = self.state
        for s in range(4):
            state.scores[s] += deltas[s]

    # -- kyoku / game advance ---------------------------------------------------
    def _end_kyoku(self, renchan: bool) -> None:
        state = self.state
        state.phase = PHASE_ENDED
        if renchan:
            state.honba += 1
        else:
            state.oya = (state.oya + 1) % 4
            state.honba = 0
            if state.kyoku < 4:
                state.kyoku += 1
            elif state.round_wind == "E":
                state.round_wind = "S"
                state.kyoku = 1
            elif state.round_wind == "S":
                if max(state.scores) >= 30000 or not self.rules.west_round:
                    self._end_game()
                    return
                state.round_wind = "W"
                state.kyoku = 1
            else:
                self._end_game()
                return
        self._start_kyoku()

    def _end_game(self) -> None:
        state = self.state
        state.game_over = True
        state.phase = PHASE_ENDED
        order = sorted(range(4), key=lambda s: (-state.scores[s], s))
        state.final_ranks = [0] * 4
        for rank, seat in enumerate(order):
            state.final_ranks[seat] = rank
            bonus = self.reward_config.placement_bonus[rank]
            self._accumulated[seat] = self._accumulated.get(seat, 0.0) + bonus


def _ceil100(x: int) -> int:
    return ((x + 99) // 100) * 100


class _DummyWall:
    remaining = 100  # riichi wall constraint always satisfied for replay cross-check


def legal_actions_from_replay(
    replay_state,
    *,
    seat: int,
    phase: str,
    drawn_tile: str | None = None,
    discarder: int | None = None,
    discard_tile: str | None = None,
) -> list:
    """Environment legal actions for a decision point reconstructed from a ReplayState.

    Used for the Environment <-> Replay cross-validation: the ReplayState has
    the full hands/melds/rivers/dora; the wall is a dummy (the wall-tile count
    is not part of the ReplayState), which only affects the rare "riichi with
    <4 wall tiles" rule.
    """
    env_state = EnvState()
    env_state.round_wind = replay_state.bakaze
    env_state.kyoku = replay_state.kyoku
    env_state.honba = replay_state.honba
    env_state.kyotaku = replay_state.kyotaku
    env_state.oya = replay_state.oya
    env_state.scores = list(replay_state.scores)
    env_state.dora_indicators = list(replay_state.dora_markers)
    env_state.ura_indicators = ()
    env_state.wall = _DummyWall()
    env_state.turn = seat
    env_state.phase = phase
    env_state.discarder = discarder
    env_state.discard_tile = discard_tile

    for i, ps in enumerate(replay_state.players):
        p = Player(i)
        p.hand = list(ps.hand)
        p.melds = list(ps.melds)
        p.discards = list(ps.discards)
        p.riichi = ps.riichi
        p.ippatsu = ps.ippatsu
        p.score = ps.score
        p.drawn_this_turn = phase == PHASE_DISCARD and drawn_tile is not None
        p.last_drawn = drawn_tile
        env_state.players[i] = p

    env = MahjongEnv(seed=0)
    env.state = env_state
    return env.legal_actions(seat)


__all__ = ["KYUUSHU_ACTION", "MahjongEnv", "RewardConfig", "SEAT_WINDS", "legal_actions_from_replay"]
