"""Decision extractor (Phase 3).

Walks a replayed event/state sequence and yields one
:class:`DecisionSample` per meaningful decision:

- a *discard* decision for the actor after a ``tsumo`` (draw) or after a
  ``chi``/``pon`` (no draw);
- a *response* decision for each other player after a ``dahai``, where they may
  ron / chi / pon / daiminkan / pass.

The human action is reconstructed from the immediately following event(s) and
verified against the generated legal actions; any mismatch is recorded in
:attr:`DecisionExtractor.inconsistencies` (never silently dropped).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ..parser.events import Event
from ..replay.state import ReplayState
from ..rules.legal import LEGAL_MASK_VERSION, chankan_legal_actions, discard_legal_actions, response_legal_actions
from .action import Action, ActionType
from .decision import DecisionPoint, DecisionSample
from .observation import PlayerObservation

_PASS = Action(ActionType.PASS)


def _action_sort_key(action: Action) -> tuple:
    return (
        action.type.value,
        action.tile or "",
        action.consumed,
        action.target if action.target is not None else -1,
        action.kan_kind or "",
    )


class DecisionExtractor:
    """ReplayState + Event sequence -> DecisionSample iterator."""

    def __init__(self, *, game_id: str | None = None) -> None:
        self.game_id = game_id
        #: Human-action-not-in-legal-actions reports (rule gap or data anomaly).
        self.inconsistencies: list[str] = []

    def extract(
        self,
        states: Sequence[ReplayState],
        events: Sequence[Event],
    ) -> Iterator[DecisionSample]:
        """Yield decision samples; ``states[i]`` must be the state after
        ``events[i]`` (as produced by ``ReplayEngine.replay``)."""
        if len(states) != len(events):
            raise ValueError("replay state/event lengths differ")
        self._temporary_furiten = [False] * 4
        self._riichi_furiten = [False] * 4
        self._draws = 0
        self._rinshan = False
        n = len(events)
        for i in range(n):
            event = events[i]
            self._current_event = event
            state = states[i]
            et = event.type.value
            if et == "start_kyoku":
                self._temporary_furiten = [False] * 4
                self._riichi_furiten = [False] * 4
                self._draws = 0
            if et == "tsumo":
                self._draws += 1
                self._temporary_furiten[event.actor] = False
                previous = i - 1
                while previous >= 0 and events[previous].type.value == "dora":
                    previous -= 1
                self._rinshan = previous >= 0 and events[previous].type.value in ("ankan", "kakan", "daiminkan")
            if et == "dahai":
                # A called chi/pon turn has no self-draw; the caller's own
                # discard still ends same-turn furiten (Tenhou rules).
                self._temporary_furiten[event.actor] = False

            if et == "tsumo":
                yield from self._discard_decision(state, event, i, events, drawn_tile=event.pai)
            elif et in ("chi", "pon"):
                yield from self._discard_decision(state, event, i, events, drawn_tile=None)
            elif et == "dahai":
                yield from self._response_decisions(state, event, i, events)
            elif et == "kakan":
                yield from self._chankan_decisions(state, event, i, events)

    # -- discard decisions -----------------------------------------------------
    def _discard_decision(self, state, event, i, events, drawn_tile):
        actor = event.actor
        ps = state.players[actor]

        # The last draw of the wall ends the kyoku without a discard
        # (tsumo -> ryukyoku), so there is no discard decision to extract.
        j = self._next_meaningful_index(i, events)
        if j >= len(events) or events[j].type.value in ("ryukyoku", "end_kyoku"):
            return

        human = self._resolve_discard_action(actor, i, events)
        if human is None:
            self.inconsistencies.append(
                f"step {i}: could not resolve human discard action for seat {actor} "
                f"after {event.type.value}"
            )
            return

        legal = discard_legal_actions(
            hand=ps.hand,
            melds=ps.melds,
            riichi=ps.riichi,
            score=ps.score,
            drawn_tile=drawn_tile,
            round_wind=state.bakaze, seat_wind="ESWN"[(actor - state.oya) % 4],
            ippatsu=ps.ippatsu, is_rinshan=self._rinshan,
            is_haitei=self._draws >= 70, wall_remaining=70 - self._draws,
        )
        legal = self._normalise_legal(legal)
        yield self._make_sample(state, actor, "discard", i, legal, human)

    def _next_meaningful_index(self, i, events) -> int:
        """Skip ``dora`` events (emitted between a kan replacement draw and the
        following discard) when looking for the action that resolves a window."""
        j = i + 1
        while j < len(events) and events[j].type.value == "dora":
            j += 1
        return j

    def _resolve_discard_action(self, actor, i, events) -> Action | None:
        j = self._next_meaningful_index(i, events)
        if j >= len(events):
            return None
        nxt = events[j]
        t = nxt.type.value

        if t == "reach" and getattr(nxt, "actor", None) == actor:
            # riichi: the tile is the following dahai
            if j + 1 < len(events) and events[j + 1].type.value == "dahai":
                return Action(ActionType.RIICHI, tile=events[j + 1].pai)
            return None
        if t == "dahai" and getattr(nxt, "actor", None) == actor:
            return Action(ActionType.DISCARD, tile=nxt.pai)
        if t == "ankan" and getattr(nxt, "actor", None) == actor:
            return Action(ActionType.KAN, kan_kind="ankan", consumed=tuple(sorted(nxt.consumed)))
        if t == "kakan" and getattr(nxt, "actor", None) == actor:
            return Action(
                ActionType.KAN, tile=nxt.pai, consumed=tuple(sorted(nxt.consumed)), kan_kind="kakan"
            )
        if t == "hora" and getattr(nxt, "actor", None) == actor and getattr(nxt, "target", None) == actor:
            return Action(ActionType.TSUMO)
        return None

    # -- response decisions ----------------------------------------------------
    def _response_decisions(self, state, event, i, events):
        discarder = event.actor
        claimed_tile = event.pai
        acted = self._resolve_responses(i, events)

        for seat in range(4):
            if seat == discarder:
                continue
            ps = state.players[seat]
            legal = response_legal_actions(
                hand=ps.hand,
                melds=ps.melds,
                own_discards=ps.discards,
                riichi=ps.riichi,
                claimed_tile=claimed_tile,
                discarder=discarder,
                is_next=(seat == (discarder + 1) % 4),
                round_wind=state.bakaze, seat_wind="ESWN"[(seat - state.oya) % 4],
                ippatsu=ps.ippatsu, is_houtei=self._draws >= 70,
                temporary_furiten=self._temporary_furiten[seat], riichi_furiten=self._riichi_furiten[seat],
            )
            legal = self._normalise_legal(legal)
            human = acted.get(seat, _PASS)

            # Emit only meaningful choices: the player had a real alternative
            # (more than PASS) or actually took a non-PASS action.
            if len(legal) > 1 or human.type is not ActionType.PASS:
                yield self._make_sample(state, seat, "response", i, legal, human)
                if human.type is ActionType.PASS and any(a.type is ActionType.RON for a in legal):
                    self._temporary_furiten[seat] = True
                    self._riichi_furiten[seat] |= ps.riichi

    def _resolve_responses(self, i, events) -> dict[int, Action]:
        """Map responder seat -> non-PASS action (for those who acted)."""
        acted: dict[int, Action] = {}
        j = i + 1
        # A riichi discard is followed by ``reach_accepted`` before the other
        # players respond; skip that (and dora, defensively).
        while j < len(events) and events[j].type.value in ("reach_accepted", "dora"):
            j += 1
        if j >= len(events):
            return acted
        nxt = events[j]
        t = nxt.type.value

        if t in ("chi", "pon", "daiminkan"):
            actor = nxt.actor
            if t == "chi":
                acted[actor] = Action(
                    ActionType.CHI, tile=nxt.pai, consumed=tuple(sorted(nxt.consumed)), target=nxt.target
                )
            elif t == "pon":
                acted[actor] = Action(
                    ActionType.PON, tile=nxt.pai, consumed=tuple(sorted(nxt.consumed)), target=nxt.target
                )
            else:
                acted[actor] = Action(
                    ActionType.KAN,
                    tile=nxt.pai,
                    consumed=tuple(sorted(nxt.consumed)),
                    target=nxt.target,
                    kan_kind="daiminkan",
                )
        elif t == "hora":
            k = j
            while k < len(events) and events[k].type.value == "hora":
                acted[events[k].actor] = Action(ActionType.RON, target=events[k].target)
                k += 1
        # else: tsumo / ryukyoku / end_kyoku -> everyone passed.
        return acted

    def _chankan_decisions(self, state, event, i, events):
        """Response window after a kakan: only a ron on the added tile (chankan)
        or pass is possible for the three other players."""
        declarer = event.actor
        added_tile = event.pai
        acted: dict[int, Action] = {}
        j = self._next_meaningful_index(i, events)
        k = j
        while k < len(events) and events[k].type.value == "hora":
            acted[events[k].actor] = Action(ActionType.RON, target=events[k].target)
            k += 1

        for seat in range(4):
            if seat == declarer:
                continue
            ps = state.players[seat]
            legal = chankan_legal_actions(
                hand=ps.hand,
                melds=ps.melds,
                own_discards=ps.discards,
                added_tile=added_tile,
                kan_declarer=declarer,
                riichi=ps.riichi, round_wind=state.bakaze, seat_wind="ESWN"[(seat - state.oya) % 4],
                ippatsu=ps.ippatsu, temporary_furiten=self._temporary_furiten[seat],
                riichi_furiten=self._riichi_furiten[seat],
            )
            legal = self._normalise_legal(legal)
            human = acted.get(seat, _PASS)
            if len(legal) > 1 or human.type is not ActionType.PASS:
                yield self._make_sample(state, seat, "response", i, legal, human)
                if human.type is ActionType.PASS and any(a.type is ActionType.RON for a in legal):
                    self._temporary_furiten[seat] = True
                    self._riichi_furiten[seat] |= ps.riichi

    # -- shared -----------------------------------------------------------------
    def _normalise_legal(self, legal: list[Action]) -> tuple[Action, ...]:
        return tuple(sorted(set(legal), key=_action_sort_key))

    def _make_sample(self, state, seat, kind, step, legal, human) -> DecisionSample:
        if human not in legal:
            self.inconsistencies.append(
                f"step {step}: seat {seat} {kind} action {human!r} not in legal actions "
                f"{[repr(a) for a in legal]}"
            )
        observation = PlayerObservation.from_state(state, seat)
        point = DecisionPoint(round_id=state.round_id, seat=seat, kind=kind, step=step)
        event = self._current_event
        context = {
            "round_wind": state.bakaze, "seat_wind": "ESWN"[(seat - state.oya) % 4],
            "ippatsu": state.players[seat].ippatsu,
            "temporary_furiten": self._temporary_furiten[seat],
            "riichi_furiten": self._riichi_furiten[seat],
            "wall_remaining": 70 - self._draws, "is_rinshan": self._rinshan,
            "drawn_tile": event.pai if event.type.value == "tsumo" else None,
            "claimed_tile": event.pai if kind == "response" else None,
            "discarder": event.actor if kind == "response" else None,
            "is_chankan": event.type.value == "kakan",
        }
        return DecisionSample(
            game_id=state.game_id or self.game_id,
            round_id=state.round_id,
            player_id=seat,
            observation=observation,
            legal_actions=legal,
            action=human,
            metadata={"kind": kind, "legal_mask_version": LEGAL_MASK_VERSION, "legal_context": context},
            decision_point=point,
        )
