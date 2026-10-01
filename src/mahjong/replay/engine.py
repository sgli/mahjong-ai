"""Replay Engine (Phase 2).

Deterministically rebuilds a full :class:`ReplayState` from an ordered stream
of typed Mjai events.  The engine yields one state after every event so the
caller can observe both the transition and the resulting snapshot.

Consistency checks (``docs/DATA_SPEC.md`` section 11):

- a tile removed from a hand must actually be in that hand;
- a tile claimed by chi / pon / daiminkan must be the target's latest discard;
- each hand satisfies the size invariant (13 - 3*n melds at rest,
  14 - 3*n melds right after a draw) so melds and kans are accounted for;
- the settled scores (start-of-kyoku scores + reach deductions + hora/ryukyoku
  deltas) must equal the next ``start_kyoku`` scores; mismatches are recorded in
  :attr:`ReplayEngine.inconsistencies` and never silently dropped.

Hard violations raise :class:`ReplayInconsistency` immediately.  Score/kyotaku
mismatches at the next ``start_kyoku`` are recorded (non-fatal) so a dataset
scan can continue and report every affected game.

Rule notes (verified against the actual ``tenhou-houou-2026`` data):

- riichi sequence is ``reach -> dahai -> reach_accepted``; the -1000 stick and
  ``kyotaku += 1`` happen at ``reach_accepted``, not at ``reach``.  When a
  player is ron'd on the riichi discard there is no ``reach_accepted`` and no
  stick is deducted (the data's ``hora`` deltas confirm this).
- exception: when the kyoku ends in a draw immediately after a riichi discard
  (``reach -> dahai -> ryukyoku``, 5 occurrences per ~5k files) the data omits
  ``reach_accepted`` but the stick *is* paid, so the engine applies any
  un-accepted reach retroactively at ``ryukyoku`` (never at ``hora``).
- ``hora``/``ryukyoku`` deltas are authoritative and already include riichi
  stick redistribution for the winner; the engine only applies the reach -1000
  at ``reach_accepted`` and then adds the deltas.
- on ``ryukyoku`` the riichi sticks carry over to the next kyoku (``kyotaku``
  is left untouched); on ``hora`` they are taken by the winner (``kyotaku = 0``).
- the dora indicator may appear before or after the replacement draw (ankan
  logs ``dora`` before ``tsumo``, kakan/daiminkan log it after); the engine
  handles both orders because each event is applied independently.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from ..parser.events import (
    Ankan,
    Chi,
    Dahai,
    Daiminkan,
    Dora,
    EndGame,
    EndKyoku,
    Event,
    Hora,
    Kakan,
    Pon,
    Reach,
    ReachAccepted,
    Ryukyoku,
    StartGame,
    StartKyoku,
    Tsumo,
)
from .state import Meld, PlayerState, ReplayState


class ReplayError(Exception):
    """Base error for the Replay Engine."""


class ReplayInconsistency(ReplayError):
    """A hard state inconsistency was detected while applying an event."""

    def __init__(self, message: str, *, step: int | None = None, event_type: str | None = None) -> None:
        self.step = step
        self.event_type = event_type
        super().__init__(message)


class ReplayEngine:
    """Deterministic Mjai event stream -> ReplayState iterator."""

    def __init__(self, *, game_id: str | None = None) -> None:
        #: Optional game id (the raw event schema has none; the file name is the
        #: natural id, so callers pass it in when they have it).
        self.game_id = game_id
        #: Non-fatal inconsistencies (score / kyotaku mismatches) collected
        #: while replaying, so they can be reported instead of silently ignored.
        self.inconsistencies: list[str] = []
        self._reset()

    # -- internal working state ------------------------------------------------
    def _reset(self) -> None:
        self._names: tuple[str, str, str, str] = ("", "", "", "")
        self._aka_flag = False
        self._kyoku_first = 0
        self._round_wind = "E"  # bakaze
        self._kyoku = 0
        self._honba = 0
        self._kyotaku = 0
        self._oya = 0
        self._scores = [0, 0, 0, 0]
        self._seen_kyoku = False
        self._hands: list[list[str]] = [[], [], [], []]
        self._melds: list[list[Meld]] = [[], [], [], []]
        self._discards: list[list[str]] = [[], [], [], []]
        self._tsumogiri: list[list[bool]] = [[], [], [], []]
        self._riichi = [False] * 4
        self._reach_accepted = [False] * 4
        self._ippatsu = [False] * 4
        self._ippatsu_close_on_dahai = [False] * 4
        self._ippatsu_reach_discarded = [False] * 4
        self._dora_markers: list[str] = []
        self._ura_markers: tuple[str, ...] = ()
        self._turn = 0
        self._step = 0
        self._kyoku_open = False

    # -- public API ------------------------------------------------------------
    def replay(self, events: Iterable[Event]) -> Iterator[ReplayState]:
        """Apply ``events`` in order, yielding the state after each event."""
        for event in events:
            self._apply(event)
            yield self._snapshot(event)

    # -- helpers ---------------------------------------------------------------
    def _fail(self, message: str, event: Event) -> None:
        raise ReplayInconsistency(
            message, step=self._step, event_type=event.type.value
        )

    def _hand_remove(self, seat: int, tile: str, event: Event) -> None:
        hand = self._hands[seat]
        if tile not in hand:
            self._fail(
                f"seat {seat}: tile {tile!r} is not in hand {hand}", event
            )
        hand.remove(tile)
        hand.sort()

    def _discard_pop(self, target: int, pai: str, event: Event) -> None:
        """Remove the tile claimed from ``target``'s river.

        Mahjong only allows claiming the most recent discard, so the claimed
        tile must sit on top of the target's river.
        """
        river = self._discards[target]
        if not river or river[-1] != pai:
            self._fail(
                f"seat {target}: claimed tile {pai!r} is not the latest discard "
                f"(river={river})",
                event,
            )
        river.pop()
        self._tsumogiri[target].pop()

    def _hand_size_after_draw(self, seat: int) -> int:
        return 14 - 3 * len(self._melds[seat])

    def _hand_size_at_rest(self, seat: int) -> int:
        return 13 - 3 * len(self._melds[seat])

    def _check_hand_size(self, seat: int, event: Event, expected: int) -> None:
        actual = len(self._hands[seat])
        if actual != expected:
            self._fail(
                f"seat {seat}: hand has {actual} tiles, expected {expected} "
                f"({len(self._melds[seat])} melds)",
                event,
            )

    def _interrupt_ippatsu(self, actor: int) -> None:
        """A call/kan by ``actor`` cancels other players' ippatsu windows."""
        for seat in range(4):
            if seat != actor:
                self._ippatsu[seat] = False
                self._ippatsu_close_on_dahai[seat] = False

    # -- event handlers --------------------------------------------------------
    def _apply(self, event: Event) -> None:
        self._step += 1

        if isinstance(event, StartGame):
            self._names = event.names
            self._aka_flag = event.aka_flag
            self._kyoku_first = event.kyoku_first

        elif isinstance(event, StartKyoku):
            if self._seen_kyoku:
                settled = tuple(self._scores)
                if settled != event.scores:
                    self.inconsistencies.append(
                        f"step {self._step}: score mismatch at start_kyoku "
                        f"(settled {settled}, data {event.scores})"
                    )
                if self._kyotaku != event.kyotaku:
                    self.inconsistencies.append(
                        f"step {self._step}: kyotaku mismatch at start_kyoku "
                        f"(tracked {self._kyotaku}, data {event.kyotaku})"
                    )
            self._seen_kyoku = True
            self._round_wind = event.bakaze
            self._kyoku = event.kyoku
            self._honba = event.honba
            self._kyotaku = event.kyotaku
            self._oya = event.oya
            self._scores = list(event.scores)
            for seat in range(4):
                self._hands[seat] = sorted(event.tehais[seat])
                self._melds[seat] = []
                self._discards[seat] = []
                self._tsumogiri[seat] = []
                self._riichi[seat] = False
                self._reach_accepted[seat] = False
                self._ippatsu[seat] = False
                self._ippatsu_close_on_dahai[seat] = False
            self._dora_markers = [event.dora_marker]
            self._ura_markers = ()
            self._turn = event.oya
            self._kyoku_open = True

        elif isinstance(event, Tsumo):
            self._hands[event.actor].append(event.pai)
            self._hands[event.actor].sort()
            self._turn = event.actor
            self._check_hand_size(
                event.actor, event, self._hand_size_after_draw(event.actor)
            )

        elif isinstance(event, Dahai):
            self._hand_remove(event.actor, event.pai, event)
            self._discards[event.actor].append(event.pai)
            self._tsumogiri[event.actor].append(event.tsumogiri)
            self._check_hand_size(
                event.actor, event, self._hand_size_at_rest(event.actor)
            )
            if self._ippatsu_close_on_dahai[event.actor]:
                self._ippatsu[event.actor] = False
                self._ippatsu_close_on_dahai[event.actor] = False
            self._turn = (event.actor + 1) % 4

        elif isinstance(event, Chi):
            self._interrupt_ippatsu(event.actor)
            for tile in event.consumed:
                self._hand_remove(event.actor, tile, event)
            self._discard_pop(event.target, event.pai, event)
            self._melds[event.actor].append(
                Meld(
                    kind="chi",
                    tiles=tuple(sorted(event.consumed + (event.pai,))),
                    from_=event.target,
                    called=event.pai,
                )
            )
            self._turn = event.actor

        elif isinstance(event, Pon):
            self._interrupt_ippatsu(event.actor)
            for tile in event.consumed:
                self._hand_remove(event.actor, tile, event)
            self._discard_pop(event.target, event.pai, event)
            self._melds[event.actor].append(
                Meld(
                    kind="pon",
                    tiles=tuple(sorted(event.consumed + (event.pai,))),
                    from_=event.target,
                    called=event.pai,
                )
            )
            self._turn = event.actor

        elif isinstance(event, Daiminkan):
            self._interrupt_ippatsu(event.actor)
            for tile in event.consumed:
                self._hand_remove(event.actor, tile, event)
            self._discard_pop(event.target, event.pai, event)
            self._melds[event.actor].append(
                Meld(
                    kind="daiminkan",
                    tiles=tuple(sorted(event.consumed + (event.pai,))),
                    from_=event.target,
                    called=event.pai,
                )
            )
            self._turn = event.actor

        elif isinstance(event, Ankan):
            self._interrupt_ippatsu(event.actor)
            for tile in event.consumed:
                self._hand_remove(event.actor, tile, event)
            self._melds[event.actor].append(
                Meld(
                    kind="ankan",
                    tiles=tuple(sorted(event.consumed)),
                    from_=None,
                    called=None,
                )
            )
            self._turn = event.actor

        elif isinstance(event, Kakan):
            self._interrupt_ippatsu(event.actor)
            self._hand_remove(event.actor, event.pai, event)
            meld_index: int | None = None
            for i, meld in enumerate(self._melds[event.actor]):
                if meld.kind == "pon" and sorted(meld.tiles) == sorted(event.consumed):
                    meld_index = i
                    break
            if meld_index is None:
                self._fail(
                    f"seat {event.actor}: no pon meld matching {event.consumed} "
                    f"for kakan",
                    event,
                )
            old = self._melds[event.actor][meld_index]
            self._melds[event.actor][meld_index] = Meld(
                kind="kakan",
                tiles=tuple(sorted(old.tiles + (event.pai,))),
                from_=old.from_,
                called=old.called,
            )
            self._turn = event.actor

        elif isinstance(event, Dora):
            self._dora_markers.append(event.dora_marker)

        elif isinstance(event, Reach):
            self._riichi[event.actor] = True

        elif isinstance(event, ReachAccepted):
            self._riichi[event.actor] = True
            self._reach_accepted[event.actor] = True
            self._ippatsu[event.actor] = True
            self._ippatsu_close_on_dahai[event.actor] = True
            self._scores[event.actor] -= 1000
            self._kyotaku += 1

        elif isinstance(event, Hora):
            for seat in range(4):
                self._scores[seat] += event.deltas[seat]
            self._ura_markers = event.ura_markers
            self._kyotaku = 0
            self._turn = event.actor

        elif isinstance(event, Ryukyoku):
            for seat in range(4):
                self._scores[seat] += event.deltas[seat]
            # A reach whose discard was followed directly by an exhaustive
            # draw has no ``reach_accepted`` event, yet the stick is still paid
            # (verified against real data).  Apply any un-accepted reach now.
            for seat in range(4):
                if self._riichi[seat] and not self._reach_accepted[seat]:
                    self._reach_accepted[seat] = True
                    self._scores[seat] -= 1000
                    self._kyotaku += 1
            # riichi sticks stay in the pot and carry over to the next kyoku.

        elif isinstance(event, EndKyoku):
            self._kyoku_open = False

        elif isinstance(event, EndGame):
            pass  # game boundary; nothing further to update.

    # -- snapshot --------------------------------------------------------------
    def _snapshot(self, event: Event) -> ReplayState:
        players = tuple(
            PlayerState(
                seat=seat,
                hand=tuple(self._hands[seat]),
                melds=tuple(self._melds[seat]),
                discards=tuple(self._discards[seat]),
                discard_tsumogiri=tuple(self._tsumogiri[seat]),
                riichi=self._riichi[seat],
                ippatsu=self._ippatsu[seat],
                score=self._scores[seat],
            )
            for seat in range(4)
        )
        return ReplayState(
            game_id=self.game_id,
            round_id=f"{self._round_wind}{self._kyoku}" if self._kyoku else "",
            bakaze=self._round_wind,
            kyoku=self._kyoku,
            honba=self._honba,
            kyotaku=self._kyotaku,
            oya=self._oya,
            scores=tuple(self._scores),
            players=players,
            dora_markers=tuple(self._dora_markers),
            ura_markers=self._ura_markers,
            turn=self._turn,
            names=self._names,
            aka_flag=self._aka_flag,
            kyoku_first=self._kyoku_first,
            step=self._step,
            event_type=event.type.value,
        )
