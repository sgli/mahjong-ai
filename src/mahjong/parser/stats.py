"""Basic statistics over parsed Mjai event streams.

Implements the counters required by ``docs/DATA_SPEC.md`` section 12 for the
Phase 1 data inspection, without loading the whole dataset into memory.
"""

from __future__ import annotations

from collections import Counter

from .events import (
    Dahai,
    Event,
    EventType,
    KAN_EVENT_TYPES,
    StartGame,
    StartKyoku,
)


class StatsCollector:
    """Aggregate counts while streaming events."""

    def __init__(self) -> None:
        self.files = 0
        self.games = 0
        self.kyokus = 0
        self.events_total = 0
        self.event_counts: Counter[str] = Counter()
        self.aka_flag_counts: Counter[bool] = Counter()
        self.bakaze_counts: Counter[str] = Counter()
        self.kyoku_first_counts: Counter[int] = Counter()
        self.tsumogiri_counts: Counter[bool] = Counter()
        self.player_names: set[str] = set()

    def add_event(self, event: Event) -> None:
        self.events_total += 1
        self.event_counts[event.type.value] += 1

        if isinstance(event, StartGame):
            self.games += 1
            self.aka_flag_counts[event.aka_flag] += 1
            self.kyoku_first_counts[event.kyoku_first] += 1
            self.player_names.update(event.names)
        elif isinstance(event, StartKyoku):
            self.kyokus += 1
            self.bakaze_counts[event.bakaze] += 1
        elif isinstance(event, Dahai):
            self.tsumogiri_counts[event.tsumogiri] += 1

    @property
    def kan_total(self) -> int:
        return sum(self.event_counts[t.value] for t in KAN_EVENT_TYPES)

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable summary dict."""
        return {
            "files": self.files,
            "games": self.games,
            "kyokus": self.kyokus,
            "events_total": self.events_total,
            "event_counts": dict(sorted(self.event_counts.items())),
            "kan_total": self.kan_total,
            "aka_flag": {str(k): v for k, v in sorted(self.aka_flag_counts.items())},
            "bakaze": dict(sorted(self.bakaze_counts.items())),
            "kyoku_first": dict(sorted(self.kyoku_first_counts.items())),
            "tsumogiri": {str(k): v for k, v in sorted(self.tsumogiri_counts.items())},
            "distinct_players": len(self.player_names),
        }

    def render(self) -> str:
        """Render a human-readable statistics block."""
        s = self.summary()
        lines = [
            "== Mjai data statistics ==",
            f"files:            {s['files']}",
            f"games:            {s['games']}",
            f"kyokus (rounds):  {s['kyokus']}",
            f"events total:     {s['events_total']}",
            f"distinct players: {s['distinct_players']}",
            "",
            "event counts:",
        ]
        counts = s["event_counts"]
        assert isinstance(counts, dict)
        for name, count in counts.items():
            lines.append(f"  {name:16s} {count}")
        lines.extend(
            [
                f"  {'kan (total)':16s} {s['kan_total']}",
                "",
                "rule/config distribution:",
                f"  aka_flag: {s['aka_flag']}",
                f"  bakaze:   {s['bakaze']}",
                f"  kyoku_first: {s['kyoku_first']}",
                f"  tsumogiri: {s['tsumogiri']}",
            ]
        )
        return "\n".join(lines)
