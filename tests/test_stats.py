"""Tests for the statistics collector."""

from mahjong.parser import StatsCollector, parse_event


def _feed(collector, *raw_events):
    for raw in raw_events:
        collector.add_event(parse_event(raw))


def test_counts_games_and_kyokus():
    c = StatsCollector()
    _feed(
        c,
        {"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": True},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "W",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [["1m"] * 13] * 4,
        },
        {"type": "tsumo", "actor": 0, "pai": "6m"},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    )
    assert c.games == 1
    assert c.kyokus == 1
    assert c.events_total == 5
    assert c.event_counts["tsumo"] == 1
    assert c.event_counts["dahai"] == 0


def test_kan_total_sums_all_variants():
    c = StatsCollector()
    _feed(
        c,
        {"type": "daiminkan", "actor": 0, "target": 3, "pai": "1s", "consumed": ["1s", "1s", "1s"]},
        {"type": "ankan", "actor": 1, "consumed": ["E", "E", "E", "E"]},
        {"type": "kakan", "actor": 2, "pai": "P", "consumed": ["P", "P", "P"]},
    )
    assert c.kan_total == 3


def test_tsumogiri_and_aka_flag_counts():
    c = StatsCollector()
    _feed(
        c,
        {"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": True},
        {"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "6m", "tsumogiri": False},
    )
    assert c.aka_flag_counts[True] == 1
    assert c.tsumogiri_counts[True] == 1
    assert c.tsumogiri_counts[False] == 1


def test_summary_and_render():
    c = StatsCollector()
    _feed(
        c,
        {"type": "start_game", "names": ["a", "b", "c", "d"], "kyoku_first": 0, "aka_flag": True},
        {"type": "tsumo", "actor": 0, "pai": "6m"},
    )
    c.files = 1
    summary = c.summary()
    assert summary["games"] == 1
    assert summary["events_total"] == 2
    assert "event_counts" in summary
    rendered = c.render()
    assert "games:" in rendered
    assert "tsumo" in rendered
