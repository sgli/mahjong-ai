#!/usr/bin/env python
"""Replay <-> Environment cross-validation (Phase 10.2 §8).

For real ``.mjai.json`` files, reconstruct each decision point and assert the
recorded human action is in the Environment's legal actions.  Mismatches are
classified as data/replay/environment issue.

Usage::

    python scripts/cross_validate_replay.py \
        --data-dir "F:/Mahjong AI/mahjong DB/tenhou-houou-2026" --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mahjong.decision.extractor import DecisionExtractor  # noqa: E402
from mahjong.environment import PHASE_CHANKAN, PHASE_DISCARD, PHASE_RESPONSE, legal_actions_from_replay  # noqa: E402
from mahjong.parser import MjaiParser  # noqa: E402
from mahjong.replay import ReplayEngine  # noqa: E402


def cross_validate(data_dir: Path, limit: int) -> dict:
    files = sorted(data_dir.glob("*.mjai.json"))[:limit]
    parser = MjaiParser()
    checked = 0
    action_kinds = Counter()
    mismatches: list[dict] = []

    for path in files:
        events = list(parser.parse_file(path))
        states = list(ReplayEngine(game_id=path.stem).replay(events))
        extractor = DecisionExtractor(game_id=path.stem)
        for sample in extractor.extract(states, events):
            i = sample.decision_point.step
            replay_state = states[i]
            event = events[i]
            kind = sample.decision_point.kind
            if kind == "discard":
                drawn_tile = event.pai if event.type.value == "tsumo" else None
                legal = legal_actions_from_replay(
                    replay_state, seat=sample.player_id, phase=PHASE_DISCARD, drawn_tile=drawn_tile
                )
            else:
                phase = PHASE_CHANKAN if event.type.value == "kakan" else PHASE_RESPONSE
                legal = legal_actions_from_replay(
                    replay_state,
                    seat=sample.player_id,
                    phase=phase,
                    discarder=event.actor,
                    discard_tile=event.pai,
                )
            action_kinds[sample.action.type.value] += 1
            checked += 1
            if sample.action not in legal:
                mismatches.append(
                    {
                        "game_id": path.stem,
                        "kyoku": replay_state.kyoku,
                        "event_index": i,
                        "seat": sample.player_id,
                        "recorded_action": repr(sample.action),
                        "legal_actions": [repr(a) for a in legal[:20]],
                    }
                )

    return {
        "files": len(files),
        "checked_decision_points": checked,
        "action_kinds": dict(action_kinds),
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay <-> Environment cross-validation.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args(argv)

    report = cross_validate(Path(args.data_dir), args.limit)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
