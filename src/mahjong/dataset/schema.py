"""Parquet schema and row (de)serialisation for the Decision Dataset.

Schema design (``decision-v1``)
-------------------------------

Each row is one :class:`~mahjong.decision.decision.DecisionSample`.  Columns are
split into four groups:

* **identity / provenance** — ``game_id``, ``round_id``, ``player_id``,
  ``decision_kind``, ``step``.
* **observation** — the *visible* projection only.  Scalar fields (``bakaze``,
  ``kyoku``, ``honba``, ``kyotaku``, ``oya``, ``turn``) and simple vectors
  (``scores``, ``riichi``, ``hand``, ``dora_markers``) are native Arrow columns;
  the three nested structures (``melds``, ``opponents_melds``, ``discards``)
  are stored as JSON strings to keep the schema flat and avoid deeply nested
  Arrow structs.  They only contain public information (own hand/melds, the four
  rivers, dora indicators, riichi flags, and each player's *revealed* melds).
  Opponents' concealed hands, the wall and ura markers are structurally absent.
* **actions** — ``legal_actions`` is a JSON string (list of action dicts);
  the human ``action`` is split into scalar columns (``action_type``,
  ``action_tile``, ``action_consumed``, ``action_target``, ``action_kan_kind``)
  so the target action is columnar for training.
* **metadata** — a JSON string for offline-only fields (data source, rule
  version, timestamps); future information may only live here, never in the
  observation columns.

The JSON columns are deterministic (``sort_keys``) so identical samples produce
byte-identical rows, and ``row_to_sample`` reconstructs the original
:class:`DecisionSample` losslessly.
"""

from __future__ import annotations

import json

import pyarrow as pa

from ..decision.action import Action, ActionType
from ..decision.decision import DecisionPoint, DecisionSample
from ..decision.observation import PlayerObservation
from ..features.action_space import FEATURE_VERSION  # 单一版本来源（feature-v1）
from ..replay.state import Meld

DATASET_VERSION = "decision-v1"


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(text: str):
    return json.loads(text)


# -- Meld / Action <-> plain dict ----------------------------------------------
def meld_to_dict(meld: Meld) -> dict:
    return {"kind": meld.kind, "tiles": list(meld.tiles), "from_": meld.from_, "called": meld.called}


def meld_from_dict(d: dict) -> Meld:
    return Meld(kind=d["kind"], tiles=tuple(d["tiles"]), from_=d.get("from_"), called=d.get("called"))


def action_to_dict(action: Action) -> dict:
    return {
        "type": action.type.value,
        "tile": action.tile,
        "consumed": list(action.consumed),
        "target": action.target,
        "kan_kind": action.kan_kind,
    }


def action_from_dict(d: dict) -> Action:
    return Action(
        type=ActionType(d["type"]),
        tile=d.get("tile"),
        consumed=tuple(d.get("consumed") or []),
        target=d.get("target"),
        kan_kind=d.get("kan_kind"),
    )


# -- sample -> row / row -> sample ---------------------------------------------
def observation_to_columns(obs: PlayerObservation) -> dict:
    return {
        "bakaze": obs.bakaze,
        "kyoku": obs.kyoku,
        "honba": obs.honba,
        "kyotaku": obs.kyotaku,
        "oya": obs.oya,
        "scores": list(obs.scores),
        "riichi": list(obs.riichi),
        "hand": list(obs.hand),
        "melds": _json([meld_to_dict(m) for m in obs.melds]),
        "opponents_melds": _json(
            [[meld_to_dict(m) for m in slot] for slot in obs.opponents_melds]
        ),
        "discards": _json([list(river) for river in obs.discards]),
        "dora_markers": list(obs.dora_markers),
        "turn": obs.turn,
    }


def sample_to_row(sample: DecisionSample) -> dict:
    """Serialize one DecisionSample to a plain-dict Parquet row."""
    point = sample.decision_point
    return {
        "game_id": sample.game_id,
        "round_id": sample.round_id,
        "player_id": sample.player_id,
        "decision_kind": point.kind if point else "",
        "step": point.step if point else -1,
        **observation_to_columns(sample.observation),
        "legal_actions": _json([action_to_dict(a) for a in sample.legal_actions]),
        "action_type": sample.action.type.value,
        "action_tile": sample.action.tile,
        "action_consumed": list(sample.action.consumed),
        "action_target": sample.action.target,
        "action_kan_kind": sample.action.kan_kind,
        "metadata": _json(sample.metadata),
    }


def row_to_sample(row: dict) -> DecisionSample:
    """Reconstruct a DecisionSample from a (plain-dict) Parquet row."""
    obs = PlayerObservation(
        seat=row["player_id"],
        bakaze=row["bakaze"],
        kyoku=row["kyoku"],
        honba=row["honba"],
        kyotaku=row["kyotaku"],
        oya=row["oya"],
        scores=tuple(row["scores"]),
        riichi=tuple(row["riichi"]),
        hand=tuple(row["hand"]),
        melds=tuple(meld_from_dict(m) for m in _loads(row["melds"])),
        opponents_melds=tuple(
            tuple(meld_from_dict(m) for m in slot) for slot in _loads(row["opponents_melds"])
        ),
        discards=tuple(tuple(r) for r in _loads(row["discards"])),
        dora_markers=tuple(row["dora_markers"]),
        turn=row["turn"],
    )
    legal_actions = tuple(action_from_dict(a) for a in _loads(row["legal_actions"]))
    action = action_from_dict(
        {
            "type": row["action_type"],
            "tile": row["action_tile"],
            "consumed": row["action_consumed"],
            "target": row["action_target"],
            "kan_kind": row["action_kan_kind"],
        }
    )
    point = DecisionPoint(
        round_id=row["round_id"],
        seat=row["player_id"],
        kind=row["decision_kind"],
        step=row["step"],
    )
    return DecisionSample(
        game_id=row["game_id"],
        round_id=row["round_id"],
        player_id=row["player_id"],
        observation=obs,
        legal_actions=legal_actions,
        action=action,
        metadata=_loads(row["metadata"]),
        decision_point=point,
    )


# -- Arrow schema --------------------------------------------------------------
def parquet_schema() -> pa.Schema:
    """Arrow schema for one DecisionSample row (documented above)."""
    return pa.schema(
        [
            pa.field("game_id", pa.string()),
            pa.field("round_id", pa.string()),
            pa.field("player_id", pa.int32()),
            pa.field("decision_kind", pa.string()),
            pa.field("step", pa.int32()),
            # observation (visible projection only)
            pa.field("bakaze", pa.string()),
            pa.field("kyoku", pa.int32()),
            pa.field("honba", pa.int32()),
            pa.field("kyotaku", pa.int32()),
            pa.field("oya", pa.int32()),
            pa.field("scores", pa.list_(pa.int32())),
            pa.field("riichi", pa.list_(pa.bool_())),
            pa.field("hand", pa.list_(pa.string())),
            pa.field("melds", pa.string()),
            pa.field("opponents_melds", pa.string()),
            pa.field("discards", pa.string()),
            pa.field("dora_markers", pa.list_(pa.string())),
            pa.field("turn", pa.int32()),
            # actions
            pa.field("legal_actions", pa.string()),
            pa.field("action_type", pa.string()),
            pa.field("action_tile", pa.string()),
            pa.field("action_consumed", pa.list_(pa.string())),
            pa.field("action_target", pa.int32()),
            pa.field("action_kan_kind", pa.string()),
            # offline metadata
            pa.field("metadata", pa.string()),
        ]
    )


#: Human-readable documentation for every column (visible vs offline).
COLUMN_DOCS: dict[str, str] = {
    "game_id": "identifier of the game (file stem); identity, not part of observation",
    "round_id": "round id (e.g. E1); identity",
    "player_id": "decision-making seat 0..3; identity",
    "decision_kind": "discard | response; provenance",
    "step": "0-based event index that opened the decision window; provenance",
    "bakaze": "round wind E/S/W; visible",
    "kyoku": "round number 1..4; visible",
    "honba": "counter sticks; visible",
    "kyotaku": "riichi sticks in the pot; visible",
    "oya": "dealer seat; visible",
    "scores": "4 players' points; visible",
    "riichi": "4 players' riichi flags; visible",
    "hand": "decision player's concealed hand; visible (own)",
    "melds": "decision player's melds (JSON); visible (own)",
    "opponents_melds": "other players' revealed melds (JSON); visible",
    "discards": "4 players' rivers (JSON); visible",
    "dora_markers": "revealed dora indicators; visible",
    "turn": "seat expected to act; visible",
    "legal_actions": "legal action set (JSON of action dicts); derived from visible state",
    "action_type": "human action type; target label",
    "action_tile": "human action tile; target label",
    "action_consumed": "human action consumed tiles; target label",
    "action_target": "human action target seat; target label",
    "action_kan_kind": "kan sub-kind for KAN actions; target label",
    "metadata": "offline-only extras (JSON); may hold future info, never in observation",
}
