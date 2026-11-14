"""Rebuild rule-aware menus only when the necessary replay context was retained."""
from .schema import row_to_sample
from ..decision.action import ActionType
from ..rules.legal import LEGAL_MASK_VERSION, discard_legal_actions, response_legal_actions, chankan_legal_actions
from ..rules.scoring import score_hand
from ..rules.yaku import WinContext


def rebuild_legal_actions(row):
    sample = row_to_sample(row)
    if sample.metadata.get("legal_mask_version") != LEGAL_MASK_VERSION:
        raise ValueError("legacy legal menu lacks verified replay context")
    c = sample.metadata["legal_context"]
    obs = sample.observation
    common = dict(hand=obs.hand, melds=obs.melds, riichi=obs.riichi[obs.seat],
                  round_wind=c["round_wind"], seat_wind=c["seat_wind"], ippatsu=c["ippatsu"])
    if row["decision_kind"] == "discard":
        return discard_legal_actions(**common, score=obs.scores[obs.seat], drawn_tile=c["drawn_tile"],
                                     is_rinshan=c["is_rinshan"], is_haitei=c["wall_remaining"] == 0,
                                     wall_remaining=c["wall_remaining"])
    common.update(own_discards=obs.discards[obs.seat], temporary_furiten=c["temporary_furiten"],
                  riichi_furiten=c["riichi_furiten"])
    if c["is_chankan"]:
        return chankan_legal_actions(**common, added_tile=c["claimed_tile"], kan_declarer=c["discarder"])
    return response_legal_actions(**common, claimed_tile=c["claimed_tile"], discarder=c["discarder"],
                                  is_next=obs.seat == (c["discarder"] + 1) % 4,
                                  is_houtei=c["wall_remaining"] == 0)


def audit_legal_row(row):
    sample = row_to_sample(row)
    if sample.metadata.get("legal_mask_version") == LEGAL_MASK_VERSION:
        if set(rebuild_legal_actions(row)) != set(sample.legal_actions):
            raise ValueError("stored menu disagrees with context-aware reconstruction")
        return {"context_verified_rows": 1}
    # Legacy Parquet cannot establish last-wall/rinshan/chankan/temporary-furiten
    # context. Count candidates needing investigation, never silently edit labels.
    obs = sample.observation
    candidates = 0
    for action in sample.legal_actions:
        if action.type not in (ActionType.RON, ActionType.TSUMO):
            continue
        tsumo = action.type is ActionType.TSUMO
        tile = row["hand"][-1] if tsumo else None
        if not tsumo and action.target is not None and obs.discards[action.target]:
            tile = obs.discards[action.target][-1]
        if tile is None:
            continue
        hand = list(obs.hand) if tsumo else list(obs.hand) + [tile]
        ctx = WinContext(obs.bakaze, "ESWN"[(obs.seat - obs.oya) % 4], tile,
                         is_tsumo=tsumo, is_menzen=all(m.kind == "ankan" for m in obs.melds),
                         is_riichi=obs.riichi[obs.seat])
        if not score_hand(hand, obs.melds, tile, ctx).has_yaku:
            candidates += 1
    return {"legacy_context_missing_rows": 1, "win_actions_without_observable_yaku": candidates}
