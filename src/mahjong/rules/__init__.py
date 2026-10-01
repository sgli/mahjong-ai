"""Unified rule core (Phase 3).

Hand-shape logic (agari / shanten / tenpai / furiten) plus legal-action
generation.  The Environment and Evaluation phases should reuse this module so
Replay / Environment / Evaluation agree on the same rules (ENVIRONMENT_SPEC
section 7).
"""

from .agari import (
    is_agari,
    is_chiitoi,
    is_furiten,
    is_kokushi,
    is_standard_agari,
    is_tenpai,
    shanten,
    tenpai_tiles,
)
from .config import MAJSOUL_RULES, TENHOU_RULES, RulesConfig
from .legal import (
    ankan_actions,
    can_riichi,
    chi_actions,
    daiminkan_actions,
    discard_legal_actions,
    kakan_actions,
    pon_actions,
    response_legal_actions,
    validate_ankan_consumed,
)
from .decompose import Group, Structure, iter_structures
from .dora import aka_dora_count, count_dora, dora_tile
from .scoring import (
    ScoreResult,
    base_points_from,
    ron_payment,
    score_hand,
    tsumo_payments,
)
from .tiles import ALL_TILES, HONORS, ORPHANS, normalize, suit_of, to_counts
from .yaku import WinContext, YakuResult, compute_yaku

__all__ = [
    "ALL_TILES",
    "Group",
    "HONORS",
    "MAJSOUL_RULES",
    "ORPHANS",
    "RulesConfig",
    "ScoreResult",
    "Structure",
    "TENHOU_RULES",
    "WinContext",
    "YakuResult",
    "aka_dora_count",
    "ankan_actions",
    "base_points_from",
    "can_riichi",
    "chi_actions",
    "compute_yaku",
    "count_dora",
    "daiminkan_actions",
    "discard_legal_actions",
    "dora_tile",
    "is_agari",
    "is_chiitoi",
    "is_furiten",
    "is_kokushi",
    "is_standard_agari",
    "is_tenpai",
    "iter_structures",
    "kakan_actions",
    "normalize",
    "pon_actions",
    "response_legal_actions",
    "ron_payment",
    "score_hand",
    "shanten",
    "suit_of",
    "tenpai_tiles",
    "to_counts",
    "tsumo_payments",
    "validate_ankan_consumed",
]
