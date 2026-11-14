"""Tests for Feature Encoder and Action Space."""

import dataclasses

import torch

from mahjong.decision import Action, ActionType, PlayerObservation
from mahjong.features import (
    ACTION_SPACE_SIZE,
    FEATURE_DIM,
    FEATURE_VERSION,
    action_from_id,
    action_to_id,
    canonical_action,
    encode_observation,
    legal_mask,
)
from mahjong.features.action_space_v2 import (
    ACTION_SPACE_SIZE_V2,
    action_from_id_v2,
    action_to_id_v2,
    legal_mask_v2,
)
from mahjong.features.encoder_v2 import FEATURE_DIM_V2, ObservationEncoderV2


def _obs(**overrides):
    kwargs = dict(
        seat=0,
        bakaze="E",
        kyoku=1,
        honba=0,
        kyotaku=0,
        oya=0,
        scores=(25000, 25000, 25000, 25000),
        riichi=(False, False, False, False),
        hand=("1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"),
        melds=(),
        opponents_melds=((), (), (), ()),
        discards=(("9p",), (), (), ()),
        dora_markers=("2m",),
        turn=0,
    )
    kwargs.update(overrides)
    return PlayerObservation(**kwargs)


# -- action space --------------------------------------------------------------
def test_action_space_size_and_version():
    assert ACTION_SPACE_SIZE == 270
    assert FEATURE_VERSION == "feature-v1"


def test_action_id_roundtrip_all_ids():
    for action_id in range(ACTION_SPACE_SIZE):
        action = action_from_id(action_id)
        assert action_to_id(action) == action_id


def test_action_id_roundtrip_eight_types():
    actions = [
        Action(ActionType.DISCARD, tile="5m"),
        Action(ActionType.RIICHI, tile="9s"),
        Action(ActionType.TSUMO),
        Action(ActionType.PASS),
        Action(ActionType.RON, target=2),
        Action(ActionType.PON, tile="C", consumed=("C", "C"), target=1),
        Action(ActionType.CHI, tile="8p", consumed=("6p", "7p"), target=0),
        Action(ActionType.KAN, kan_kind="ankan", consumed=("E", "E", "E", "E")),
        Action(ActionType.KAN, tile="1s", consumed=("1s", "1s", "1s"), target=3, kan_kind="daiminkan"),
        Action(ActionType.KAN, tile="P", consumed=("P", "P", "P"), kan_kind="kakan"),
    ]
    for action in actions:
        assert canonical_action(action) == action_from_id(action_to_id(action))
        assert action_to_id(canonical_action(action)) == action_to_id(action)


def test_action_red_five_normalized_to_base_five():
    assert action_to_id(Action(ActionType.DISCARD, tile="5mr")) == action_to_id(
        Action(ActionType.DISCARD, tile="5m")
    )


def test_action_target_is_not_part_of_identity():
    assert action_to_id(Action(ActionType.RON, target=0)) == action_to_id(
        Action(ActionType.RON, target=3)
    )


def test_legal_mask():
    legal = [Action(ActionType.DISCARD, tile="1m"), Action(ActionType.PASS)]
    mask = legal_mask(legal)
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[action_to_id(Action(ActionType.DISCARD, tile="1m"))]
    assert mask[action_to_id(Action(ActionType.PASS))]
    assert not mask[action_to_id(Action(ActionType.DISCARD, tile="9m"))]
    assert not mask[action_to_id(Action(ActionType.TSUMO))]


# -- encoder -------------------------------------------------------------------
def test_encoder_shape_and_determinism():
    obs = _obs()
    out1 = encode_observation(obs)
    out2 = encode_observation(obs)
    assert out1.shape == (FEATURE_DIM,)
    assert torch.equal(out1, out2)


def test_encoder_only_reads_observation_fields():
    class _Probe:
        def __init__(self, obs):
            object.__setattr__(self, "_obs", obs)
            object.__setattr__(self, "accessed", set())

        def __getattribute__(self, name):
            if name in ("_obs", "accessed", "__class__"):
                return object.__getattribute__(self, name)
            object.__getattribute__(self, "accessed").add(name)
            return getattr(object.__getattribute__(self, "_obs"), name)

    probe = _Probe(_obs())
    out = encode_observation(probe)
    assert out.shape == (FEATURE_DIM,)

    allowed = {f.name for f in dataclasses.fields(PlayerObservation)}
    assert probe.accessed <= allowed
    assert not {"opponents_hands", "wall", "ura_markers"} & probe.accessed


def test_action_v2_distinguishes_red_five_roles_and_roundtrips():
    normal = Action(ActionType.DISCARD, tile="5m")
    red = Action(ActionType.DISCARD, tile="5mr")
    assert action_to_id_v2(normal) != action_to_id_v2(red)
    assert action_from_id_v2(action_to_id_v2(red)) == red

    normal_pon = Action(ActionType.PON, tile="5m", consumed=("5m", "5m"))
    red_pon = Action(ActionType.PON, tile="5m", consumed=("5m", "5mr"))
    assert action_to_id_v2(normal_pon) != action_to_id_v2(red_pon)
    assert action_from_id_v2(action_to_id_v2(red_pon)) == red_pon
    assert ACTION_SPACE_SIZE_V2 > ACTION_SPACE_SIZE
    assert action_from_id_v2(ACTION_SPACE_SIZE_V2 - 1).type is ActionType.KYUUSHU_KYUHAI


def test_action_v2_has_typed_kyuushu_id_and_mask_gate():
    action = Action(ActionType.KYUUSHU_KYUHAI)
    assert action_from_id_v2(action_to_id_v2(action)) == action
    assert not legal_mask_v2([action]).any()
    assert legal_mask_v2([action], allow_kyuushu=True)[action_to_id_v2(action)]
    for action_id in range(ACTION_SPACE_SIZE_V2):
        assert action_to_id_v2(action_from_id_v2(action_id)) == action_id


def test_feature_v2_preserves_red_tile_and_relative_river_order():
    encoder = ObservationEncoderV2()
    normal = _obs(hand=("5m",), discards=((), ("1m", "2m"), (), ()))
    red = _obs(hand=("5mr",), discards=((), ("1m", "2m"), (), ()))
    changed_order = _obs(hand=("5mr",), discards=((), ("2m", "1m"), (), ()))
    normal_features = encoder.encode(normal)
    red_features = encoder.encode(red)
    order_features = encoder.encode(changed_order)
    assert normal_features.shape == (FEATURE_DIM_V2,)
    assert not torch.equal(normal_features, red_features)
    assert not torch.equal(red_features, order_features)


def test_feature_v2_rejects_river_overflow_instead_of_truncating():
    import pytest

    obs = _obs(discards=((), tuple(["1m"] * 129), (), ()))
    with pytest.raises(ValueError, match="limit is 128"):
        ObservationEncoderV2().encode(obs)
