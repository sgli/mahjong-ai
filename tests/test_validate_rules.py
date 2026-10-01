"""Tests for the rule-validation tool (scripts/validate_rules.py)."""

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "validate_rules.py"
_spec = importlib.util.spec_from_file_location("validate_rules", str(_SCRIPT))
vr = importlib.util.module_from_spec(_spec)
sys.modules["validate_rules"] = vr
_spec.loader.exec_module(vr)


def _write_synthetic_hora(tmp_path):
    lines = [
        '{"type":"start_game","names":["A","B","C","D"],"kyoku_first":0,"aka_flag":true}',
        '{"type":"start_kyoku","bakaze":"E","dora_marker":"W","kyoku":1,"honba":0,"kyotaku":0,"oya":0,'
        '"scores":[25000,25000,25000,25000],"tehais":['
        '["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","5p"],'
        '["1m","1m","3m","3m","5m","5m","7m","7m","9m","9m","1p","1p","3p"],'
        '["2m","2m","4m","4m","6m","6m","8m","8m","1p","1p","2p","2p","3p"],'
        '["3m","3m","5m","5m","7m","7m","9m","9m","2p","2p","3p","3p","4p"]]}',
        '{"type":"tsumo","actor":0,"pai":"5p"}',
        '{"type":"hora","actor":0,"target":0,"deltas":[6000,-2000,-2000,-2000],"ura_markers":[]}',
        '{"type":"end_kyoku"}',
    ]
    path = tmp_path / "synthetic.mjai.json"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_validate_file_reconstructs_hora(tmp_path):
    path = _write_synthetic_hora(tmp_path)
    report = vr.validate_file(path, "tenhou", vr.TENHOU_RULES)
    assert len(report.horas) == 1
    h = report.horas[0]
    assert h.match is True
    assert h.is_tsumo is True
    assert h.win_tile == "5p"
    assert ("ittsuu", 2) in h.yaku
    assert ("menzen_tsumo", 1) in h.yaku
    assert h.han == 3
    assert h.fu == 30
    assert h.expected == [6000, -2000, -2000, -2000]


def test_expected_hora_deltas_ron():
    # base 240 (1 han 30 fu), non-dealer ron, honba 0, kyotaku 1
    d = vr.expected_hora_deltas(
        240, is_tsumo=False, is_dealer=False, winner=1, target=0, oya=0, honba=0, kyotaku=1
    )
    assert d == [-1000, 2000, 0, 0]  # discarder pays 960->1000; winner +1000 +1000 kyotaku


def test_expected_hora_deltas_tsumo_dealer():
    d = vr.expected_hora_deltas(
        960, is_tsumo=True, is_dealer=True, winner=0, target=0, oya=0, honba=0, kyotaku=0
    )
    assert d == [6000, -2000, -2000, -2000]


def test_reconstruct_ryukyoku_nagashi_true_positive():
    from types import SimpleNamespace

    class P:
        def __init__(self, hand, discards):
            self.hand = hand
            self.discards = discards
            self.melds = []
            self.discard_called = False

    # 子家 seat0 全幺九舍牌未被叫 → nagashi；庄家 seat1
    state = SimpleNamespace(
        oya=1,
        honba=0,
        kyotaku=0,
        players=[
            P(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"], ["1m", "9m", "1p", "9p", "E", "S", "W", "N", "P", "F", "C"]),
            P(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"], ["2m"]),
            P(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"], ["2m"]),
            P(["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "5p"], ["2m"]),
        ],
    )
    ev = SimpleNamespace(deltas=[0, 0, 0, 0])
    deltas, nagashi = vr.reconstruct_ryukyoku([ev], [state], 0, vr.MAJSOUL_RULES, [False, False, False, False])
    # 子家 nagashi：+8000，庄家 -4000，其余 -2000
    assert nagashi == [0]
    assert deltas == [8000, -4000, -2000, -2000]


def test_sample_files_reproducible(tmp_path):
    root = tmp_path / "data"
    (root / "tenhou-a").mkdir(parents=True)
    for i in range(10):
        (root / "tenhou-a" / f"f{i}.mjai.json").write_text("{}\n", encoding="utf-8")
    classified = vr.classify_files(root)
    assert len(classified["tenhou"]) == 10
    s1 = vr.sample_files(classified["tenhou"], 4, seed=7)
    s2 = vr.sample_files(classified["tenhou"], 4, seed=7)
    assert [p.name for p in s1] == [p.name for p in s2]
