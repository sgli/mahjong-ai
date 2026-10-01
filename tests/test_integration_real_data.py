"""Integration test: parse a few real ``.mjai.json`` files if available.

Skipped automatically when the dataset is not present on this machine, so the
rest of the suite stays deterministic on a clean checkout.
"""

import os
from pathlib import Path

import pytest

from mahjong.parser import MjaiParser

# Test convenience only: ``MAHJONG_TEST_DATA_DIR`` always takes priority so the
# test works on any machine; the hard-coded path below is just a fallback for
# the author's checkout and is never used when the environment variable is set.
DEFAULT_DATA_DIR = Path("F:/Mahjong AI/mahjong DB/tenhou-houou-2026")


def _data_dir() -> Path:
    env = os.environ.get("MAHJONG_TEST_DATA_DIR")
    return Path(env) if env else DEFAULT_DATA_DIR


def test_real_data_roundtrip():
    data_dir = _data_dir()
    if not data_dir.is_dir():
        pytest.skip(f"real data not found at {data_dir}")

    files = sorted(data_dir.glob("*.mjai.json"))[:3]
    if not files:
        pytest.skip("no .mjai.json files found")

    parser = MjaiParser()
    for path in files:
        types = [e.type.value for e in parser.parse_file(path)]
        assert types[0] == "start_game", path.name
        assert types[-1] == "end_game", path.name
        assert "start_kyoku" in types
        assert types.count("start_kyoku") == types.count("end_kyoku")
        assert len(types) > 0
