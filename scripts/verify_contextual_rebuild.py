"""Bounded raw replay -> contextual menu audit, optionally paired with a legacy shard.

No source Parquet/raw files are modified. Output is diagnostic only.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pyarrow as pa
import pyarrow.parquet as pq
from mahjong.parser import MjaiParser
from mahjong.replay import ReplayEngine
from mahjong.decision.extractor import DecisionExtractor
from mahjong.dataset.schema import sample_to_row, parquet_schema
from mahjong.dataset.legality import audit_legal_row


def verify(raw_file, output_dir, legacy_shard=None):
    game_id = raw_file.stem
    events = [MjaiParser().parse_line(line, n) for n, line in enumerate(raw_file.open(encoding='utf-8'), 1)]
    states = list(ReplayEngine(game_id=game_id).replay(events))
    extractor = DecisionExtractor(game_id=game_id)
    samples = list(extractor.extract(states, events))
    if extractor.inconsistencies:
        raise ValueError(f'{game_id}: {extractor.inconsistencies[:3]}')
    rows = [sample_to_row(s) for s in samples]
    for row in rows:
        audit_legal_row(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=parquet_schema()), output_dir / (game_id + '.parquet'))
    result = {'game_id': game_id, 'events': len(events), 'rows': len(rows), 'context_verified_rows': len(rows)}
    if legacy_shard:
        old = pq.read_table(legacy_shard, filters=[('game_id', '=', game_id)]).to_pylist()
        def key(row):
            return row['step'], row['player_id'], row['decision_kind']
        fresh = {key(r): r for r in rows}
        common = [(r, fresh[key(r)]) for r in old if key(r) in fresh]
        def menu(r):
            return {json.dumps(a, sort_keys=True) for a in json.loads(r['legal_actions'])}
        label_fields = ('action_type','action_tile','action_consumed','action_target','action_kan_kind')
        result.update(legacy_rows=len(old), common_rows=len(common),
                      changed_menus=sum(menu(a) != menu(b) for a,b in common),
                      changed_labels=sum(any(a[k] != b[k] for k in label_fields) for a,b in common),
                      removed_windows=len(old) - len(common))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-file', type=Path, action='append', required=True)
    ap.add_argument('--legacy-shard', type=Path)
    ap.add_argument('--out', type=Path, default=ROOT / 'experiments/phase10_3_8/contextual_rebuild')
    args = ap.parse_args()
    results = [verify(path, args.out, args.legacy_shard) for path in args.raw_file]
    (args.out / 'result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
