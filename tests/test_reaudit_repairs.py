"""Regression cases that the original 454-test suite did not cover."""
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from mahjong.decision.action import Action, ActionType
from mahjong.decision.extractor import DecisionExtractor
from mahjong.dataset.legality import audit_legal_row, rebuild_legal_actions
from mahjong.evaluation.opponent import RuleOpponent
from mahjong.evaluation.validation import validate_game_rows
from mahjong.features.encoder_v2 import ObservationEncoderV2
from mahjong.model import MLPPolicy
from mahjong.rules.decompose import Group, Structure
from mahjong.rules.legal import LEGAL_MASK_VERSION, response_legal_actions
from mahjong.rules.scoring import score_hand
from mahjong.rules.yaku import WinContext, winning_interpretations
from mahjong.training.bc import MetricAccumulator, collate_record_batches, prepare_bc_batch, train_dataloader
from mahjong.training.data_gate import require_full_audit

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location('repair_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row():
    return dict(hand=['1m', '2m', '3m', '4m', '5m', '6m', '7p', '8p', '9p', '2s', '3s', '4s', '5p', '5p'],
                player_id=2, melds='[]', opponents_melds='[[],[],[],[]]',
                discards='[[],[],[],[]]', dora_markers=['1m'], scores=[-12300, 25100, 33300, 53900],
                riichi=[False]*4, bakaze='E', kyoku=1, honba=2, kyotaku=1, oya=0, turn=2,
                action_type='discard', action_tile='1m', action_consumed=[], action_kan_kind=None,
                action_target=None, legal_actions=json.dumps([{'type': 'discard', 'tile': '1m'}, {'type': 'discard', 'tile': '2m'}]),
                decision_kind='discard', game_id='g', round_id='E1', step=1, metadata='{}')


def test_ron_sequence_assignment_retains_three_concealed_triplets():
    hand = ['5m']*3 + ['3m', '4m', '5m'] + ['1p']*3 + ['9s']*3 + ['E']*2
    result = score_hand(hand, [], '5m', WinContext('E', 'S', '5m', is_riichi=True))
    assert ('sanankou', 2) in result.yaku
    assert (result.han, result.fu, result.base_points) == (3, 60, 1920)


def test_existing_ankan_never_receives_winning_tile():
    structure = Structure((Group('quad', ('5m',)*4, False), Group('run', ('3m','4m','5m'), False)), 'E')
    candidates = list(winning_interpretations(structure, '5m'))
    assert [(s.winning_group, wait) for s, wait in candidates] == [(1, 'ryanmen')]


def test_no_yaku_ron_requires_contextual_yaku_and_respects_furiten():
    hand = ['1m','2m','4m','5m','6m','7p','8p','9p','2s','3s','4s','5p','5p']
    kwargs = dict(hand=hand, melds=[], own_discards=[], riichi=False, claimed_tile='3m', discarder=0, is_next=False)
    assert all(a.type is not ActionType.RON for a in response_legal_actions(**kwargs))
    assert any(a.type is ActionType.RON for a in response_legal_actions(**kwargs, is_houtei=True))
    assert all(a.type is not ActionType.RON for a in response_legal_actions(**kwargs, is_houtei=True, temporary_furiten=True))


def test_compact_multi_batch_exact_and_gold_gate():
    r = row()
    rb = pa.RecordBatch.from_pylist([r, r])
    dense = collate_record_batches([rb, rb], ObservationEncoderV2())
    compact = collate_record_batches([rb, rb], ObservationEncoderV2(), compact_v2_transport=True)
    restored = prepare_bc_batch(compact, torch.device('cpu'))
    assert compact[0].dtype == torch.uint8
    for expected, actual in zip(dense, restored):
        assert torch.equal(expected, actual)
    r['legal_actions'] = '[]'
    with pytest.raises(ValueError, match='gold label'):
        collate_record_batches([pa.RecordBatch.from_pylist([r])], ObservationEncoderV2(), compact_v2_transport=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_compact_cuda_exact():
    rb = pa.RecordBatch.from_pylist([row()])
    dense = collate_record_batches([rb], ObservationEncoderV2())
    restored = prepare_bc_batch(collate_record_batches([rb], ObservationEncoderV2(), compact_v2_transport=True), torch.device('cuda'))
    assert all(torch.equal(a, b.cpu()) for a, b in zip(dense, restored))


def test_weighted_metrics_and_exact_validation_budget(tmp_path):
    accum = MetricAccumulator()
    accum.update({'accuracy': 1.0}, 100)
    accum.update({'accuracy': 0.0}, 1)
    assert accum.compute()['accuracy'] == pytest.approx(100 / 101)
    shard = tmp_path / 'part.parquet'
    pq.write_table(pa.Table.from_pylist([row()] * 5), shard)
    model = MLPPolicy(14605, 325, (4,))
    history = train_dataloader(model, torch.optim.Adam(model.parameters()), [shard], [shard], ObservationEncoderV2(),
                               epochs=1, batch_size=2, device=torch.device('cpu'), val_max_samples=3)
    assert history[0]['val']['sampled'] == 3
    prof = history[0]['profile']
    assert prof['samples'] == 5 and prof['batch_time'] == prof['train_wall_seconds'] > 0


def test_bc_epoch_snapshots_accumulate_and_resume_preserves_best(tmp_path, monkeypatch):
    module = script('train_bc')
    config = yaml.safe_load((ROOT / 'configs/train_bc_v3.yaml').read_text(encoding='utf-8'))
    config['model']['hidden_sizes'] = [4]
    config['dataset_dir'] = str(tmp_path / 'data')
    config['checkpoint_dir'] = str(tmp_path / 'checkpoints')
    config['training'].update(device='cpu', epochs=3, limit_train_shards=1)
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setattr(module, '_parquet_files', lambda *a: [Path('fake.parquet')])
    def fake_train(*args, **kw):
        history = []
        for epoch in range(kw['start_epoch'] + 1, 4):
            entry = {'epoch': epoch, 'train': {}, 'val': {'loss': [1., .8, .9][epoch-1]}, 'profile': {'samples': 100}}
            kw['on_epoch_end'](epoch, entry['train'], entry['val'], entry['profile'])
            history.append(entry)
        return history
    monkeypatch.setattr(module, 'train_dataloader', fake_train)
    module.main(['--config', str(config_path)])
    folder = tmp_path / 'checkpoints'
    def load(name):
        return torch.load(folder / name, weights_only=False)
    epoch2 = load('epoch_2.pt')
    assert epoch2['global_step'] == 200 and epoch2['best_val_loss'] == .8 and epoch2['best_epoch'] == 2
    assert [h['epoch'] for h in epoch2['metrics']['history']] == [1, 2]
    module.main(['--config', str(config_path), '--resume', str(folder / 'best.pt')])
    assert load('latest.pt')['global_step'] == 300
    assert load('best.pt')['best_val_loss'] == .8
    bad = dict(epoch2, config={**epoch2['config'], 'feature_version': 'feature-v1'})
    with pytest.raises(ValueError, match='feature_version'):
        module._validate_resume(bad, config, (4,), 2048, 42, .001, None)


@pytest.mark.parametrize('change', [{'error':'crash'}, {'completed':False}, {'illegal_actions':1}, {'final_ranks':[0,0,1,2]}])
def test_benchmark_rejects_failed_games(change):
    r = dict(seed=320000, completed=True, error=None, illegal_actions=0, seat=0, final_ranks=[0,1,2,3])
    r.update(change)
    with pytest.raises(ValueError):
        validate_game_rows([r])


def test_registered_bc_effect_is_not_difference_of_differences(tmp_path):
    module = script('analyze_phase10_3_7')
    def payload(rank):
        rows = []
        for i in range(1000):
            seat = i % 4
            ranks = [r for r in range(4) if r != rank]
            ranks.insert(seat, rank)
            rows.append(dict(seed=320000+i, seat=seat, final_ranks=ranks, completed=True, error=None, illegal_actions=0))
        return dict(candidate_checkpoint_sha256='a'*64, environment_source_sha256={'env':'x'}, benchmark_runner_sha256='r'*64, candidate_mode='greedy',
                    opponent_mode='greedy', candidate_temperature=1., opponent_temperature=1.,
                    matchups={'bc_v2_1': {'raw_games':rows, 'opponent_checkpoint_sha256':'a'*64}})
    control, treatment = tmp_path/'c.json', tmp_path/'t.json'
    control.write_text(json.dumps(payload(3)), encoding='utf-8')
    treatment.write_text(json.dumps(payload(0)), encoding='utf-8')
    result = module.paired_analysis(control, treatment, 'bc_v2_1', seed=1, anchor_sha256='a'*64)
    assert result['effect_rank'] == -2 and result['passed'] is True
    bad = payload(0)
    bad['matchups']['bc_v2_1']['raw_games'][1]['seed'] = 320000
    treatment.write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        module._read(treatment, 'bc_v2_1')


def test_rule_win_first_and_explicit_legacy_version():
    actions = [Action(ActionType.KYUUSHU_KYUHAI), Action(ActionType.TSUMO)]
    assert RuleOpponent().decide(None, actions, random.Random()).type is ActionType.TSUMO
    assert RuleOpponent(model_version='rule-v1').decide(None, actions, random.Random()).type is ActionType.KYUUSHU_KYUHAI


def test_legacy_data_cannot_pass_context_gate(tmp_path):
    r = row()
    assert audit_legal_row(r)['legacy_context_missing_rows'] == 1
    with pytest.raises(ValueError, match='legacy'):
        rebuild_legal_actions(r)
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'passed':False, 'limit_per_split':None}), encoding='utf-8')
    with pytest.raises(ValueError, match='complete contextual'):
        require_full_audit(audit, tmp_path)


@pytest.mark.parametrize('riichi', [False, True])
def test_live_furiten_after_passing_ron_and_next_draw(riichi):
    from mahjong.environment import MahjongEnv
    from mahjong.environment.state import PHASE_RESPONSE
    env = MahjongEnv(seed=4)
    env.reset()
    hand = ['1m','2m','3m','4m','5m','6m','7m','8m','9m','7p','8p','5p','5p']
    p = env.state.players[1]
    p.hand, p.melds, p.discards, p.riichi = list(hand), [], [], riichi
    env.state.phase, env.state.turn = PHASE_RESPONSE, 1
    env.state.discarder, env.state.discard_tile = 0, '9p'
    env.state.pending_responders = [1, 2]
    assert env._can_ron(1, '9p', 0)
    env.step(Action(ActionType.PASS))
    assert not env._can_ron(1, '9p', 0)
    env._draw_for(1)
    p.hand = list(hand)
    assert env._can_ron(1, '9p', 0) is (not riichi)
    env._start_kyoku()
    assert not p.temporary_furiten and not p.riichi_furiten
    assert env.state.players[env.state.oya].last_drawn in env.state.players[env.state.oya].hand


def test_temporary_furiten_clears_after_own_call_discard(monkeypatch):
    extractor = DecisionExtractor(game_id='g')
    events = [SimpleNamespace(type=SimpleNamespace(value='dahai'), actor=seat)
              for seat in (0, 1, 2)]
    seen = []
    def responses(state, event, index, events):
        if index == 0:
            extractor._temporary_furiten[1] = True
        if index == 2:
            seen.append(extractor._temporary_furiten[1])
        return iter(())
    monkeypatch.setattr(extractor, '_response_decisions', responses)
    list(extractor.extract([object()] * len(events), events))
    assert seen == [False]

    from mahjong.environment import MahjongEnv
    env = MahjongEnv(seed=4)
    env.reset()
    player = env.state.players[1]
    player.hand = ['5p']
    player.temporary_furiten = True
    player.riichi_furiten = True
    env._discard_tile(1, '5p')
    assert not player.temporary_furiten
    assert player.riichi_furiten


def test_diagnostics_collect_once_and_decompose_same_sample_variance(monkeypatch):
    module = script('diag_advantage')
    calls = []
    class Env:
        def __init__(self, **kw):
            self.state = SimpleNamespace(final_ranks=[0,1,2,3])
        def done(self):
            return True
    monkeypatch.setattr(module, 'MahjongEnv', Env)
    monkeypatch.setattr(module, 'load_policy', lambda *a: (object(), SimpleNamespace(eval=lambda: None)))
    def collect(*a, **kw):
        calls.append(kw['rng_seed'])
        return {s: SimpleNamespace(steps=[SimpleNamespace(reward=s+1., value=.3, done=False), SimpleNamespace(reward=s+2., value=.4, done=True)]) for s in range(4)}, 8, 0
    monkeypatch.setattr(module, 'collect_trajectory', collect)
    result = module.audit('fake', torch.device('cpu'), games=2)
    assert calls == [0,1]
    stats = result['advantage']
    assert stats['var_total'] == pytest.approx(stats['var_between_episodes'] + stats['var_within'])


def test_context_menu_roundtrip_and_tampering_rejected():
    r = row()
    context = dict(round_wind='E', seat_wind='W', ippatsu=False, temporary_furiten=False,
                   riichi_furiten=False, wall_remaining=60, is_rinshan=False,
                   drawn_tile='5p', claimed_tile=None, discarder=None, is_chankan=False)
    r['metadata'] = json.dumps({'legal_mask_version':LEGAL_MASK_VERSION, 'legal_context':context})
    actions = rebuild_legal_actions(r)
    r['legal_actions'] = json.dumps([a.to_dict() for a in actions])
    assert audit_legal_row(r) == {'context_verified_rows':1}
    r['legal_actions'] = '[]'
    with pytest.raises(ValueError, match='disagrees'):
        audit_legal_row(r)


def test_ppo_loader_restores_independent_value_trunk(tmp_path):
    from mahjong.evaluation.opponent import load_policy
    model = MLPPolicy(234, 270, (4,), value_separate_trunk=True)
    checkpoint = tmp_path / 'ppo.pt'
    torch.save({'checkpoint_type':'ppo', 'model_state_dict':model.state_dict(),
                'config':{'model':{'hidden_sizes':[4], 'value_separate_trunk':True}}}, checkpoint)
    _, restored = load_policy(checkpoint)
    features, mask = torch.randn(2,234), torch.ones(2,270, dtype=torch.bool)
    assert restored.value_separate_trunk
    assert torch.equal(model(features,mask).value, restored(features,mask).value)


def test_full_contextual_audit_accepts_verified_data_and_rejects_drift(tmp_path):
    from mahjong.training.data_gate import dataset_identity, source_hashes
    r = row()
    context = dict(round_wind='E', seat_wind='W', ippatsu=False, temporary_furiten=False,
                   riichi_furiten=False, wall_remaining=60, is_rinshan=False,
                   drawn_tile='5p', claimed_tile=None, discarder=None, is_chankan=False)
    r['metadata'] = json.dumps({'legal_mask_version':LEGAL_MASK_VERSION, 'legal_context':context})
    r['legal_actions'] = json.dumps([a.to_dict() for a in rebuild_legal_actions(r)])
    module = script('audit_action_feature_v2')
    results = {}
    for split in ('train', 'validation', 'test'):
        folder = tmp_path / split
        folder.mkdir()
        pq.write_table(pa.Table.from_pylist([r]), folder / 'part.parquet')
        results[split] = module.audit_split(folder, None)
    manifest = dict(dataset_version='decision-v3', feature_version='feature-v2',
                    action_schema_version='action-v2', legal_mask_version=LEGAL_MASK_VERSION,
                    config={'diagnostic_only':False}, counts={'samples':3,
                    'splits':{s:{'samples':1,'files':1} for s in results}})
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps(dict(passed=True, limit_per_split=None, splits=results,
                                    dataset_manifest=manifest, dataset_identity=dataset_identity(tmp_path),
                                    source_sha256=source_hashes())), encoding='utf-8')
    assert require_full_audit(audit, tmp_path)['passed']
    manifest['config']['diagnostic_only'] = True
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    audit.write_text(json.dumps(dict(passed=True, limit_per_split=None, splits=results,
                                    dataset_manifest=manifest, dataset_identity=dataset_identity(tmp_path),
                                    source_sha256=source_hashes())), encoding='utf-8')
    with pytest.raises(ValueError, match='non-diagnostic'):
        require_full_audit(audit, tmp_path)
    (tmp_path / 'manifest.json').write_text('{"changed":true}', encoding='utf-8')
    with pytest.raises(ValueError, match='dataset changed'):
        require_full_audit(audit, tmp_path)
