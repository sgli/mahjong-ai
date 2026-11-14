import hashlib, json, math
from pathlib import Path
import torch, yaml

cfg_path = Path('configs/train_ppo_v4_pure_rank_lambda099_audit_seed46.yaml')
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
root = Path(cfg['checkpoint_dir'])
bc = Path(cfg['bc_checkpoint'])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
latest = torch.load(root/'latest.pt', map_location='cpu', weights_only=False)
best = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
snap = torch.load(root/'epoch_10.pt', map_location='cpu', weights_only=False)
hist = latest['metrics_history']
assert len(hist) == 10 and sum(int(x['learner_transitions']) for x in hist) == 100000
assert all(int(x['learner_transitions']) == 10000 and float(x['illegal_rate']) == 0 for x in hist)
assert all(math.isfinite(float(x['policy_loss'])) and math.isfinite(float(x['value_loss'])) and int(x['optimizer_updates']) > 0 for x in hist)
assert latest['epoch'] == best['epoch'] == snap['epoch'] == 10
assert all(torch.equal(v, best['model_state_dict'][k]) and torch.equal(v, snap['model_state_dict'][k]) for k,v in latest['model_state_dict'].items())
assert latest['config']['training']['lam'] == cfg['training']['lam'] == 0.99
assert latest['config']['training']['reward_score_scale'] == 0.0
assert latest['reward_version'] == 'reward-v1-pure-placement-v1'
assert latest['config']['provenance']['bc_checkpoint_sha256'] == sha(bc)
assert latest['config']['training']['seed'] == 46 and latest['config']['training']['env_seed'] == 660000
out = {'status':'PASS','config_sha256':sha(cfg_path),'checkpoint_sha256':sha(root/'best.pt'),'learner_transitions':100000,
       'environment_steps':sum(int(x['environment_steps']) for x in hist),'complete_games':sum(int(x['complete_games']) for x in hist),
       'optimizer_updates':sum(int(x['optimizer_updates']) for x in hist),
       'kl_early_stop_epochs':[x['epoch'] for x in hist if x.get('target_kl_stopped')],
       'illegal_actions':0,'lambda':0.99,'seed':46}
(root/'training_verification.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps(out, indent=2))
