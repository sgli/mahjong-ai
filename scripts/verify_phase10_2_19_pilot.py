import json, math
from pathlib import Path
import torch, yaml

cfg_path = Path('configs/train_ppo_v4_pure_rank_lambda099_audit_seed46_pilot.yaml')
cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
root = Path(cfg['checkpoint_dir'])
latest = torch.load(root / 'latest.pt', map_location='cpu', weights_only=False)
hist = latest['metrics_history']
assert len(hist) == 2 and sum(int(x['learner_transitions']) for x in hist) == 4096
assert all(float(x['illegal_rate']) == 0 and math.isfinite(float(x['policy_loss'])) and math.isfinite(float(x['value_loss'])) for x in hist)
assert latest['epoch'] == 2 and latest['config']['training']['lam'] == 0.99
assert latest['config']['training']['seed'] == 46 and latest['config']['training']['reward_score_scale'] == 0.0
assert latest['reward_version'] == 'reward-v1-pure-placement-v1'
print(json.dumps({'status':'PASS','learner_transitions':4096,'illegal_actions':0,'lambda':0.99,'seed':46,'checkpoint_dir':str(root)}, indent=2))
