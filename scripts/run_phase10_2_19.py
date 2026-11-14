from pathlib import Path
import json, subprocess, sys, datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/ppo_v4_rank_lambda099_audit'
OUT.mkdir(parents=True, exist_ok=True)
LOG = (OUT/'pipeline.log').open('a',encoding='utf-8',buffering=1)

def run(label,*args):
    stamp=datetime.datetime.now().astimezone().isoformat()
    print(f'{stamp} START {label}: {" ".join(map(str,args))}',file=LOG,flush=True)
    p=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=LOG,stderr=subprocess.STDOUT)
    print(f'{datetime.datetime.now().astimezone().isoformat()} END {label} exit={p.returncode}',file=LOG,flush=True)
    if p.returncode: raise SystemExit(p.returncode)

def benchmark(label,config_path):
    cfg=json.loads((ROOT/config_path).read_text(encoding='utf-8'))
    run(label,'scripts/benchmark_paired_ppo.py',
        '--candidate-checkpoint',cfg['candidate_checkpoint'],
        '--mirror-checkpoint',cfg['mirror_checkpoint'],
        '--opponent-checkpoint',cfg['opponent_checkpoint'],
        '--games-per-arm',str(cfg['games_per_arm']),
        '--seed',str(cfg['seed']),
        '--num-workers',str(cfg['num_workers']),
        '--bootstrap-seed',str(cfg['bootstrap_seed']),
        '--bootstrap-replicates',str(cfg['bootstrap_replicates']),
        '--mirror-sanity-mode',cfg['mirror_sanity_mode'],
        '--stage-id',cfg['stage_id'],
        '--candidate-version',cfg['candidate_version'],
        '--out',cfg['out'])

run('pilot','scripts/train_ppo.py','--config','configs/train_ppo_v4_pure_rank_lambda099_audit_seed46_pilot.yaml')
run('verify-pilot','scripts/verify_phase10_2_19_pilot.py')
run('train-100k','scripts/train_ppo.py','--config','configs/train_ppo_v4_pure_rank_lambda099_audit_seed46.yaml')
run('verify-training','scripts/verify_phase10_2_19_training.py')
benchmark('screen-2000','configs/benchmark_phase10_2_19_lambda099_screen.json')
run('verify-screen','scripts/verify_phase10_2_19_eval.py','screen')
screen=json.loads((OUT/'screen_2000_seed1140000_v1/independent_verification.json').read_text(encoding='utf-8'))
if screen['screen_pass']:
    print('SCREEN_GO: launching preregistered 10k/arm strength evaluation',file=LOG,flush=True)
    benchmark('strength-10000','configs/benchmark_phase10_2_19_lambda099_strength.json')
    run('verify-strength','scripts/verify_phase10_2_19_eval.py','strength')
else:
    print('SCREEN_NO_GO: formal strength evaluation not launched',file=LOG,flush=True)
print(f'{datetime.datetime.now().astimezone().isoformat()} PIPELINE COMPLETE',file=LOG,flush=True)
