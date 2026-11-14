import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BC = ROOT / 'experiments/bc_v4/checkpoints_20w/best.pt'
CANDIDATE = ROOT / 'experiments/ppo_v4_rank_lambda099_audit/seed46_100k/best.pt'
SEEDS = {'screen': (1140000, 2000, 10400), 'strength': (1150000, 10000, 10401)}
REPS = 10000

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('screen','strength'))
    args = ap.parse_args()
    start, n, bootstrap_seed = SEEDS[args.kind]
    out = ROOT / ('experiments/ppo_v4_rank_lambda099_audit/screen_2000_seed1140000_v1' if args.kind == 'screen'
                  else 'experiments/ppo_v4_rank_lambda099_audit/strength_10000_seed1150000_v1')
    meta = json.loads((out/'run_metadata.json').read_text(encoding='utf-8'))
    result = json.loads((out/'result.json').read_text(encoding='utf-8'))
    c = json.loads((out/'candidate_arm.json').read_text(encoding='utf-8'))['raw_games']
    m = json.loads((out/'mirror_arm.json').read_text(encoding='utf-8'))['raw_games']
    assert meta['formal_preregistered_run'] is True
    assert meta['seed_start'] == start and meta['seed_end_inclusive'] == start+n-1
    assert meta['games_per_arm'] == n and meta['bootstrap_seed'] == bootstrap_seed and meta['bootstrap_replicates'] == REPS
    assert meta['mirror_sanity_mode'] == 'normal99-v2'
    assert meta['candidate_sha256'] == sha(CANDIDATE)
    assert meta['mirror_sha256'] == meta['opponent_sha256'] == sha(BC)
    assert len(c) == len(m) == n
    cm, mm = ({r['seed']:r for r in rows} for rows in (c,m))
    expected = set(range(start,start+n))
    assert set(cm) == set(mm) == expected and len(cm) == len(mm) == n
    d = np.empty(n,dtype=np.float64); cr=np.empty(n,dtype=np.int8); mr=np.empty(n,dtype=np.int8)
    for i,seed in enumerate(range(start,start+n)):
        a,b=cm[seed],mm[seed]
        assert a['game_index']==b['game_index']==i and a['seat']==b['seat']==i%4
        for row in (a,b):
            assert row['completed'] and row['error'] is None and row['illegal_actions']==0
            assert sorted(row['final_ranks'])==[0,1,2,3]
        cr[i]=int(a['final_ranks'][i%4]);mr[i]=int(b['final_ranks'][i%4]);d[i]=cr[i]-mr[i]
    with (out/'paired_rank_differences.csv').open(newline='',encoding='utf-8') as f:
        pairs=list(csv.DictReader(f))
    assert len(pairs)==n
    for i,row in enumerate(pairs):
        assert tuple(int(row[k]) for k in ('seed','seat','candidate_rank','mirror_rank','rank_difference')) == (start+i,i%4,int(cr[i]),int(mr[i]),int(d[i]))
    rng=np.random.Generator(np.random.PCG64(bootstrap_seed));boot=np.empty(REPS,dtype=np.float64)
    for begin in range(0,REPS,100):
        end=min(begin+100,REPS);idx=rng.integers(0,n,size=(end-begin,n),dtype=np.int32);boot[begin:end]=d[idx].mean(axis=1)
    ci=np.quantile(boot,[.025,.975],method='linear').tolist()
    mean=float(d.mean()); mirror_mean=float(mr.mean())
    tol=2.5758293035489004*(1.25/n)**0.5
    mirror_ok=abs(mirror_mean-1.5)<=tol
    passed=mean<=-.020 and ci[1]<0 and mirror_ok
    analysis=result['analysis']
    assert np.allclose(ci,analysis['paired_rank_difference_bootstrap_ci95'],atol=1e-12)
    assert abs(mean-analysis['paired_rank_difference_mean'])<1e-12
    assert mirror_ok==analysis['mirror_sanity_passed'] and result['candidate_sha256']==meta['candidate_sha256']
    assert result['mirror_sha256']==meta['mirror_sha256']
    report={'status':'PASS','kind':args.kind,'games_per_arm':n,'seed_range':[start,start+n-1],
            'candidate_mean_rank':float(cr.mean()),'mirror_mean_rank':mirror_mean,'paired_difference_mean':mean,
            'ci95':ci,'mirror_tolerance':tol,'mirror_sanity_passed':mirror_ok,
            'illegal_actions':0,'screen_pass' if args.kind=='screen' else 'strength_gate_pass':passed,
            'decision':'GO' if passed else 'NO_GO'}
    (out/'independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
