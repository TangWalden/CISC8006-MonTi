"""Freeze source, complete parent caches and three-repeat protocol before calls."""
import argparse,datetime,pathlib,subprocess
from helpers import read,write,sha,ROOT
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);rt=s.parent.parent
    parent=rt/'runs/local_swap_20261113_attempt1';q=read(parent/'registration/protocol.json')
    assert read(parent/'driver_receipt.json')['status']=='completed' and read(parent/'audit.json')['status']=='passed'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==q['runtime_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
    manifest=read(ROOT/'source_manifest.json')
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h
    refs=list(q['input_refs']);up=read(parent/'main/upstream_seal.json')
    for name in ['registration/protocol.json','main/upstream_seal.json','main/summary.json','audit.json','reachability_v1/summary.json']:
        f=parent/name;refs.append(dict(path=f.relative_to(rt).as_posix(),sha256=sha(f)))
    for r in up['cases']:
        f=parent/'main'/r['path'];assert sha(f)==r['sha256'];refs.append(dict(path=f.relative_to(rt).as_posix(),sha256=r['sha256']))
    for r in refs:assert sha(rt/r['path'])==r['sha256']
    p=dict(schema='monti_block_swap_v1',status='frozen',owner_execution_authorized=True,final_TEST_allowed=False,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),runtime_root=str(rt),runtime_commit=q['runtime_commit'],
        parent_stage=str(parent),source_manifest=manifest,input_refs=refs,
        seeds=q['seeds'],train_groups=q['train_groups'],valid_groups=q['valid_groups'],train_members=q['train_members'],valid_members=q['valid_members'],
        arms=['native','block_rank','matched_random','always_random'],
        settings=dict(pool_size=8,regularization=.1,features='Same 32 v1 features, mean over donor rows',
            donor='Uniform eligible donor column, preferring >=2 incident rows if available; chosen without outcomes.',
            recipients='At most 8 uniformly sampled candidates absent in every donor row; same pool for learned/random.',
            policy='One whole donor-column replacement if largest predicted TRAIN-GCN loss gain exceeds max(1e-6,.1*TRAIN gain RMS); else identity.',
            controls='Matched random shares exact donor, removed rows, activation and number of additions; always random auxiliary only.',
            preserved='Author checkpoint outputs/features/target edges/injected edges/individual physical injected degrees/candidate order/total budget.',
            learning='Three float64 ridge solves using only TRAIN hard-graph finite differences; no ST gradient or victim supervision.'),
        resources=dict(preflight_seconds=300,main_seconds=7200,CPU_threads=4,GPU_hours=0,API_calls=0,automatic_retry=False),
        caps=dict(generations=0,train_GCN=3456,evaluation_GCN=832,official_GAGA_VALID=832,ridge_fits=3,TEST_queries=0),
        qualification=q['qualification'],
        limitations=['Reused development; prior outcomes influenced design, not untouched confirmation.',
            'One author upstream checkpoint and three cached noise repeats, not three independently trained upstreams.',
            'Block vs single-edge studies differ intervention scale; primary learning contrast is only against exact-dose block random.',
            'Unchanged structural budget is not equal training computation. Nominal three-repeat intervals do not correct adaptive search.'],
        cost_snapshot=q['ledger_snapshot'])
    p['qualification']['primary']='block_rank-minus-matched_random official GAGA occurrence MR'
    p['qualification']['secondary']='block_rank-minus-native'
    (s/'registration').mkdir(exist_ok=False);write(s/'registration/protocol.json',p)
    print(__import__('json').dumps(dict(status='frozen',protocol_sha256=sha(s/'registration/protocol.json'),inputs=len(refs),model_calls=0)))
if __name__=='__main__':main()
