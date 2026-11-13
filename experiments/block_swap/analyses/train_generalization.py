"""Planned five-fold group-held-out TRAIN ranking diagnostic; no graph/model calls.

Fifteen diagnostic ridge solves reuse saved TRAIN action labels. They are not
new deployed arms, do not choose a validation checkpoint, and do not establish
article-independent generalization because TRAIN groups can share nodes.
"""
import argparse,json,pathlib,sys,time,hashlib
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage)
    start=time.monotonic();cpu=time.process_time();read=lambda p:json.loads(p.read_text())
    p=read(s/'registration/protocol.json');assert read(s/'main/result.json')['status']=='completed'
    sys.path.insert(0,p['source_root'])
    import numpy as np
    from experiments.local_swap.core import fit,predict
    rows=[json.loads(line) for line in (s/'main/raw/training.jsonl').read_text().splitlines()]
    fold=lambda g:int(hashlib.sha256(('block_training_cv|'+str(g)).encode()).hexdigest()[:8],16)%5
    all_rows=[];fits=0
    for seed in p['seeds']:
        rr=[r for r in rows if r['seed']==seed];assert len(rr)==128
        for k in range(5):
            fitrows=[r for r in rr if fold(r['group'])!=k];held=[r for r in rr if fold(r['group'])==k]
            X=[x['features'] for r in fitrows for x in r['swaps']];y=[x['gain'] for r in fitrows for x in r['swaps']]
            model=fit(X,y,.1);fits+=1
            for r in held:
                if not r['swaps']:
                    all_rows.append(dict(seed=seed,group=r['group'],fold=k,actions=0,active=False,selected_gain=0.,matched_random_expected_gain=0.));continue
                pred=predict(model,[x['features'] for x in r['swaps']]);true=np.array([x['gain'] for x in r['swaps']]);j=int(np.argmax(pred));active=float(pred[j])>model['threshold']
                all_rows.append(dict(seed=seed,group=r['group'],fold=k,actions=len(true),active=active,
                    selected_gain=float(true[j]) if active else 0.,matched_random_expected_gain=float(true.mean()) if active else 0.,
                    best_measured_gain=float(true.max()),squared_error=float(((pred-true)**2).sum()),target_squared=float((true**2).sum())))
    summary=[]
    for seed in p['seeds']:
        rr=[r for r in all_rows if r['seed']==seed];assert len(rr)==128
        actions=sum(r['actions'] for r in rr);error=sum(r.get('squared_error',0) for r in rr);zero=sum(r.get('target_squared',0) for r in rr)
        summary.append(dict(seed=seed,groups=128,eligible_groups=sum(r['actions']>0 for r in rr),active_groups=sum(r['active'] for r in rr),
            action_rows=actions,heldout_rmse=(error/actions)**.5,zero_prediction_rmse=(zero/actions)**.5,
            selected_GCN_gain_per_all_group=sum(r['selected_gain'] for r in rr)/128,
            matched_random_expected_GCN_gain_per_all_group=sum(r['matched_random_expected_gain'] for r in rr)/128,
            rank_beats_random_expectation_groups=sum(r['selected_gain']>r['matched_random_expected_gain']+1e-6 for r in rr),
            rank_loses_random_expectation_groups=sum(r['selected_gain']<r['matched_random_expected_gain']-1e-6 for r in rr)))
    result=dict(status='completed',diagnostic_ridge_fits=fits,summary=summary,graph_classifier_forwards=0,new_graphs=0,TEST_queries=0,GPU_hours=0,
        wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        scope='Group-held-out TRAIN only; fixed five folds, no parameter grid, no VALID labels. GCN finite-difference fit diagnosis, not GAGA efficacy or independent-news confirmation.')
    for name,value in [('train_generalization.json',result),('train_generalization_cases.json',all_rows)]:
        with (s/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
