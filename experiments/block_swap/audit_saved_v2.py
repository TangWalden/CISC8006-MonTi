"""Audit-only amendment: restore empty JSON feature matrices to shape (0,32).

Original audit/source/failure and all scientific outputs remain unchanged.
Run against the original frozen source root through PYTHONPATH.
"""
import argparse,json,pathlib,time,math,statistics
from helpers import verify,read,write,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    start=time.monotonic();cpu=time.process_time();p,prov=verify(a.protocol);root=pathlib.Path(a.run);result=read(root/'result.json');assert result['status']=='completed'
    import torch,numpy as np
    torch.set_num_threads(1)
    from experiments.block_swap.core import pool,exchange,check,fit,choose
    from experiments.downstream_cooperation.common import tensor_hash
    checks=0
    def ok(v):
        nonlocal checks
        assert v;checks+=1
    def loss(values):
        z=np.asarray(values,dtype=np.float64);m=z[:,1]-z[:,0]
        # Match the executed torch Softplus threshold=20 exactly.
        return float(np.where(m>20,m,np.logaddexp(0,m)).mean())
    train=[json.loads(s) for s in (root/'raw/training.jsonl').read_text().splitlines()]
    ok(len(train)==384);forwards=0;fits={}
    for seed in p['seeds']:
        rows=[r for r in train if r['seed']==seed];ok([r['group'] for r in rows]==p['train_groups']);X=[];y=[]
        for r in rows:
            ok(abs(loss(r['native_logits'])-r['native_loss'])<1e-10);forwards+=1
            for x in r['swaps']:
                ok(abs(loss(x['logits'])-x['loss'])<1e-10);ok(abs(r['native_loss']-x['loss']-x['gain'])<1e-10)
                X.append(x['features']);y.append(x['gain']);forwards+=1
        model=read(root/'models'/('s%d.json'%seed));refit=fit(X,y,p['settings']['regularization'])
        ok(np.allclose(model['weights'],refit['weights'],rtol=1e-8,atol=1e-10));fits[seed]=model
    ok(forwards==result['counters']['train_GCN'])
    parent=pathlib.Path(p['parent_stage'])/'main';up=read(parent/'upstream_seal.json');idx={(r['seed'],r['phase'],r['group']):r for r in up['cases']}
    seal=read(root/'sealed_before_victim.json');plans={};totals={}
    for r in seal['plans']:
        seed,group=r['seed'],r['group'];ref=idx[seed,'valid',group];path=parent/ref['path'];ok(sha(path)==ref['sha256']);v=torch.load(path,map_location='cpu')
        ok(v['ids'].tolist()==p['valid_members'][str(group)]);t,c=len(v['ids']),len(v['candidate_ids']);base=v['native_adjacency'];b=len(v['source_ids'])
        ok(pool(v)==r['pool']);ok(choose(fits[seed],np.asarray(r['features'],dtype=np.float64).reshape((-1,32)),r['pool'],seed,group)==r['decision'])
        path=root/r['path'];ok(sha(path)==r['sha256']);pp=torch.load(path,map_location='cpu')
        learned,random=r['selected']['block_rank'],r['selected']['matched_random'];ok((learned is None)==(random is None))
        if learned is not None:ok(learned['rows']==random['rows'] and learned['old']==random['old'])
        original_ids=v['ids'].tolist()+v['candidate_ids'].tolist();ok(len(original_ids)==len(set(original_ids)))
        ok(torch.equal(base[:,t+c:],torch.tril(base[:,t+c:],diagonal=-1)))
        degree=base.sum(1)+base[:,t+c:].sum(0)
        for arm,mask in pp.items():
            action=r['selected'][arm];check(base,mask,t,c,action);checks+=1
            ok(torch.equal(degree,mask.sum(1)+mask[:,t+c:].sum(0)));ok(int(mask.sum())==v['quota']['edge_budget'])
            plans[seed,group,arm]=(tensor_hash(mask),v['feature_sha256'],action)
            key=str(seed)+'_'+arm;totals.setdefault(key,dict(nodes=0,edges=0));totals[key]['nodes']+=b;totals[key]['edges']+=int(mask.sum())
    raw=[json.loads(s) for s in (root/'raw/evaluation.jsonl').read_text().splitlines()];ok(len(raw)==832)
    attacked=[r for r in raw if r['stage']=='attacked'];ok(len(attacked)==768);ok(len(plans)==768)
    for r in attacked:
        k=r['seed'],r['group'],r['arm'];top,feature,action=plans[k];ok((r['topology_hash'],r['feature_hash'],r['action'])==(top,feature,action))
        for name in ('gcn','gaga'):ok(int((np.asarray(r[name+'_logits']).argmax(1)==0).sum())==r[name+'_wrong'])
        ok(r['node_ids']==p['valid_members'][str(r['group'])]);ok(r['exchanges']==(0 if action is None else len(action['rows'])))
    for metric in result['metrics']:
        rr=[r for r in attacked if r['seed']==metric['seed'] and r['arm']==metric['arm']];n=sum(r['targets'] for r in rr)
        ok(n==571 and len(rr)==64)
        for name in ('gcn','gaga'):ok(abs(100*sum(r[name+'_wrong'] for r in rr)/n-metric[name+'_mr'])<1e-10)
    contrasts={}
    for control in ('native','matched_random','always_random'):
        ds=[]
        for seed in p['seeds']:
            wrong=lambda arm:sum(r['gaga_wrong'] for r in attacked if r['seed']==seed and r['arm']==arm)
            ds.append(100*(wrong('block_rank')-wrong(control))/571)
        mean=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5)
        ok(np.allclose(ds,result['contrasts'][control]['deltas_pp'],rtol=0,atol=1e-10));contrasts[control]=[mean-half,mean+half]
    qualified=all(contrasts[c][0]>0 and min(result['contrasts'][c]['deltas_pp'])>0 for c in ('native','matched_random'))
    ok(qualified==result['qualification_met'])
    report=dict(status='passed',checks=checks,training_GCN_recount=forwards,evaluation_records=len(raw),physical_totals=totals,
        model_calls=0,raw_dataset_loads=0,pretrained_parameter_loads=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
        scope='Root-authored saved fit/decision/logit and cached-tensor recount, not independent human verification. Feature descriptors are not independently regenerated.')
    write(a.output,report);print(json.dumps(report))
if __name__=='__main__':main()
