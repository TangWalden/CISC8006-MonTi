"""Saved logits/fit/geometry recount; no dataset or model parameter loads."""
import argparse,hashlib,json,math,pathlib,time
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--run',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    start=time.monotonic();cpu=time.process_time();p=read(a.protocol);root=pathlib.Path(a.run);result=read(root/'result.json')
    assert result['status']=='completed' and result['provenance']['protocol_sha256']==sha(a.protocol)
    import numpy as np,torch,random
    from experiments.local_swap.core import proposal_pool
    from experiments.downstream_cooperation.common import tensor_hash
    checks=0;hashes={}
    def hashed(name,wanted):
        nonlocal checks
        assert sha(root/name)==wanted,name;hashes[name]=wanted;checks+=1
    seal=read(root/'sealed_before_victim.json');up=read(root/'upstream_seal.json');hashed('upstream_seal.json',seal['upstream_seal_sha256'])
    caches={(r['seed'],r['phase'],r['group']):r for r in up['cases']};assert len(caches)==576
    for r in caches.values():hashed(r['path'],r['sha256'])
    fits={s:([],[]) for s in p['seeds']};train_calls=0
    for line in (root/'raw/training.jsonl').read_text().splitlines():
        r=json.loads(line);assert r['node_ids']==p['train_members'][str(r['group'])];base=np.asarray(r['native_logits'])
        bl=float(np.logaddexp(0.,base[:,1]-base[:,0]).mean());assert abs(bl-r['native_loss'])<1e-10;train_calls+=1
        entry=caches[r['seed'],'train',r['group']];case=torch.load(root/entry['path'],map_location='cpu')
        assert [v['triple'] for v in r['swaps']]==case['pool'];assert len(case['pool'])==entry['pool_count']
        for v in r['swaps']:
            z=np.asarray(v['logits']);loss=float(np.logaddexp(0.,z[:,1]-z[:,0]).mean());gain=bl-loss
            assert abs(loss-v['loss'])<1e-10 and abs(gain-v['gain'])<1e-10
            fits[r['seed']][0].append(v['features']);fits[r['seed']][1].append(gain);train_calls+=1;checks+=1
    models={}
    for row in seal['models']:
        hashed(row['path'],row['sha256']);s=row['seed'];m=read(root/row['path'])['model'];models[s]=m
        X=np.asarray(fits[s][0]);y=np.asarray(fits[s][1]);scale=np.maximum(np.sqrt((X*X).mean(0)),1e-8);ys=max(float(np.sqrt((y*y).mean())),1e-12);z=X/scale
        w=np.linalg.solve(z.T@z/len(y)+.1*np.eye(32),z.T@(y/ys)/len(y))
        np.testing.assert_allclose(w,m['weights'],rtol=1e-8,atol=1e-10)
        np.testing.assert_allclose(scale,m['feature_rms'],rtol=1e-10,atol=1e-12)
        assert abs(m['threshold']-max(1e-6,.1*ys))<1e-12;checks+=3
    plan_index={}
    for row in seal['plans']:
        s,g=row['seed'],row['group'];hashed(row['path'],row['sha256']);plans=torch.load(root/row['path'],map_location='cpu')
        entry=caches[s,'valid',g];case=torch.load(root/entry['path'],map_location='cpu');base=case['native_adjacency'];t=len(case['ids']);c=len(case['candidate_ids'])
        assert case['ids'].tolist()==p['valid_members'][str(g)]
        pool=proposal_pool(base[:,t:t+c].tolist(),s,g,8);assert pool==row['pool']==case['pool']
        X=np.asarray(row['features']).reshape((-1,32));m=models[s];scores=X/np.asarray(m['feature_rms'])@np.asarray(m['weights'])*m['gain_rms']
        chosen=int(scores.argmax()) if len(pool) else None;active=chosen is not None and float(scores[chosen])>m['threshold']
        key=int(hashlib.sha256(repr((s,g,'local_swap_random_v1')).encode('ascii')).hexdigest()[:16],16)
        ri=random.Random(key).randrange(len(pool)) if pool else None
        wanted=dict(native=None,local_rank=pool[chosen] if active else None,matched_random=pool[ri] if active else None,always_random=pool[ri] if pool else None)
        assert row['selected']==wanted;np.testing.assert_allclose(scores,row['decision']['predicted_gains'],rtol=1e-9,atol=1e-12)
        for arm,v in plans.items():
            exp=base.clone();swap=wanted[arm]
            if swap is not None:
                r,old,new=swap;assert base[r,t+old]==1 and base[r,t+new]==0
                exp[r,t+old]=0;exp[r,t+new]=1
            assert torch.equal(exp,v) and torch.equal(base.sum(1),v.sum(1))
            assert torch.equal(base[:,:t],v[:,:t]) and torch.equal(base[:,t+c:],v[:,t+c:])
            assert int((base!=v).sum())==(0 if swap is None else 2)
            assert tensor_hash(v)==row['topology_hashes'][arm]
            assert tensor_hash(case['x_aux'])==row['feature_hash']==case['feature_sha256'];checks+=6
        plan_index[s,g]=row
    assert len(plan_index)==192
    clean={};metrics={(s,arm):dict(n=0,gaga_wrong=0,gcn_wrong=0,cases=0,swaps=0) for s in p['seeds'] for arm in p['arms']}
    raw_count=0;seen=set()
    for line in (root/'raw/evaluation.jsonl').read_text().splitlines():
        r=json.loads(line);g=r['group'];raw_count+=1;assert r['node_ids']==p['valid_members'][str(g)]
        pred={name:np.asarray(r[name+'_logits']).argmax(1) for name in ('gcn','gaga')}
        if r['stage']=='clean':assert g not in clean;clean[g]=pred;continue
        s,arm=r['seed'],r['arm'];assert (s,g,arm) not in seen;seen.add((s,g,arm));m=metrics[s,arm]
        assert r['topology_hash']==plan_index[s,g]['topology_hashes'][arm] and r['swap']==plan_index[s,g]['selected'][arm]
        for name in ('gcn','gaga'):
            cw=clean[g][name]!=1;aw=pred[name]!=1
            expected=dict(occurrences=len(aw),clean_wrong=int(cw.sum()),attacked_wrong=int(aw.sum()),new_wrong=int((~cw&aw).sum()),recoveries=int((cw&~aw).sum()))
            assert expected==r[name];m[name+'_wrong']+=int(aw.sum());checks+=1
        m['n']+=len(r['node_ids']);m['cases']+=1;m['swaps']+=r['swap'] is not None
    for row in result['metrics']:
        m=metrics[row['seed'],row['arm']];assert m['n']==571 and m['cases']==64
        assert abs(row['gaga_mr']-100*m['gaga_wrong']/571)<1e-10 and abs(row['gcn_mr']-100*m['gcn_wrong']/571)<1e-10
        assert row['exchanges']==m['swaps'];checks+=3
    assert len(clean)==64 and raw_count==832 and len(seen)==768
    assert train_calls==result['counters']['train_GCN'] and result['counters']['official_GAGA_VALID']==raw_count==result['counters']['evaluation_GCN']
    assert result['counters']['TEST_queries']==result['counters']['API_calls']==result['counters']['upstream_updates']==0
    report=dict(status='passed',checks=checks,hashes_checked=len(hashes),evaluation_records=raw_count,training_GCN_recount=train_calls,
        model_calls=0,pretrained_parameter_loads=0,raw_graph_loads=0,GPU_hours=0,process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        scope='Root-authored saved fit/logit and cached-tensor geometry recount, not independent human verification. Original graph descriptors not independently regenerated; on-execution physical graph guards supply that scope.',
        checked_hashes=hashes)
    with pathlib.Path(a.output).open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checked_hashes'}))
if __name__=='__main__':main()
