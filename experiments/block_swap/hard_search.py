"""Registered direct-GCN block search; original GAGA is evaluation only."""
import argparse,json,pathlib,sys,time,signal,random,statistics,hashlib,os
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);a=ap.parse_args();reg=read(a.registration);stage=pathlib.Path(a.registration).parent
    assert sha(__file__)==reg['source_sha256'];parent=pathlib.Path(reg['parent_stage']);p=read(parent/'registration/protocol.json')
    for path,h in reg['input_hashes'].items():assert sha(path)==h
    sys.path[:0]=[str(parent/'source/experiments/block_swap'),str(parent/'source'),p['runtime_root']]
    from helpers import verify
    verify(parent/'registration/protocol.json')
    assert read(parent/'audit_closure.json')['amended_audit_status']=='passed'
    out=stage/'execution';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    def timeout(*a):raise TimeoutError('Frozen direct-search deadline')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(3600)
    counts=dict(search_GCN=0,evaluation_GCN=0,official_GAGA_VALID=0,generations=0,updates=0,TEST_queries=0)
    result=dict(status='failed',counters=counts,registration_sha256=sha(a.registration),GPU_hours=0)
    try:
        import torch,numpy as np,scipy.sparse as sp
        from experiments.downstream_cooperation.common import check_device,make_graph,canonical,state_hash,assert_frozen,tensor_hash
        check_device()
        from experiments.local_swap.loader import CONFIG,DEFAULTS
        from experiments.local_swap.core import keyed
        from experiments.block_swap.core import exchange,check
        from modules.data_handler import DataHandler
        from course.dual_surrogate.runtime import load_classifier,gcn_logits,load_original_view
        config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config);data.target_idx_test=np.empty(0,dtype=np.int64)
        teacher=load_classifier('gcn_base',data);th=state_hash(teacher)
        original=pathlib.Path(p['parent_stage'])/'main';up=read(original/'upstream_seal.json');idx={(r['seed'],r['phase'],r['group']):r for r in up['cases']}
        oldseal=read(parent/'main/sealed_before_victim.json');(out/'plans').mkdir();sealed=[]
        def case(seed,group):
            r=idx[seed,'valid',group];f=original/r['path'];assert sha(f)==r['sha256'];return torch.load(f,map_location='cpu')
        def gcn(g,ids,phase):
            key='search_GCN' if phase=='search' else 'evaluation_GCN';counts[key]+=1;assert counts[key]<=reg['caps'][key]
            with torch.no_grad():return gcn_logits(teacher,g,ids)
        def loss(x):return float(torch.nn.functional.softplus(x.double()[:,1]-x.double()[:,0]).mean())
        print('STAGE direct_surrogate_search',flush=True)
        with (out/'search.jsonl').open('x') as stream:
            for i,entry in enumerate(oldseal['plans']):
                seed,group=entry['seed'],entry['group'];v=case(seed,group);base=v['native_adjacency'];t,c=len(v['ids']),len(v['candidate_ids'])
                native=gcn(make_graph(data,v,base),v['ids'],'search');base_loss=loss(native);gains=[];records=[]
                for action in entry['pool']:
                    mask=exchange(base,t,c,action);check(base,mask,t,c,action);z=gcn(make_graph(data,v,mask),v['ids'],'search');value=loss(z);gains.append(base_loss-value);records.append(dict(action=action,logits=z.tolist(),loss=value))
                threshold=entry['decision']['threshold'];best=max(range(len(gains)),key=lambda j:gains[j]) if gains else None
                active=best is not None and gains[best]>threshold
                ri=random.Random(keyed(seed,group,'block_swap_random_v1')).randrange(len(gains)) if gains else None
                selected=dict(native=None,direct_search=entry['pool'][best] if active else None,matched_random=entry['pool'][ri] if active else None)
                plans={}
                for arm,action in selected.items():
                    mask=exchange(base,t,c,action);check(base,mask,t,c,action);plans[arm]=mask
                path=out/'plans'/('s%d_g%d.pt'%(seed,group));torch.save(plans,path)
                sealed.append(dict(seed=seed,group=group,path=str(path.relative_to(out)),sha256=sha(path),selected=selected,active=active,threshold=threshold))
                stream.write(json.dumps(dict(seed=seed,group=group,native_logits=native.tolist(),native_loss=base_loss,options=records,gains=gains,selected=selected,threshold=threshold),allow_nan=False)+'\n');stream.flush()
                if (i+1)%32==0:print('SEARCH',i+1,192,flush=True)
        assert_frozen(teacher,th);write(out/'seal.json',dict(plans=sealed,official_GAGA_loaded=False));seal_hash=sha(out/'seal.json')
        adj=sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros()
        view=load_original_view(data,adj=adj);victim=load_classifier('gaga_victim',data);vh=state_hash(victim);rows=[]
        print('STAGE sealed_VALID',flush=True)
        with (out/'evaluation.jsonl').open('x') as stream:
            for i,entry in enumerate(sealed):
                seed,group=entry['seed'],entry['group'];v=case(seed,group);path=out/entry['path'];assert sha(path)==entry['sha256'];plans=torch.load(path,map_location='cpu')
                for arm,mask in plans.items():
                    graph=make_graph(data,v,mask);x=gcn(graph,v['ids'],'eval')
                    with torch.no_grad():seq=canonical(data,view,graph,v);counts['official_GAGA_VALID']+=1;assert counts['official_GAGA_VALID']<=576;z=victim(seq)
                    row=dict(seed=seed,group=group,arm=arm,node_ids=v['ids'].tolist(),gcn_logits=x.tolist(),gaga_logits=z.tolist(),wrong=int((z.argmax(1)==0).sum()),action=entry['selected'][arm],actual_nodes=len(v['source_ids']),actual_edges=int(mask.sum()),topology_hash=tensor_hash(mask))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
        assert_frozen(teacher,th);assert_frozen(victim,vh);assert sha(out/'seal.json')==seal_hash
        metrics=[]
        for seed in p['seeds']:
            for arm in ('native','direct_search','matched_random'):
                rr=[r for r in rows if r['seed']==seed and r['arm']==arm];n=sum(len(r['node_ids']) for r in rr);assert len(rr)==64 and n==571
                metrics.append(dict(seed=seed,arm=arm,MR=100*sum(r['wrong'] for r in rr)/n,active_cases=sum(r['action'] is not None for r in rr)))
        contrasts={}
        for control in ('native','matched_random'):
            ds=[next(r['MR'] for r in metrics if r['seed']==s and r['arm']=='direct_search')-next(r['MR'] for r in metrics if r['seed']==s and r['arm']==control) for s in p['seeds']]
            mean=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrasts[control]=dict(deltas_pp=ds,mean_pp=mean,nominal_t95=[mean-half,mean+half])
        result.update(status='completed',metrics=metrics,contrasts=contrasts,qualification_met=all(min(v['deltas_pp'])>0 and v['nominal_t95'][0]>0 for v in contrasts.values()))
    finally:
        signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
