"""Additional saved-tensor physical-edge ownership check; no model/graph loads."""
import argparse,json,pathlib,hashlib,time
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();stage=pathlib.Path(a.stage);root=stage/'main'
    start=time.monotonic();cpu=time.process_time();read=lambda p:json.loads(p.read_text())
    assert read(root/'result.json')['status']=='completed'
    import torch
    torch.set_num_threads(1)
    up=read(root/'upstream_seal.json');seal=read(root/'sealed_before_victim.json');index={}
    totals={};count=0
    for row in up['cases']:
        path=root/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        case=torch.load(path,map_location='cpu');t=len(case['ids']);c=len(case['candidate_ids']);sources=case['source_ids'];b=len(sources);N=sources[0]
        assert sources==list(range(N,N+b))
        original=case['ids'].tolist()+case['candidate_ids'].tolist()
        assert len(original)==len(set(original)) and all(0<=v<N for v in original)
        adj=case['native_adjacency'];assert adj.shape==(b,t+c+b) and bool(((adj==0)|(adj==1)).all())
        inj=adj[:,t+c:];assert torch.equal(inj,torch.tril(inj,diagonal=-1))
        assert int(adj.sum())==case['quota']['edge_budget']
        degree=adj.sum(1)+inj.sum(0)
        index[row['seed'],row['phase'],row['group']]=(adj,t,c,degree)
        key=str(row['seed'])+'_'+row['phase'];totals.setdefault(key,dict(scenarios=0,actual_nodes=0,actual_unique_nonself_edges=0))
        totals[key]['scenarios']+=1;totals[key]['actual_nodes']+=b;totals[key]['actual_unique_nonself_edges']+=int(adj.sum());count+=1
    changes={}
    for row in seal['plans']:
        path=root/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'];plans=torch.load(path,map_location='cpu')
        base,t,c,degree=index[row['seed'],'valid',row['group']]
        for arm,adj in plans.items():
            inj=adj[:,t+c:];assert torch.equal(inj,torch.tril(inj,diagonal=-1))
            assert torch.equal(degree,adj.sum(1)+inj.sum(0)) and int(adj.sum())==int(base.sum())
            changed=int((adj!=base).sum());assert changed in (0,2)
            key=str(row['seed'])+'_'+arm;changes.setdefault(key,dict(cases=0,changed_edges=0));changes[key]['cases']+=1;changes[key]['changed_edges']+=changed
    report=dict(status='passed',native_scenarios=count,final_plans=sum(v['cases'] for v in changes.values()),
        totals=totals,turnover=changes,physical_scope='Distinct original IDs, contiguous injected IDs, binary cells and strict lower-triangular injected-edge ownership prove unique nonself unordered pairs in saved masks. Injected degree includes both row-owned edges and incoming injected edges. Totals sum independent scenarios, not one graph.',
        model_calls=0,raw_dataset_loads=0,parameter_checkpoint_loads=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    with (stage/'geometry_extension.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
