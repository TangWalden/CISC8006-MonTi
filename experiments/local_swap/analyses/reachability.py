"""Post-hoc, model-free reachability diagnosis and TRAIN-only block feasibility."""
import argparse,hashlib,json,pathlib,time,sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args()
    stage=pathlib.Path(a.stage);out=stage/'reachability_v1';out.mkdir(exist_ok=False)
    start=time.monotonic();cpu=time.process_time()
    read=lambda p:json.loads(p.read_text())
    p=read(stage/'registration/protocol.json');root=stage/'main';rt=pathlib.Path(p['runtime_root'])
    assert read(root/'summary.json')['status']=='completed'
    refs={r['path']:r['sha256'] for r in p['input_refs']}
    def checked(name):
        path=rt/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==refs[name];return path
    import torch,dgl,numpy as np,scipy.sparse as sp
    torch.set_num_threads(4)
    graph=dgl.load_graphs(str(checked('Datasets/GossipCop/GossipCop_0.05.bin')))[0][0]
    # Same incoming adjacency convention as DataHandler; self-loops excluded.
    adj=sp.csr_matrix(graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros()
    adj2=sp.load_npz(checked('Datasets/GossipCop/data_split/gossipcop_0.05_gaga_adj2_0.npz')).tocsr()
    up=read(root/'upstream_seal.json');seal=read(root/'sealed_before_victim.json')
    plan_index={(r['seed'],r['group']):r for r in seal['plans']}
    raw=[json.loads(s) for s in (root/'raw/evaluation.jsonl').read_text().splitlines()]
    predictions={(r['seed'],r['group'],r['arm']):r['gaga_logits'] for r in raw if r['stage']=='attacked'}
    rows=[];train=[]
    for index,ref in enumerate(up['cases']):
        path=root/ref['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ref['sha256']
        case=torch.load(path,map_location='cpu');ids=case['ids'].numpy();cids=case['candidate_ids'].numpy()
        t,c,b=len(ids),len(cids),len(case['source_ids']);base=case['native_adjacency'].numpy().astype(np.int64)
        if ref['phase']=='train':
            block=base[:,t:t+c];pairs=[]
            for old in range(c):
                source=np.flatnonzero(block[:,old]);k=len(source)
                if not k:continue
                valid=np.flatnonzero(block[source].sum(0)==0)
                pairs.extend((old,int(new),k) for new in valid)
            train.append(dict(seed=ref['seed'],group=ref['group'],targets=t,aux=b,candidates=c,
                feasible_pairs=len(pairs),multirow_pairs=sum(k>1 for _,_,k in pairs),
                min_cost=min((k for _,_,k in pairs),default=0),max_cost=max((k for _,_,k in pairs),default=0),
                used_columns=int((block.sum(0)>0).sum()),fully_absent_columns=int((block.sum(0)==0).sum())))
            continue
        orig=np.concatenate((ids,cids));n=t+c+b
        original=np.zeros((n,n),dtype=np.int64);original[:t+c,:t+c]=adj[orig][:,orig].toarray()!=0
        old1=np.zeros((t,n),dtype=bool);old1[:,:t+c]=adj[ids][:,orig].toarray()!=0
        old2=np.zeros((t,n),dtype=bool);old2[:,:t+c]=adj2[ids][:,orig].toarray()!=0
        def masks(mask):
            a=original.copy();a[t+c:,:]+=mask;a[:,t+c:]+=mask.T
            assert np.isin(a,[0,1]).all()
            one=a[:t].astype(bool);two=(a[:t]@a)>0;two[np.arange(t),np.arange(t)]=False
            return one&~old1,two&~old2
        native=masks(base);entry=plan_index[ref['seed'],ref['group']];path=root/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];plans=torch.load(path,map_location='cpu')
        for arm in ('local_rank','matched_random','always_random'):
            mask=plans[arm].numpy().astype(np.int64);active=bool((mask!=base).any());changed=masks(mask)
            first=native[0]!=changed[0];second=native[1]!=changed[1]
            same=~(first.any(1)|second.any(1));a0=np.asarray(predictions[ref['seed'],ref['group'],'native']);a1=np.asarray(predictions[ref['seed'],ref['group'],arm])
            rows.append(dict(seed=ref['seed'],group=ref['group'],arm=arm,active=active,targets=t,
                exact_same_membership_targets=int(same.sum()),changed_onehop_cells=int(first.sum()),changed_twohop_cells=int(second.sum()),
                changed_candidate_twohop_cells=int(second[:,t:t+c].sum()),changed_aux_twohop_cells=int(second[:,t+c:].sum()),
                identical_all_memberships=bool(same.all()),max_logit_difference=float(np.abs(a1-a0).max()),
                class_changes=int((a1.argmax(1)!=a0.argmax(1)).sum()),
                invariant_membership_max_logit_difference=float(np.abs(a1[same]-a0[same]).max()) if same.any() else None))
        if (index+1)%32==0:print('CASES',index+1,flush=True)
    summary={}
    for arm in ('local_rank','matched_random','always_random'):
        rr=[r for r in rows if r['arm']==arm and r['active']]
        summary[arm]=dict(active_cases=len(rr),active_occurrences=sum(r['targets'] for r in rr),
            invariant_cases=sum(r['identical_all_memberships'] for r in rr),invariant_occurrences=sum(r['exact_same_membership_targets'] for r in rr),
            class_changes=sum(r['class_changes'] for r in rr),
            max_invariant_logit_difference=max((r['invariant_membership_max_logit_difference'] or 0 for r in rr),default=0))
    report=dict(status='completed',scope='Post-hoc structural diagnosis, not a new efficacy test. Exact canonical added-neighbor membership is reconstructed from the frozen original graph and saved masks. With fixed features/group labels, unchanged memberships imply unchanged mathematical aggregate inputs; floating-point ordering is not asserted identical.',
        summary=summary,train_feasibility=dict(cases=len(train),feasible_cases=sum(r['feasible_pairs']>0 for r in train),multirow_feasible_cases=sum(r['multirow_pairs']>0 for r in train)),
        raw_graph_loads=1,model_loads=0,model_calls=0,new_generation=0,TEST_queries=0,GPU_hours=0,
        process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
    for name,value in [('summary.json',report),('valid_membership.json',rows),('train_feasibility.json',train)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
