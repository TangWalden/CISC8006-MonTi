"""Single fixed author checkpoint, hard-swap TRAIN supervision, sealed VALID."""
import argparse,datetime,json,pathlib,signal,statistics,time,traceback
from helpers import verify,read,write,sha,capture

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--preflight',action='store_true');ap.add_argument('--preflight-result');args=ap.parse_args()
    start,cpu=time.monotonic(),time.process_time();p,prov=verify(args.protocol)
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False)
    counts=dict(generations=0,train_GCN=0,evaluation_GCN=0,official_GAGA_VALID=0,ridge_fits=0,upstream_updates=0,autograd_calls=0,API_calls=0,TEST_queries=0)
    result=dict(status='failed',provenance=prov,counters=counts,GPU_hours=0)
    limit=p['resources']['preflight_seconds' if args.preflight else 'main_seconds'];deadline=start+limit
    def stop(*a):raise TimeoutError('Frozen local-swap time limit or termination.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(limit)
    def check():
        if time.monotonic()>=deadline:stop()
    try:
        from experiments.downstream_cooperation.common import check_device,make_graph,canonical,tensor_hash,state_hash,assert_frozen
        result['environment']=check_device()
        import torch,numpy as np
        import scipy.sparse as sp
        from course.dual_surrogate.runtime import gcn_logits,load_classifier,load_original_view,transitions
        from experiments.local_swap.loader import load_engine
        from experiments.local_swap.core import exchange,assert_exchange,features,fit,choose
        if args.preflight:
            import unittest
            from experiments.local_swap.checks import Checks
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
            checked=unittest.TextTestRunner(verbosity=2).run(suite)
            if not checked.wasSuccessful():raise RuntimeError('Synthetic contracts failed.')
            result['synthetic_tests']=checked.testsRun
            engine,data,receipt,frozen=load_engine(p,0,out,deadline);checks=[]
            for group in p['train_groups'][:2]:
                case,original=capture(engine,data,0,group,'train',p);counts['generations']+=1
                rebuilt=make_graph(data,case,case['native_adjacency'])
                with torch.no_grad():
                    counts['train_GCN']+=2
                    a=gcn_logits(engine.teachers['gcn_base'],original,case['ids'])
                    b=gcn_logits(engine.teachers['gcn_base'],rebuilt,case['ids'])
                error=float((a-b).abs().max())
                if error>2e-5 or not torch.equal(a.argmax(1),b.argmax(1)):raise RuntimeError('Native reconstruction changes the measured classifier output.')
                adj=sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros()
                f=features(case,adj)
                for triple in case['pool'][:1]:
                    changed=exchange(case['native_adjacency'],len(case['ids']),len(case['candidate_ids']),triple)
                    assert_exchange(case['native_adjacency'],changed,len(case['ids']),len(case['candidate_ids']),triple)
                    graph=make_graph(data,case,changed)
                    with torch.no_grad():counts['train_GCN']+=1;gcn_logits(engine.teachers['gcn_base'],graph,case['ids'])
                    if not torch.equal(original.in_degrees()[case['source_ids']],graph.in_degrees()[case['source_ids']]):raise RuntimeError('Injected physical degree changed.')
                checks.append(dict(group=group,features_shape=list(f.shape),legal_swaps=len(case['pool']),max_logit_difference=error,upstream_unchanged=True))
            assert_frozen(engine.model,frozen['upstream']);assert_frozen(engine.teachers['gcn_base'],frozen['teacher'])
            result.update(status='passed',checks=checks,author_receipt=receipt)
            return
        if not args.preflight_result:raise RuntimeError('Preflight receipt required.')
        gate=read(args.preflight_result)
        if gate['status']!='passed' or gate['provenance']!=prov:raise RuntimeError('Same-protocol preflight required.')
        for name in ('cache','models','plans','raw'):(out/name).mkdir()
        cache=[];receipts=[]
        def cache_path(seed,phase,group):return out/'cache'/('s%d_%s_g%d.pt'%(seed,phase,group))
        def load_case(seed,phase,group):
            path=cache_path(seed,phase,group);entry=cache_index[(seed,phase,group)]
            if sha(path)!=entry['sha256']:raise RuntimeError('Frozen upstream cache changed.')
            return torch.load(path,map_location='cpu')
        print('STAGE upstream_capture',flush=True)
        for seed in p['seeds']:
            engine,data,receipt,frozen=load_engine(p,seed,out,deadline)
            for phase in ('train','valid'):
                for i,group in enumerate(p[phase+'_groups']):
                    check();case,graph=capture(engine,data,seed,group,phase,p);counts['generations']+=1;del graph
                    path=cache_path(seed,phase,group);torch.save(case,path)
                    cache.append(dict(seed=seed,phase=phase,group=group,path=str(path.relative_to(out)),sha256=sha(path),targets=len(case['ids']),pool_count=len(case['pool'])))
                    if (i+1)%32==0:print('CAPTURE',seed,phase,i+1,flush=True)
            assert_frozen(engine.model,frozen['upstream']);assert_frozen(engine.teachers['gcn_base'],frozen['teacher']);receipts.append(receipt)
        if counts['generations']!=p['caps']['main_generations']:raise RuntimeError('Incomplete capture cohort.')
        write(out/'upstream_seal.json',dict(provenance=prov,cases=cache,receipts=receipts,official_GAGA_loaded=False))
        cache_index={(r['seed'],r['phase'],r['group']):r for r in cache}
        teacher=engine.teachers['gcn_base'];teacher_hash=state_hash(teacher)
        adjacency=sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr();adjacency.setdiag(0);adjacency.eliminate_zeros()
        def gcn(graph,ids,phase):
            check();key='train_GCN' if phase=='train' else 'evaluation_GCN';counts[key]+=1
            if counts[key]>p['caps'][key]:raise RuntimeError('GCN query cap exceeded.')
            with torch.no_grad():return gcn_logits(teacher,graph,ids)
        def loss(logits):return float(torch.nn.functional.softplus(logits.double()[:,1]-logits.double()[:,0]).mean())
        print('STAGE train_hard_swap_supervision',flush=True)
        models=[];sealed=[]
        with (out/'raw'/'training.jsonl').open('x',encoding='utf-8') as stream:
            for seed in p['seeds']:
                X=[];y=[]
                for i,group in enumerate(p['train_groups']):
                    check();case=load_case(seed,'train',group);base=case['native_adjacency'];f=features(case,adjacency)
                    graph=make_graph(data,case,base);native=gcn(graph,case['ids'],'train');base_loss=loss(native);del graph
                    row=dict(seed=seed,group=group,node_ids=case['ids'].tolist(),native_logits=native.tolist(),native_loss=base_loss,swaps=[])
                    for j,triple in enumerate(case['pool']):
                        changed=exchange(base,len(case['ids']),len(case['candidate_ids']),triple)
                        assert_exchange(base,changed,len(case['ids']),len(case['candidate_ids']),triple)
                        graph=make_graph(data,case,changed);logits=gcn(graph,case['ids'],'train');value=loss(logits);delta=base_loss-value
                        X.append(f[j]);y.append(delta);row['swaps'].append(dict(triple=triple,features=f[j].tolist(),logits=logits.tolist(),loss=value,gain=delta));del graph
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if (i+1)%32==0:print('LABELS',seed,i+1,flush=True)
                model=fit(X,y,p['settings']['ridge_regularization']);counts['ridge_fits']+=1
                path=out/'models'/('s%d.json'%seed);write(path,dict(provenance=prov,seed=seed,model=model))
                models.append(dict(seed=seed,path=str(path.relative_to(out)),sha256=sha(path),training_rows=len(y),threshold=model['threshold'],train_rmse=model['train_rmse']))
                for group in p['valid_groups']:
                    case=load_case(seed,'valid',group);f=features(case,adjacency);decision=choose(model,f,case['pool'],seed,group)
                    selected=dict(native=None,local_rank=decision['learned'],matched_random=decision['matched_random'],always_random=decision['always_random'])
                    plans={}
                    for arm,triple in selected.items():
                        changed=exchange(case['native_adjacency'],len(case['ids']),len(case['candidate_ids']),triple)
                        assert_exchange(case['native_adjacency'],changed,len(case['ids']),len(case['candidate_ids']),triple);plans[arm]=changed
                    path=out/'plans'/('s%d_g%d.pt'%(seed,group));torch.save(plans,path)
                    sealed.append(dict(seed=seed,group=group,path=str(path.relative_to(out)),sha256=sha(path),pool=case['pool'],features=f.tolist(),decision=decision,selected=selected,
                        topology_hashes={a:tensor_hash(v) for a,v in plans.items()},feature_hash=case['feature_sha256']))
        assert_frozen(teacher,teacher_hash);assert_frozen(engine.model,frozen['upstream'])
        write(out/'sealed_before_victim.json',dict(provenance=prov,models=models,plans=sealed,upstream_seal_sha256=sha(out/'upstream_seal.json'),official_GAGA_loaded=False))
        seal_sha=sha(out/'sealed_before_victim.json')
        print('STAGE sealed_VALID_evaluation',flush=True)
        view=load_original_view(data,adj=adjacency);victim=load_classifier('gaga_victim',data);victim_hash=state_hash(victim)
        clean_seq=np.load(pathlib.Path(p['runtime_root'])/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',mmap_mode='r',allow_pickle=False)
        if clean_seq.shape!=(data.graph.num_nodes(),7,data.feat_dim):raise RuntimeError('Clean sequence shape changed.')
        def gaga(sequence):
            check();counts['official_GAGA_VALID']+=1
            if counts['official_GAGA_VALID']>p['caps']['official_GAGA_VALID']:raise RuntimeError('GAGA VALID cap exceeded.')
            with torch.no_grad():return victim(sequence)
        clean={};rows=[]
        with (out/'raw'/'evaluation.jsonl').open('x',encoding='utf-8') as stream:
            for group in p['valid_groups']:
                case=load_case(0,'valid',group);ids=case['ids'];a=gcn(data.graph,ids,'valid');b=gaga(torch.from_numpy(clean_seq[ids.numpy()].copy()))
                clean[group]=(a.argmax(1),b.argmax(1))
                stream.write(json.dumps(dict(stage='clean',group=group,node_ids=ids.tolist(),gcn_logits=a.tolist(),gaga_logits=b.tolist()),allow_nan=False)+'\n');stream.flush()
            for i,entry in enumerate(sealed):
                seed,group=entry['seed'],entry['group'];case=load_case(seed,'valid',group);path=out/entry['path']
                if sha(path)!=entry['sha256']:raise RuntimeError('Sealed plan changed.')
                plans=torch.load(path,map_location='cpu');ids=case['ids'];labels=data.labels[ids]
                for arm in p['arms']:
                    graph=make_graph(data,case,plans[arm]);a=gcn(graph,ids,'valid')
                    with torch.no_grad():sequence=canonical(data,view,graph,case);b=gaga(sequence)
                    row=dict(stage='attacked',seed=seed,group=group,arm=arm,node_ids=ids.tolist(),targets=len(ids),
                        gcn_logits=a.tolist(),gaga_logits=b.tolist(),gcn=transitions(labels,clean[group][0],a.argmax(1)),
                        gaga=transitions(labels,clean[group][1],b.argmax(1)),swap=entry['selected'][arm],actual_nodes=len(case['source_ids']),
                        actual_edges=int(plans[arm].sum()),topology_hash=tensor_hash(plans[arm]),feature_hash=case['feature_sha256'])
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append({k:v for k,v in row.items() if k not in ('gcn_logits','gaga_logits','node_ids')});del graph,sequence
                if (i+1)%16==0:print('EVALUATED_CASES',i+1,len(sealed),flush=True)
        assert_frozen(victim,victim_hash);assert_frozen(teacher,teacher_hash)
        if sha(out/'sealed_before_victim.json')!=seal_sha:raise RuntimeError('Final seal mutated.')
        metrics=[]
        for seed in p['seeds']:
            for arm in p['arms']:
                chosen=[r for r in rows if r['seed']==seed and r['arm']==arm]
                total=sum(r['targets'] for r in chosen)
                if len(chosen)!=64 or total!=571:raise RuntimeError('Incomplete evaluation denominator.')
                metrics.append(dict(seed=seed,arm=arm,groups=64,occurrences=total,gaga_mr=100*sum(r['gaga']['attacked_wrong'] for r in chosen)/total,
                    gcn_mr=100*sum(r['gcn']['attacked_wrong'] for r in chosen)/total,exchanges=sum(r['swap'] is not None for r in chosen)))
        contrasts={}
        for control in ('native','matched_random','always_random'):
            deltas=[next(r['gaga_mr'] for r in metrics if r['seed']==s and r['arm']=='local_rank')-next(r['gaga_mr'] for r in metrics if r['seed']==s and r['arm']==control) for s in p['seeds']]
            mean=statistics.mean(deltas);half=4.302652729911275*statistics.stdev(deltas)/(3**.5)
            contrasts[control]=dict(deltas_pp=deltas,mean_pp=mean,nominal_t95=[mean-half,mean+half])
        qualified=all(contrasts[c]['nominal_t95'][0]>0 and min(contrasts[c]['deltas_pp'])>0 for c in ('native','matched_random'))
        result.update(status='completed',metrics=metrics,means={a:statistics.mean(r['gaga_mr'] for r in metrics if r['arm']==a) for a in p['arms']},
            contrasts=contrasts,qualification_met=qualified,cases=rows,models=models,author_receipts=receipts,
            limitations=['Development reused; no TEST calls.','One author state; conditional repeats, not independent author models.',
                'A TRAIN GCN gain prediction does not guarantee GAGA transfer.','Matched random shares learned activation; always-random is auxiliary, not a replacement candidate.'])
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error));traceback.print_exc();raise
    finally:
        signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(out/'result.json',result)
        if result['status']=='completed':write(out/'summary.json',{k:v for k,v in result.items() if k!='cases'})
if __name__=='__main__':main()
