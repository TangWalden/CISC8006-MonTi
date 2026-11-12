"""Hard-graph TRAIN block supervision, sealed decision sets, complete VALID."""
import argparse,json,pathlib,signal,statistics,time,traceback
from helpers import verify,read,write,sha
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--output',required=True);ap.add_argument('--preflight',action='store_true');ap.add_argument('--preflight-result');a=ap.parse_args()
    start=time.monotonic();cpu=time.process_time();p,prov=verify(a.protocol);out=pathlib.Path(a.output);out.mkdir(exist_ok=False)
    counts=dict(generations=0,train_GCN=0,evaluation_GCN=0,official_GAGA_VALID=0,ridge_fits=0,upstream_updates=0,TEST_queries=0)
    result=dict(status='failed',provenance=prov,counters=counts,GPU_hours=0)
    def stop(*args):raise TimeoutError('Frozen block study deadline reached.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(p['resources']['preflight_seconds' if a.preflight else 'main_seconds'])
    try:
        from experiments.downstream_cooperation.common import check_device,make_graph,canonical,tensor_hash,state_hash,assert_frozen
        result['environment']=check_device()
        import torch,numpy as np,scipy.sparse as sp
        from experiments.local_swap.loader import CONFIG,DEFAULTS
        from modules.data_handler import DataHandler
        from course.dual_surrogate.runtime import load_classifier,gcn_logits,load_original_view
        from experiments.block_swap.core import pool,exchange,check,features,fit,choose
        parent=pathlib.Path(p['parent_stage'])/'main';up=read(parent/'upstream_seal.json');index={(r['seed'],r['phase'],r['group']):r for r in up['cases']}
        def case(seed,phase,group):
            r=index[seed,phase,group];f=parent/r['path'];assert sha(f)==r['sha256'];v=torch.load(f,map_location='cpu');v['block_pool']=pool(v)
            assert v['ids'].tolist()==p[phase+'_members'][str(group)]
            return v
        config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config)
        data.target_idx_test=np.empty(0,dtype=np.int64)
        adjacency=sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr();adjacency.setdiag(0);adjacency.eliminate_zeros()
        teacher=load_classifier('gcn_base',data);teacher_hash=state_hash(teacher)
        def evaluate_gcn(graph,ids,phase):
            key='train_GCN' if phase=='train' else 'evaluation_GCN';counts[key]+=1
            assert counts[key]<=p['caps'][key]
            with torch.no_grad():return gcn_logits(teacher,graph,ids)
        def loss(logits):return float(torch.nn.functional.softplus(logits.double()[:,1]-logits.double()[:,0]).mean())
        if a.preflight:
            import unittest
            from experiments.block_swap.checks import Checks
            tests=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks));assert tests.wasSuccessful()
            previous={(r['seed'],r['group']):r for r in [json.loads(line) for line in (parent/'raw/training.jsonl').read_text().splitlines()]}
            checks=[]
            for group in p['train_groups'][:2]:
                v=case(0,'train',group);base=v['native_adjacency'];g=make_graph(data,v,base);native=evaluate_gcn(g,v['ids'],'train');d=g.in_degrees()[v['source_ids']]
                prior=torch.tensor(previous[0,group]['native_logits'],dtype=native.dtype)
                error=float((native-prior).abs().max())
                assert error<=1e-4 and torch.equal(native.argmax(1),prior.argmax(1))
                f=features(v,adjacency)
                for action in v['block_pool'][:1]:
                    mask=exchange(base,len(v['ids']),len(v['candidate_ids']),action);check(base,mask,len(v['ids']),len(v['candidate_ids']),action)
                    changed=make_graph(data,v,mask);assert torch.equal(changed.in_degrees()[v['source_ids']],d)
                    evaluate_gcn(changed,v['ids'],'train')
                checks.append(dict(group=group,pool=len(v['block_pool']),features_shape=list(f.shape),unchanged_physical_degrees=True,parent_native_max_logit_difference=error))
            result.update(status='passed',synthetic_tests=tests.testsRun,checks=checks);return
        assert a.preflight_result and read(a.preflight_result)['status']=='passed' and read(a.preflight_result)['provenance']==prov
        for n in ('models','plans','raw'):(out/n).mkdir()
        models=[];sealed=[]
        print('STAGE TRAIN_block_supervision',flush=True)
        with (out/'raw/training.jsonl').open('x') as stream:
            for seed in p['seeds']:
                X=[];y=[]
                for i,group in enumerate(p['train_groups']):
                    v=case(seed,'train',group);base=v['native_adjacency'];f=features(v,adjacency);g=make_graph(data,v,base)
                    logits=evaluate_gcn(g,v['ids'],'train');base_loss=loss(logits);row=dict(seed=seed,group=group,node_ids=v['ids'].tolist(),native_logits=logits.tolist(),native_loss=base_loss,swaps=[])
                    for j,action in enumerate(v['block_pool']):
                        mask=exchange(base,len(v['ids']),len(v['candidate_ids']),action);check(base,mask,len(v['ids']),len(v['candidate_ids']),action)
                        g=make_graph(data,v,mask);logits=evaluate_gcn(g,v['ids'],'train');value=loss(logits);gain=base_loss-value
                        X.append(f[j]);y.append(gain);row['swaps'].append(dict(action=action,features=f[j].tolist(),logits=logits.tolist(),loss=value,gain=gain))
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if (i+1)%32==0:print('TRAIN',seed,i+1,flush=True)
                model=fit(X,y,p['settings']['regularization']);counts['ridge_fits']+=1;path=out/'models'/('s%d.json'%seed);write(path,model)
                models.append(dict(seed=seed,path=str(path.relative_to(out)),sha256=sha(path),training_rows=len(y),threshold=model['threshold']))
                for group in p['valid_groups']:
                    v=case(seed,'valid',group);f=features(v,adjacency);decision=choose(model,f,v['block_pool'],seed,group)
                    selected=dict(native=None,block_rank=decision['learned'],matched_random=decision['matched_random'],always_random=decision['always_random']);plans={}
                    for arm,action in selected.items():
                        mask=exchange(v['native_adjacency'],len(v['ids']),len(v['candidate_ids']),action);check(v['native_adjacency'],mask,len(v['ids']),len(v['candidate_ids']),action);plans[arm]=mask
                    path=out/'plans'/('s%d_g%d.pt'%(seed,group));torch.save(plans,path)
                    sealed.append(dict(seed=seed,group=group,path=str(path.relative_to(out)),sha256=sha(path),pool=v['block_pool'],features=f.tolist(),decision=decision,selected=selected))
        assert_frozen(teacher,teacher_hash)
        write(out/'sealed_before_victim.json',dict(provenance=prov,models=models,plans=sealed,official_GAGA_loaded=False));seal_hash=sha(out/'sealed_before_victim.json')
        print('STAGE sealed_VALID_evaluation',flush=True)
        view=load_original_view(data,adj=adjacency);victim=load_classifier('gaga_victim',data);victim_hash=state_hash(victim)
        clean_seq=np.load(pathlib.Path(p['runtime_root'])/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',mmap_mode='r',allow_pickle=False)
        def evaluate_gaga(seq):
            counts['official_GAGA_VALID']+=1;assert counts['official_GAGA_VALID']<=832
            with torch.no_grad():return victim(seq)
        rows=[]
        with (out/'raw/evaluation.jsonl').open('x') as stream:
            for group in p['valid_groups']:
                v=case(0,'valid',group);ids=v['ids'];x=evaluate_gcn(data.graph,ids,'valid');z=evaluate_gaga(torch.from_numpy(clean_seq[ids.numpy()].copy()))
                stream.write(json.dumps(dict(stage='clean',group=group,node_ids=ids.tolist(),gcn_logits=x.tolist(),gaga_logits=z.tolist()))+'\n');stream.flush()
            for i,entry in enumerate(sealed):
                seed,group=entry['seed'],entry['group'];v=case(seed,'valid',group);path=out/entry['path'];assert sha(path)==entry['sha256'];plans=torch.load(path,map_location='cpu')
                for arm in p['arms']:
                    mask=plans[arm];g=make_graph(data,v,mask);x=evaluate_gcn(g,v['ids'],'valid')
                    with torch.no_grad():z=evaluate_gaga(canonical(data,view,g,v))
                    action=entry['selected'][arm];row=dict(stage='attacked',seed=seed,group=group,arm=arm,node_ids=v['ids'].tolist(),targets=len(v['ids']),
                        gcn_logits=x.tolist(),gaga_logits=z.tolist(),gcn_wrong=int((x.argmax(1)==0).sum()),gaga_wrong=int((z.argmax(1)==0).sum()),
                        action=action,exchanges=0 if action is None else len(action['rows']),actual_nodes=len(v['source_ids']),actual_edges=int(mask.sum()),topology_hash=tensor_hash(mask),feature_hash=v['feature_sha256'])
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows.append(row)
                if (i+1)%16==0:print('VALID',i+1,192,flush=True)
        assert_frozen(victim,victim_hash);assert_frozen(teacher,teacher_hash);assert sha(out/'sealed_before_victim.json')==seal_hash
        metrics=[]
        for seed in p['seeds']:
            for arm in p['arms']:
                rr=[r for r in rows if r['seed']==seed and r['arm']==arm];total=sum(r['targets'] for r in rr);assert len(rr)==64 and total==571
                metrics.append(dict(seed=seed,arm=arm,groups=64,occurrences=total,gaga_mr=100*sum(r['gaga_wrong'] for r in rr)/total,gcn_mr=100*sum(r['gcn_wrong'] for r in rr)/total,
                    active_cases=sum(r['action'] is not None for r in rr),exchanges=sum(r['exchanges'] for r in rr)))
        contrasts={}
        for control in ('native','matched_random','always_random'):
            ds=[next(r['gaga_mr'] for r in metrics if r['seed']==s and r['arm']=='block_rank')-next(r['gaga_mr'] for r in metrics if r['seed']==s and r['arm']==control) for s in p['seeds']]
            avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrasts[control]=dict(deltas_pp=ds,mean_pp=avg,nominal_t95=[avg-half,avg+half])
        qualified=all(contrasts[c]['nominal_t95'][0]>0 and min(contrasts[c]['deltas_pp'])>0 for c in ('native','matched_random'))
        result.update(status='completed',metrics=metrics,means={arm:statistics.mean(r['gaga_mr'] for r in metrics if r['arm']==arm) for arm in p['arms']},contrasts=contrasts,qualification_met=qualified,models=models)
    except BaseException as e:result.update(error=str(e),error_type=type(e).__name__);traceback.print_exc();raise
    finally:
        signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
        if result['status']=='completed':write(out/'summary.json',result)
if __name__=='__main__':main()
