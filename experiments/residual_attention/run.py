"""Frozen-topology contextual residuals; all terminal weights sealed before GAGA."""
import pathlib,json,sys,time,argparse,hashlib,signal,statistics
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
    with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=read(s/'downstream_protocol.json')
    for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
    assert read(s/'preflight_result.json')['status']=='passed';teacher_receipt=read(s/'teacher/result.json');assert teacher_receipt['quality_passed'] and sha(s/'teacher/model.pt')==teacher_receipt['checkpoint_sha256']
    sys.path[:0]=[str(s/'source'),p['support_source']+'/experiments/block_swap',p['support_source'],p['runtime_root']]
    from helpers import verify
    verify(pathlib.Path(p['support_source']).parent/'registration/protocol.json')
    import torch,numpy as np
    from models.victim_model import GAGA
    from modules.data_handler import DataHandler
    from experiments.local_swap.loader import CONFIG,DEFAULTS
    from experiments.local_swap.core import keyed
    from experiments.downstream_cooperation.common import check_device,make_graph,canonical,tensor_hash,state_hash,assert_frozen
    from course.dual_surrogate.runtime import load_original_view,load_classifier
    from model import Residual,feature_update
    from coefficients import coefficients
    check_device();out=s/'downstream';out.mkdir(exist_ok=False)
    for n in ('cards','models','plans'):(out/n).mkdir()
    start=time.monotonic();cpu=time.process_time();counts=dict(cards=0,updates=0,own_teacher_TRAIN=0,own_teacher_VALID=0,official_GAGA_VALID=0,upstream_updates=0,generations=0,TEST_queries=0)
    result=dict(status='failed',counters=counts,GPU_hours=0,protocol_sha256=sha(s/'downstream_protocol.json'))
    signal.signal(signal.SIGALRM,lambda *x:(_ for _ in ()).throw(TimeoutError('Frozen residual deadline')));signal.alarm(3600)
    try:
        config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config);data.target_idx_test=np.empty(0,dtype=np.int64);view=load_original_view(data)
        teacher=GAGA(data.feat_dim,64,2);teacher.load_state_dict(torch.load(s/'teacher/model.pt',map_location='cpu'));teacher.eval()
        for param in teacher.parameters():param.requires_grad_(False)
        teacher_hash=state_hash(teacher);train_ids=torch.as_tensor(data.idx_train).long();std=data.features[train_ids].std(0,unbiased=False);low=data.features.min(0).values;high=data.features.max(0).values
        parent=pathlib.Path(p['capture_parent']);up=read(parent/'upstream_seal.json');card_index={}
        print('STAGE fixed_topology_cards',flush=True)
        for i,ref in enumerate(up['cases']):
            f=parent/ref['path'];assert sha(f)==ref['sha256'];v=torch.load(f,map_location='cpu');L1,L2=coefficients(view,v)
            with torch.no_grad():base=canonical(data,view,make_graph(data,v,v['native_adjacency']),v)
            assert bool(((v['x_aux']>=low-1e-6)&(v['x_aux']<=high+1e-6)).all())
            card=dict(case=v,sequence=base,L1=L1,L2=L2);path=out/'cards'/('s%d_%s_g%d.pt'%(ref['seed'],ref['phase'],ref['group']));torch.save(card,path);card_index[ref['seed'],ref['phase'],ref['group']]=dict(path=str(path.relative_to(out)),sha256=sha(path));counts['cards']+=1
            if (i+1)%64==0:print('CARDS',i+1,576,flush=True)
        assert counts['cards']==576
        write(out/'card_seal.json',dict(cards=[dict(key=list(k),**v) for k,v in card_index.items()],official_victim_loaded=False))
        def card(seed,phase,group):
            ref=card_index[seed,phase,group];path=out/ref['path'];assert sha(path)==ref['sha256'];return torch.load(path,map_location='cpu')
        plans={};model_refs=[]
        print('STAGE TRAIN_residual_networks',flush=True)
        with (out/'training.jsonl').open('x') as stream:
            for seed in p['seeds']:
                for mode in ('attention','mean'):
                    torch.manual_seed(2026111300+seed);net=Residual(data.feat_dim,mode=='attention');initial_hash=state_hash(net);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
                    for epoch in range(4):
                        generator=torch.Generator().manual_seed(keyed(seed,epoch,'residual_training_order')%(2**63-1));order=torch.randperm(128,generator=generator).tolist()
                        for position in order:
                            group=p['train_groups'][position];v=card(seed,'train',group);case=v['case'];opt.zero_grad();delta=net(case,std,low,high)
                            logits=teacher(feature_update(v['sequence'],v['L1'],v['L2'],delta));counts['own_teacher_TRAIN']+=1
                            attack=torch.nn.functional.softplus(logits[:,1]-logits[:,0]).mean();penalty=.01*(delta/std.clamp_min(1e-8)).square().mean();loss=attack+penalty
                            assert bool(torch.isfinite(loss));loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();counts['updates']+=1
                            stream.write(json.dumps(dict(seed=seed,mode=mode,epoch=epoch+1,group=group,attack_loss=float(attack),penalty=float(penalty),teacher_wrong=int((logits.argmax(1)==0).sum()),targets=len(case['ids'])))+'\n')
                        stream.flush();print('TRAIN',seed,mode,epoch+1,flush=True)
                    assert state_hash(net)!=initial_hash;net.eval();path=out/'models'/('s%d_%s.pt'%(seed,mode));torch.save(net.state_dict(),path);model_refs.append(dict(seed=seed,mode=mode,path=str(path.relative_to(out)),sha256=sha(path)))
                    for group in p['valid_groups']:
                        v=card(seed,'valid',group);case=v['case']
                        with torch.no_grad():delta=net(case,std,low,high)
                        radius=torch.minimum(1.0*std,torch.minimum(case['x_aux']-low,high-case['x_aux']).clamp_min(0));assert bool((delta.abs()<=radius+1e-7).all())
                        generator=torch.Generator().manual_seed(keyed(seed,group,mode,'matched_random_sign')%(2**63-1));sign=torch.randint(0,2,delta.shape,generator=generator).float()*2-1
                        random_delta=delta.abs()*sign;assert torch.equal(random_delta.abs(),delta.abs());key=(seed,group)
                        plans.setdefault(key,dict(native=torch.zeros_like(delta)));plans[key][mode]=delta;plans[key]['random_'+mode]=random_delta
        assert counts['updates']==3072 and counts['own_teacher_TRAIN']==3072;assert_frozen(teacher,teacher_hash)
        sealed=[]
        for (seed,group),deltas in plans.items():
            path=out/'plans'/('s%d_g%d.pt'%(seed,group));torch.save(deltas,path);sealed.append(dict(seed=seed,group=group,path=str(path.relative_to(out)),sha256=sha(path)))
        write(out/'sealed_before_victim.json',dict(plans=sealed,models=model_refs,official_victim_loaded=False));seal_hash=sha(out/'sealed_before_victim.json')
        victim=load_classifier('gaga_victim',data);victim_hash=state_hash(victim);rows=[]
        print('STAGE sealed_VALID',flush=True)
        with (out/'evaluation.jsonl').open('x') as stream:
            for i,entry in enumerate(sealed):
                seed,group=entry['seed'],entry['group'];v=card(seed,'valid',group);case=v['case'];path=out/entry['path'];assert sha(path)==entry['sha256'];deltas=torch.load(path,map_location='cpu')
                for arm in p['arms']:
                    delta=deltas[arm];assert bool(((case['x_aux']+delta>=low-1e-6)&(case['x_aux']+delta<=high+1e-6)).all())
                    graph=make_graph(data,case,case['native_adjacency']);graph.ndata['x']=torch.cat((data.features,case['x_aux']+delta))
                    with torch.no_grad():seq=canonical(data,view,graph,case);z=victim(seq);own=teacher(seq)
                    counts['official_GAGA_VALID']+=1;counts['own_teacher_VALID']+=1;assert counts['official_GAGA_VALID']<=960
                    row=dict(seed=seed,group=group,arm=arm,node_ids=case['ids'].tolist(),gaga_logits=z.tolist(),teacher_logits=own.tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_wrong=int((own.argmax(1)==0).sum()),delta_sha256=tensor_hash(delta),topology_sha256=tensor_hash(case['native_adjacency']),delta_L2=float(delta.norm()),actual_nodes=len(case['source_ids']),actual_edges=int(case['native_adjacency'].sum()))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
        assert_frozen(teacher,teacher_hash);assert_frozen(victim,victim_hash);assert sha(out/'sealed_before_victim.json')==seal_hash
        metrics=[]
        for seed in p['seeds']:
            for arm in p['arms']:
                rr=[r for r in rows if r['seed']==seed and r['arm']==arm];total=sum(len(r['node_ids']) for r in rr);assert len(rr)==64 and total==571
                metrics.append(dict(seed=seed,arm=arm,MR=100*sum(r['wrong'] for r in rr)/total,teacher_MR=100*sum(r['teacher_wrong'] for r in rr)/total))
        means={arm:statistics.mean(r['MR'] for r in metrics if r['arm']==arm) for arm in p['arms']};contrasts={};owner={}
        total_wrong=lambda arm:sum(r['wrong'] for r in rows if r['arm']==arm)
        for mode in ('attention','mean'):
            owner[mode]=total_wrong(mode)>max(total_wrong('native'),total_wrong('random_'+mode))
            for control in ('native','random_'+mode):
                ds=[next(r['MR'] for r in metrics if r['seed']==seed and r['arm']==mode)-next(r['MR'] for r in metrics if r['seed']==seed and r['arm']==control) for seed in p['seeds']]
                mean=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrasts[mode+'__'+control]=dict(deltas_pp=ds,mean_pp=mean,nominal_t95=[mean-half,mean+half])
        result.update(status='completed',metrics=metrics,means=means,contrasts=contrasts,owner_mean_criterion=owner,limitations=p['limitations'])
    finally:
        signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
