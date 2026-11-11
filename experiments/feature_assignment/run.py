"""One fixed graph, two own-model feature derivatives and three common-filtered permutations.

All native raw/final inputs seal before loading the official GAGA. The original
fraud-target cohort is checked, but its truth never selects a proposal/action.
"""
import argparse
import datetime
import json
import pathlib
import signal
import time
import traceback

from runtime import (ARMS,PROPOSALS,SEEDS,verify_protocol,check_device,write_json,load_json,sha,
    tensor_hash,source_state,assert_frozen,data_hashes,load_engine,load_own_teacher,make_case,
    canonical,softplus,check_permuted)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):return value.detach().cpu().contiguous()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [cpu_tree(v) for v in value]
    return value


def serial(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def tensor_inventory(value,prefix=''):
    import torch
    if torch.is_tensor(value):return {prefix:tensor_hash(value)}
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(tensor_inventory(item,(prefix+'.' if prefix else '')+key))
    return result


def append(stream,row):
    stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--preflight',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'artifacts').mkdir();(out/'engine_receipts').mkdir()
    start,cpu_start=time.monotonic(),time.process_time();began=utc()
    counters=dict(dataset_loads=0,generator_models_loaded=0,proposal_GCN_models_loaded=0,own_GAGA_models_loaded=0,
        generations=0,generator_forward_dispatches=0,generator_completed_forwards=0,proposal_generation_GCN_forwards=0,
        own_forward_requests=0,own_GAGA_physical_forwards=0,own_GCN_physical_forwards=0,own_GAGA_completed_forwards=0,own_GCN_completed_forwards=0,own_cache_hits=0,native_sequence_cache_hits=0,
        autograd_calls=0,native_canonical_updates=0,final_canonical_sequences=0,sealed_cases=0,
        official_GAGA_models_loaded=0,official_GAGA_forwards=0,clean_GCN_forwards=0,cached_attacked_GCN_requests=0,
        artifact_tensor_loads=0,optimizer_updates=0,API_calls=0,TEST_queries=0)
    result=dict(status='failed',counters=counters,started_utc=began,active_case=None,official_victim_loaded=False)
    records=[];teacher=None;engine=data=view=None;handles=[];deadline=None
    def check():
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Frozen assignment main deadline reached.')
    def stop(signum,frame):raise TimeoutError('Assignment guardian/inner deadline signal received.')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    try:
        p,provenance=verify_protocol(args.protocol);result['provenance']=provenance
        if pathlib.Path(p['asset_root'])/'runs' not in out.parents:raise RuntimeError('Outputs must remain in ignored historical runs/.')
        gate=load_json(args.preflight)
        if (gate['status']!='passed' or gate['provenance']!=provenance or gate['counters']['optimizer_updates'] or
                gate['counters']['official_GAGA_forwards']):raise RuntimeError('Passed same-source/protocol zero-victim preflight required.')
        write_json(out/'registered.json',dict(provenance=provenance,preflight=dict(path=str(pathlib.Path(args.preflight).resolve()),sha256=sha(args.preflight)),
            protocol_path=str(pathlib.Path(args.protocol).resolve()),arms=list(ARMS),valid_groups=p['valid_groups'],
            acceptance='Actual native own-GAGA input; strict1e-7 loss improvement and preserve every baseline GCN class0 target',
            native_inputs_seal_before_official_victim=True,cached_GCN_attacked_channel=True))
        deadline=start+p['resource']['main_inner_seconds'];signal.alarm(max(1,int(deadline-time.monotonic())))
        result['device']=check_device(p)
        result['counter_semantics']='physical_forwards count dispatches; completed counters separately confirm successful own-model returns; failed dispatch may have unknown internal partial work'
        import dgl
        import numpy as np
        import torch
        from utils.utils import set_seed,get_n_ids
        from course.dual_surrogate.runtime import load_original_view,checked_logits,load_classifier,gcn_logits,transitions
        from methods.feature_assignment import propose_assignment,cycle_matched_random,accept_if_improved
        original_hashes=None;teacher_hash=None;last_engine_hash=None
        with (out/'cases.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for seed in SEEDS:
                check();set_seed(seed);dgl.seed(seed);dgl.random.seed(seed)
                engine,data,receipt=load_engine(p,seed,out/'engine_receipts',deadline)
                counters['dataset_loads']+=1;counters['generator_models_loaded']+=1;counters['proposal_GCN_models_loaded']+=1
                view=load_original_view(data);gcn=engine.teachers['gcn_base'];gen_hash=source_state(engine.model);gcn_hash=source_state(gcn)
                current=data_hashes(data,view)
                if original_hashes is None:original_hashes=current;result['immutable_content_before']=current
                elif current!=original_hashes:raise RuntimeError('Original features/group caches differ across seed loads.')
                if teacher is None:
                    teacher,teacher_manifest=load_own_teacher(p,data);teacher_hash=source_state(teacher);counters['own_GAGA_models_loaded']+=1
                    result['own_teacher']=dict(manifest_sha256=p['own_teacher_manifest']['sha256'],state_sha256=teacher_hash)
                if set(engine.teachers)!={'gcn_base'} or engine.victim_model is not None:raise RuntimeError('Baseline helper loaded unexpected teacher/victim.')
                def gen_begin(model,inputs):check();counters['generator_forward_dispatches']+=1
                def gen_end(model,inputs,outputs):counters['generator_completed_forwards']+=1
                def forbid(model,inputs):
                    counters['proposal_generation_GCN_forwards']+=1
                    raise RuntimeError('Generation must use frozen h_GCN cache without GCN query.')
                handles=[engine.model.register_forward_pre_hook(gen_begin),engine.model.register_forward_hook(gen_end)]
                forbidden=gcn.register_forward_pre_hook(forbid)
                write_json(out/'engine_receipts'/('seed%d_before.json'%seed),dict(terminal_receipt=receipt,generator_state=gen_hash,gcn_state=gcn_hash,
                    scheduler=dict(temperature=float(engine.model.temp_scheduler.get_temperature()),epsilon=float(engine.model.eps_scheduler.get_epsilon()),
                    restoration_ticks=8),original_content=current))
                for ordinal,group in enumerate(p['valid_groups']):
                    check();result['active_case']=dict(seed=seed,group=group,ordinal=ordinal)
                    case=make_case(p,engine,data,view,seed,group);counters['generations']+=1;counters['native_canonical_updates']+=1
                    forbidden.remove();x0=case['x0'];n=len(case['ids']);identity=torch.arange(len(x0),device=x0.device)
                    if case['info']['node_ids']!=p['cohort']['members'][str(group)]:raise RuntimeError('Frozen full target list changed.')
                    raw_logits={};gradients={};caches={};cache_receipts=[]
                    for role in ('gaga','gcn'):
                        check();x=x0.detach().clone().requires_grad_(True)
                        counters['own_forward_requests']+=1;counters['own_'+role.upper()+'_physical_forwards']+=1
                        if role=='gaga':
                            native_equal_gradient_input=case['affine'].sequence(x)
                            if not torch.equal(native_equal_gradient_input.detach(),case['baseline_sequence']):
                                raise RuntimeError('Baseline affine gradient input is not exactly the actual native canonical baseline.')
                            logits=checked_logits(teacher(native_equal_gradient_input),n)
                        else:logits=checked_logits(case['gcn_forward'](gcn,x),n)
                        counters['own_'+role.upper()+'_completed_forwards']+=1
                        loss=softplus(logits);counters['autograd_calls']+=1
                        gradient,=torch.autograd.grad(loss,x)
                        if not bool(torch.isfinite(gradient).all()):raise RuntimeError('Nonfinite actual input-feature derivative.')
                        gradients[role]=gradient.detach();raw_logits.setdefault('baseline',{})[role]=logits.detach()
                    caches[tensor_hash(x0)]={k:v for k,v in raw_logits['baseline'].items()}
                    sequence_cache={tensor_hash(x0):case['baseline_sequence']}
                    proposals={
                        'relation_assignment':propose_assignment(x0.detach().cpu().numpy(),gradients['gaga'].cpu().numpy()),
                        'gcn_assignment':propose_assignment(x0.detach().cpu().numpy(),gradients['gcn'].cpu().numpy())}
                    proposals['matched_random_permutation']=cycle_matched_random(proposals['relation_assignment']['permutation'],seed=int(seed),group_key=int(group))
                    raw_perms={arm:torch.as_tensor(proposals[arm]['permutation'],dtype=torch.long,device=x0.device) for arm in PROPOSALS}
                    raw_features={'baseline':x0};raw_sequences={'baseline':case['baseline_sequence']};raw_invariants={'baseline':check_permuted(case,x0,identity)}
                    final_perms={'baseline':identity};features={'baseline':x0};sequences={'baseline':case['baseline_sequence']};final_logits={'baseline':raw_logits['baseline']}
                    acceptance={};affine_error={}
                    for arm in PROPOSALS:
                        check();perm=raw_perms[arm];candidate=x0.index_select(0,perm).detach();raw_features[arm]=candidate
                        raw_invariants[arm]=check_permuted(case,candidate,perm);key=tensor_hash(candidate)
                        with torch.no_grad():
                            if key in sequence_cache:seq=sequence_cache[key];counters['native_sequence_cache_hits']+=1
                            else:
                                seq=canonical(data,view,case['graph'],case['ids'],case['candidates'],case['injected'],candidate)
                                counters['native_canonical_updates']+=1;sequence_cache[key]=seq
                            raw_sequences[arm]=seq
                            affine_error[arm]=float((seq-case['affine'].sequence(candidate)).abs().max())
                            counters['own_forward_requests']+=2
                            if key in caches:
                                logits_pair=caches[key];counters['own_cache_hits']+=2
                                cache_receipts.append(dict(arm=arm,feature_sha256=key,reused=True))
                            else:
                                logits_pair={}
                                for role in ('gaga','gcn'):
                                    check();counters['own_'+role.upper()+'_physical_forwards']+=1
                                    logits_pair[role]=checked_logits(teacher(seq) if role=='gaga' else case['gcn_forward'](gcn,candidate),n).detach()
                                    counters['own_'+role.upper()+'_completed_forwards']+=1
                                caches[key]=logits_pair;cache_receipts.append(dict(arm=arm,feature_sha256=key,reused=False))
                            raw_logits[arm]=logits_pair
                        decision=accept_if_improved(raw_logits['baseline']['gaga'].cpu().numpy(),logits_pair['gaga'].cpu().numpy(),
                            raw_logits['baseline']['gcn'].cpu().numpy(),logits_pair['gcn'].cpu().numpy())
                        acceptance[arm]=decision;accepted=decision['accepted']
                        final_perms[arm]=perm if accepted else identity
                        features[arm]=candidate if accepted else x0;sequences[arm]=seq if accepted else case['baseline_sequence']
                        final_logits[arm]=logits_pair if accepted else raw_logits['baseline']
                    for arm in ARMS:check_permuted(case,features[arm],final_perms[arm])
                    artifact=dict(capture=case['capture'],random_fill_adjacency=case['random_fill_adjacency'],
                        gradients=gradients,raw_permutations=raw_perms,final_permutations=final_perms,raw_features=raw_features,features=features,
                        raw_canonical_sequences=raw_sequences,canonical_sequences=sequences,own_logits_raw=raw_logits,own_logits_final=final_logits)
                    path=out/'artifacts'/('seed%d_group%d.pt'%(seed,group))
                    with path.open('xb') as destination:torch.save(cpu_tree(artifact),destination)
                    row=dict(seed=seed,ordinal=ordinal,original_node_count=data.graph.num_nodes(),**case['info'],
                        physical_injected_degrees=case['physical_degrees'].cpu().tolist(),allocation={k:v for k,v in case['allocation'].items() if k!='arms'},
                        artifact=dict(path=path.relative_to(out).as_posix(),sha256=sha(path)),block_signature=case['block_signature'],
                        tensor_hashes=tensor_inventory(artifact),
                        proposal_metadata=serial(proposals),acceptance=acceptance,raw_invariants=raw_invariants,
                        final_invariants={arm:check_permuted(case,features[arm],final_perms[arm]) for arm in ARMS},
                        raw_sequence_hashes={arm:tensor_hash(v) for arm,v in raw_sequences.items()},
                        final_sequence_hashes={arm:tensor_hash(v) for arm,v in sequences.items()},
                        gradients_hashes={role:tensor_hash(v) for role,v in gradients.items()},
                        own_logits_raw={arm:{role:v.cpu().tolist() for role,v in values.items()} for arm,values in raw_logits.items()},
                        own_logits_final={arm:{role:v.cpu().tolist() for role,v in values.items()} for arm,values in final_logits.items()},
                        cache_receipts=cache_receipts,affine_raw_sequence_max_error=affine_error,
                        logical_own_forward_requests=8,logical_autograd_calls=2,final_inputs_created_utc=utc())
                    append(stream,row);records.append(row);counters['sealed_cases']+=1;counters['final_canonical_sequences']+=4
                    assert_frozen(gcn,gcn_hash);assert_frozen(teacher,teacher_hash)
                    if source_state(engine.model)!=gen_hash or any(v.grad is not None for v in engine.model.parameters()):raise RuntimeError('Generator parameters/buffers/gradient state changed.')
                    del case,artifact,gradients,raw_logits,final_logits,features,sequences,raw_sequences,raw_features,caches,sequence_cache,x,x0
                    forbidden=gcn.register_forward_pre_hook(forbid)
                    if (ordinal+1)%16==0:print('ASSIGNMENT_PREPARED',seed,ordinal+1,flush=True)
                forbidden.remove()
                for handle in handles:handle.remove()
                handles=[]
                if data_hashes(data,view)!=original_hashes:raise RuntimeError('Original features/group caches mutated.')
                write_json(out/'engine_receipts'/('seed%d_after.json'%seed),dict(generator_state=source_state(engine.model),gcn_state=source_state(gcn),
                    original_content=data_hashes(data,view),engine_generator_forwards=engine.generator_forwards,
                    scheduler=dict(temperature=float(engine.model.temp_scheduler.get_temperature()),epsilon=float(engine.model.eps_scheduler.get_epsilon()),restoration_ticks=8)))
                if seed!=SEEDS[-1]:del engine,gcn,data,view
        if counters['generations']!=384 or counters['autograd_calls']!=768 or counters['own_forward_requests']!=3072 or counters['final_canonical_sequences']!=1536:
            raise RuntimeError('Complete fixed cohort logical execution counts differ.')
        manifest=[row['artifact'] for row in records]
        write_json(out/'geometry_sealed.json',dict(status='sealed',provenance=provenance,completed_utc=utc(),cases=384,final_canonical_sequences=1536,
            official_GAGA_loaded=False,official_GAGA_forwards=0,cases_jsonl_sha256=sha(out/'cases.jsonl'),artifacts=manifest,
            immutable_original_content=data_hashes(data,view),counters=dict(counters)))
        result['geometry_sealed_sha256']=sha(out/'geometry_sealed.json');result['geometry_sealed_before_victim_load']=True
        check();(out/'evaluation').mkdir();victim=load_classifier('gaga_victim',data);counters['official_GAGA_models_loaded']+=1
        result['official_victim_loaded']=True;result['official_victim_load_utc']=utc();victim_hash=source_state(victim)
        result['official_victim_state_before']=victim_hash
        clean={}
        with torch.no_grad(),(out/'evaluation'/'clean.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            original_sequences=np.load(pathlib.Path(p['asset_root'])/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',mmap_mode='r',allow_pickle=False)
            for group in p['valid_groups']:
                check();ids=torch.tensor(get_n_ids(data.target_assignment,group),dtype=torch.long,device=data.device);truth=data.labels[ids].detach()
                seq=torch.as_tensor(np.array(original_sequences[ids.cpu().numpy()],copy=True),device=data.device,dtype=torch.float32)
                counters['official_GAGA_forwards']+=1;vg=checked_logits(victim(seq),len(ids))
                counters['clean_GCN_forwards']+=1;cg=gcn_logits(gcn,data.graph,ids)
                row=dict(group=group,node_ids=ids.cpu().tolist(),labels=truth.cpu().tolist(),gaga_logits=vg.cpu().tolist(),gcn_logits=cg.cpu().tolist())
                clean[group]=dict(gaga=vg.argmax(1),gcn=cg.argmax(1));append(stream,row)
        streams={(seed,arm):(out/'evaluation'/('%s_seed%d.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n') for seed in SEEDS for arm in ARMS}
        try:
            with torch.no_grad():
                for row in records:
                    check();ref=row['artifact'];path=out/ref['path']
                    if sha(path)!=ref['sha256']:raise RuntimeError('Sealed artifact changed before victim use.')
                    artifact=torch.load(path,map_location=data.device);counters['artifact_tensor_loads']+=1
                    ids=torch.tensor(row['node_ids'],dtype=torch.long,device=data.device);truth=data.labels[ids]
                    for arm in ARMS:
                        check();seq=artifact['canonical_sequences'][arm]
                        if tensor_hash(seq)!=row['final_sequence_hashes'][arm]:raise RuntimeError('Sealed final native input changed.')
                        counters['official_GAGA_forwards']+=1;vg=checked_logits(victim(seq),len(ids))
                        cg=artifact['own_logits_final'][arm]['gcn'];counters['cached_attacked_GCN_requests']+=1
                        prediction=dict(seed=row['seed'],group=row['group'],arm=arm,node_ids=row['node_ids'],labels=row['labels'],
                            node_budget=row['node_budget'],edge_budget=row['edge_budget'],gaga_logits=vg.cpu().tolist(),gcn_logits=cg.cpu().tolist(),
                            gcn_source='Cached own original-GCN fixed-block logits, not a fresh independent replay',
                            artifact_sha256=ref['sha256'],canonical_sequence_sha256=tensor_hash(seq),feature_sha256=tensor_hash(artifact['features'][arm]),
                            gaga=transitions(truth,clean[row['group']]['gaga'],vg.argmax(1)),gcn=transitions(truth,clean[row['group']]['gcn'],cg.argmax(1)))
                        append(streams[(row['seed'],arm)],prediction)
        finally:
            for stream in streams.values():stream.close()
        if counters['official_GAGA_forwards']!=1664 or counters['clean_GCN_forwards']!=128 or counters['cached_attacked_GCN_requests']!=1536:
            raise RuntimeError('Frozen full-cohort evaluation counts differ.')
        assert_frozen(victim,victim_hash);assert_frozen(teacher,teacher_hash);assert_frozen(gcn,gcn_hash)
        result['immutable_content_after']=data_hashes(data,view)
        if result['immutable_content_after']!=original_hashes:raise RuntimeError('Original features/cache mutated during evaluation.')
        result.update(status='completed',completed_utc=utc(),official_victim_state_after=source_state(victim),original_node_count=data.graph.num_nodes(),
            own_teacher_state_after=source_state(teacher),final_original_GCN_state_after=source_state(gcn),
            evaluation_files={path.name:sha(path) for path in (out/'evaluation').glob('*.jsonl')},full_case_count=len(records),
            own_model_filter_not_victim_feedback=True,cached_GCN_not_independent_neural_replay=True)
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        signal.alarm(0)
        for handle in handles:handle.remove()
        result.update(counters=counters,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu_start,finished_utc=utc())
        write_json(out/'result.json',result);print('ASSIGNMENT_END',json.dumps(dict(status=result['status'],counters=counters,wall_seconds=result['wall_seconds'])),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
