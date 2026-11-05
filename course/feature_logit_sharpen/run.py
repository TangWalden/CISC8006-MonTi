"""Fresh-context feature contrasts, sealed before dedicated evaluation models.

Exactly one unchanged terminal-eight legacy generator supplies each seed/group.
No proposal uses classifier predictions, gradients, labels, fitting or API calls.
The original cached h_GCN conditions generation. All four feature branches share
one freshly generated hard graph, complete target list and physical budgets.
"""
import argparse
import copy
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback


ROOT=pathlib.Path(__file__).resolve().parents[2]
SCHEMA='monti_feature_logit_sharpen_v1'
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
ARMS=('baseline','logit_sharp','linear_contrast','shuffled_delta')
SEEDS=(3,4,5)
FIXED_TOKENS=(0,1,2,4,5)
EXPECTED_COUNTS=dict(
    baseline_engine_load_dispatches=3,baseline_engines_loaded=3,dataset_loads=3,
    proposal_gcn_models_loaded=3,generator_models_loaded=3,
    generator_forward_dispatches=384,generator_completed_forwards=384,
    baseline_generations=384,proposal_gcn_forward_attempts=0,
    scheduler_restoration_ticks_reported=24,feature_cases_prepared=384,
    geometry_feature_graph_variants=1536,canonical_sequence_updates=1536,
    fixed_token_invariant_checks=1536,prepared_artifact_tensor_loads=384,
    evaluation_base_graph_reconstructions=384,evaluation_feature_graph_variants=1536,
    original_gcn_load_dispatches=1,original_gcn_models_loaded=1,
    official_gaga_load_dispatches=1,official_gaga_models_loaded=1,
    original_gcn_forward_dispatches=1664,original_gcn_completed_forwards=1664,
    official_gaga_forward_dispatches=1664,official_gaga_completed_forwards=1664,
    original_gcn_clean_forwards=128,official_gaga_clean_forwards=128,
    original_gcn_attacked_forwards=1536,official_gaga_attacked_forwards=1536,
    generator_optimizer_steps=0,fresh_teacher_model_constructions=0,
    fresh_teacher_forwards=0,autograd_calls=0,API_calls=0,TEST_queries=0)


def sha256(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def checked(reference,receipts):
    relative=pathlib.Path(reference['path'])
    if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Frozen inputs must be repository-relative.')
    path=(ROOT/relative).resolve()
    if ROOT not in path.parents or not path.is_file() or sha256(path)!=reference['sha256']:
        raise RuntimeError('A frozen input is unsafe, missing or changed: '+reference['path'])
    if receipts.setdefault(reference['path'],reference['sha256'])!=reference['sha256']:
        raise RuntimeError('Conflicting input byte hashes.')
    return path


def validate_registration(path,receipts):
    p=read_json(path)
    if (p.get('schema')!=SCHEMA or p.get('status')!='frozen' or p.get('post_hoc') is not True or
            p.get('owner_execution_authorized') is not True or p.get('final_TEST_allowed') is not False):
        raise RuntimeError('A frozen owner-authorized post-hoc/no-TEST registration is required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit tracked source before execution.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip()
    if head!=p['source_commit']:raise RuntimeError('Execution HEAD differs from the frozen registration.')
    required={f.relative_to(ROOT).as_posix() for f in pathlib.Path(__file__).parent.glob('*.py')}
    if not required<=set(p['source_files']):raise RuntimeError('New package source freeze is incomplete.')
    for name,wanted in p['source_files'].items():
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        checked(dict(path=name,sha256=wanted),receipts)
    for reference in p['inputs']+p.get('retained_inputs',[]):checked(reference,receipts)
    if (tuple(p['arms'])!=ARMS or tuple(p['seeds'])!=SEEDS or p['beta']!=2.0 or p['linear_slope']!=1.5 or
            p['deadline_seconds']!=5400 or p['cpu_threads']!=8 or p['execution_device']!='cpu' or
            p['cuda_visible_devices']!='' or len(p['valid_groups'])!=128 or len(set(p['valid_groups']))!=128 or
            not isinstance(p['selection'],dict)):
        raise RuntimeError('Fixed feature arms, parameters, fresh cohort or CPU bounds changed.')
    if p['expected_if_evaluated']!=EXPECTED_COUNTS:raise RuntimeError('Registered execution counters differ.')
    old=read_json(checked(p['base_experiment_ref'],receipts))
    if (p['base_experiment_ref']['sha256']!=BASE_SHA or old['source_commit']!=BASE_SOURCE or
            tuple(old['seeds'])!=SEEDS or old['execution_device']!='cpu' or
            old['final_test_allowed'] is not False or len(old['train_groups'])!=256 or
            p['excluded_valid_groups']!=old['valid_groups'] or set(p['valid_groups'])&set(old['valid_groups'])):
        raise RuntimeError('The exact legacy generator family or unused VALID-context exclusion differs.')
    for name,wanted in old['source_files'].items():
        if p['source_files'].get(name)!=wanted:raise RuntimeError('Original scientific source changed: '+name)
    for reference in old['inputs']+old.get('retained_inputs',[]):checked(reference,receipts)
    if set(old['baseline_runs'])!={'3','4','5'}:raise RuntimeError('Three specified legacy terminal checkpoints are required.')
    for seed in SEEDS:
        ref=old['baseline_runs'][str(seed)];directory=pathlib.Path(ref['directory'])
        checked(dict(path=(directory/'result.json').as_posix(),sha256=ref['result_sha256']),receipts)
        checked(dict(path=(directory/'final_model.pt').as_posix(),sha256=ref['final_model_sha256']),receipts)
    engine_protocol=copy.deepcopy(old);engine_protocol['valid_groups']=list(p['valid_groups'])
    changed={key for key in old if old[key]!=engine_protocol[key]}
    if changed!={'valid_groups'}:raise RuntimeError('Only VALID group IDs may change in the baseline-engine protocol clone.')
    return p,old,engine_protocol,dict(source_commit=head,protocol_sha256=sha256(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:raise RuntimeError('Private outputs must remain below ignored runs/.')
    output.mkdir(parents=True,exist_ok=False)
    for name in ('geometry','generation_starts','engine_receipts','partial'):(output/name).mkdir()
    started,cpu_started=time.monotonic(),time.process_time();counts={name:0 for name in EXPECTED_COUNTS}
    runtime=dict(cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_cuda_available=None,
        torch_visible_gpu_count=None,native_visible_gpu_count=None)
    result=dict(status='failed',counters=counts,classifier_loading_after_geometry_seal=False,
        immutable_content_hashes=dict(before=None,after=None),active_case=None)
    receipts,prepared,engine_records={},[],[];engine=data=view=None;handles=[];torch=None;active_capture=None
    def stop(signum,frame):raise TimeoutError('Frozen5400secondCPUfeature-logit-sharpen deadline reached.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(5400)
    def check():
        if time.monotonic()-started>=5400:raise TimeoutError('Frozen CPU phase deadline reached.')
    try:
        protocol_path=pathlib.Path(args.protocol).resolve();p,old,engine_protocol,provenance=validate_registration(protocol_path,receipts)
        if runtime['cuda_visibility']!='':raise RuntimeError('Explicitly empty CUDA visibility is required.')
        write_json(output/'registered.json',dict(provenance=provenance,own_source_sha256=sha256(__file__),
            base_experiment_ref=p['base_experiment_ref'],source_and_input_hashes=receipts,
            arms=list(ARMS),beta=2.0,linear_slope=1.5,valid_groups=p['valid_groups'],
            excluded_valid_groups=p['excluded_valid_groups'],selection=p['selection'],
            baseline_engine_protocol_override={'valid_groups':list(p['valid_groups'])},
            no_training_gradients_classifier_proposals_or_API=True,
            model_gate='Proposal construction loads three frozen GCN surrogates and legacy generators; GCN hooks prohibit forwards. All384graphs/1536features/canonical sequences seal before dedicated evaluators and any official victim load.'))
        import dgl
        import numpy as np
        import torch
        from utils.utils import get_n_ids,set_seed
        from course.dual_surrogate.common import case_seed
        from course.dual_surrogate.runtime import load_original_view,load_classifier,model_hash,gcn_logits,checked_logits,transitions
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.feature_polish.refine import make_baseline_engine,tensor_sha256
        from course.source_role_fix.saved_teacher_diagnostic import reconstruct
        from course.feature_logit_sharpen.core import transform_features
        runtime['torch_cuda_available']=bool(torch.cuda.is_available())
        runtime['torch_visible_gpu_count']=int(torch.cuda.device_count());runtime['native_visible_gpu_count']=int(torch._C._cuda_getDeviceCount())
        if runtime['torch_cuda_available'] or runtime['torch_visible_gpu_count'] or runtime['native_visible_gpu_count']:
            raise RuntimeError('Torch/native GPU counts must both be zero.')
        torch.set_num_threads(8)
        original_hashes=None;original_group_identity={};bounds=None
        def feature_graph(base_graph,features):
            if features.shape!=(base_graph.num_nodes()-data.graph.num_nodes(),data.features.shape[1]):
                raise RuntimeError('Fixed-graph injected features have the wrong shape.')
            local=base_graph.local_var()
            local.ndata['x']=torch.cat((data.features.detach(),features),dim=0)
            return local
        def state_receipt(current):
            return dict(temperature=float(current.model.temp_scheduler.get_temperature()),
                epsilon=float(current.model.eps_scheduler.get_epsilon()),phase=current.phase,
                current_epoch=current.current_epoch,model_training=bool(current.model.training))
        def reject_proposal_gcn(model,inputs):
            counts['proposal_gcn_forward_attempts']+=1
            raise RuntimeError('Proposal GCN inference is forbidden; generation must use cached h_GCN only.')
        def generation_started(model,inputs):
            check();counts['generator_forward_dispatches']+=1
            if counts['generator_forward_dispatches']>384:raise RuntimeError('Frozen384generator-forward cap exceeded.')
        def generation_completed(model,inputs,outputs):counts['generator_completed_forwards']+=1
        with (output/'geometry.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for seed in SEEDS:
                check();set_seed(seed);dgl.seed(seed);dgl.random.seed(seed)
                counts['baseline_engine_load_dispatches']+=1
                engine,data,baseline_receipt=make_baseline_engine(engine_protocol,seed,output,started+5400)
                counts['baseline_engines_loaded']+=1;counts['dataset_loads']+=1
                counts['proposal_gcn_models_loaded']+=1;counts['generator_models_loaded']+=1
                counts['scheduler_restoration_ticks_reported']+=8
                if data.device.type!='cpu' or data.discrete:raise RuntimeError('Only CPU continuous GossipCop is registered.')
                if (set(engine.teachers)!={'gcn_base'} or engine.victim_model is not None or engine.fit_view is not None or
                        engine.generator_forwards or engine.optimizer_steps or any(engine.classifier_forwards.values())):
                    raise RuntimeError('Baseline engine loaded an unregistered branch or performed a model call/update.')
                view=load_original_view(data)
                current_hashes=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d))
                current_bounds=(data.features.min(0).values,data.features.max(0).values)
                if original_hashes is None:
                    original_hashes=current_hashes;bounds=tuple(v.detach().clone() for v in current_bounds)
                    result['immutable_content_hashes']['before']=dict(original_hashes)
                elif current_hashes!=original_hashes or any(tensor_sha256(x)!=tensor_sha256(y) for x,y in zip(bounds,current_bounds)):
                    raise RuntimeError('Repeated baseline loaders changed original features/cache/box.')
                model_before=model_hash(engine.model);gcn_before=model_hash(engine.teachers['gcn_base'])
                schedule_before=state_receipt(engine)
                handles=[engine.teachers['gcn_base'].register_forward_pre_hook(reject_proposal_gcn),
                    engine.model.register_forward_pre_hook(generation_started),engine.model.register_forward_hook(generation_completed)]
                record=dict(seed=seed,generator_checkpoint_sha256=baseline_receipt['final_model_sha256'],
                    generator_state_before=model_before,proposal_gcn_state_before=gcn_before,
                    scheduler_restoration_ticks_reported_by_helper=8,scheduler_getters_before=schedule_before,
                    scheduler_evidence='The unchanged helper executes8step calls; recorded current getters are actual runtime values, not a claim that historical receipts prove scheduler identity.',
                    data_hashes_before=current_hashes,baseline_engine_protocol_override={'valid_groups':list(p['valid_groups'])})
                write_json(output/'engine_receipts'/('seed%d_registered.json'%seed),record)
                for group in p['valid_groups']:
                    check();engine.current_group=group
                    ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                    labels=data.labels[ids].detach().cpu()
                    if not len(ids) or not bool((labels==1).all()):raise RuntimeError('Complete registered fraud-target group is required.')
                    nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
                    identity=dict(node_ids=ids.tolist(),labels=labels.tolist(),node_budget=nb,edge_budget=eb)
                    if group in original_group_identity and original_group_identity[group]!=identity:
                        raise RuntimeError('Target identities/order or budgets changed across seeds.')
                    original_group_identity[group]=identity
                    active=dict(seed=seed,group=group,**identity,generation_seed=case_seed(seed,'validation',8,group),
                        counters_before=dict(counts),stage='generation_dispatched')
                    result['active_case']=active
                    write_json(output/'generation_starts'/('seed%d_group%d.json'%(seed,group)),active)
                    with torch.no_grad():graph=engine.get_perturbed_graph(ids,nb,eb,False)
                    counts['baseline_generations']+=1
                    capture={key:value.detach().clone() for key,value in engine.capture.items()}
                    active_capture=capture
                    engine.capture=engine.live_capture=None
                    result['active_case']=dict(active,stage='generated_features_being_prepared')
                    original=data.graph.num_nodes();injected=torch.arange(original,graph.num_nodes(),device=data.device)
                    x0=capture['x_atk'];adj=capture['adj_attack'];candidates=capture['candidate_ids']
                    identity_hashes={key:tensor_sha256(value) for key,value in capture.items()}
                    original_ids=torch.cat((ids,candidates));legal=torch.tril(torch.ones_like(adj,dtype=torch.bool),diagonal=len(original_ids)-1)
                    if (not torch.equal(capture['target_ids'],ids) or x0.shape!=(nb,data.feat_dim) or x0.dtype!=torch.float32 or
                            not bool(torch.isfinite(x0).all()) or len(torch.unique(original_ids))!=len(original_ids) or
                            not bool(((original_ids>=0)&(original_ids<original)).all()) or
                            adj.shape!=(nb,len(ids)+len(candidates)+nb) or not bool(torch.isfinite(adj).all()) or
                            not bool(((adj==0)|(adj==1)).all()) or int(adj.sum())!=eb or
                            bool((adj[~legal]!=0).any()) or not bool((adj[:,:len(ids)].sum(1)>=1).all())):
                        raise RuntimeError('Fresh generator capture does not match the exact hard graph budgets/target order.')
                    physical_degrees=(graph.in_degrees(injected)-1).to(torch.int64)
                    with torch.no_grad():proposal=transform_features(x0,bounds[0],bounds[1],seed=seed,group=group,original_count=original)
                    if tuple(proposal['arms'])!=ARMS or proposal['beta']!=2.0 or proposal['linear_slope']!=1.5:
                        raise RuntimeError('The frozen four feature transformations changed.')
                    features=proposal['arms']
                    if tensor_sha256(features['baseline'])!=identity_hashes['x_atk']:raise RuntimeError('Baseline bytes changed.')
                    sequences,variant_metadata={},{};baseline_sequence=None
                    with torch.no_grad():
                        for arm in ARMS:
                            local=feature_graph(graph,features[arm]);counts['geometry_feature_graph_variants']+=1
                            if (local.num_nodes()!=graph.num_nodes() or local.num_edges()!=graph.num_edges() or
                                    not torch.equal(local.in_degrees(injected)-1,physical_degrees)):
                                raise RuntimeError('Feature transformation changed topology/physical degrees.')
                            sequence=updated_sequences(view,local,ids,candidates,injected,training=False)
                            counts['canonical_sequence_updates']+=1
                            if arm=='baseline':baseline_sequence=sequence
                            if not torch.equal(sequence[:,FIXED_TOKENS],baseline_sequence[:,FIXED_TOKENS]):
                                raise RuntimeError('SELF/knownTRAIN first/second-hop canonical tokens changed.')
                            counts['fixed_token_invariant_checks']+=1;sequences[arm]=sequence.detach().cpu()
                            variant_metadata[arm]=dict(feature_content_sha256=tensor_sha256(features[arm]),
                                canonical_sequence_content_sha256=tensor_sha256(sequence),
                                unchanged_tokens0_1_2_4_5_exact=True,actual_injected_nodes=len(injected),
                                actual_injected_nonself_undirected_edges=(local.num_edges()-data.graph.num_edges()-len(injected))//2,
                                physical_injected_degrees=physical_degrees.tolist(),identity=proposal['identity'][arm])
                            del local
                    if identity_hashes!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Feature transformations mutated original capture tensors.')
                    artifact_path=output/'geometry'/('seed%d_group%d.pt'%(seed,group))
                    artifact=dict(capture={key:value.detach().cpu() for key,value in capture.items()},
                        features={arm:value.detach().cpu() for arm,value in features.items()},canonical_sequences=sequences)
                    with artifact_path.open('xb') as handle:torch.save(artifact,handle)
                    row=dict(seed=seed,group=group,node_ids=ids.tolist(),labels=labels.tolist(),targets=len(ids),
                        node_budget=nb,edge_budget=eb,actual_injected_nodes=nb,
                        actual_injected_nonself_undirected_edges=eb,generation_seed=case_seed(seed,'validation',8,group),
                        immutable_capture_content_hashes=identity_hashes,
                        artifact=dict(path=artifact_path.relative_to(output).as_posix(),sha256=sha256(artifact_path)),
                        variants=variant_metadata,feature_transform={key:value for key,value in proposal.items() if key!='arms'})
                    prepared.append(row);counts['feature_cases_prepared']+=1
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();result['active_case']=None;active_capture=None
                    if len(prepared)%32==0:print('FEATURE_LOGIT_SHARPEN_GEOMETRY',len(prepared),flush=True)
                    del graph,capture,artifact,sequences,features,x0,adj,proposal
                record.update(generator_state_after=model_hash(engine.model),proposal_gcn_state_after=model_hash(engine.teachers['gcn_base']),
                    scheduler_getters_after=state_receipt(engine),generator_completed_forwards=engine.generator_forwards,
                    classifier_forwards=dict(engine.classifier_forwards),optimizer_steps=engine.optimizer_steps,
                    data_hashes_after=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d)))
                if (record['generator_state_after']!=model_before or record['proposal_gcn_state_after']!=gcn_before or
                        record['scheduler_getters_after']!=schedule_before or record['data_hashes_after']!=current_hashes or
                        engine.generator_forwards!=128 or engine.optimizer_steps or any(engine.classifier_forwards.values()) or
                        any(parameter.grad is not None for model in (engine.model,engine.teachers['gcn_base']) for parameter in model.parameters())):
                    raise RuntimeError('Fresh baseline generation changed frozen parameters/schedule/data or invoked a classifier/update.')
                write_json(output/'engine_receipts'/('seed%d_completed.json'%seed),record);engine_records.append(record)
                for handle in handles:handle.remove()
                handles=[];engine=None
        for key in ('baseline_generations','generator_forward_dispatches','generator_completed_forwards','feature_cases_prepared'):
            if counts[key]!=384:raise RuntimeError('All384fresh complete captures must seal before any victim.')
        if counts['proposal_gcn_forward_attempts'] or any(counts[name] for name in ('original_gcn_load_dispatches','official_gaga_load_dispatches')):
            raise RuntimeError('An unregistered proposal classifier was used before sealing.')
        seed_totals=[]
        for seed in SEEDS:
            rows=[row for row in prepared if row['seed']==seed]
            if len(rows)!=128:raise RuntimeError('Fresh-cohort group denominator changed.')
            for arm in ARMS:
                seed_totals.append(dict(seed=seed,arm=arm,groups=128,target_occurrences=sum(row['targets'] for row in rows),
                    unique_targets=len({node for row in rows for node in row['node_ids']}),
                    summed_injected_nodes=sum(row['variants'][arm]['actual_injected_nodes'] for row in rows),
                    summed_nonself_undirected_edges=sum(row['variants'][arm]['actual_injected_nonself_undirected_edges'] for row in rows)))
        marker=dict(status='all_fresh_graph_feature_proposals_sealed_before_dedicated_evaluators_and_any_official_victim',
            provenance=provenance,source_captures=384,variants=1536,canonical_sequences=1536,fixed_tokens=list(FIXED_TOKENS),
            counters_before_evaluation=dict(counts),geometry_jsonl_sha256=sha256(output/'geometry.jsonl'),
            artifacts=[dict(seed=row['seed'],group=row['group'],**row['artifact']) for row in prepared],
            original_content_hashes=original_hashes,whole_seed_geometry_totals=seed_totals,
            engine_receipt_hashes={('engine_receipts/seed%d_completed.json'%seed):sha256(output/'engine_receipts'/('seed%d_completed.json'%seed)) for seed in SEEDS},
            minimum_content_sha256=tensor_sha256(bounds[0]),maximum_content_sha256=tensor_sha256(bounds[1]))
        write_json(output/'geometry_sealed.json',marker)
        result.update(provenance=provenance,geometry_marker_sha256=sha256(output/'geometry_sealed.json'),
            all_geometry_prepared_before_dedicated_evaluators=True,all_geometry_prepared_before_official_victim=True,
            generator_receipts=engine_records,whole_seed_geometry_totals=seed_totals)
        nonidentity_by_seed={str(seed):sum(not row['feature_transform']['identity']['logit_sharp']['exact_equals_baseline']
            for row in prepared if row['seed']==seed) for seed in SEEDS}
        no_headroom=any(value==0 for value in nonidentity_by_seed.values())
        headroom=dict(provenance=provenance,groups_per_seed=128,sealed_cases=384,
            primary='logit_sharp',nonidentity_cases_by_seed=nonidentity_by_seed,
            rule='Stop before dedicated evaluators if ANY seed has zero nonidentity primary feature proposals; retain all384 generated cases and complete denominators.',
            stop_before_evaluation=no_headroom,uses_model_predictions=False,
            original_generator_calls_already_spent=counts['generator_completed_forwards'],
            proposal_gcn_models_already_loaded=counts['proposal_gcn_models_loaded'])
        write_json(output/'headroom.json',headroom);result['headroom']=headroom
        if no_headroom:
            immutable_after=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d))
            result['immutable_content_hashes']['after']=immutable_after
            if immutable_after!=result['immutable_content_hashes']['before']:
                raise RuntimeError('Original features/cache changed before the planned no-headroom stop.')
            result.update(status='completed_no_headroom',immutable_content_hashes_retained=True,
                planned_stop_before_dedicated_evaluation=True,
                interpretation='All384generator calls and1536feature/canonical preparations completed; a seed has zero primary feature changes. No dedicated GCN or official GAGA loaded or queried. No performance result or winner.',
                final_TEST_used=False,experimental_API_calls=0)
            return
        counts['original_gcn_load_dispatches']+=1;gcn=load_classifier('gcn_base',data);counts['original_gcn_models_loaded']+=1
        counts['official_gaga_load_dispatches']+=1;victim=load_classifier('gaga_victim',data);counts['official_gaga_models_loaded']+=1
        result['classifier_loading_after_geometry_seal']=True
        result['immutable_content_hashes']['before'].update(original_gcn=model_hash(gcn),official_gaga=model_hash(victim))
        def classify(name,graph=None,ids=None,sequence=None,clean=False):
            check();key='original_gcn' if name=='gcn' else 'official_gaga'
            if counts[key+'_forward_dispatches']>=1664:raise RuntimeError('Frozen classifier forward cap exhausted.')
            counts[key+'_forward_dispatches']+=1
            logits=gcn_logits(gcn,graph,ids,training=False) if name=='gcn' else checked_logits(victim(sequence),len(ids))
            counts[key+'_completed_forwards']+=1;counts[key+('_clean_forwards' if clean else '_attacked_forwards')]+=1
            return logits
        clean_sequence=np.load(ROOT/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',allow_pickle=False,mmap_mode='r')
        clean_predictions={}
        with torch.no_grad(),(output/'clean_predictions.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for group in p['valid_groups']:
                ids=torch.tensor(original_group_identity[group]['node_ids'],device=data.device,dtype=torch.long)
                ga=classify('gcn',graph=data.graph,ids=ids,clean=True)
                sequence=torch.from_numpy(np.array(clean_sequence[ids.cpu().numpy()],copy=True)).to(data.device)
                gb=classify('gaga',ids=ids,sequence=sequence,clean=True)
                labels=data.labels[ids].cpu();clean_predictions[group]=(ga.argmax(1).cpu(),gb.argmax(1).cpu())
                stream.write(json.dumps(dict(group=group,node_ids=ids.tolist(),labels=labels.tolist(),gcn_logits=ga.tolist(),gaga_logits=gb.tolist()),allow_nan=False)+'\n');stream.flush()
        totals={(arm,seed):{name:dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0) for name in ('gcn','gaga')} for arm in ARMS for seed in SEEDS}
        streams={(arm,seed):(output/('%s_seed%d_predictions.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n') for arm in ARMS for seed in SEEDS}
        try:
            with torch.no_grad():
                for row in prepared:
                    check();seed,group=row['seed'],row['group'];artifact_path=output/row['artifact']['path']
                    if sha256(artifact_path)!=row['artifact']['sha256']:raise RuntimeError('A sealed proposal artifact changed.')
                    counts['prepared_artifact_tensor_loads']+=1;payload=torch.load(artifact_path,map_location='cpu')
                    if set(payload)!={'capture','features','canonical_sequences'}:raise RuntimeError('Prepared schema differs.')
                    capture=payload['capture']
                    if row['immutable_capture_content_hashes']!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Baseline capture changed after proposal sealing.')
                    identity=dict(group=group,node_ids=row['node_ids'],labels=row['labels'])
                    graph_record=dict(row,actual_injected_edges=row['edge_budget'])
                    graph,ids,candidates,injected=reconstruct(data,capture,graph_record,identity)
                    counts['evaluation_base_graph_reconstructions']+=1
                    for arm in ARMS:
                        features=payload['features'][arm];sequence=payload['canonical_sequences'][arm];meta=row['variants'][arm]
                        if tensor_sha256(features)!=meta['feature_content_sha256'] or tensor_sha256(sequence)!=meta['canonical_sequence_content_sha256']:
                            raise RuntimeError('Sealed features/canonical sequence changed.')
                        local=feature_graph(graph,features);counts['evaluation_feature_graph_variants']+=1
                        ga=classify('gcn',graph=local,ids=ids);gb=classify('gaga',ids=ids,sequence=sequence.to(data.device))
                        if tensor_sha256(features)!=meta['feature_content_sha256'] or tensor_sha256(sequence)!=meta['canonical_sequence_content_sha256']:
                            raise RuntimeError('Classifier mutated a sealed proposal.')
                        labels=data.labels[ids].cpu()
                        predicted_row=dict(seed=seed,group=group,arm=arm,node_ids=ids.tolist(),labels=labels.tolist(),gcn_logits=ga.tolist(),gaga_logits=gb.tolist(),
                            node_budget=row['node_budget'],edge_budget=row['edge_budget'],actual_injected_nodes=len(injected),
                            actual_injected_edges=(local.num_edges()-data.graph.num_edges()-len(injected))//2,
                            prepared_artifact_sha256=row['artifact']['sha256'],original_adjacency_content_sha256=row['immutable_capture_content_hashes']['adj_attack'],
                            target_order_content_sha256=row['immutable_capture_content_hashes']['target_ids'],candidate_order_content_sha256=row['immutable_capture_content_hashes']['candidate_ids'],
                            injected_feature_content_sha256=meta['feature_content_sha256'],canonical_sequence_content_sha256=meta['canonical_sequence_content_sha256'])
                        for index,name in enumerate(('gcn','gaga')):
                            transition=transitions(labels,clean_predictions[group][index],(ga if name=='gcn' else gb).argmax(1).cpu())
                            predicted_row[name+'_transitions']=transition
                            for key,value in transition.items():totals[arm,seed][name][key]+=value
                        streams[arm,seed].write(json.dumps(predicted_row,allow_nan=False)+'\n');streams[arm,seed].flush();del local
                    if row['immutable_capture_content_hashes']!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Evaluation mutated original capture bytes.')
                    del graph,payload,capture
        finally:
            for stream in streams.values():stream.close()
        if counts!=EXPECTED_COUNTS:raise RuntimeError('Fixed fresh-context model/canonical counters differ.')
        summaries=[]
        for seed in SEEDS:
            geom=next(row for row in seed_totals if row['seed']==seed and row['arm']=='baseline')
            for arm in ARMS:
                if any(values['occurrences']!=geom['target_occurrences'] for values in totals[arm,seed].values()):
                    raise RuntimeError('A measured complete-cohort denominator changed.')
                summaries.append(dict(seed=seed,arm=arm,groups=128,unique_targets=geom['unique_targets'],totals=totals[arm,seed],
                    gcn_mr_percent=100*totals[arm,seed]['gcn']['attacked_wrong']/geom['target_occurrences'],
                    gaga_mr_percent=100*totals[arm,seed]['gaga']['attacked_wrong']/geom['target_occurrences']))
        immutable_after=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d),original_gcn=model_hash(gcn),official_gaga=model_hash(victim))
        result['immutable_content_hashes']['after']=immutable_after
        if immutable_after!=result['immutable_content_hashes']['before']:raise RuntimeError('Original features/cache/classifiers changed.')
        if any(parameter.grad is not None or parameter.requires_grad for model in (gcn,victim) for parameter in model.parameters()):
            raise RuntimeError('Dedicated evaluator gradient/frozen flags changed.')
        result.update(status='completed',summaries=summaries,immutable_content_hashes_retained=True,
            baseline_channel='Fresh legacy terminal8 generators on ID-only fresh VALID contexts; no historical absolute score treated as paired',
            official_gaga_channel='All1536canonical sequences sealed before any official-victim loading',
            proposal_gcn_channel='Three frozen constructor loads; forward hooks and inherited counters require zero classifier calls during generation/proposals',
            RNG_limitation='Unchanged per-case dual-surrogate namespace/validation/epoch8; DGL native scheduling state is not claimed bitwise identical across separate processes or historical device runs',
            final_TEST_used=False,experimental_API_calls=0)
    except BaseException as error:
        traceback.print_exc();result.update(status='failed',error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
        if engine is not None:
            result['partial_engine_counters']=dict(generator_forwards=engine.generator_forwards,optimizer_steps=engine.optimizer_steps,classifier_forwards=dict(engine.classifier_forwards))
            partial_capture=engine.capture if engine.capture is not None else active_capture
            if partial_capture is not None and torch is not None:
                try:
                    path=output/'partial'/'failed_active_capture.pt'
                    with path.open('xb') as stream:torch.save({key:value.detach().cpu() for key,value in partial_capture.items()},stream)
                    result['failed_active_capture']=dict(path=path.relative_to(output).as_posix(),sha256=sha256(path))
                except BaseException as capture_error:result['partial_capture_write_error']=repr(capture_error)
    finally:
        signal.alarm(0)
        for handle in handles:handle.remove()
        write_json(output/'input_hashes.json',receipts)
        result.update(counters=counts,prepared_cases=len(prepared),process_cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.monotonic()-started,GPU_hours=0,**runtime)
        names=['registered.json','geometry.jsonl','geometry_sealed.json','headroom.json','clean_predictions.jsonl','input_hashes.json']
        names.extend('%s_seed%d_predictions.jsonl'%(arm,seed) for arm in ARMS for seed in SEEDS)
        names.extend('engine_receipts/seed%d_%s.json'%(seed,state) for seed in SEEDS for state in ('registered','completed'))
        result['output_hashes']={name:sha256(output/name) for name in names if (output/name).is_file()}
        write_json(output/'result.json',result);print('FEATURE_LOGIT_SHARPEN_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
