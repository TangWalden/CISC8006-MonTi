"""Seal TRAIN-only reference prototypes and all feature variants before models.

No fitting, new generator, gradients, surrogate guidance or API is used.
The original hard graphs and x0 baseline are reused from one fixed CPU phase.
Only unique original TRAIN labels select prototype pools; all controls are fixed.
"""
import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback


ROOT=pathlib.Path(__file__).resolve().parents[2]
SCHEMA='monti_train_prototype_mix_v1'
ARMS=('baseline','benign_mix','all_train_mix','fraud_mix')
SEEDS=(3,4,5)
FIXED_TOKENS=(0,1,2,4,5)


def sha256(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def checked(reference,receipts):
    relative=pathlib.Path(reference['path'])
    if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Frozen inputs must be repository-relative.')
    path=(ROOT/relative).resolve()
    if ROOT not in path.parents or not path.is_file() or sha256(path)!=reference['sha256']:
        raise RuntimeError('A frozen input is unsafe, missing or changed: '+reference['path'])
    receipts[reference['path']]=reference['sha256'];return path


def index_jsonl(path,groups):
    indexed={}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row=json.loads(line);group=int(row['group'])
            if group in indexed:raise RuntimeError('Duplicate original group row.')
            indexed[group]=row
    if len(indexed)!=128 or set(indexed)!=set(groups):raise RuntimeError('Retain all128fixedVALIDgroups.')
    return indexed


def validate_registration(path,receipts):
    p=read_json(path)
    if (p.get('schema')!=SCHEMA or p.get('status')!='frozen' or p.get('post_hoc') is not True or
            p.get('owner_execution_authorized') is not True or p.get('final_TEST_allowed') is not False):
        raise RuntimeError('A frozen owner-authorized post-hoc/no-TEST prototype registration is required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit tracked source before execution.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip()
    if head!=p['source_commit']:raise RuntimeError('Execution HEAD differs from the frozen registration.')
    required={f.relative_to(ROOT).as_posix() for f in pathlib.Path(__file__).parent.glob('*.py')}
    if not required<=set(p['source_files']):raise RuntimeError('Prototype package source freeze is incomplete.')
    for name,wanted in p['source_files'].items():
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        checked(dict(path=name,sha256=wanted),receipts)
    for reference in p['inputs']+p.get('retained_inputs',[]):checked(reference,receipts)
    if (tuple(p['arms'])!=ARMS or tuple(p['seeds'])!=SEEDS or p['alpha']!=.5 or
            p['deadline_seconds']!=5400 or p['cpu_threads']!=8 or len(p['valid_groups'])!=128 or
            len(set(p['valid_groups']))!=128 or set(p['runs'])!={'3','4','5'}):
        raise RuntimeError('The fixed four arms, alpha, cohort or CPU bounds changed.')
    old_path=checked(p['base_experiment_ref'],receipts);old=read_json(old_path)
    if (tuple(old['seeds'])!=SEEDS or old['valid_groups']!=p['valid_groups'] or old['execution_device']!='cpu' or
            old['final_test_allowed'] is not False):
        raise RuntimeError('Original CPU baseline experiment differs.')
    for name,wanted in old['source_files'].items():
        if p['source_files'].get(name)!=wanted:raise RuntimeError('Original scientific source changed: '+name)
    for reference in old['inputs']+old.get('retained_inputs',[]):checked(reference,receipts)
    if 'parity_result_ref' in p:
        parity=read_json(checked(p['parity_result_ref'],receipts))
        if (parity['status']!='completed' or parity['parity_passed'] is not True or parity['violation_count']!=0 or
                parity['completed_comparison_rows']!=768 or parity['teacher_argmax_identity'] is not True):
            raise RuntimeError('The declared prior all-case parity rationale is incomplete.')
    runs={}
    for seed in SEEDS:
        spec=p['runs'][str(seed)];result_path=checked(spec['result_ref'],receipts);result=read_json(result_path)
        rows_path=checked(spec['groups_ref'],receipts)
        if (result['status']!='completed' or result['seed']!=seed or result['groups']!=128 or
                result['source_commit']!=old['source_commit'] or
                result['protocol_sha256']!=p['base_experiment_ref']['sha256'] or
                result['groups_jsonl_sha256']!=sha256(rows_path)):
            raise RuntimeError('A retained successful baseline capture run differs.')
        rows=index_jsonl(rows_path,p['valid_groups'])
        captures={int(row['group']):row for row in spec['captures']}
        originals={int(row['group']):row for row in result['exports']}
        if (len(spec['captures'])!=128 or len(result['exports'])!=128 or set(captures)!=set(rows) or set(originals)!=set(rows)):
            raise RuntimeError('All128captures per original seed must be frozen.')
        paths={}
        for group,reference in captures.items():
            capture_path=checked(reference,receipts)
            if (originals[group]!=rows[group]['graph_capture'] or reference['sha256']!=originals[group]['sha256'] or
                    capture_path!=(result_path.parent/originals[group]['path']).resolve()):
                raise RuntimeError('A capture differs from its original result/row/byte identity.')
            paths[group]=capture_path
        runs[seed]=dict(rows=rows,captures=paths)
    return p,old,runs,dict(source_commit=head,protocol_sha256=sha256(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:raise RuntimeError('Private outputs must remain below ignored runs/.')
    output.mkdir(parents=True,exist_ok=False);(output/'geometry').mkdir()
    started,cpu_started=time.monotonic(),time.process_time()
    counts=dict(reference_selections=0,reference_tensor_saves=0,source_capture_tensor_loads=0,source_captures_prepared=0,
        geometry_base_graph_reconstructions=0,geometry_feature_graph_variants=0,canonical_sequence_updates=0,fixed_token_invariant_checks=0,
        prepared_artifact_tensor_loads=0,evaluation_base_graph_reconstructions=0,evaluation_feature_graph_variants=0,
        original_gcn_load_dispatches=0,original_gcn_models_loaded=0,official_gaga_load_dispatches=0,official_gaga_models_loaded=0,
        original_gcn_forward_dispatches=0,original_gcn_completed_forwards=0,official_gaga_forward_dispatches=0,official_gaga_completed_forwards=0,
        original_gcn_clean_forwards=0,official_gaga_clean_forwards=0,original_gcn_attacked_forwards=0,official_gaga_attacked_forwards=0,
        fresh_teacher_model_constructions=0,fresh_teacher_forwards=0,generator_model_constructions=0,generator_forwards=0,
        optimizer_steps=0,autograd_calls=0,API_calls=0,TEST_queries=0)
    runtime=dict(cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),torch_cuda_available=None,
        torch_visible_gpu_count=None,native_visible_gpu_count=None)
    result=dict(status='failed',counters=counts,classifier_loading_after_geometry_seal=False,
        immutable_content_hashes=dict(before=None,after=None))
    receipts,prepared={},[]
    def stop(signum,frame):raise TimeoutError('Frozen5400secondCPUprototype-mixingdeadline reached.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(5400)
    try:
        protocol_path=pathlib.Path(args.protocol).resolve();p,old,runs,provenance=validate_registration(protocol_path,receipts)
        if runtime['cuda_visibility']!='':raise RuntimeError('Explicitly empty CUDA visibility is required.')
        write_json(output/'registered.json',dict(provenance=provenance,own_source_sha256=sha256(__file__),
            base_experiment_ref=p['base_experiment_ref'],source_and_input_hashes=receipts,
            arms=list(ARMS),alpha=.5,source_captures=384,
            model_load_gate='Seal one TRAIN-only reference and all384graphs/1536featurevariants/canonicalsequences/fixedtokensfirst',
            reference_selection='Only unique originalTRAIN rows/labels; full unlabeled feature box; no outcome-based reference or arm selection',
            no_new_teacher_generator_training_gradient_or_API=True))
        import dgl
        import numpy as np
        import torch
        from utils.utils import get_n_ids,set_seed
        from course.dual_surrogate.runtime import load_data,load_original_view,load_classifier,model_hash,gcn_logits,checked_logits,transitions
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.source_role_fix.saved_teacher_diagnostic import reconstruct
        from course.feature_polish.refine import tensor_sha256
        from course.train_prototype_mix.core import selectreference,mixfeatures
        runtime['torch_cuda_available']=bool(torch.cuda.is_available())
        runtime['torch_visible_gpu_count']=int(torch.cuda.device_count());runtime['native_visible_gpu_count']=int(torch._C._cuda_getDeviceCount())
        if runtime['torch_cuda_available'] or runtime['torch_visible_gpu_count'] or runtime['native_visible_gpu_count']:
            raise RuntimeError('Torch/native visible GPU counts must both be zero.')
        torch.set_num_threads(8);set_seed(20261105);dgl.seed(20261105);dgl.random.seed(20261105)
        data=load_data(old,20261105);view=load_original_view(data)
        if data.device.type!='cpu' or data.discrete:raise RuntimeError('This phase requires CPU continuous GossipCop features.')
        original=data.graph.num_nodes()
        immutable_before=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d))
        result['immutable_content_hashes']['before']=dict(immutable_before)
        counts['reference_selections']+=1
        reference=selectreference(data.features,data.labels,data.idx_train,data.idx_valid,data.idx_test)
        if len(reference['train_ids'])!=6595 or len(reference['benign_ids'])+len(reference['fraud_ids'])!=6595:
            raise RuntimeError('Reference must retain the6595uniqueoriginalTRAINrows.')
        reference_hashes={key:tensor_sha256(value) for key,value in reference.items() if torch.is_tensor(value)}
        reference_path=output/'reference.pt'
        with reference_path.open('xb') as handle:torch.save(reference,handle)
        counts['reference_tensor_saves']+=1
        reference_metadata=dict(status='reference_sealed_before_classifier',provenance=provenance,alpha=.5,
            counts=reference['counts'],labels_read_scope=reference['labels_read_scope'],
            reference_scope=reference.get('reference_scope','Unique originalTRAIN class/all means; full unlabeled original feature box'),
            arithmetic='Unique TRAIN means accumulatedfloat64 thenfloat32; raw features, no L2 normalization',
            content_hashes=reference_hashes,artifact=dict(path=reference_path.name,sha256=sha256(reference_path)),
            no_model_outcome_reference_selection=True)
        write_json(output/'reference.json',reference_metadata)
        reference_manifest=dict(status='sealed',reference_pt_sha256=sha256(reference_path),
            reference_json_sha256=sha256(output/'reference.json'),content_hashes=reference_hashes,
            counters_before_models=dict(counts))
        write_json(output/'reference_sealed.json',reference_manifest)

        def feature_graph(base_graph,features):
            if features.shape!=(base_graph.num_nodes()-original,data.feat_dim):raise RuntimeError('Variant feature shape differs.')
            local=base_graph.local_var()
            # local_var shares underlying tensors: assign a NEW complete tensor,
            # never slice-write or otherwise mutate a shared original feature.
            local.ndata['x']=torch.cat((data.features.detach(),features),dim=0)
            return local

        with torch.no_grad(),(output/'geometry.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for seed in SEEDS:
                for group in p['valid_groups']:
                    saved=runs[seed]['rows'][group];path=runs[seed]['captures'][group]
                    if sha256(path)!=saved['graph_capture']['sha256']:raise RuntimeError('Source capture changed after freeze.')
                    counts['source_capture_tensor_loads']+=1;payload=torch.load(path,map_location='cpu')
                    if set(payload)!={'capture','features'}:raise RuntimeError('Original capture schema differs.')
                    capture=payload['capture'];x0=capture['x_atk'];adjacency=capture['adj_attack']
                    if (not bool(torch.isfinite(adjacency).all()) or not bool(((adjacency==0)|(adjacency==1)).all()) or
                            tensor_sha256(adjacency)!=saved['adjacency_content_sha256'] or
                            not torch.equal(payload['features']['baseline'],x0) or
                            tensor_sha256(x0)!=saved['feature_statistics']['baseline']['content_sha256'] or
                            capture['target_ids'].tolist()!=saved['node_ids']):
                        raise RuntimeError('Baseline hard adjacency, x0 or target identities differ.')
                    identity_hashes={key:tensor_sha256(value) for key,value in capture.items()}
                    proposal=mixfeatures(x0,reference)
                    if tuple(proposal['arms'])!=ARMS or proposal['alpha']!=.5:raise RuntimeError('Fixed prototype arm/alpha contract differs.')
                    features=proposal['arms']
                    if tensor_sha256(features['baseline'])!=identity_hashes['x_atk']:raise RuntimeError('The baseline feature bytes changed.')
                    graph_record=dict(saved,actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                    identity=dict(group=group,node_ids=saved['node_ids'],labels=saved['labels'])
                    graph,ids,candidates,injected=reconstruct(data,capture,graph_record,identity)
                    counts['geometry_base_graph_reconstructions']+=1
                    physical_degrees=(graph.in_degrees(injected)-1).to(torch.int64)
                    sequences,variant_metadata={},{};baseline_sequence=None
                    for arm in ARMS:
                        local=feature_graph(graph,features[arm]);counts['geometry_feature_graph_variants']+=1
                        if (local.num_nodes()!=graph.num_nodes() or local.num_edges()!=graph.num_edges() or
                                not torch.equal(local.in_degrees(injected)-1,physical_degrees)):
                            raise RuntimeError('Prototype mixing changed hard topology or physical budgets/degrees.')
                        sequence=updated_sequences(view,local,ids,candidates,injected,training=False)
                        counts['canonical_sequence_updates']+=1
                        if arm=='baseline':baseline_sequence=sequence
                        if not torch.equal(sequence[:,FIXED_TOKENS],baseline_sequence[:,FIXED_TOKENS]):
                            raise RuntimeError('SELF/knownTRAIN first/second-hop tokens changed before model construction.')
                        counts['fixed_token_invariant_checks']+=1;sequences[arm]=sequence.detach().cpu()
                        delta=(features[arm]-x0).double()
                        variant_metadata[arm]=dict(feature_content_sha256=tensor_sha256(features[arm]),
                            canonical_sequence_content_sha256=tensor_sha256(sequence),
                            unchanged_tokens0_1_2_4_5_exact=True,actual_injected_nodes=len(injected),
                            actual_injected_nonself_undirected_edges=(local.num_edges()-data.graph.num_edges()-len(injected))//2,
                            physical_injected_degrees=physical_degrees.tolist(),raw_displacement_linf=float(delta.abs().max()),
                            raw_displacement_l2=float(delta.norm()),identity=proposal['identity'][arm])
                        del local
                    if identity_hashes!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Feature proposals mutated original x0, adjacency or ID tensors.')
                    artifact_path=output/'geometry'/('seed%d_group%d.pt'%(seed,group))
                    artifact=dict(capture={key:value.detach().cpu() for key,value in capture.items()},
                        features={arm:value.detach().cpu() for arm,value in features.items()},canonical_sequences=sequences)
                    with artifact_path.open('xb') as handle:torch.save(artifact,handle)
                    row=dict(seed=seed,group=group,node_ids=saved['node_ids'],labels=saved['labels'],targets=len(ids),
                        node_budget=saved['node_budget'],edge_budget=saved['edge_budget'],source_capture_sha256=saved['graph_capture']['sha256'],
                        immutable_capture_content_hashes=identity_hashes,reference_pt_sha256=reference_manifest['reference_pt_sha256'],
                        artifact=dict(path=artifact_path.relative_to(output).as_posix(),sha256=sha256(artifact_path)),
                        variants=variant_metadata,alpha=.5,zero_width_dimensions=proposal['zero_width_dimensions'],
                        baseline_box_max_violation=proposal['baseline_box_max_violation'],baseline_exact_box=proposal['baseline_exact_box'])
                    prepared.append(row);counts['source_captures_prepared']+=1
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    if len(prepared)%32==0:print('TRAIN_PROTOTYPE_GEOMETRY',len(prepared),flush=True)
                    del graph,payload,capture,artifact,sequences,features,x0,adjacency
        if (counts['reference_selections']!=1 or counts['source_captures_prepared']!=384 or
                counts['geometry_base_graph_reconstructions']!=384 or counts['geometry_feature_graph_variants']!=1536 or
                counts['canonical_sequence_updates']!=1536 or counts['fixed_token_invariant_checks']!=1536 or
                any(counts[name] for name in ('original_gcn_load_dispatches','official_gaga_load_dispatches',
                    'original_gcn_forward_dispatches','official_gaga_forward_dispatches'))):
            raise RuntimeError('All reference/proposal/canonical invariants must seal before any classifier.')
        seed_totals=[]
        for seed in SEEDS:
            selected=[row for row in prepared if row['seed']==seed]
            if len(selected)!=128 or sum(row['targets'] for row in selected)!=1091 or len({node for row in selected for node in row['node_ids']})!=374:
                raise RuntimeError('Prototype preparation changed original cohort denominators.')
            for arm in ARMS:
                nodes=sum(row['variants'][arm]['actual_injected_nodes'] for row in selected)
                edges=sum(row['variants'][arm]['actual_injected_nonself_undirected_edges'] for row in selected)
                if (nodes,edges)!=(7776,358491):raise RuntimeError('Hard physical budgets changed before model gate.')
                seed_totals.append(dict(seed=seed,arm=arm,groups=128,target_occurrences=1091,unique_targets=374,
                    summed_injected_nodes=nodes,summed_nonself_undirected_edges=edges))
        marker=dict(status='all_feature_proposals_sealed_before_any_classifier',provenance=provenance,
            reference_seal_sha256=sha256(output/'reference_sealed.json'),source_captures=384,variants=1536,
            canonical_sequences=1536,fixed_tokens=list(FIXED_TOKENS),counters_before_models=dict(counts),
            geometry_jsonl_sha256=sha256(output/'geometry.jsonl'),
            artifacts=[dict(seed=row['seed'],group=row['group'],**row['artifact']) for row in prepared],
            original_content_hashes=immutable_before,whole_seed_geometry_totals=seed_totals)
        marker_path=output/'geometry_sealed.json';write_json(marker_path,marker)
        result.update(provenance=provenance,geometry_marker_sha256=sha256(marker_path),all_geometry_prepared_before_classifier=True)
        if (sha256(reference_path)!=reference_manifest['reference_pt_sha256'] or
                reference_hashes!={key:tensor_sha256(value) for key,value in reference.items() if torch.is_tensor(value)}):
            raise RuntimeError('TRAIN reference changed after proposal sealing.')

        result['classifier_loading_after_geometry_seal']=True
        counts['original_gcn_load_dispatches']+=1;gcn=load_classifier('gcn_base',data);counts['original_gcn_models_loaded']+=1
        counts['official_gaga_load_dispatches']+=1;victim=load_classifier('gaga_victim',data);counts['official_gaga_models_loaded']+=1
        result['immutable_content_hashes']['before'].update(original_gcn=model_hash(gcn),official_gaga=model_hash(victim))
        for model in (gcn,victim):
            if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError('Original evaluation classifiers must stay frozen in eval mode.')
        def classify(name,graph=None,ids=None,sequence=None,clean=False):
            key='original_gcn' if name=='gcn' else 'official_gaga'
            if counts[key+'_forward_dispatches']>=1664:raise RuntimeError('Frozen1664classifierforwardcap exhausted.')
            counts[key+'_forward_dispatches']+=1
            logits=gcn_logits(gcn,graph,ids,training=False) if name=='gcn' else checked_logits(victim(sequence),len(ids))
            counts[key+'_completed_forwards']+=1;counts[key+('_clean_forwards' if clean else '_attacked_forwards')]+=1
            return logits
        clean_sequence=np.load(ROOT/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',allow_pickle=False,mmap_mode='r')
        clean_predictions={}
        with torch.no_grad(),(output/'clean_predictions.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for group in p['valid_groups']:
                ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                ga=classify('gcn',graph=data.graph,ids=ids,clean=True)
                sequence=torch.from_numpy(np.array(clean_sequence[ids.cpu().numpy()],copy=True)).to(data.device)
                gb=classify('gaga',ids=ids,sequence=sequence,clean=True)
                labels=data.labels[ids].cpu();clean_predictions[group]=(ga.argmax(1).cpu(),gb.argmax(1).cpu())
                stream.write(json.dumps(dict(group=group,node_ids=ids.tolist(),labels=labels.tolist(),
                    gcn_logits=ga.tolist(),gaga_logits=gb.tolist()),allow_nan=False)+'\n');stream.flush()
        totals={(arm,seed):{name:dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0)
            for name in ('gcn','gaga')} for arm in ARMS for seed in SEEDS}
        streams={(arm,seed):(output/('%s_seed%d_predictions.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n')
            for arm in ARMS for seed in SEEDS}
        try:
            with torch.no_grad():
                for row in prepared:
                    seed,group=row['seed'],row['group'];artifact_path=output/row['artifact']['path']
                    if sha256(artifact_path)!=row['artifact']['sha256']:raise RuntimeError('A sealed feature/sequence artifact changed.')
                    counts['prepared_artifact_tensor_loads']+=1;payload=torch.load(artifact_path,map_location='cpu')
                    if set(payload)!={'capture','features','canonical_sequences'}:raise RuntimeError('Prepared artifact schema differs.')
                    capture=payload['capture']
                    if row['immutable_capture_content_hashes']!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Original capture x0, adjacency or ID tensors changed after seal.')
                    saved=runs[seed]['rows'][group]
                    graph_record=dict(saved,actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                    identity=dict(group=group,node_ids=saved['node_ids'],labels=saved['labels'])
                    graph,ids,candidates,injected=reconstruct(data,capture,graph_record,identity)
                    counts['evaluation_base_graph_reconstructions']+=1
                    for arm in ARMS:
                        features=payload['features'][arm];sequence=payload['canonical_sequences'][arm]
                        if (tensor_sha256(features)!=row['variants'][arm]['feature_content_sha256'] or
                                tensor_sha256(sequence)!=row['variants'][arm]['canonical_sequence_content_sha256']):
                            raise RuntimeError('Sealed feature values or canonical sequence changed.')
                        local=feature_graph(graph,features);counts['evaluation_feature_graph_variants']+=1
                        ga=classify('gcn',graph=local,ids=ids);gb=classify('gaga',ids=ids,sequence=sequence.to(data.device))
                        if (tensor_sha256(features)!=row['variants'][arm]['feature_content_sha256'] or
                                tensor_sha256(sequence)!=row['variants'][arm]['canonical_sequence_content_sha256']):
                            raise RuntimeError('Classifier inference mutated a sealed feature or sequence tensor.')
                        labels=data.labels[ids].cpu()
                        predicted_row=dict(seed=seed,group=group,arm=arm,node_ids=ids.tolist(),labels=labels.tolist(),
                            gcn_logits=ga.tolist(),gaga_logits=gb.tolist(),alpha=.5,
                            node_budget=saved['node_budget'],edge_budget=saved['edge_budget'],actual_injected_nodes=len(injected),
                            actual_injected_edges=(local.num_edges()-data.graph.num_edges()-len(injected))//2,
                            source_capture_sha256=row['source_capture_sha256'],prepared_artifact_sha256=row['artifact']['sha256'],
                            reference_pt_sha256=reference_manifest['reference_pt_sha256'],
                            original_adjacency_content_sha256=row['immutable_capture_content_hashes']['adj_attack'],
                            target_order_content_sha256=row['immutable_capture_content_hashes']['target_ids'],
                            candidate_order_content_sha256=row['immutable_capture_content_hashes']['candidate_ids'],
                            injected_feature_content_sha256=row['variants'][arm]['feature_content_sha256'],
                            canonical_sequence_content_sha256=row['variants'][arm]['canonical_sequence_content_sha256'])
                        for index,name in enumerate(('gcn','gaga')):
                            predicted=(ga if name=='gcn' else gb).argmax(1).cpu()
                            transition=transitions(labels,clean_predictions[group][index],predicted)
                            predicted_row[name+'_transitions']=transition
                            for key,value in transition.items():totals[arm,seed][name][key]+=value
                        streams[arm,seed].write(json.dumps(predicted_row,allow_nan=False)+'\n');streams[arm,seed].flush()
                        del local
                    if row['immutable_capture_content_hashes']!={key:tensor_sha256(value) for key,value in capture.items()}:
                        raise RuntimeError('Evaluation mutated the baseline x0, hard adjacency or original ID tensors.')
                    del graph,payload,capture
        finally:
            for stream in streams.values():stream.close()
        if (counts['prepared_artifact_tensor_loads']!=384 or counts['evaluation_base_graph_reconstructions']!=384 or
                counts['evaluation_feature_graph_variants']!=1536 or counts['original_gcn_completed_forwards']!=1664 or
                counts['official_gaga_completed_forwards']!=1664 or counts['original_gcn_clean_forwards']!=128 or
                counts['official_gaga_clean_forwards']!=128 or counts['original_gcn_attacked_forwards']!=1536 or
                counts['official_gaga_attacked_forwards']!=1536):
            raise RuntimeError('Complete fixed-cohort prototype evaluation counters differ.')
        summaries=[]
        for seed in SEEDS:
            for arm in ARMS:
                if any(values['occurrences']!=1091 for values in totals[arm,seed].values()):
                    raise RuntimeError('A fixed occurrence denominator differs.')
                summaries.append(dict(seed=seed,arm=arm,groups=128,unique_targets=374,totals=totals[arm,seed],
                    gcn_mr_percent=100*totals[arm,seed]['gcn']['attacked_wrong']/1091,
                    gaga_mr_percent=100*totals[arm,seed]['gaga']['attacked_wrong']/1091))
        immutable_after=dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d),
            original_gcn=model_hash(gcn),official_gaga=model_hash(victim))
        result['immutable_content_hashes']['after']=immutable_after
        if (immutable_after!=result['immutable_content_hashes']['before'] or
                reference_hashes!={key:tensor_sha256(value) for key,value in reference.items() if torch.is_tensor(value)} or
                sha256(reference_path)!=reference_manifest['reference_pt_sha256'] or
                sha256(output/'reference.json')!=reference_manifest['reference_json_sha256']):
            raise RuntimeError('Original features/cache/models or TRAIN prototypes changed.')
        if any(parameter.grad is not None or parameter.requires_grad for model in (gcn,victim) for parameter in model.parameters()):
            raise RuntimeError('Evaluation classifier flags or gradients changed.')
        result.update(status='completed',summaries=summaries,reference_counts=reference['counts'],
            reference_pt_sha256=reference_manifest['reference_pt_sha256'],reference_content_hashes=reference_hashes,
            immutable_content_hashes_retained=True,all_reference_and_feature_proposals_sealed_before_models=True,
            baseline_channel='FreshCPUreconstruction/inferenceonoriginalsavedx0graphs;notstoredpredictions',
            official_gaga_channel='All1536canonicalsequencessealedbeforeclassifierconstruction',
            interpretation='Post-hocfixedalphaprototypemix;onlyTRAINlabelsselectmeans,notnovelty/generalization/coursecompletion',
            final_TEST_used=False,experimental_API_calls=0)
    except BaseException as error:
        traceback.print_exc();result.update(status='failed',error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);write_json(output/'input_hashes.json',receipts)
        result.update(counters=counts,prepared_cases=len(prepared),process_cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.monotonic()-started,GPU_hours=0,**runtime)
        names=['registered.json','reference.pt','reference.json','reference_sealed.json','geometry.jsonl','geometry_sealed.json',
            'clean_predictions.jsonl','input_hashes.json']
        names.extend('%s_seed%d_predictions.jsonl'%(arm,seed) for arm in ARMS for seed in SEEDS)
        result['output_hashes']={name:sha256(output/name) for name in names if (output/name).is_file()}
        write_json(output/'result.json',result);print('TRAIN_PROTOTYPE_MIX_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
