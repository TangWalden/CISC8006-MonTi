"""Observe original edge scoring and a finite generator/detector device matrix."""
import argparse
import datetime as dt
import inspect
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback

import dgl
import numpy as np
import scipy
import torch
import torch.nn.functional as functional

import models.attack_model as attack_module
from modules.data_handler import DataHandler
from course.one_edge_diagnosis_v2 import SurrogateOnly, sha, write, csv_write, tensor_sha
from utils.utils import get_n_ids, set_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]


def fingerprint(graph):
    source, destination = graph.edges(order='eid')
    return dict(nodes=graph.num_nodes(),edges=graph.num_edges(),features=tensor_sha(graph.ndata['x']),
                ordered_sources=tensor_sha(source),ordered_destinations=tensor_sha(destination))


def score_stats(vector):
    vector=vector.detach().cpu().flatten()
    maximum=float(vector.max())
    ordered=torch.sort(vector,descending=True).values
    return dict(maximum=maximum,first=float(vector[0]),exact_maxima=int((vector==vector.max()).sum()),
                within_1e_minus6=int((vector.max()-vector<=1e-6).sum()),
                within_1e_minus5=int((vector.max()-vector<=1e-5).sum()),top_gap=float(ordered[0]-ordered[1]))


def observe(experiment, n_ids):
    capture={}
    def projection_hook(module,arguments,output):
        assert 'projected' not in capture
        capture['edge_input']=arguments[0].detach().clone()
        capture['projected']=output.detach().clone()
        return None
    def model_hook(module,arguments,output):
        assert 'target_order' not in capture
        subgraph,targets=arguments[:2]
        assert int(arguments[2])==int(arguments[3])==1
        capture['target_order']=subgraph.ndata[dgl.NID][targets].detach().cpu().tolist()
        candidates=output[2]
        capture['candidate_order']=subgraph.ndata[dgl.NID][candidates].detach().cpu().tolist()
        capture['adjacency']=output[1].detach().clone()
        return None
    original_gumbel,original_topk=attack_module.gumbel_softmax,attack_module.st_top_k
    def gumbel(logits,*args,**kwargs):
        assert 'raw_scores' not in capture and kwargs.get('randomness') is False
        caller=inspect.currentframe().f_back
        try:
            assert caller.f_code.co_name=='edge_generation' and caller.f_globals['__name__']=='models.attack_model'
            capture['score_src']=caller.f_locals['src'].detach().clone()
            capture['score_dst']=caller.f_locals['dst'].detach().clone()
            capture['normalized']=caller.f_locals['e'].detach().clone()
            capture['executed_num_targets']=int(caller.f_locals['num_targets'])
            capture['executed_num_candidates']=int(caller.f_locals['num_candidates'])
        finally:
            del caller
        capture['raw_scores']=logits.detach().clone()
        output=original_gumbel(logits,*args,**kwargs)
        capture['soft_scores']=output.detach().clone()
        return output
    def topk(values,k,dim=-1):
        assert 'topk_input' not in capture and k==1 and dim==-1
        capture['topk_native']=dict(shape=list(values.shape),strides=list(values.stride()),
                                    dtype=str(values.dtype),contiguous=values.is_contiguous())
        capture['topk_input']=values.detach().clone()
        output=original_topk(values,k,dim)
        capture['topk_indices']=output[1].detach().clone()
        return output
    handles=[experiment.model.linear_edge.register_forward_hook(projection_hook),
             experiment.model.register_forward_hook(model_hook)]
    try:
        attack_module.gumbel_softmax,attack_module.st_top_k=gumbel,topk
        graph=experiment.get_perturbed_graph(n_ids,1,1,training=False,need_sequence=False)
    finally:
        attack_module.gumbel_softmax,attack_module.st_top_k=original_gumbel,original_topk
        for handle in handles:
            handle.remove()
    assert graph is not None and capture['target_order']==n_ids.cpu().tolist()
    targets,candidates=len(n_ids),len(capture['candidate_order'])
    start=targets+candidates
    assert capture['executed_num_targets']==targets and capture['executed_num_candidates']==candidates
    projected=capture['projected']
    assert projected.shape[0]==start+1 and capture['raw_scores'].shape==(start,)
    assert capture['topk_input'].shape==(1,targets)
    assert torch.equal(capture['soft_scores'][:targets],capture['topk_input'][0])
    normalized=capture['normalized']
    assert all(torch.isfinite(value).all() for value in [projected,normalized,capture['raw_scores'],capture['soft_scores'],capture['topk_input']])
    assert capture['score_src'].unique().cpu().tolist()==[0]
    assert capture['score_dst'].cpu().tolist()==list(range(start))
    derived_actual=(normalized[capture['score_src']]*normalized[capture['score_dst']]).sum(-1)
    paper_reference=(normalized[start:start+1]*normalized[:targets]).sum(-1)
    chosen=int(capture['topk_indices'][0,0])
    assert float(capture['topk_input'][0,chosen])==float(capture['topk_input'].max())
    total=experiment.data.graph.num_nodes()
    assert graph.num_nodes()==total+1 and graph.num_edges()==experiment.data.graph.num_edges()+3
    assert [x for x in graph.successors(total).tolist() if x!=total]==[int(n_ids[chosen])]
    assert int(torch.count_nonzero(capture['adjacency']))==1
    # Algebra-only same-input top-k contrasts do not change any generated attack.
    probability=capture['topk_input']
    cpu_probability=probability.cpu()
    gpu_probability=probability.to('cuda:0')
    assert tensor_sha(cpu_probability)==tensor_sha(gpu_probability)
    cpu_choice=int(cpu_probability.topk(1,dim=-1).indices[0,0])
    gpu_choice=int(gpu_probability.topk(1,dim=-1).indices[0,0])
    actual=score_stats(capture['raw_scores'][:targets])
    soft=score_stats(probability)
    reference=score_stats(paper_reference)
    fields=dict(targets=targets,candidates=candidates,start_atk=start,executed_source_index=0,
          executed_source_role='first_target',physical_source_role='injected',paper_source_index=start,
          source_projection_norm=float(projected[0].norm()),injected_projection_norm=float(projected[start].norm()),
          source_normalized_norm=float(normalized[0].norm()),injected_normalized_norm=float(normalized[start].norm()),
          topk_native_shape=json.dumps(capture['topk_native']['shape']),topk_native_strides=json.dumps(capture['topk_native']['strides']),
          topk_native_dtype=capture['topk_native']['dtype'],topk_native_contiguous=capture['topk_native']['contiguous'],
          selected_endpoint_position=chosen,target_order_sha256=sha_list(capture['target_order']),
          candidate_order_sha256=sha_list(capture['candidate_order']),candidate_set_sha256=sha_list(sorted(capture['candidate_order'])),
          feature_sha256=tensor_sha(graph.ndata['x'][total:total+1]),
          raw_first_score=actual['first'],raw_max_score=actual['maximum'],raw_exact_maxima=actual['exact_maxima'],
          raw_near_max_1e_minus6=actual['within_1e_minus6'],raw_near_max_1e_minus5=actual['within_1e_minus5'],raw_top_gap=actual['top_gap'],
          soft_exact_maxima=soft['exact_maxima'],soft_near_max_1e_minus6=soft['within_1e_minus6'],
          soft_near_max_1e_minus5=soft['within_1e_minus5'],soft_top_gap=soft['top_gap'],
          derived_actual_max_difference=float((derived_actual-capture['raw_scores']).abs().max()),
          captured_normalization_vs_derived_difference=float((normalized-functional.normalize(projected,dim=-1)).abs().max()),
          paper_reference_exact_maxima=reference['exact_maxima'],paper_reference_top_gap=reference['top_gap'],
          same_probability_cpu_topk=cpu_choice,same_probability_gpu_topk=gpu_choice,
          same_device_reference_matches_executed=(cpu_choice if probability.device.type=='cpu' else gpu_choice)==chosen,
          same_probability_device_topk_disagrees=cpu_choice!=gpu_choice)
    private=dict(target_order=capture['target_order'],candidate_order=capture['candidate_order'],
          edge_input=capture['edge_input'].cpu().tolist(),projected=projected.cpu().tolist(),normalized=normalized.cpu().tolist(),
          actual_src_indices=capture['score_src'].cpu().tolist(),actual_dst_indices=capture['score_dst'].cpu().tolist(),
          topk_native=capture['topk_native'],actual_raw_scores=capture['raw_scores'].cpu().tolist(),
          actual_soft_scores=capture['soft_scores'].cpu().tolist(),actual_topk_input=probability.cpu().tolist(),
          paper_role_reference_scores=paper_reference.cpu().tolist(),feature=graph.ndata['x'][total:total+1].cpu().tolist())
    return graph,fields,private


def sha_list(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',default='course/score_role_replay_protocol_v2.json')
    parser.add_argument('--output',default='evidence/score_role_replay_v2')
    args=parser.parse_args()
    output=ROOT/args.output
    assert str(output.resolve()).startswith(str(ROOT.resolve())+os.sep)
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw').mkdir()
    protocol_path=ROOT/args.protocol
    protocol=json.loads(protocol_path.read_text())
    started=time.monotonic()
    calls=generations=0
    write(output/'registered.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         tracked_changes=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),
         registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),pid=os.getpid(),protocol=args.protocol,
         protocol_sha256=sha(protocol_path),script_sha256=sha(pathlib.Path(__file__)),
         command=['env','CUDA_VISIBLE_DEVICES=0','bash','course/run.sh','-m','course.score_role_replay_v2']+__import__('sys').argv[1:],
         gpu=0,torch=torch.__version__,dgl=dgl.__version__,numpy=np.__version__,scipy=scipy.__version__,
         cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),threads=8,
         gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True).strip(),
         historical_devices=protocol['historical_devices'],
         environment={key:os.environ.get(key) for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED','DGLBACKEND','CUDA_VISIBLE_DEVICES']},
         backend_before_upstream_initialization=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
           cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
           deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),
         cost_policy='Conservatively charge entire main elapsed time as one GPU wall hour, including CPU preparation and all transfers/math.'))
    def expired(signum,frame):
        raise TimeoutError('Registered 300-second conservative GPU wall-time limit reached.')
    signal.signal(signal.SIGALRM,expired)
    signal.signal(signal.SIGTERM,expired)
    signal.alarm(protocol['main_wall_seconds_limit'])
    raw_path=output/'raw/measurements.jsonl'
    generator_rows,detector_rows,pair_rows=[],[],[]
    try:
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
        assert torch.cuda.is_available() and protocol['gpu']==0
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and torch.cuda.device_count()==torch._C._cuda_getDeviceCount()==1
        torch.set_num_threads(8)
        frozen=json.loads((ROOT/'course/frozen_protocol.json').read_text())
        inputs={}
        for item in frozen['inputs']+frozen['caches']+frozen['scientific_code']+protocol['source_files']:
            assert sha(ROOT/item['path'])==item['sha256'],item['path']
            inputs[item['path']]=item['sha256']
        for run in protocol['runs']:
            for key in ['config','checkpoint','source_prediction','source_prediction_summary','source_result']:
                item=run[key]
                assert '_test_' not in item['path'] and sha(ROOT/item['path'])==item['sha256']
                inputs[item['path']]=item['sha256']
        previous_path=ROOT/'evidence/one_edge_diagnosis_v2/raw/predictions.jsonl'
        assert sha(previous_path)==protocol['previous_raw_prediction_sha256']
        inputs[previous_path.relative_to(ROOT).as_posix()]=protocol['previous_raw_prediction_sha256']
        write(output/'verified_inputs.json',inputs)
        ledger=json.loads((ROOT/'evidence/research_cost_ledger.json').read_text())
        assert protocol['gpu_hour_limit']<=ledger['limit']-ledger['measured_or_conservatively_charged']-ledger['unmetered_setup_reserve']
        prior_forensic_cost=sum(row['charged_gpu_wall_hours'] for row in ledger.get('score_role_replay_attempts',[]))
        assert protocol['gpu_hour_limit']<=protocol['aggregate_forensic_gpu_hour_limit']-prior_forensic_cost
        runtime=json.loads((ROOT/'environment/runtime.json').read_text())
        assert all(value==runtime[key] for key,value in [('torch',torch.__version__),('dgl',dgl.__version__),('numpy',np.__version__),('scipy',scipy.__version__)])
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        identity_allocation=torch.zeros(1,device='cuda:0')
        torch.cuda.synchronize(0)
        own_uuids=[]
        for metadata_attempt in range(1,4):
            process_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
            own_uuids=[parts[1].strip() for parts in (line.split(',') for line in process_rows)
                       if len(parts)>=2 and parts[0].strip().isdigit() and int(parts[0].strip())==os.getpid()]
            if own_uuids:
                break
            time.sleep(0.1)
        assert own_uuids==[protocol['gpu_uuid']], 'Physical GPU identity could not be established.'
        write(output/'gpu_identity.json',dict(pid=os.getpid(),visibility='0',expected_uuid=protocol['gpu_uuid'],
              observed_own_process_uuids=own_uuids,metadata_queries=metadata_attempt,pytorch_visible_count=1,native_runtime_visible_count=1))
        del identity_allocation
        config=json.loads((ROOT/protocol['runs'][0]['config']['path']).read_text())
        data={device:DataHandler(dict(config,cuda_id='cpu' if device=='cpu' else 'cuda:0')) for device in ['cpu','gpu']}
        groups=protocol['groups']
        assert len(groups)==16 and groups==sorted(set(groups))
        assert all(set(groups).issubset(set(value.target_idx_valid.tolist())) for value in data.values())
        assert all(int(value.node_budgets[group])==int(value.edge_budgets[group])==1 for value in data.values() for group in groups)
        base_cpu,base_gpu=fingerprint(data['cpu'].graph),fingerprint(data['gpu'].graph)
        assert all(base_cpu[key]==base_gpu[key] for key in ['nodes','edges','features'])
        if base_cpu!=base_gpu:
            cs,cd=data['cpu'].graph.edges(order='eid')
            gs,gd=data['gpu'].graph.edges(order='eid')
            count=data['cpu'].graph.num_nodes()
            assert torch.equal(torch.sort(cs*count+cd).values,torch.sort(gs*count+gd).values.cpu())
        write(output/'base_graph_identity.json',dict(cpu=base_cpu,gpu=base_gpu,ordered_identity=base_cpu==base_gpu,
            feature_and_edge_multiset_identity=True,feature_dtype=str(data['cpu'].features.dtype)))
        previous={(row['attacker_seed'],row['target_group']):row for row in map(json.loads,previous_path.read_text().splitlines())}
        clean={}
        with raw_path.open('x') as raw,torch.no_grad():
            def record(value):
                raw.write(json.dumps(value)+'\n')
                raw.flush()
            for run in protocol['runs']:
                config=json.loads((ROOT/run['config']['path']).read_text())
                experiments={}
                for device in ['cpu','gpu']:
                    current=dict(config,cuda_id='cpu' if device=='cpu' else 'cuda:0')
                    experiment=SurrogateOnly(current,data[device])
                    experiment.model.load_state_dict(torch.load(ROOT/run['checkpoint']['path'],map_location='cpu'))
                    experiment.model.eval()
                    def forbidden_step(*step_args,**step_kwargs):
                        raise RuntimeError('Optimizer steps are prohibited in this diagnostic.')
                    experiment.optimizer.step=forbidden_step
                    for parameter in experiment.model.parameters():
                        parameter.requires_grad=False
                    assert experiment.victim_model is None
                    experiments[device]=experiment
                for model_name in ['model','surrogate_model']:
                    cpu_state=getattr(experiments['cpu'],model_name).state_dict()
                    gpu_state=getattr(experiments['gpu'],model_name).state_dict()
                    assert cpu_state.keys()==gpu_state.keys()
                    assert all(torch.equal(cpu_state[key].cpu(),gpu_state[key].cpu()) for key in cpu_state)
                assert torch.equal(experiments['cpu'].model.min_feat.cpu(),experiments['gpu'].model.min_feat.cpu())
                assert torch.equal(experiments['cpu'].model.max_feat.cpu(),experiments['gpu'].model.max_feat.cpu())
                if run['seed']==0:
                    write(output/'backend_measurement.json',dict(
                      upstream_seed_policy='The unchanged upstream set_seed called by ExperimentHandler and replay resets sets cudnn.deterministic=True and benchmark=False. No manual backend tuning.',
                      matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                      cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
                      deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                      parameter_dtype=str(next(experiments['cpu'].model.parameters()).dtype),default_dtype=str(torch.get_default_dtype()),
                      pythonhashseed_environment_after_upstream_reset=os.environ.get('PYTHONHASHSEED'),
                      interpreter_hash_seed='Set to 0 at process start by course/run.sh; later environment assignments do not reset interpreter hashing'))
                source={row['target_group']:row for row in map(json.loads,(ROOT/run['source_prediction']['path']).read_text().splitlines()) if row['target_group'] in groups}
                assert set(source)==set(groups)
                def logits(graph,n_ids,detector,phase,group,generator=None):
                    nonlocal calls
                    assert calls<protocol['gcn_forward_limit'] and time.monotonic()-started<protocol['main_wall_seconds_limit']
                    calls+=1
                    context=dict(attacker_seed=run['seed'],target_group=group,detector=detector,generator=generator,phase=phase,call=calls)
                    record(dict(event='gcn_started',**context))
                    sampler=experiments[detector].sampler_surrogate
                    had_override='sample_blocks' in sampler.__dict__
                    original_sample=sampler.sample_blocks
                    alignment={}
                    def observed_sample(*sample_args,**sample_kwargs):
                        result=original_sample(*sample_args,**sample_kwargs)
                        _,output_nodes,blocks=result
                        output_order=output_nodes.detach().cpu().tolist()
                        block_order=blocks[-1].dstdata[dgl.NID].detach().cpu().tolist()
                        assert output_order==block_order==n_ids.cpu().tolist()
                        alignment['output_order_sha256']=sha_list(output_order)
                        alignment['block_dst_order_sha256']=sha_list(block_order)
                        return result
                    sampler.sample_blocks=observed_sample
                    forward_count=[]
                    def counted_forward(module,arguments,result):
                        forward_count.append(1)
                        return None
                    handle=experiments[detector].surrogate_model.register_forward_hook(counted_forward)
                    try:
                        value=experiments[detector].get_logits(graph,n_ids,training=False).detach().cpu()
                    finally:
                        if had_override:
                            sampler.sample_blocks=original_sample
                        else:
                            del sampler.sample_blocks
                        handle.remove()
                    assert len(forward_count)==1 and alignment
                    assert torch.isfinite(value).all()
                    record(dict(event='gcn_completed',logits=value.tolist(),alignment=alignment,**context))
                    return value
                for group in groups:
                    n_ids=torch.tensor(get_n_ids(data['cpu'].target_assignment,group),dtype=torch.long)
                    labels=data['cpu'].labels[n_ids].cpu()
                    assert source[group]['node_ids']==n_ids.tolist() and source[group]['labels']==labels.tolist()
                    record(dict(event='identity',attacker_seed=run['seed'],target_group=group,node_ids=n_ids.tolist(),labels=labels.tolist(),source_predictions=source[group]['predictions']))
                    for detector in ['cpu','gpu']:
                        if (detector,group) not in clean:
                            clean[detector,group]=logits(data[detector].graph,n_ids,detector,'clean',group)
                    captured={}
                    for generator in ['cpu','gpu']:
                        set_seed(protocol['reconstruction_seed'])
                        generations+=1
                        assert generations<=protocol['generator_forward_limit']
                        record(dict(event='generation_started',attacker_seed=run['seed'],target_group=group,generator=generator,generation=generations))
                        graph,fields,private=observe(experiments[generator],n_ids)
                        signature=fingerprint(graph)
                        fields=dict(target_group=group,attacker_seed=run['seed'],generator=generator,**fields)
                        generator_rows.append(fields)
                        record(dict(event='generation_completed',public=fields,private=private,graph_signature=signature,generation=generations))
                        results={}
                        for detector in ['cpu','gpu']:
                            transferred=graph if generator==detector else graph.to(experiments[detector].device)
                            if generator!=detector:
                                assert fingerprint(transferred)==signature
                            value=logits(transferred,n_ids,detector,'attacked',group,generator)
                            results[detector]=value
                            predicted=value.argmax(1)
                            correct=predicted.eq(labels)
                            clean_correct=clean[detector,group].argmax(1).eq(labels)
                            detector_rows.append(dict(target_group=group,attacker_seed=run['seed'],generator=generator,detector=detector,
                               target_occurrences=len(n_ids),residuals=int(correct.sum()),clean_correct=int(clean_correct.sum()),
                               clean_correct_to_wrong=int((clean_correct & ~correct).sum()),clean_wrong_to_correct=int((~clean_correct & correct).sum()),
                               source_prediction_match=predicted.tolist()==source[group]['predictions'],
                               source_prediction_disagreements=sum(a!=b for a,b in zip(predicted.tolist(),source[group]['predictions'])),
                               previous_cpu_prediction_match=predicted.tolist()==np.asarray(previous[run['seed'],group]['reconstructed_logits']).argmax(1).tolist()))
                        captured[generator]=dict(fields=fields,feature=np.asarray(private['feature'],dtype=np.float32),
                                                 cpu=results['cpu'],gpu=results['gpu'])
                    left,right=captured['cpu'],captured['gpu']
                    previous_row=previous[run['seed'],group]
                    old_position=previous_row['node_ids'].index(previous_row['endpoint_node_id'])
                    pair_rows.append(dict(target_group=group,attacker_seed=run['seed'],
                       cpu_gpu_same_target_order=left['fields']['target_order_sha256']==right['fields']['target_order_sha256'],
                       cpu_gpu_same_candidate_order=left['fields']['candidate_order_sha256']==right['fields']['candidate_order_sha256'],
                       cpu_gpu_same_candidate_set=left['fields']['candidate_set_sha256']==right['fields']['candidate_set_sha256'],
                       cpu_gpu_same_selected_endpoint=left['fields']['selected_endpoint_position']==right['fields']['selected_endpoint_position'],
                       cpu_gpu_feature_max_difference=float(np.max(np.abs(left['feature']-right['feature']))),
                       fresh_cpu_previous_feature_exact_match=bool(np.array_equal(left['feature'],np.asarray(previous_row['feature'],dtype=np.float32))),
                       fresh_cpu_previous_endpoint_match=left['fields']['selected_endpoint_position']==old_position,
                       cpu_attack_detector_prediction_disagreements=int((left['cpu'].argmax(1)!=left['gpu'].argmax(1)).sum()),
                       gpu_attack_detector_prediction_disagreements=int((right['cpu'].argmax(1)!=right['gpu'].argmax(1)).sum()),
                       cpu_detector_generator_prediction_disagreements=int((left['cpu'].argmax(1)!=right['cpu'].argmax(1)).sum()),
                       gpu_detector_generator_prediction_disagreements=int((left['gpu'].argmax(1)!=right['gpu'].argmax(1)).sum())))
                    print('CASE',run['seed'],group,'complete',flush=True)
        assert generations==96 and calls==224 and len(generator_rows)==96 and len(detector_rows)==192 and len(pair_rows)==48
        csv_write(output/'generators.csv',generator_rows)
        csv_write(output/'detectors.csv',detector_rows)
        csv_write(output/'pairs.csv',pair_rows)
        matrix=[]
        for generator in ['cpu','gpu']:
            for detector in ['cpu','gpu']:
                selected=[row for row in detector_rows if row['generator']==generator and row['detector']==detector]
                matrix.append(dict(generator=generator,detector=detector,cases=len(selected),target_occurrences=sum(row['target_occurrences'] for row in selected),
                   residuals=sum(row['residuals'] for row in selected),source_concordant_cases=sum(row['source_prediction_match'] for row in selected),
                   source_prediction_disagreements=sum(row['source_prediction_disagreements'] for row in selected),
                   previous_cpu_concordant_cases=sum(row['previous_cpu_prediction_match'] for row in selected)))
        torch.cuda.synchronize(0)
        for item in frozen['scientific_code']+protocol['source_files']:
            assert sha(ROOT/item['path'])==item['sha256'],item['path']
        elapsed=time.monotonic()-started
        summary=dict(status='completed',gcn_forward_calls=calls,generator_forward_calls=generations,group_checkpoint_cases=48,
            matrix=matrix,source_role_first_target_cases=sum(row['executed_source_role']=='first_target' for row in generator_rows),
            actual_soft_multiple_exact_maxima=sum(row['soft_exact_maxima']>1 for row in generator_rows),
            same_probability_device_topk_disagreement_cases=sum(row['same_probability_device_topk_disagrees'] for row in generator_rows),
            fresh_cpu_previous_feature_exact_cases=sum(row['fresh_cpu_previous_feature_exact_match'] for row in pair_rows),
            fresh_cpu_previous_endpoint_match_cases=sum(row['fresh_cpu_previous_endpoint_match'] for row in pair_rows),
            cross_device_endpoint_match_cases=sum(row['cpu_gpu_same_selected_endpoint'] for row in pair_rows),
            raw_sha256=sha(raw_path),elapsed_seconds=elapsed,gpu_wall_hours=elapsed/3600,conservative_gpu_devices=1,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            interpretation='Observed source-role discrepancy and captured device contrasts; no corrected edges, training, victim/test queries or demonstrated improvement. Prediction concordance is not historical perturbation identity.')
        assert summary['gpu_wall_hours']<=protocol['gpu_hour_limit']
        write(output/'summary.json',summary)
        write(output/'output_checksums.json',{name:sha(output/name) for name in ['registered.json','verified_inputs.json','gpu_identity.json','base_graph_identity.json','backend_measurement.json','generators.csv','detectors.csv','pairs.csv','summary.json']})
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        elapsed=time.monotonic()-started
        write(output/'failure.json',dict(status='failed',traceback=traceback.format_exc(),gcn_forward_calls=calls,generator_forward_calls=generations,
             elapsed_seconds=elapsed,gpu_wall_hours=elapsed/3600,conservative_gpu_devices=1))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':
    main()
