"""Independent saved-record and CPU-tensor audit of target-degree balancing.

Freeze before new real captures/outcomes are read. Allowed deserializations:
384 frozen BASE capture payloads and384 NEW sealed geometry artifacts only.
No original dataset/graph/truth array, classifier weights, model construction,
inference, gradient, optimizer, API/network or CUDA tensors are permitted.
Original-original graph/canonical construction and temporal model gating are
producer/source assertions, not independently replayed with a dataset here.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
import traceback

ARMS=('baseline','balanced_target','matched_random')
SEEDS=(3,4,5)
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
NO_MODEL=('fresh_teacher_model_constructions','fresh_teacher_forwards','generator_model_constructions',
    'generator_forwards','optimizer_steps','autograd_calls','API_calls','TEST_queries')
COUNTER_KEYS=('source_capture_tensor_loads','source_captures_prepared','geometry_graph_reconstructions',
    'canonical_sequence_updates','first_hop_invariant_checks','prepared_artifact_tensor_loads','evaluation_graph_reconstructions',
    'original_gcn_load_dispatches','original_gcn_models_loaded','official_gaga_load_dispatches','official_gaga_models_loaded',
    'original_gcn_forward_dispatches','original_gcn_completed_forwards','official_gaga_forward_dispatches','official_gaga_completed_forwards',
    'original_gcn_clean_forwards','official_gaga_clean_forwards','original_gcn_attacked_forwards','official_gaga_attacked_forwards')+NO_MODEL


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024*1024),b''):h.update(value)
    return h.hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


class Audit:
    def __init__(self,root):self.root=root;self.files={};self.counts=collections.Counter()
    def check(self,value,name):
        self.counts[name]+=1
        if not value:raise AssertionError('Target-balance audit failed: '+name)
    def path(self,name,directory=False):
        raw=pathlib.Path(name);p=raw.resolve() if raw.is_absolute() else (self.root/raw).resolve()
        self.check(self.root in p.parents and (p.is_dir() if directory else p.is_file()),'repository_input_boundary')
        return p
    def file(self,name,wanted=None):
        p=self.path(name);actual=sha(p);key=p.relative_to(self.root).as_posix()
        self.check(self.files.setdefault(key,actual)==actual,'bytes_unchanged_during_audit')
        if wanted is not None:self.check(actual==wanted,'frozen_byte_hash')
        return p
    def json(self,name,wanted=None):return json.loads(self.file(name,wanted).read_text(encoding='utf-8'))
    def rows(self,name):
        with self.file(name).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
    def close(self,x,y,name):self.check(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-9+1e-8*abs(y),name)


def keyed(a,rows,groups,name):
    values={row['group']:row for row in rows}
    a.check(len(values)==len(rows)==128 and set(values)==set(groups),name)
    return values


def tensor_hash(value):
    value=value.detach().cpu().contiguous()
    header=json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode()
    return hashlib.sha256(len(header).to_bytes(8,'big')+header+value.numpy().tobytes()).hexdigest()


def sample_role(values,k,seed,group,source,role):
    payload=json.dumps([seed,str(group),source,role],separators=(',',':')).encode('ascii')
    key=int.from_bytes(hashlib.sha256(payload).digest(),'big')
    return random.Random(key).sample(sorted(values),k)


def independent_geometry(a,base,targets,candidates,injected,seed,group):
    """Independent integer replay of the specified ties, swaps and control."""
    t,c,n=len(targets),len(candidates),len(injected)
    original=[list(map(int,row)) for row in base];current=[row[:] for row in original]
    degree=[sum(row[j] for row in original) for j in range(t)];before=degree[:];history=[]
    while max(degree)-min(degree)>1:
        high_degree=max(degree);low_degree=min(degree)
        high=min((j for j in range(t) if degree[j]==high_degree),key=lambda j:targets[j])
        low=min((j for j in range(t) if degree[j]==low_degree),key=lambda j:targets[j])
        eligible=[i for i in range(n) if current[i][high]==1 and current[i][low]==0]
        a.check(bool(eligible),'strict_column_difference_has_eligible_row')
        chosen=min(eligible,key=lambda i:injected[i]);potential=sum(v*v for v in degree)
        current[chosen][high]=0;current[chosen][low]=1;degree[high]-=1;degree[low]+=1
        delta=sum(v*v for v in degree)-potential
        a.check(delta==-2*(high_degree-low_degree-1) and delta<0 and degree[high]>=1,'strict_potential_descent_without_uncovering')
        history.append(dict(iteration=len(history)+1,row=chosen,source_id=injected[chosen],
            removed_target_id=targets[high],added_target_id=targets[low],previous_high_degree=high_degree,
            previous_low_degree=low_degree,delta_sum_squared_target_degree=delta))
    q,remainder=divmod(sum(before),t);optimum=remainder*(q+1)**2+(t-remainder)*q*q
    a.check(sum(v*v for v in degree)==optimum and max(degree)-min(degree)<=1,'attained_fixed_total_convex_column_optimum')
    random_matrix=[row[:] for row in original];plan=[];by_id={value:j for j,value in enumerate(targets)}
    for i,(left,right) in enumerate(zip(original,current)):
        removed=sorted(targets[j] for j in range(t) if left[j] and not right[j])
        added=sorted(targets[j] for j in range(t) if not left[j] and right[j]);k=len(removed)
        a.check(k==len(added),'balanced_final_row_turnover_equal_sides')
        take=sample_role([targets[j] for j in range(t) if left[j]],k,seed,group,injected[i],'remove-target')
        put=sample_role([targets[j] for j in range(t) if not left[j]],k,seed,group,injected[i],'add-target')
        for node in take:random_matrix[i][by_id[node]]=0
        for node in put:random_matrix[i][by_id[node]]=1
        plan.append(dict(row=i,source_id=injected[i],original_target_count=sum(left[:t]),final_k=k,
            balanced_removed_target_ids=removed,balanced_added_target_ids=added,random_removed_target_ids=take,
            random_added_target_ids=put,final_target_hamming_distance=2*k))
    arms=dict(baseline=original,balanced_target=current,matched_random=random_matrix)
    after={arm:[sum(row[j] for row in matrix) for j in range(t)] for arm,matrix in arms.items()}
    old_covered={targets[j] for j,d in enumerate(before) if d};covered={arm:{targets[j] for j,d in enumerate(v) if d} for arm,v in after.items()}
    physical=[sum(row)+sum(original[j][t+c+i] for j in range(i+1,n)) for i,row in enumerate(original)]
    expected=dict(row_plan=plan,iteration_history=history,target_incoming_injected_degrees_before=before,
        target_incoming_injected_degrees_after=after,total_target_incidence=sum(before),
        sum_squared_target_degree_before=sum(v*v for v in before),sum_squared_target_degree_balanced=optimum,
        balanced_degree_range=max(degree)-min(degree),balancing_iterations=len(history),
        final_rewired_edges=sum(row['final_k'] for row in plan),original_row_edge_counts=[sum(row) for row in original],
        physical_injected_degrees=physical,newly_covered_target_ids={arm:sorted(v-old_covered) for arm,v in covered.items()},
        previously_covered_target_ids_lost={arm:sorted(old_covered-v) for arm,v in covered.items()},
        balanced_equals_baseline=current==original,balanced_equals_random=current==random_matrix,
        matched_random_rule='Per-role SHA256-derived seed; Random.sample on sorted ORIGINAL target ones/zeros; freeze Python runtime for replay',
        model_outputs_used=False,features_read_or_changed=False)
    a.check(old_covered<=covered['balanced_target'],'balanced_preserves_every_previously_covered_target')
    return arms,expected


def classify(a,values,n):
    a.check(len(values)==n,'binary_logit_occurrence_count')
    for row in values:a.check(len(row)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in row),'finite_binary_raw_logits')
    return [0 if z[0]>=z[1] else 1 for z in values]


def transitions(labels,clean,attacked):
    return dict(occurrences=len(labels),clean_wrong=sum(p!=y for p,y in zip(clean,labels)),
        attacked_wrong=sum(p!=y for p,y in zip(attacked,labels)),
        new_wrong=sum(c==y and p!=y for c,p,y in zip(clean,attacked,labels)),
        recoveries=sum(c!=y and p==y for c,p,y in zip(clean,attacked,labels)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--execution-source',required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--audit-source-sha256',required=True);args=parser.parse_args()
    root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
    out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh audit output must remain below ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);wall,cpu=time.monotonic(),time.process_time();a=Audit(root)
    counters=dict(base_capture_tensor_loads=0,new_geometry_tensor_loads=0,original_dataset_loads=0,
        original_graph_loads=0,original_label_array_loads=0,classifier_weight_loads=0,model_constructions=0,
        model_forwards=0,gradients=0,optimizer_updates=0,API_calls=0,GPU_hours=0)
    result=dict(status='failed',schema='independent_target_degree_balance_audit_v1',post_hoc=True,
        independent_of_method_implementation_authorship=True,human_verification=False)
    try:
        own=sha(__file__);a.check(own==args.audit_source_sha256,'externally_frozen_independent_auditor')
        write(out/'registered_scope.json',dict(audit_source_sha256=own,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol_sha256=args.protocol_sha256,execution_source=args.execution_source,new_output_reads_started=False,
            allowed_tensor_reads='Exactly384BASEcapture payloads and384NEWgeometry artifacts after byte hashes; CPU only',
            forbidden='Original datasets/graphs/label arrays, classifier weights or models, inference, autograd/updates, CUDA, API/network',
            caveat='Dataset-level canonical reconstruction and pre-classifier temporal order are producer/source assertions, not replayed here.'))
        p=a.json(args.protocol,args.protocol_sha256);stage=a.path(args.stage,directory=True)
        a.check(p['schema']=='monti_target_degree_balance_v1' and p['status']=='frozen' and p['post_hoc'] is True
            and p['owner_execution_authorized'] is True and p['source_commit']==args.execution_source,'explicit_posthoc_registered_execution')
        a.check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and set(p['runs'])=={'3','4','5'}
            and len(p['valid_groups'])==len(set(p['valid_groups']))==128 and p['deadline_seconds']==5400
            and p['cpu_threads']==8 and p['execution_device']=='cpu' and p['cuda_visible_devices']==''
            and p['final_TEST_allowed'] is False,'fixed_three_arm_cohort_resources')
        a.check(p['qualification']['primary']=='balanced_target' and p['qualification']['min_mean_gain_pp']==1
            and p['qualification']['require_all_seed_deltas_positive'] is True
            and p['qualification']['require_mean_above_matched_random'] is True
            and p['qualification']['control_not_candidate'] is True,'frozen_primary_only_no_control_winner_rule')
        expected_inputs={}
        def retain(ref):
            path=a.file(ref['path'],ref['sha256']);previous=expected_inputs.setdefault(ref['path'],ref['sha256'])
            a.check(previous==ref['sha256'],'consistent_registered_input_hashes');return path
        for name,wanted in p['source_files'].items():
            retain(dict(path=name,sha256=wanted));source=subprocess.check_output(['git','show',args.execution_source+':'+name],cwd=root)
            a.check(hashlib.sha256(source).hexdigest()==wanted,'registered_source_committed_bytes')
        a.check({'course/target_degree_balance/'+name for name in ('__init__.py','balance_target.py','prepare.py','run.py','test_balance_target.py')}
            <=set(p['source_files']),'complete_new_balancing_package_source_freeze')
        snapshot=p['budget_snapshot'];ledger_source=subprocess.check_output(['git','show',args.execution_source+':evidence/research_cost_ledger.json'],cwd=root)
        a.check(hashlib.sha256(ledger_source).hexdigest()==snapshot['ledger_sha256_at_registered_source']
            and snapshot['new_phase_GPU_hours']==0 and snapshot['charged']+snapshot['reserve']<=snapshot['limit'],
            'source_bound_existing_resource_snapshot_and_cpu_phase')
        a.close(snapshot['remaining'],snapshot['limit']-snapshot['charged']-snapshot['reserve'],'budget_remaining_recount')
        for ref in p['inputs']:retain(ref)
        a.check(p['base_experiment_ref']['sha256']==BASE_SHA,'original_completed_base_protocol')
        base=a.json(retain(p['base_experiment_ref']));a.check(base['source_commit']==BASE_SOURCE
            and base['valid_groups']==p['valid_groups'] and base['final_test_allowed'] is False,'base_cohort_and_closed_test_gate')
        for name,wanted in base['source_files'].items():a.check(p['source_files'].get(name)==wanted,'all_reused_scientific_bytes_unchanged')
        for ref in base['inputs']+base['retained_inputs']:retain(ref)
        parity=a.json(retain(p['parity_result_ref']))
        a.check(parity['status']=='completed' and parity['parity_passed'] is True and parity['violation_count']==0
            and parity['completed_comparison_rows']==768,'prior_all_case_implementation_parity_only_not_efficacy_selection')
        runs={}
        for seed in SEEDS:
            spec=p['runs'][str(seed)];run=a.json(retain(spec['result_ref']));rows=keyed(a,a.rows(retain(spec['groups_ref'])),p['valid_groups'],'source_full128_group_cover')
            a.check(run['status']=='completed' and run['seed']==seed and run['source_commit']==BASE_SOURCE
                and run['protocol_sha256']==BASE_SHA and run['groups_jsonl_sha256']==spec['groups_ref']['sha256'],'source_refinement_exact_provenance')
            original={r['group']:r for r in run['exports']};captures={r['group']:r for r in spec['captures']}
            a.check(len(original)==len(run['exports'])==len(captures)==len(spec['captures'])==128
                and set(original)==set(captures)==set(rows),'source384_capture_reference_cover')
            directory=a.path(spec['result_ref']['path']).parent
            for group,ref in captures.items():
                path=retain(ref);a.check(rows[group]['graph_capture']==original[group] and ref['sha256']==original[group]['sha256']
                    and path==(directory/original[group]['path']).resolve(),'source_capture_row_and_manifest_binding')
            runs[seed]=dict(rows=rows,captures=captures)
        receipt=a.json(stage/'registered.json');r=a.json(stage/'result.json');provenance=dict(source_commit=args.execution_source,protocol_sha256=args.protocol_sha256)
        a.check(r['status'] in ('completed','completed_no_headroom') and r['provenance']==receipt['provenance']==provenance,
            'completed_or_no_headroom_preserved_status')
        for name,wanted in r['output_hashes'].items():a.file(stage/name,wanted)
        a.check(a.json(stage/'input_hashes.json')==receipt['source_and_input_hashes']==expected_inputs,'all_input_refs_reconciled')
        a.check(receipt['own_source_sha256']==p['source_files']['course/target_degree_balance/run.py']
            and receipt['base_experiment_ref']==p['base_experiment_ref'] and receipt['parity_result_ref']==p['parity_result_ref'],
            'executed_source_and_registered_rationale')
        marker=a.json(stage/'geometry_sealed.json',r['geometry_marker_sha256'])
        a.check(marker['status']=='geometry_sealed_before_any_classifier' and marker['provenance']==provenance
            and marker['source_captures']==384 and marker['variants']==marker['canonical_sequences']==1152,
            'all_geometry_sealed_before_classifier_marker')
        geometry=a.rows(stage/'geometry.jsonl');a.check(len(geometry)==384 and r['prepared_cases']==384,'complete384_geometry_rows')
        records={(row['seed'],row['group']):row for row in geometry}
        a.check(len(records)==384 and set(records)=={(s,g) for s in SEEDS for g in p['valid_groups']},'exact_geometry_case_cover')
        a.file(stage/'geometry.jsonl',marker['geometry_jsonl_sha256'])
        exports={(row['seed'],row['group']):row for row in marker['artifacts']}
        a.check(len(exports)==len(marker['artifacts'])==384 and set(exports)==set(records),'sealed384_artifact_manifest')
        for key,row in records.items():
            export=exports[key];a.check(export==dict(seed=key[0],group=key[1],**row['artifact']),'sealed_artifact_row_binding')
            a.check(row['artifact']['path']=='geometry/seed%d_group%d.pt'%key,'new_artifact_strict_relative_geometry_path')
            a.file(stage/row['artifact']['path'],row['artifact']['sha256'])
        write(out/'authorized_input_manifest.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())])
        # Every allowed tensor payload has now been named/byte-verified. This
        # audit imports Torch for CPU tensor operations only; no DGL/data/model.
        a.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit_cpu_audit_visibility')
        import torch
        a.check(sys.version_info[:3]==(3,8,20) and torch.__version__=='1.13.1+cu117','frozen_python_random_replay_and_torch_runtime')
        a.check(not torch.cuda.is_available() and torch.cuda.device_count()==torch._C._cuda_getDeviceCount()==0,'zero_visible_cuda_devices')
        torch.set_num_threads(8);nonidentity={str(s):0 for s in SEEDS};original_node_count=None;geometry_totals=[]
        geometry_statistics=collections.Counter()
        for seed in SEEDS:
            nodes=edges=occurrences=0;unique=set()
            for group in p['valid_groups']:
                row=records[seed,group];saved=runs[seed]['rows'][group];source_ref=runs[seed]['captures'][group]
                source_path=a.file(source_ref['path'],source_ref['sha256']);artifact_path=a.file(stage/row['artifact']['path'],row['artifact']['sha256'])
                old=torch.load(source_path,map_location='cpu');counters['base_capture_tensor_loads']+=1
                new=torch.load(artifact_path,map_location='cpu');counters['new_geometry_tensor_loads']+=1
                a.check(set(old)=={'capture','features'} and set(new)=={'capture','adjacencies','canonical_sequences'}
                    and set(new['adjacencies'])==set(new['canonical_sequences'])==set(ARMS),'exact_saved_tensor_schemas')
                cap=old['capture'];a.check(set(cap)==set(new['capture'])=={'x_atk','adj_attack','target_ids','candidate_ids'},'exact_source_capture_schema')
                for name,value in cap.items():a.check(torch.equal(value,new['capture'][name])
                    and tensor_hash(value)==tensor_hash(new['capture'][name]),'entire_original_capture_shape_dtype_bytes_unchanged')
                x0,adj=cap['x_atk'],cap['adj_attack'];targets=cap['target_ids'].tolist();candidates=cap['candidate_ids'].tolist()
                t,c,n=len(targets),len(candidates),len(x0);eb=saved['edge_budget']
                a.check(t>0 and n>0 and targets==row['node_ids']==saved['node_ids'] and row['labels']==saved['labels'] and len(row['labels'])==t
                    and t==row['targets'] and all(y==1 for y in row['labels']),'complete_stored_target_ids_labels_order')
                a.check(cap['target_ids'].dtype==cap['candidate_ids'].dtype==torch.int64 and x0.dtype==adj.dtype==torch.float32
                    and tuple(x0.shape)==(n,768) and tuple(adj.shape)==(n,t+c+n) and bool(torch.isfinite(x0).all())
                    and bool(torch.isfinite(adj).all()) and bool(((adj==0)|(adj==1)).all()),'source_shapes_dtypes_finite_binary')
                a.check(n==row['node_budget']==saved['node_budget'] and eb==row['edge_budget'] and int(adj.sum())==eb
                    and len(set(targets+candidates))==t+c,'physical_counts_and_disjoint_original_ids')
                a.check(torch.equal(old['features']['baseline'],x0),'original_features_equal_baseline_x0')
                identity=dict(features=tensor_hash(x0),targets=tensor_hash(cap['target_ids']),candidates=tensor_hash(cap['candidate_ids']))
                a.check(identity==row['immutable_capture_content_hashes'] and identity['features']==saved['feature_statistics']['baseline']['content_sha256']
                    and row['source_capture_sha256']==source_ref['sha256'],'all_immutable_content_hashes')
                injected=[plan['source_id'] for plan in row['geometry']['row_plan']]
                a.check(len(injected)==n and injected==list(range(injected[0],injected[0]+n))
                    and all(0<=v<injected[0] for v in targets+candidates),'consecutive_injected_ids_after_original_id_space')
                if original_node_count is None:original_node_count=injected[0]
                else:a.check(injected[0]==original_node_count,'same_original_node_boundary_all_cases')
                matrices,expected_geometry=independent_geometry(a,adj.to(torch.int64).tolist(),targets,candidates,injected,seed,group)
                a.check(row['geometry']==expected_geometry,'independent_full_geometry_random_and_history_replay')
                if not expected_geometry['balanced_equals_baseline']:nonidentity[str(seed)]+=1
                geometry_statistics['balancing_iterations']+=expected_geometry['balancing_iterations']
                geometry_statistics['final_rewired_edges']+=expected_geometry['final_rewired_edges']
                geometry_statistics['potential_reduction']+=expected_geometry['sum_squared_target_degree_before']-expected_geometry['sum_squared_target_degree_balanced']
                for arm in ARMS:
                    matrix=new['adjacencies'][arm];sequence=new['canonical_sequences'][arm];meta=row['variants'][arm]
                    a.check(matrix.dtype==torch.float32 and tuple(matrix.shape)==tuple(adj.shape)
                        and matrix.to(torch.int64).tolist()==matrices[arm] and bool(((matrix==0)|(matrix==1)).all()),'actual_variant_equals_independent_geometry')
                    legal=torch.tril(torch.ones_like(matrix,dtype=torch.bool),diagonal=t+c-1)
                    a.check(not bool((matrix[~legal]!=0).any()) and bool((matrix[:,:t].sum(1)>=1).all())
                        and int(matrix.sum())==eb and torch.equal(matrix[:,t:],adj[:,t:])
                        and torch.equal(matrix.sum(1),adj.sum(1)) and torch.equal(matrix[:,:t].sum(1),adj[:,:t].sum(1)),
                        'legal_lower_triangle_role_blocks_row_target_and_total_counts')
                    for i in range(n):
                        expected_k=0 if arm=='baseline' else expected_geometry['row_plan'][i]['final_k']
                        removed=int(((adj[i,:t]==1)&(matrix[i,:t]==0)).sum())
                        added=int(((adj[i,:t]==0)&(matrix[i,:t]==1)).sum())
                        a.check(removed==added==expected_k and int((adj[i,:t]!=matrix[i,:t]).sum())==2*expected_k,
                            'actual_balanced_random_final_row_hamming_match')
                    physical=[sum(v)+sum(matrices[arm][j][t+c+i] for j in range(i+1,n)) for i,v in enumerate(matrices[arm])]
                    a.check(physical==expected_geometry['physical_injected_degrees']==meta['physical_injected_degrees'],
                        'every_physical_injected_degree_independently_preserved')
                    a.check(meta['actual_injected_nodes']==n and meta['actual_injected_nonself_undirected_edges']==eb
                        and meta['adjacency_content_sha256']==tensor_hash(matrix),'actual_budget_adjacency_hash_recount')
                    a.check(sequence.dtype==torch.float32 and tuple(sequence.shape)==(t,7,768) and bool(torch.isfinite(sequence).all())
                        and tensor_hash(sequence)==meta['canonical_sequence_content_sha256'],'sealed_finite_canonical_input_shape_hash')
                    a.check(torch.equal(sequence[:,[0,1,2]],new['canonical_sequences']['baseline'][:,[0,1,2]])
                        and meta['first_hop_tokens0_1_2_exact'] is True,'actual_self_known_hop_tokens_bit_identity')
                nodes+=n;edges+=eb;occurrences+=t;unique.update(targets)
                del old,new,cap,x0,adj,matrix,sequence
            a.check((nodes,edges,occurrences,len(unique))==(7776,358491,1091,374),'all128_scenario_budgets_occurrences_and_unique_targets')
            geometry_totals.extend(dict(seed=seed,arm=arm,groups=128,target_occurrences=1091,unique_targets=374,
                summed_injected_nodes=nodes,summed_nonself_undirected_edges=edges) for arm in ARMS)
        a.check(counters['base_capture_tensor_loads']==counters['new_geometry_tensor_loads']==384,'exact768_allowed_tensor_deserializations')
        a.check(nonidentity==marker['balanced_nonidentity_cases_by_seed']==r['balanced_nonidentity_cases_by_seed']
            and geometry_totals==marker['whole_seed_geometry_totals'],'headroom_and_budget_totals_recount')
        geom_counts=dict(source_capture_tensor_loads=384,source_captures_prepared=384,geometry_graph_reconstructions=1152,
            canonical_sequence_updates=1152,first_hop_invariant_checks=1152)
        a.check(set(marker['counters'])==set(r['counters'])==set(COUNTER_KEYS),'all_declared_operation_counter_keys')
        for name,value in marker['counters'].items():a.check(value==geom_counts.get(name,0),'geometry_stage_has_zero_classifier_queries')
        a.check(r['all_geometry_prepared_before_classifier'] is True and marker['original_content_hashes']=={
            key:r['immutable_content_hashes']['before'][key] for key in ('features','feat_sum','degrees')},'geometry_seal_original_input_hashes')
        a.check(r['immutable_content_hashes_retained'] is True and r['immutable_content_hashes']['before']==r['immutable_content_hashes']['after'],
            'producer_model_and_input_hash_immutability_receipts')
        a.check(r['GPU_hours']==0 and r['cuda_visibility']=='' and r['torch_cuda_available'] is False
            and r['torch_visible_gpu_count']==r['native_visible_gpu_count']==0 and r['final_TEST_used'] is False
            and r['experimental_API_calls']==0 and r['process_cpu_seconds']>=0 and 0<r['wall_seconds']<=5415,'cpu_resource_clocks_and_no_test_api')
        metrics=[];contrasts={};primary=None
        counts=r['counters'];expected_counts={name:geom_counts.get(name,0) for name in counts}
        no_headroom=any(value==0 for value in nonidentity.values())
        if no_headroom:
            a.check(r['status']=='completed_no_headroom' and r['classifier_loading_after_geometry_seal'] is False,'zero_headroom_stops_before_classifiers')
            a.check(not (stage/'clean_predictions.jsonl').exists() and all(not (stage/('%s_seed%d_predictions.jsonl'%(arm,seed))).exists()
                for arm in ARMS for seed in SEEDS),'no_prediction_files_when_gate_closed')
        else:
            a.check(r['status']=='completed' and r['classifier_loading_after_geometry_seal'] is True,'full_cohort_queries_after_nonzero_headroom')
            expected_counts.update(prepared_artifact_tensor_loads=384,evaluation_graph_reconstructions=1152,
                original_gcn_load_dispatches=1,original_gcn_models_loaded=1,official_gaga_load_dispatches=1,official_gaga_models_loaded=1,
                original_gcn_forward_dispatches=1280,original_gcn_completed_forwards=1280,
                official_gaga_forward_dispatches=1280,official_gaga_completed_forwards=1280,
                original_gcn_clean_forwards=128,official_gaga_clean_forwards=128,
                original_gcn_attacked_forwards=1152,official_gaga_attacked_forwards=1152)
            clean=keyed(a,a.rows(stage/'clean_predictions.jsonl'),p['valid_groups'],'fresh128_clean_case_cover')
            clean_pred={}
            for group,row in clean.items():
                a.check(row['node_ids']==records[3,group]['node_ids'] and row['labels']==records[3,group]['labels'],'fresh_clean_identity_order')
                for detector in ('gcn','gaga'):clean_pred[group,detector]=classify(a,row[detector+'_logits'],len(row['labels']))
            outputs={};summaries={(row['seed'],row['arm']):row for row in r['summaries']}
            a.check(len(summaries)==len(r['summaries'])==9,'all9_seed_arm_summaries')
            for seed in SEEDS:
                for arm in ARMS:
                    raw=keyed(a,a.rows(stage/('%s_seed%d_predictions.jsonl'%(arm,seed))),p['valid_groups'],'all1152_fresh_attacked_prediction_rows')
                    totals={name:collections.Counter() for name in ('gcn','gaga')};macro={name:[] for name in ('gcn','gaga')}
                    for group,row in raw.items():
                        geometry=records[seed,group];n=geometry['targets'];labels=geometry['labels'];meta=geometry['variants'][arm]
                        a.check(row['seed']==seed and row['arm']==arm and row['node_ids']==geometry['node_ids'] and row['labels']==labels,
                            'fresh_variant_complete_target_identity')
                        a.check(row['node_budget']==row['actual_injected_nodes']==geometry['node_budget']
                            and row['edge_budget']==row['actual_injected_edges']==geometry['edge_budget']
                            and row['source_capture_sha256']==geometry['source_capture_sha256']
                            and row['prepared_artifact_sha256']==geometry['artifact']['sha256']
                            and row['variant_adjacency_content_sha256']==meta['adjacency_content_sha256']
                            and row['canonical_sequence_content_sha256']==meta['canonical_sequence_content_sha256'],
                            'prediction_bound_to_actual_sealed_variant_and_cost')
                        for key,name in (('injected_feature_content_sha256','features'),('candidate_order_content_sha256','candidates'),('target_order_content_sha256','targets')):
                            a.check(row[key]==geometry['immutable_capture_content_hashes'][name],'prediction_immutable_feature_and_id_binding')
                        for detector in ('gcn','gaga'):
                            pred=classify(a,row[detector+'_logits'],n);count=transitions(labels,clean_pred[group,detector],pred)
                            a.check(row[detector+'_transitions']==count,'fresh_raw_argmax_transition_recount')
                            totals[detector].update(count);macro[detector].append(100*count['attacked_wrong']/n)
                    summary=summaries[seed,arm];a.check(summary['groups']==128 and summary['unique_targets']==374,'summary_fixed_denominators')
                    for detector in ('gcn','gaga'):
                        total={key:totals[detector][key] for key in FIELDS};mr=100*total['attacked_wrong']/1091
                        a.check(total['occurrences']==1091 and summary['totals'][detector]==total,'raw_totals_reconcile_summary')
                        a.close(summary[detector+'_mr_percent'],mr,'raw_mr_recount')
                        metrics.append(dict(seed=seed,arm=arm,detector=detector,**total,mr_percent=mr,group_macro_mr_percent=statistics.fmean(macro[detector])))
                    outputs[seed,arm]=raw
            lookup={(row['seed'],row['arm'],row['detector']):row for row in metrics}
            t95=math.sqrt(2)*.95/math.sqrt(1-.95**2)
            for arm,reference in (('balanced_target','baseline'),('matched_random','baseline'),('balanced_target','matched_random')):
                for detector in ('gcn','gaga'):
                    values=[lookup[s,arm,detector]['mr_percent']-lookup[s,reference,detector]['mr_percent'] for s in SEEDS]
                    mean=statistics.fmean(values);sd=statistics.stdev(values);half=t95*sd/math.sqrt(3)
                    paired=collections.Counter()
                    for seed in SEEDS:
                        for group in p['valid_groups']:
                            left=outputs[seed,reference][group];right=outputs[seed,arm][group];labels=left['labels']
                            first=classify(a,left[detector+'_logits'],len(labels));second=classify(a,right[detector+'_logits'],len(labels))
                            delta=sum(x!=y for x,y in zip(second,labels))-sum(x!=y for x,y in zip(first,labels))
                            paired['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']+=1
                            for y,c,x,z in zip(labels,clean_pred[group,detector],first,second):
                                paired['occurrences']+=1;paired['gained_wrong']+=x==y and z!=y;paired['lost_wrong']+=x!=y and z==y
                                if c==y:
                                    paired['new_success_gained']+=x==y and z!=y;paired['baseline_new_success_lost']+=x!=y and z==y
                                else:
                                    paired['additional_recovery']+=x!=y and z==y;paired['previous_recovery_lost']+=x==y and z!=y
                    a.check(paired['occurrences']==3273 and sum(paired[k] for k in ('group_wins','group_ties','group_losses'))==384,'paired_correlated_complete_denominators')
                    contrasts[arm+'-minus-'+reference+'_'+detector]=dict(seed_differences_pp=values,mean_difference_pp=mean,
                        sample_sd_pp=sd,paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts=dict(paired))
            deltas=contrasts['balanced_target-minus-baseline_gaga']['seed_differences_pp']
            advantage=contrasts['balanced_target-minus-matched_random_gaga']['mean_difference_pp']
            primary=dict(mean_delta_pp=statistics.fmean(deltas),all_three_seed_deltas_positive=all(v>0 for v in deltas),
                mean_advantage_over_random_pp=advantage,criterion_met=statistics.fmean(deltas)>=1 and all(v>0 for v in deltas) and advantage>0,
                definition='Mean>=1pp official-GAGA MR versus freshbaseline, all3seed differences>0, mean>matchedrandom; no forced winner or finalTEST claim')
        expected_outputs={'registered.json','geometry.jsonl','geometry_sealed.json','input_hashes.json'}
        if not no_headroom:expected_outputs.update({'clean_predictions.jsonl'}|{'%s_seed%d_predictions.jsonl'%(arm,seed) for arm in ARMS for seed in SEEDS})
        a.check(set(r['output_hashes'])==expected_outputs,'all_geometry_and_prediction_outputs_hash_bound')
        a.check(counts==expected_counts and all(counts[name]==0 for name in NO_MODEL),'exact_geometry_and_classifier_query_counters')
        result.update(status='passed',producer_status=r['status'],provenance=provenance,headroom_nonidentity=nonidentity,
            independently_replayed_geometry=dict(geometry_statistics),geometry_totals=geometry_totals,metrics=metrics,
            paired_contrasts=contrasts,primary_development_criterion=primary,producer_counters=counts,
            immutable_content_hash_receipts=r['immutable_content_hashes'],producer_cpu_seconds=r['process_cpu_seconds'],producer_wall_seconds=r['wall_seconds'],
            audit_runtime=dict(python=sys.version,torch=torch.__version__,cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')),
            limitations=['POST HOC exploratory study after prior negative outcomes; mathematical target-degree optimum is not attack efficacy or novelty.',
                'Exactly384original capture and384new artifact tensors are inspected on CPU. No original dataset/graph/label array or classifier weight is loaded and no classifier is called.',
                'Relative hard-edge legality, actual incidence/physical injected degrees, all x0/ID bytes, random replay, Phi and tokens0..2 bits are independently checked. Original-original arcs, whole canonical input construction and timing of the seal before model loads follow producer/source assertions, not an independent graph/model replay.',
                'Fresh baseline predictions remain distinct from old saved predictions. No class-protection property is assumed from target-degree invariants.',
                '3273 repeated target occurrences and384cases across3seeds are correlated. Conditional paired t95 intervals do not provide IID news or fresh-teacher uncertainty.',
                'No efficacy/novelty/course acceptance or human verification is implied; finalTEST/tag checkpoints remain separate.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(audit_operation_counters=counters,check_counts=dict(a.counts),byte_hashed_files=len(a.files),
            process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'input_hashes.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())])
        write(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
