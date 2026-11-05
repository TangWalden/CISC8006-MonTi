"""Independent role-reallocation saved/tensor audit, no model execution.

Freeze before new ID metadata/generation. After authorization, allowed reads:
original DGL raw x (discard y unused), VALID IDs/CSR membership,384FRESHartifacts,
new saved VALID logits, frozen checkpoint opaque bytes/state receipt fields.
No old capture or prediction decoding, classifier/weight deserialization, model,
training/gradients/API/GPU/TEST prediction queries. Target ranking uses baseline
INJECTED-ATTACHMENT column counts, never total clean physical target degree.
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

ARMS=('baseline','balanced_target_fill','random_target_fill','candidate_rewire')
SEEDS=(3,4,5)
FIXED_TOKENS=(0,)
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
SHARP_SOURCE='43e46e37621f203a51bbdfdb1667f5466e73ad68'
SHARP_SHA='d40fc7d8c01b1eac1f8afac72e215161897fe8296206bdc194e4821d0a050447'
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
FULL_COUNTS={'baseline_engine_load_dispatches': 3, 'baseline_engines_loaded': 3, 'dataset_loads': 3, 'proposal_gcn_models_loaded': 3, 'generator_models_loaded': 3, 'generator_forward_dispatches': 384, 'generator_completed_forwards': 384, 'baseline_generations': 384, 'proposal_gcn_forward_attempts': 0, 'scheduler_restoration_ticks_reported': 24, 'geometry_cases_prepared': 384, 'geometry_graph_reconstructions': 1536, 'canonical_sequence_updates': 1536, 'self_token_invariant_checks': 1536, 'prepared_artifact_tensor_loads': 384, 'evaluation_graph_reconstructions': 1536, 'original_gcn_load_dispatches': 1, 'original_gcn_models_loaded': 1, 'official_gaga_load_dispatches': 1, 'official_gaga_models_loaded': 1, 'original_gcn_forward_dispatches': 1664, 'original_gcn_completed_forwards': 1664, 'official_gaga_forward_dispatches': 1664, 'official_gaga_completed_forwards': 1664, 'original_gcn_clean_forwards': 128, 'official_gaga_clean_forwards': 128, 'original_gcn_attacked_forwards': 1536, 'official_gaga_attacked_forwards': 1536, 'generator_optimizer_steps': 0, 'fresh_teacher_model_constructions': 0, 'fresh_teacher_forwards': 0, 'autograd_calls': 0, 'API_calls': 0, 'TEST_queries': 0}
GEOMETRY_COUNTS={k:FULL_COUNTS[k] for k in ('baseline_engine_load_dispatches', 'baseline_engines_loaded', 'dataset_loads', 'proposal_gcn_models_loaded', 'generator_models_loaded', 'generator_forward_dispatches', 'generator_completed_forwards', 'baseline_generations', 'proposal_gcn_forward_attempts', 'scheduler_restoration_ticks_reported', 'geometry_cases_prepared', 'geometry_graph_reconstructions', 'canonical_sequence_updates', 'self_token_invariant_checks')}


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
        if not value:raise AssertionError('Reallocation audit failed: '+name)
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

def classify(a,values,n):
    a.check(len(values)==n,'binary_logit_occurrence_count')
    for row in values:a.check(len(row)==2 and all(isinstance(v,(int,float)) and math.isfinite(v) for v in row),'finite_binary_raw_logits')
    return [0 if z[0]>=z[1] else 1 for z in values]

def transitions(labels,clean,attacked):
    return dict(occurrences=len(labels),clean_wrong=sum(p!=y for p,y in zip(clean,labels)),
        attacked_wrong=sum(p!=y for p,y in zip(attacked,labels)),
        new_wrong=sum(c==y and p!=y for c,p,y in zip(clean,attacked,labels)),
        recoveries=sum(c!=y and p==y for c,p,y in zip(clean,attacked,labels)))

def compact_hash(value):return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()

def reallocate_independently(a,baseline,targets,candidates,injected,seed,group):
    """Independent integer/count replay; no method-module import or labels."""
    t,c,n=len(targets),len(candidates),len(injected);matrix=[list(map(int,row)) for row in baseline]
    arms={arm:[row[:] for row in matrix] for arm in ARMS};tcol={v:j for j,v in enumerate(targets)};ccol={v:t+j for j,v in enumerate(candidates)}
    attachment=[sum(row[j] for row in matrix) for j in range(t)];plans=[]
    def digest(source,purpose,endpoint=None):
        value=[seed,str(group),source,purpose,endpoint,'candidate-target-reallocation-v1']
        return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()
    for i,row in enumerate(matrix):
        c1=[v for v in candidates if row[ccol[v]]];t0=[v for v in targets if not row[tcol[v]]];c0=[v for v in candidates if not row[ccol[v]]]
        k=min(len(c1),len(t0),len(c0));source=injected[i]
        dk=digest(source,'candidate-donors');rk=digest(source,'random-target-addition');ck=digest(source,'control-candidate-addition')
        donors=random.Random(int(dk,16)).sample(sorted(c1),k)
        primary=sorted(t0,key=lambda v:(attachment[tcol[v]],digest(source,'primary-target-tie',v),v))[:k]
        random_targets=random.Random(int(rk,16)).sample(sorted(t0),k);control=random.Random(int(ck,16)).sample(sorted(c0),k)
        for arm in ARMS[1:]:
            for donor in donors:arms[arm][i][ccol[donor]]=0
        for v in primary:arms['balanced_target_fill'][i][tcol[v]]=1
        for v in random_targets:arms['random_target_fill'][i][tcol[v]]=1
        for v in control:arms['candidate_rewire'][i][ccol[v]]=1
        plans.append(dict(row=i,source_id=source,baseline_candidate_count=len(c1),missing_target_count=len(t0),missing_candidate_count=len(c0),k=k,
            candidate_donor_fraction=k/len(c1) if c1 else None,removed_candidate_ids=donors,primary_added_target_ids=primary,
            random_added_target_ids=random_targets,control_added_candidate_ids=control,donor_key_sha256=dk,random_target_key_sha256=rk,
            control_candidate_key_sha256=ck,primary_selected_target_priorities=[dict(target_id=v,baseline_injected_attachment_count=attachment[tcol[v]],
                tie_key_sha256=digest(source,'primary-target-tie',v)) for v in primary]))
    total_k=sum(v['k'] for v in plans);candidate_before=sum(sum(row[t:t+c]) for row in matrix)
    target_after={arm:[sum(row[j] for row in rows) for j in range(t)] for arm,rows in arms.items()}
    candidate_after={arm:sum(sum(row[t:t+c]) for row in rows) for arm,rows in arms.items()}
    degrees=[sum(row)+sum(matrix[j][t+c+i] for j in range(i+1,n)) for i,row in enumerate(matrix)]
    covered={targets[j] for j,d in enumerate(attachment) if d}
    coverage={arm:{targets[j] for j,d in enumerate(values) if d} for arm,values in target_after.items()};identity={}
    for arm,rows in arms.items():
        differences=[sum(x!=y for x,y in zip(left,right)) for left,right in zip(rows,matrix)]
        identity[arm]=dict(exact_equals_baseline=not any(differences),changed_rows=sum(v>0 for v in differences),changed_coordinates=sum(differences),comparison='Exact binary owned-adjacency entries')
        a.check(covered<=coverage[arm],'no_previously_covered_target_lost')
        a.check(sum(target_after[arm])==sum(attachment)+(total_k if arm in ARMS[1:3] else 0)
            and candidate_after[arm]==candidate_before-(total_k if arm in ARMS[1:3] else 0),'target_candidate_role_incidence_accounting')
    expected=dict(identity=identity,row_plan=plans,total_reallocated_edges=total_k,
        target_incoming_injected_degrees_before=attachment,target_incoming_injected_degrees_after=target_after,
        total_target_incidence_before=sum(attachment),total_target_incidence_after={arm:sum(v) for arm,v in target_after.items()},
        candidate_incidence_before=candidate_before,candidate_incidence_after=candidate_after,
        original_row_edge_counts=[sum(row) for row in matrix],physical_injected_degrees=degrees,total_owned_nonself_undirected_edges=sum(map(sum,matrix)),
        newly_covered_target_ids={arm:sorted(values-covered) for arm,values in coverage.items()},
        previously_covered_target_ids_lost={arm:sorted(covered-values) for arm,values in coverage.items()},primary_equals_random=arms['balanced_target_fill']==arms['random_target_fill'])
    return arms,expected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--protocol',required=True);parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--execution-source',required=True);parser.add_argument('--stage',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--audit-source-sha256',required=True);args=parser.parse_args()
    root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output);out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh audit output must remain below ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);wall,cpu=time.monotonic(),time.process_time();a=Audit(root)
    counts=dict(original_DGL_graph_loads=0,VALID_group_ID_array_loads=0,target_assignment_loads=0,label_values_indexed=0,
        fresh_artifact_tensor_loads=0,old_capture_tensor_loads=0,old_prediction_streams_decoded=0,classifier_weight_loads=0,
        model_constructions=0,model_forwards=0,autograd_calls=0,optimizer_updates=0,API_calls=0,TEST_prediction_reads=0,GPU_hours=0)
    result=dict(status='failed',schema='independent_candidate_target_reallocation_audit_v1',post_hoc=True,
        independent_of_method_implementation_authorship=True,human_verification=False)
    try:
        a.check(sha(__file__)==args.audit_source_sha256,'externally_frozen_auditor')
        write(out/'registered_scope.json',dict(audit_source_sha256=args.audit_source_sha256,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            expected_source=args.execution_source,protocol_sha256=args.protocol_sha256,new_scientific_reads_started=False,
            allowed='Original DGL raw x/box; discard y unused; original VALID group IDs and membership metadata;384FRESHartifacts; saved new VALID logits.',
            forbidden='Original label operations; old capture/prediction decoding; model/checkpoint deserialization or inference; gradients/updates; API/TEST predictions/CUDA.',
            limitation='Source/state receipts verify generator/scheduler and temporal model gates; no model replay or historical bitwise trajectory identity is claimed.'))
        p=a.json(args.protocol,args.protocol_sha256);stage=a.path(args.stage,directory=True);protocol_path=a.path(args.protocol)
        a.check(p['schema']=='monti_candidate_target_reallocation_v1' and p['status']=='frozen' and p['source_commit']==args.execution_source
            and p['post_hoc'] is True and p['owner_execution_authorized'] is True and p['final_TEST_allowed'] is False,'frozen_posthoc_owner_authorized_source')
        a.check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['scientific_version']=='candidate-target-reallocation-v1'
            and p['expected_if_evaluated']==FULL_COUNTS and p['deadline_seconds']==5400 and p['cpu_threads']==8
            and p['execution_device']=='cpu' and p['cuda_visible_devices']=='' and len(p['valid_groups'])==len(set(p['valid_groups']))==128,'fixed_parameters_and34_counters')
        q=p['qualification'];a.check(q['primary']=='balanced_target_fill' and q['min_mean_gain_pp']==1
            and all(q[k] is True for k in ('require_all_seed_deltas_positive','require_paired_t95_lower_above_zero',
                'require_mean_above_candidate_rewire','controls_not_candidates','random_target_comparator_need_not_be_worse')),'frozen_role_allocation_primary_no_priority_superiority_or_control_winner')
        a.check(not q.get('require_mean_above_random_target_fill',False),'no_required_priority_superiority_over_random_target_fill')
        amendment=p['pre_outcome_design_amendment'];a.check(amendment['new_context_metadata_opened'] is False
            and amendment['new_model_calls']==0,'qualification_algebraic_design_amendment_before_new_metadata_and_calls')
        inputs={}
        def retain(ref):
            path=a.file(ref['path'],ref['sha256']);prior=inputs.setdefault(ref['path'],ref['sha256']);a.check(prior==ref['sha256'],'consistent_input_refs');return path
        for name,wanted in p['source_files'].items():
            retain(dict(path=name,sha256=wanted));data=subprocess.check_output(['git','show',args.execution_source+':'+name],cwd=root)
            a.check(hashlib.sha256(data).hexdigest()==wanted,'registered_source_committed_bytes')
        for ref in p['inputs']+p.get('retained_inputs',[]):retain(ref)
        synthetic=a.json(retain(p['synthetic_receipt_ref']))
        a.check(synthetic['status']=='passed' and synthetic['tests_run']==11 and not synthetic['failures'] and not synthetic['errors']
            and synthetic['real_data_reads']==synthetic['model_calls']==synthetic['GPU_hours']==synthetic.get('API_calls',0)==0
            and all(synthetic['source_files']['course/candidate_target_reallocation/'+name]==p['source_files']['course/candidate_target_reallocation/'+name]
                for name in ('core.py','synthetic_tests.py')),'frozen11_synthetic_checks_no_real_data_or_models_receipt')
        snapshot=p['budget_snapshot'];ledger_bytes=subprocess.check_output(['git','show',args.execution_source+':evidence/research_cost_ledger.json'],cwd=root)
        a.check(hashlib.sha256(ledger_bytes).hexdigest()==snapshot['ledger_sha256_at_registered_source']
            and snapshot['new_phase_GPU_hours']==0 and snapshot['charged']+snapshot['reserve']<=snapshot['limit'],'source_bound_resource_snapshot')
        a.close(snapshot['remaining'],snapshot['limit']-snapshot['charged']-snapshot['reserve'],'resource_remaining_recount')
        a.check({'course/candidate_target_reallocation/'+name for name in ('__init__.py','prepare.py','run.py','core.py','synthetic_tests.py')}<=set(p['source_files']),'complete_new_package_source_freeze')
        a.check(p['base_experiment_ref']['sha256']==BASE_SHA,'exact_frozen_legacy_generator_family_protocol')
        old=a.json(retain(p['base_experiment_ref']));a.check(old['source_commit']==BASE_SOURCE and p['excluded_original_valid_groups']==old['valid_groups']
            and len(old['valid_groups'])==128 and old['final_test_allowed'] is False,'same_legacy_family_and_first128_context_exclusion')
        a.check(p['excluded_sharpen_protocol_ref']['sha256']==SHARP_SHA,'exact_second_context_registration')
        sharpen=a.json(retain(p['excluded_sharpen_protocol_ref']))
        a.check(sharpen['schema']=='monti_feature_logit_sharpen_v1' and sharpen['source_commit']==SHARP_SOURCE
            and sharpen['excluded_valid_groups']==old['valid_groups'] and p['excluded_sharpen_valid_groups']==sharpen['valid_groups']
            and not set(old['valid_groups'])&set(sharpen['valid_groups']) and len(sharpen['valid_groups'])==128
            and p['excluded_valid_groups']==sorted(set(old['valid_groups'])|set(sharpen['valid_groups']))
            and len(p['excluded_valid_groups'])==256 and not set(p['excluded_valid_groups'])&set(p['valid_groups']),
            'both_exact_old_context_lists_excluded_no_regrouping')
        for name,wanted in p['excluded_sharpen_source_files'].items():
            a.check(sharpen['source_files'].get(name)==wanted,'second_context_source_inventory');retain(dict(path=name,sha256=wanted))
            data=subprocess.check_output(['git','show',SHARP_SOURCE+':'+name],cwd=root)
            a.check(hashlib.sha256(data).hexdigest()==wanted,'second_context_committed_source_hash')
        for name,wanted in old['source_files'].items():a.check(p['source_files'].get(name)==wanted,'reused_scientific_bytes_unchanged')
        for ref in old['inputs']+old['retained_inputs']:retain(ref)
        baselines={}
        for seed in SEEDS:
            ref=old['baseline_runs'][str(seed)];directory=pathlib.Path(ref['directory'])
            baselines[seed]=a.json(retain(dict(path=(directory/'result.json').as_posix(),sha256=ref['result_sha256'])))
            retain(dict(path=(directory/'final_model.pt').as_posix(),sha256=ref['final_model_sha256']))
            a.check(baselines[seed]['status']=='completed' and baselines[seed]['seed']==seed and baselines[seed]['arm']=='legacy'
                and baselines[seed]['source_mode']=='legacy' and baselines[seed]['epochs']==8 and baselines[seed]['optimizer_steps']==2048,
                'same_terminal8_legacy_checkpoint_receipt_metadata')
        metadata_scope=a.json(protocol_path.parent/'metadata_scope.json',p['metadata_scope_sha256'])
        a.check(metadata_scope['source_commit']==args.execution_source and metadata_scope['source_files']==p['source_files']
            and metadata_scope['reference_data']==p['reference_data'] and metadata_scope['new_group_ID_arrays_opened'] is False,'source_metadata_scope_precedes_ID_array_selection')
        for ref in p['reference_data'].values():retain(ref)
        r=a.json(stage/'result.json');registration=a.json(stage/'registered.json');provenance=dict(source_commit=args.execution_source,protocol_sha256=args.protocol_sha256)
        a.check(r['status'] in ('completed','completed_no_headroom') and r['provenance']==registration['provenance']==provenance,'producer_completed_or_planned_no_headroom')
        a.check(registration['own_source_sha256']==p['source_files']['course/candidate_target_reallocation/run.py'] and registration['selection']==p['selection']
            and registration['valid_groups']==p['valid_groups'] and registration['excluded_valid_groups']==p['excluded_valid_groups']
            and registration['baseline_engine_protocol_override']=={'valid_groups':p['valid_groups']},'new_only_VALID_group_override_and_registered_source')
        for name,wanted in r['output_hashes'].items():a.file(stage/name,wanted)
        a.check(a.json(stage/'input_hashes.json')==registration['source_and_input_hashes']==inputs,'all_source_and_opaque_inputs_reconciled')
        marker=a.json(stage/'geometry_sealed.json',r['geometry_marker_sha256'])
        a.check(marker['status']=='all_fresh_graph_topology_proposals_sealed_before_dedicated_evaluators_and_any_official_victim'
            and marker['provenance']==provenance and marker['source_captures']==384 and marker['variants']==marker['canonical_sequences']==1536
            and marker['fixed_tokens']==list(FIXED_TOKENS),'all384GEN_and1536_proposals_sealed_before_dedicated_evaluators')
        geometry=a.rows(stage/'geometry.jsonl');records={(v['seed'],v['group']):v for v in geometry}
        a.check(len(records)==len(geometry)==r['prepared_cases']==384 and set(records)=={(s,g) for s in SEEDS for g in p['valid_groups']},'all_new384_cases_no_filtering')
        a.file(stage/'geometry.jsonl',marker['geometry_jsonl_sha256']);exports={(v['seed'],v['group']):v for v in marker['artifacts']}
        a.check(len(exports)==len(marker['artifacts'])==384 and set(exports)==set(records),'complete384_artifact_manifest')
        for key,row in records.items():
            a.check(row['artifact']['path']=='geometry/seed%d_group%d.pt'%key and exports[key]==dict(seed=key[0],group=key[1],**row['artifact']),'strict_fresh_artifact_paths')
            a.file(stage/row['artifact']['path'],row['artifact']['sha256'])
        write(out/'authorized_input_manifest.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())])
        a.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit_CPU_visibility')
        import dgl
        import numpy as np
        import scipy.sparse as sp
        import torch
        a.check(sys.version_info[:3]==(3,8,20) and torch.__version__=='1.13.1+cu117' and np.__version__=='1.24.4' and dgl.__version__.startswith('1.1.2'),'locked_runtime_and_Python_random_replay')
        a.check(not torch.cuda.is_available() and torch.cuda.device_count()==torch._C._cuda_getDeviceCount()==0,'zero_visible_Torch_native_CUDA')
        torch.set_num_threads(8);refpaths={key:a.file(ref['path'],ref['sha256']) for key,ref in p['reference_data'].items()}
        a.check(set(refpaths)=={'graph','valid_group_ids','target_assignment'},'only_raw_graph_and_ID_membership_metadata')
        graphs,_=dgl.load_graphs(str(refpaths['graph']));counts['original_DGL_graph_loads']+=1;graph=graphs[0].to('cpu');del graphs
        if 'y' in graph.ndata:discarded=graph.ndata.pop('y');del discarded
        raw=graph.ndata['x'];original,width=raw.shape;a.check(raw.dtype==torch.float32 and width==768 and bool(torch.isfinite(raw).all()),'unlabelled_original_raw_feature_matrix')
        original_feature_hash=tensor_hash(raw);minimum=raw.min(0).values;maximum=raw.max(0).values;del graph,raw
        a.check(original_feature_hash==marker['original_content_hashes']['features'],'independent_original_feature_hash')
        valid=np.load(refpaths['valid_group_ids'],allow_pickle=False);counts['VALID_group_ID_array_loads']+=1
        a.check(valid.ndim==1 and valid.dtype.kind in 'iuf' and bool(np.isfinite(valid).all()),'original_VALID_group_ID_vector')
        ints=valid.astype(np.int64);population=set(map(int,ints))
        a.check(np.array_equal(valid.astype(np.float64),ints.astype(np.float64)) and len(population)==len(ints) and bool((ints>=0).all()),'exact_unique_nonnegative_group_IDs')
        excluded=p['excluded_valid_groups'];a.check(set(excluded)<=population and len(population)==592,'both_old128_lists_are_original_VALID_contexts')
        eligible=sorted(population-set(excluded));selected=random.Random(20261106).sample(eligible,128)
        selection=p['selection'];a.check(selection['algorithm']=='python_random_sample_sorted_eligible_v1' and selection['rng_seed']==20261106
            and selection['draw_count']==128 and selection['original_valid_group_count']==len(population) and selection['eligible_count']==len(eligible)
            and len(eligible)==336 and selection['excluded_group_count']==256 and selected==p['valid_groups']
            and selection['ordered_groups_sha256']==compact_hash(selected),'independent_third_ID_only_cohort_sampling_replay')
        assignment=sp.load_npz(refpaths['target_assignment']).tocsr();counts['target_assignment_loads']+=1
        a.check(assignment.shape[1]==original and max(population)<assignment.shape[0] and bool(np.isfinite(assignment.data).all()),'original_group_node_assignment_shape')
        def members(group):return list(map(int,np.flatnonzero(assignment.getrow(group).toarray().reshape(-1))))
        old_union={v for g in excluded for v in members(g)};new_members={g:members(g) for g in selected};new_union={v for ids in new_members.values() for v in ids}
        n_occ=sum(map(len,new_members.values()))
        cm=p['cohort_metadata'];a.check(cm['target_occurrences_per_seed']==n_occ and cm['unique_targets']==len(new_union)
            and cm['original_node_count']==original and cm['overlap_unique_targets_with_either_previous_context_list']==len(new_union&old_union)
            and cm['new_unique_targets_relative_to_previous_context_lists']==len(new_union-old_union) and cm['group_target_counts']=={str(g):len(v) for g,v in new_members.items()}
            and cm['target_ids_sha256']==compact_hash(sorted(new_union)) and cm['excluded_target_ids_sha256']==compact_hash(sorted(old_union)),
            'independent_new_complete_membership_and_overlap_metadata')
        a.check(all(new_members.values()),'no_empty_selected_original_target_lists')
        nonidentity={str(s):0 for s in SEEDS};geometry_totals=[];base_ids={};role_counts=collections.Counter();donor_fractions=[]
        for seed in SEEDS:
            node_sum=edge_sum=occurrences=0
            for ordinal,group in enumerate(selected):
                row=records[seed,group];path=a.file(stage/row['artifact']['path'],row['artifact']['sha256']);payload=torch.load(path,map_location='cpu');counts['fresh_artifact_tensor_loads']+=1
                a.check(set(payload)=={'capture','adjacencies','canonical_sequences'} and set(payload['adjacencies'])==set(payload['canonical_sequences'])==set(ARMS),'fresh_artifact_tensor_schema')
                cap=payload['capture'];a.check(set(cap)=={'x_atk','adj_attack','target_ids','candidate_ids'},'fresh_capture_schema')
                hashes={k:tensor_hash(v) for k,v in cap.items()};a.check(hashes==row['immutable_capture_content_hashes'],'all_fresh_capture_dtype_shape_bits_bound')
                x,adj=cap['x_atk'],cap['adj_attack'];targets=cap['target_ids'].tolist();candidates=cap['candidate_ids'].tolist();n=len(x);t=len(targets);c=len(candidates)
                a.check(targets==new_members[group]==row['node_ids'] and t==row['targets'] and len(row['labels'])==t and all(y==1 for y in row['labels']),'unchanged_original_new_context_membership')
                a.check(x.dtype==adj.dtype==torch.float32 and tuple(x.shape)==(n,768) and n>0 and tuple(adj.shape)==(n,t+c+n)
                    and cap['target_ids'].dtype==cap['candidate_ids'].dtype==torch.int64 and bool(torch.isfinite(x).all()) and bool(torch.isfinite(adj).all())
                    and bool(((adj==0)|(adj==1)).all()),'fresh_finite_binary_capture_shapes')
                legal=torch.tril(torch.ones_like(adj,dtype=torch.bool),diagonal=t+c-1)
                a.check(len(set(targets+candidates))==t+c and all(0<=v<original for v in targets+candidates) and not bool((adj[~legal]!=0).any())
                    and bool((adj[:,:t].sum(1)>=1).all()) and n==row['node_budget']==row['actual_injected_nodes']
                    and int(adj.sum())==row['edge_budget']==row['actual_injected_nonself_undirected_edges'],'actual_legal_fresh_hard_graph_budgets')
                signature=(targets,row['labels'],row['node_budget'],row['edge_budget'])
                a.check(base_ids.setdefault(group,signature)==signature,'fixed_new_membership_order_and_budgets_across_seeds')
                key='%d|validation|8|%d|dual-surrogate-v1'%(seed,group);rng=int(hashlib.sha256(key.encode('ascii')).hexdigest()[:8],16)
                a.check(row['generation_seed']==rng,'unchanged_per_case_generation_RNG_namespace')
                start=a.json(stage/'generation_starts'/('seed%d_group%d.json'%(seed,group)))
                a.check(start['seed']==seed and start['group']==group and start['node_ids']==targets and start['labels']==row['labels']
                    and start['generation_seed']==rng and start['node_budget']==n and start['edge_budget']==int(adj.sum()) and start['stage']=='generation_dispatched','per_case_generation_start_record')
                before=start['counters_before'];global_ordinal=SEEDS.index(seed)*128+ordinal
                for name in ('generator_forward_dispatches','generator_completed_forwards','baseline_generations','geometry_cases_prepared'):
                    a.check(before[name]==global_ordinal,'per_case_generation_progress_before_dispatch')
                injected=list(range(original,original+n));matrices,expected=reallocate_independently(a,adj.to(torch.int64).tolist(),targets,candidates,injected,seed,group)
                report=row['topology_transform']
                for name,value in expected.items():a.check(report[name]==value,'independent_role_plan_keys_counts_priorities_and_geometry')
                a.check(report['namespace']=='candidate-target-reallocation-v1' and report['uses_full_clean_target_degree'] is False
                    and report['labels_read'] is False and report['model_outputs_used'] is False and report['features_read_or_changed'] is False
                    and report['all_old_target_edges_preserved'] is True and report['injected_block_preserved'] is True
                    and report['row_and_total_edge_counts_preserved'] is True and report['physical_injected_degrees_preserved'] is True
                    and report['maximum_common_feasible_k'] is True and report['tunable_fraction'] is False,'label_blind_fixed_maximal_common_feasibility_contract')
                a.check(bool(((x.double()>=minimum.double()-1e-6)&(x.double()<=maximum.double()+1e-6)).all()),'unchanged_generated_x0_original_raw_box')
                for arm in ARMS:
                    actual=payload['adjacencies'][arm];sequence=payload['canonical_sequences'][arm];meta=row['variants'][arm]
                    a.check(actual.dtype==torch.float32 and tuple(actual.shape)==tuple(adj.shape) and bool(torch.isfinite(actual).all())
                        and bool(((actual==0)|(actual==1)).all()) and actual.to(torch.int64).tolist()==matrices[arm]
                        and tensor_hash(actual)==meta['adjacency_content_sha256'],'actual_matrix_equals_independent_shared_donor_target_and_control_replay')
                    if arm=='baseline':a.check(tensor_hash(actual)==hashes['adj_attack'],'baseline_adjacency_capture_bits_preserved')
                    a.check(not bool((actual[~legal]!=0).any()) and tensor_hash(actual[:,t+c:])==tensor_hash(adj[:,t+c:])
                        and torch.equal(actual.sum(1),adj.sum(1)) and int(actual.sum())==row['edge_budget']
                        and not bool(((adj[:,:t]==1)&(actual[:,:t]!=1)).any()),'old_targets_injected_block_and_every_row_cost_preserved')
                    for i,plan in enumerate(expected['row_plan']):
                        k=0 if arm=='baseline' else plan['k']
                        removed=[candidates[j] for j in range(c) if adj[i,t+j] and not actual[i,t+j]]
                        a.check(set(removed)==set([] if arm=='baseline' else plan['removed_candidate_ids'])
                            and len(removed)==k and int((actual[i]!=adj[i]).sum())==2*k,'exact_common_candidate_donors_and_row_Hamming_match')
                    degree=[sum(v)+sum(matrices[arm][j][t+c+i] for j in range(i+1,n)) for i,v in enumerate(matrices[arm])]
                    a.check(degree==expected['physical_injected_degrees']==meta['physical_injected_degrees']
                        and meta['actual_injected_nodes']==n and meta['actual_injected_nonself_undirected_edges']==int(adj.sum())
                        and meta['feature_content_sha256']==hashes['x_atk'] and meta['identity']==expected['identity'][arm],
                        'all_feature_bits_physical_injected_degrees_and_actual_costs_fixed')
                    if arm=='balanced_target_fill':nonidentity[str(seed)]+=int(not expected['identity'][arm]['exact_equals_baseline'])
                    a.check(sequence.dtype==torch.float32 and tuple(sequence.shape)==(t,7,768) and bool(torch.isfinite(sequence).all())
                        and tensor_hash(sequence)==meta['canonical_sequence_content_sha256'] and meta['self_token0_exact'] is True
                        and tensor_hash(sequence[:,0])==tensor_hash(payload['canonical_sequences']['baseline'][:,0]),'actual_SELF_token0_bits_no_other_token_invariance_assumed')
                    a.check(meta['changed_coordinates_by_token']==[int((sequence[:,i]!=payload['canonical_sequences']['baseline'][:,i]).sum()) for i in range(7)],'all7_actual_token_change_counts')
                role_counts['baseline_candidate_incidence']+=expected['candidate_incidence_before'];role_counts['reallocated_edges']+=expected['total_reallocated_edges']
                role_counts['baseline_target_incidence']+=expected['total_target_incidence_before'];role_counts['primary_target_incidence']+=expected['total_target_incidence_after']['balanced_target_fill']
                role_counts['primary_candidate_incidence_after']+=expected['candidate_incidence_after']['balanced_target_fill']
                role_counts['candidate_control_incidence_after']+=expected['candidate_incidence_after']['candidate_rewire']
                role_counts['all_injected_rows']+=n;role_counts['primary_equals_random_cases']+=int(expected['primary_equals_random'])
                for plan in expected['row_plan']:
                    if plan['baseline_candidate_count']:
                        donor_fractions.append(plan['candidate_donor_fraction']);role_counts['rows_with_candidate_edges']+=1
                        role_counts['rows_losing_all_candidate_neighbors']+=int(plan['k']==plan['baseline_candidate_count'])
                    else:role_counts['rows_without_candidate_edges']+=1
                    role_counts['rows_with_positive_k']+=int(plan['k']>0)
                    role_counts['positive_k_rows_filling_all_missing_targets']+=int(plan['k']>0 and plan['k']==plan['missing_target_count'])
                node_sum+=n;edge_sum+=int(adj.sum());occurrences+=t
                del payload,cap,x,adj,actual,sequence,expected,matrices
            a.check(occurrences==n_occ,'measured_new_occurrence_denominator_all_seeds')
            geometry_totals.extend(dict(seed=seed,arm=arm,groups=128,target_occurrences=occurrences,unique_targets=len(new_union),summed_injected_nodes=node_sum,summed_nonself_undirected_edges=edge_sum) for arm in ARMS)
        a.check(geometry_totals==marker['whole_seed_geometry_totals']==r['whole_seed_geometry_totals'],'dynamic_all12_resource_denominators_no_old1091_or7776_assumption')
        a.check(marker['counters_before_evaluation']=={k:GEOMETRY_COUNTS.get(k,0) for k in FULL_COUNTS},'GEN_and_proposal_GCN_loaded_but_zero_proposal_GCN_forwards_before_seal')
        a.check(r['all_geometry_prepared_before_dedicated_evaluators'] is True and r['all_geometry_prepared_before_official_victim'] is True,'dedicated_model_and_official_victim_gate_scope')
        for seed in SEEDS:
            registered=a.json(stage/'engine_receipts'/('seed%d_registered.json'%seed));completed=a.json(stage/'engine_receipts'/('seed%d_completed.json'%seed),marker['engine_receipt_hashes']['engine_receipts/seed%d_completed.json'%seed])
            a.check(all(completed[k]==v for k,v in registered.items()) and completed==r['generator_receipts'][SEEDS.index(seed)],'engine_registered_completed_result_receipt_binding')
            a.check(completed['generator_checkpoint_sha256']==old['baseline_runs'][str(seed)]['final_model_sha256']
                and completed['generator_state_before']==completed['generator_state_after']==baselines[seed]['final_model_state_sha256']
                and completed['proposal_gcn_state_before']==completed['proposal_gcn_state_after'] and completed['generator_completed_forwards']==128
                and completed['optimizer_steps']==0 and all(v==0 for v in completed['classifier_forwards'].values()),'frozen_checkpoint_states_and128completed_generator_calls_no_classifier')
            schedule=completed['scheduler_getters_before'];a.check(completed['scheduler_restoration_ticks_reported_by_helper']==8
                and completed['scheduler_getters_after']==schedule and schedule['phase']=='validation' and schedule['current_epoch']==8 and schedule['model_training'] is False
                and math.isfinite(schedule['temperature']) and math.isfinite(schedule['epsilon']),'actual_scheduler_getters_unchanged_eight_ticks_receipt_not_replay')
            a.check(completed['baseline_engine_protocol_override']=={'valid_groups':selected} and completed['data_hashes_before']==completed['data_hashes_after']==marker['original_content_hashes'],'only_new_context_override_and_original_data_receipts')
        headroom=a.json(stage/'headroom.json');stop=any(v==0 for v in nonidentity.values())
        a.check(headroom==r['headroom'] and headroom['provenance']==provenance and headroom['primary']=='balanced_target_fill'
            and headroom['nonidentity_cases_by_seed']==nonidentity and headroom['stop_before_evaluation']==stop
            and headroom['original_generator_calls_already_spent']==384 and headroom['proposal_gcn_models_already_loaded']==3
            and headroom['sealed_cases']==384 and headroom['groups_per_seed']==128 and headroom['uses_model_predictions'] is False,'headroom_gate_preserves_GEN_cost_no_zero_model_claim')
        a.check(r['immutable_content_hashes_retained'] is True and r['immutable_content_hashes']['before']==r['immutable_content_hashes']['after']
            and all(r['immutable_content_hashes']['before'][k]==v for k,v in marker['original_content_hashes'].items()),'original_features_cache_and_classifier_immutability_receipts')
        a.check(r['GPU_hours']==0 and r['cuda_visibility']=='' and r['torch_cuda_available'] is False and r['torch_visible_gpu_count']==r['native_visible_gpu_count']==0
            and r['final_TEST_used'] is False and r['experimental_API_calls']==0 and r['process_cpu_seconds']>=0 and 0<r['wall_seconds']<=5415,'actual_CPU_only_resources_and_closed_TEST_API')
        metrics=[];contrasts={};criterion=None;expected_outputs={'registered.json','geometry.jsonl','geometry_sealed.json','headroom.json','input_hashes.json'}
        expected_outputs.update('engine_receipts/seed%d_%s.json'%(s,state) for s in SEEDS for state in ('registered','completed'))
        if stop:
            a.check(r['status']=='completed_no_headroom' and r['classifier_loading_after_geometry_seal'] is False
                and r['counters']=={k:GEOMETRY_COUNTS.get(k,0) for k in FULL_COUNTS},'planned_zero_headroom_no_dedicated_evaluation')
            a.check(not (stage/'clean_predictions.jsonl').exists() and all(not (stage/('%s_seed%d_predictions.jsonl'%(arm,s))).exists() for arm in ARMS for s in SEEDS),'no_prediction_files_if_headroom_stops')
        else:
            a.check(r['status']=='completed' and r['classifier_loading_after_geometry_seal'] is True and r['counters']==FULL_COUNTS,'exact34_full_counters_and1664fresh_classifier_calls_each')
            clean=keyed(a,a.rows(stage/'clean_predictions.jsonl'),selected,'all128_fresh_clean_predictions');clean_pred={}
            for g,row in clean.items():
                a.check(row['node_ids']==new_members[g] and row['labels']==records[3,g]['labels'],'new_clean_complete_ID_truth_binding')
                for detector in ('gcn','gaga'):clean_pred[g,detector]=classify(a,row[detector+'_logits'],len(row['labels']))
            outputs={};summaries={(v['seed'],v['arm']):v for v in r['summaries']};a.check(len(summaries)==len(r['summaries'])==12,'all12_new_seed_arm_summaries')
            for seed in SEEDS:
                for arm in ARMS:
                    rows=keyed(a,a.rows(stage/('%s_seed%d_predictions.jsonl'%(arm,seed))),selected,'all1536_new_attacked_prediction_rows')
                    totals={d:collections.Counter() for d in ('gcn','gaga')};macro={d:[] for d in ('gcn','gaga')}
                    for group,row in rows.items():
                        geometry=records[seed,group];meta=geometry['variants'][arm];labels=geometry['labels'];n=len(labels);hashes=geometry['immutable_capture_content_hashes']
                        a.check(row['seed']==seed and row['arm']==arm and row['node_ids']==new_members[group] and row['labels']==labels
                            and row['node_budget']==row['actual_injected_nodes']==geometry['node_budget'] and row['edge_budget']==row['actual_injected_edges']==geometry['edge_budget']
                            and row['prepared_artifact_sha256']==geometry['artifact']['sha256'] and row['original_adjacency_content_sha256']==hashes['adj_attack']
                            and row['variant_adjacency_content_sha256']==meta['adjacency_content_sha256']
                            and row['target_order_content_sha256']==hashes['target_ids'] and row['candidate_order_content_sha256']==hashes['candidate_ids']
                            and row['injected_feature_content_sha256']==meta['feature_content_sha256'] and row['canonical_sequence_content_sha256']==meta['canonical_sequence_content_sha256'],'fresh_prediction_geometry_feature_and_dynamic_cost_binding')
                        for detector in ('gcn','gaga'):
                            count=transitions(labels,clean_pred[group,detector],classify(a,row[detector+'_logits'],n));a.check(row[detector+'_transitions']==count,'new_raw_argmax_transition_recount')
                            totals[detector].update(count);macro[detector].append(100*count['attacked_wrong']/n)
                    summary=summaries[seed,arm];a.check(summary['groups']==128 and summary['unique_targets']==len(new_union),'new_summary_group_unique_counts')
                    for detector in ('gcn','gaga'):
                        total={k:totals[detector][k] for k in FIELDS};mr=100*total['attacked_wrong']/n_occ
                        a.check(total['occurrences']==n_occ and summary['totals'][detector]==total,'dynamic_allcase_summary_counts');a.close(summary[detector+'_mr_percent'],mr,'dynamic_raw_MR_recount')
                        metrics.append(dict(seed=seed,arm=arm,detector=detector,**total,mr_percent=mr,group_macro_mr_percent=statistics.fmean(macro[detector])))
                    outputs[seed,arm]=rows
            lookup={(v['seed'],v['arm'],v['detector']):v for v in metrics};t95=math.sqrt(2)*.95/math.sqrt(1-.95**2)
            for arm,control in [(arm,'baseline') for arm in ARMS[1:]]+[('balanced_target_fill','random_target_fill'),('balanced_target_fill','candidate_rewire')]:
                for detector in ('gcn','gaga'):
                    differences=[lookup[s,arm,detector]['mr_percent']-lookup[s,control,detector]['mr_percent'] for s in SEEDS]
                    mean=statistics.fmean(differences);sd=statistics.stdev(differences);half=t95*sd/math.sqrt(3);paired=collections.Counter()
                    for s in SEEDS:
                        for g in selected:
                            before=classify(a,outputs[s,control][g][detector+'_logits'],len(new_members[g]));after=classify(a,outputs[s,arm][g][detector+'_logits'],len(new_members[g]));labels=records[s,g]['labels']
                            delta=sum(x!=y for x,y in zip(after,labels))-sum(x!=y for x,y in zip(before,labels));paired['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']+=1
                            for y,c,x,z in zip(labels,clean_pred[g,detector],before,after):
                                paired['occurrences']+=1;paired['gained_wrong']+=x==y and z!=y;paired['lost_wrong']+=x!=y and z==y
                                if c==y:paired['new_success_gained']+=x==y and z!=y;paired['baseline_new_success_lost']+=x!=y and z==y
                                else:paired['additional_recovery']+=x!=y and z==y;paired['previous_recovery_lost']+=x==y and z!=y
                    a.check(paired['occurrences']==3*n_occ and sum(paired[k] for k in ('group_wins','group_ties','group_losses'))==384,'dynamic_correlated_pair_counts_not_old3273')
                    contrasts[arm+'-minus-'+control+'_'+detector]=dict(seed_differences_pp=differences,mean_difference_pp=mean,sample_sd_pp=sd,paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts=dict(paired))
            primary=contrasts['balanced_target_fill-minus-baseline_gaga'];above=contrasts['balanced_target_fill-minus-candidate_rewire_gaga']['mean_difference_pp']
            criterion=dict(primary='balanced_target_fill',mean_delta_pp=primary['mean_difference_pp'],all_three_seed_deltas_positive=all(v>0 for v in primary['seed_differences_pp']),paired_t95_lower_pp=primary['paired_t95_pp'][0],mean_advantage_over_candidate_rewire_pp=above,
                criterion_met=primary['mean_difference_pp']>=1 and all(v>0 for v in primary['seed_differences_pp']) and primary['paired_t95_pp'][0]>0 and above>0,
                random_target_comparator_need_not_be_worse=True,target_priority_superiority_required=False,controls_not_candidates=True,final_TEST_allowed=False)
            expected_outputs.add('clean_predictions.jsonl');expected_outputs.update('%s_seed%d_predictions.jsonl'%(arm,s) for arm in ARMS for s in SEEDS)
        a.check(set(r['output_hashes'])==expected_outputs,'all_geometry_engine_headroom_and_evaluation_outputs_hash_bound')
        a.check(counts['original_DGL_graph_loads']==counts['VALID_group_ID_array_loads']==counts['target_assignment_loads']==1 and counts['fresh_artifact_tensor_loads']==384
            and all(counts[k]==0 for k in ('label_values_indexed','old_capture_tensor_loads','old_prediction_streams_decoded','classifier_weight_loads','model_constructions','model_forwards','autograd_calls','optimizer_updates','API_calls','TEST_prediction_reads','GPU_hours')),'only_allowed_ID_rawbox_and384fresh_tensor_reads')
        donor_fraction_summary=dict(rows=len(donor_fractions),pooled_edge_fraction=role_counts['reallocated_edges']/role_counts['baseline_candidate_incidence'] if role_counts['baseline_candidate_incidence'] else None,
            minimum=min(donor_fractions) if donor_fractions else None,
            mean=statistics.fmean(donor_fractions) if donor_fractions else None,maximum=max(donor_fractions) if donor_fractions else None,
            scope='Per injected row with at least one original candidate edge, including k=0; pooled across3seeds and128separate scenarios each.')
        result.update(status='passed',producer_status=r['status'],provenance=provenance,cohort_metadata=cm,cohort_ID_replay=True,geometry_totals=geometry_totals,
            nonidentity_cases_by_seed=nonidentity,role_allocation_counts=dict(role_counts),candidate_donor_fraction_summary=donor_fraction_summary,
            metrics=metrics,paired_contrasts=contrasts,primary_development_criterion=criterion,
            producer_counters=r['counters'],producer_cpu_seconds=r['process_cpu_seconds'],producer_wall_seconds=r['wall_seconds'],immutable_content_hash_receipts=r['immutable_content_hashes'],
            limitations=['New ID-sampled contexts exclude exactly both earlier128 lists/256contexts, not unseen independent news or an untouched validation universe; original target membership lists and overlapping target identities are retained.',
                'Original DGL y is discarded unused by this audit; no original labels select reallocations. The producer DataHandler still materializes whole-graph label/class metadata and saved VALID truth is used only for MR. Target membership carries the existing known-fraud-target assumption.',
                'Exactly384freshartifact tensors are inspected, with no old capture/old prediction decoding or classifier weight/model replay. Raw original x supplies only the unlabelled box.',
                'All4integer adjacency variants, common candidate donors, SHA-keyed selections, once-computed baseline injected-attachment priorities, per-row costs and physical injected degrees are independently recomputed. Original clean target degrees are not used. Some rows may lose all original candidate neighbors; this risk is measured without dropping cases.',
                'Three frozen proposalGCNs and three generators load before geometry;384GENcalls are spent. Hooks/receipts report zero proposalGCNforward attempts. Dedicated evaluators and any official victim load after the seal; no zero-model claim.',
                'Generator parameter/state/scheduler receipts and reported8restorationticks are reconciled, not independently replayed. Native DGL historical trajectory identity, original-original arc preservation, full canonical construction and seal temporal ordering retain producer/source assertions.',
                'Only SELF token0 is required bit-identical; actual changes across all7canonical tokens are recounted from saved sequences. Whole canonical model inputs are not independently reconstructed using a neural model or DataHandler.',
                'Primary tests candidate-to-target role allocation, not target-priority superiority. Random target-fill can tie or outperform it; the prespecified control comparison is candidate-only rewiring. No control is promoted into a primary, and no allocation optimum, camouflage, novelty or course completion is established.',
                'Nominal n3 paired t95 is conditional on fixed checkpoints and correlated/adaptively inspected development data; no population/teacher/confirmatory claim. FullTEST/tag gates remain separate.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(audit_operation_counters=counts,check_counts=dict(a.counts),byte_hashed_files=len(a.files),process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'input_hashes.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())]);write(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],output=str(out),checks=sum(a.counts.values()),CPU_seconds=result['process_cpu_seconds'],wall_seconds=result['wall_seconds'])),flush=True)
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
