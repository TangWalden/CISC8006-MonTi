"""Independent CPU reference/tensor/saved-prediction audit for fixed prototypes.

Allowed: one original DGL graph and three ID arrays; ONLY TRAIN truth indexing;
reference.pt plus384BASEand384NEWtensor artifacts; saved VALID prediction JSON.
No classifier weights, producer imports, model construction/inference, gradients,
updates, API/network or CUDA tensors. All source/scope freeze precedes reads.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import statistics
import struct
import subprocess
import sys
import time
import traceback

ARMS=('baseline','benign_mix','all_train_mix','fraud_mix')
SEEDS=(3,4,5)
FIXED_TOKENS=(0,1,2,4,5)
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
REFERENCE_NAMES=('minimum','maximum','benign_mean','all_train_mean','fraud_mean','train_ids','train_labels','benign_ids','fraud_ids')
NO_MODEL=('fresh_teacher_model_constructions','fresh_teacher_forwards','generator_model_constructions','generator_forwards','optimizer_steps','autograd_calls','API_calls','TEST_queries')
GEOMETRY_COUNTS=dict(reference_selections=1,reference_tensor_saves=1,source_capture_tensor_loads=384,source_captures_prepared=384,geometry_base_graph_reconstructions=384,geometry_feature_graph_variants=1536,canonical_sequence_updates=1536,fixed_token_invariant_checks=1536)
MODEL_COUNTS=dict(prepared_artifact_tensor_loads=384,evaluation_base_graph_reconstructions=384,evaluation_feature_graph_variants=1536,original_gcn_load_dispatches=1,original_gcn_models_loaded=1,official_gaga_load_dispatches=1,official_gaga_models_loaded=1,original_gcn_forward_dispatches=1664,original_gcn_completed_forwards=1664,official_gaga_forward_dispatches=1664,official_gaga_completed_forwards=1664,original_gcn_clean_forwards=128,official_gaga_clean_forwards=128,original_gcn_attacked_forwards=1536,official_gaga_attacked_forwards=1536)


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
        if not value:raise AssertionError('Prototype audit failed: '+name)
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


def float32_order(value):
    bits=struct.unpack('>I',struct.pack('>f',value))[0]
    return 0x80000000-(bits&0x7fffffff) if bits&0x80000000 else 0x80000000+bits


def independent_reference(a,p,reference,torch,np,dgl,counters):
    paths={key:a.file(ref['path'],ref['sha256']) for key,ref in p['reference_data'].items()}
    a.check(set(paths)=={'graph','train_indices','valid_indices','test_indices'},'exact_reference_dataset_and_partition_scope')
    graphs,_=dgl.load_graphs(str(paths['graph']));counters['original_DGL_graph_loads']+=1
    a.check(len(graphs)==1,'single_original_graph');graph=graphs[0].to('cpu');del graphs
    raw=graph.ndata['x'];n,width=raw.shape
    a.check(raw.dtype==torch.float32 and width==768 and bool(torch.isfinite(raw).all()),'finite_original_raw_feature_matrix')
    partitions={}
    for key in ('train_indices','valid_indices','test_indices'):
        value=np.load(paths[key],allow_pickle=False).reshape(-1);counters['partition_ID_array_loads']+=1
        a.check(value.dtype.kind in 'iuf' and bool(np.isfinite(value).all()),'finite_numeric_partition_IDs')
        integer=value.astype(np.int64)
        a.check(np.array_equal(value.astype(np.float64),integer.astype(np.float64)) and len(integer)==len(set(integer.tolist()))
            and bool(((integer>=0)&(integer<n)).all()),'exact_unique_original_partition_IDs')
        partitions[key]=torch.tensor(integer,dtype=torch.long)
    joined=torch.cat(list(partitions.values()))
    a.check(joined.numel()==torch.unique(joined).numel(),'original_train_valid_test_ID_partitions_disjoint')
    train=partitions['train_indices'];a.check(len(train)==6595,'full_original_unique_TRAIN_count')
    # DGL materializes the serialized global label field. Index ONLY TRAIN and
    # immediately discard the global tensor; no held-out label operation.
    all_truth=graph.ndata.pop('y');truth=all_truth.index_select(0,train).clone();del all_truth
    counters['TRAIN_label_entries_indexed']=len(train)
    a.check(not truth.is_complex() and bool(torch.isfinite(truth).all()) and bool(((truth==0)|(truth==1)).all()),'only_TRAIN_truth_binary_values')
    truth=truth.to(torch.long);benign=train[truth==0];fraud=train[truth==1]
    a.check(len(benign)>0 and len(fraud)>0,'both_TRAIN_class_reference_pools')
    for key,value in (('train_ids',train),('train_labels',truth),('benign_ids',benign),('fraud_ids',fraud)):
        a.check(reference[key].dtype==torch.int64 and tensor_hash(reference[key])==tensor_hash(value),'independent_TRAIN_pool_ID_and_label_truth')
    minimum=raw.min(0).values;maximum=raw.max(0).values;movable=maximum>minimum
    for key,value in (('minimum',minimum),('maximum',maximum)):
        a.check(reference[key].dtype==torch.float32 and tensor_hash(reference[key])==tensor_hash(value),'full_unlabelled_original_feature_box')
    counts=dict(original_nodes=n,feature_dimension=width,train_nodes=len(train),benign_train_nodes=len(benign),fraud_train_nodes=len(fraud),
        valid_nodes=len(partitions['valid_indices']),test_nodes=len(partitions['test_indices']),label_values_read=len(train))
    a.check(reference['counts']==counts and reference['row_l2_normalization'] is False
        and reference['continuous_l0_constraint'] is False,'reference_counts_and_no_extra_feature_constraint')
    mean_checks={}
    for name,ids in (('benign_mean',benign),('all_train_mean',train),('fraud_mean',fraud)):
        matrix=raw.index_select(0,ids).numpy()
        # Compensated Python fsum is independent of the producer's Torch float64
        # parallel reduction. Do not claim identical float64 summation order.
        exact64=[math.fsum(float(v) for v in matrix[:,j])/len(ids) for j in range(width)]
        expected=torch.tensor(exact64,dtype=torch.float64).to(torch.float32)
        expected=torch.maximum(minimum,torch.minimum(maximum,expected));expected=torch.where(movable,expected,minimum)
        actual=reference[name]
        a.check(actual.dtype==torch.float32 and tuple(actual.shape)==(width,) and bool(torch.isfinite(actual).all())
            and bool(((actual>=minimum)&(actual<=maximum)).all()),'finite_bounded_reference_mean_vectors')
        actual_values=actual.tolist();expected_values=expected.tolist()
        ulps=[abs(float32_order(x)-float32_order(y)) for x,y in zip(actual_values,expected_values)]
        max_ulp=max(ulps)
        a.check(max_ulp<=1,'independent_fsum_TRAIN_mean_within_one_float32_ULP')
        a.check(tensor_hash(actual[~movable])==tensor_hash(minimum[~movable]),'reference_zero_width_coordinate_bits')
        exact=tensor_hash(actual)==tensor_hash(expected)
        mean_checks[name]=dict(pool_nodes=len(ids),max_float32_ULP_difference=max_ulp,
            max_absolute_difference=max(abs(x-y) for x,y in zip(actual_values,expected_values)),
            exact_float32_tensor_bytes_match=exact,coordinates_with_nonzero_ULP_difference=sum(v!=0 for v in ulps),
            coordinates_with_different_float32_bits=sum(struct.pack('>f',x)!=struct.pack('>f',y) for x,y in zip(actual_values,expected_values)),
            acceptance='At most1float32ULP to compensated fsum/N rounded/clamped independently; zero-width coordinates exact. No float64 reduction-order identity claim.')
        del matrix
    report=dict(counts=counts,independent_mean_checks=mean_checks,full_partition_IDs_cover_original_graph=joined.numel()==n,
        prototype_pool_labels='Original TRAIN only; held-out ID arrays for partition checks, no held-out truth indexing',
        DGL_materialization_caveat='Serialized graph initially materializes global y; audit indexes only TRAIN then discards global truth.')
    del graph,raw,truth,train,benign,fraud,partitions,joined
    return minimum,maximum,n,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--execution-source',required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--audit-source-sha256',required=True);args=parser.parse_args()
    root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
    out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh independent audit output must be below ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);wall,cpu=time.monotonic(),time.process_time();a=Audit(root)
    counters=dict(original_DGL_graph_loads=0,partition_ID_array_loads=0,TRAIN_label_entries_indexed=0,
        nonTRAIN_label_entries_indexed=0,reference_tensor_loads=0,base_capture_tensor_loads=0,new_artifact_tensor_loads=0,
        classifier_weight_deserializations=0,model_constructions=0,model_forwards=0,autograd_calls=0,optimizer_updates=0,API_calls=0,TEST_prediction_reads=0,GPU_hours=0)
    result=dict(status='failed',schema='independent_train_prototype_mix_audit_v1',post_hoc=True,
        independent_of_method_implementation_authorship=True,human_verification=False)
    try:
        own=sha(__file__);a.check(own==args.audit_source_sha256,'externally_frozen_auditor')
        write(out/'registered_scope.json',dict(audit_source_sha256=own,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            protocol_sha256=args.protocol_sha256,execution_source=args.execution_source,new_input_or_outcome_reads_started=False,
            allowed_reads='One original DGL graph;3original partition-ID arrays; TRAIN truth indexing only; reference.pt+384BASE+384NEWartifacts; saved VALID predictions.',
            forbidden='Classifier weight deserialization/producer imports/model construction or inference/heldout truth indexing/gradients/updates/API/TEST predictions/CUDA.',
            independent_reference_tolerance='Compensated math.fsum/N; stored rounded/clamped float32 mean within1ULP, zero-width exact; record exact bit agreement separately.'))
        p=a.json(args.protocol,args.protocol_sha256);stage=a.path(args.stage,directory=True)
        a.check(p['schema']=='monti_train_prototype_mix_v1' and p['status']=='frozen' and p['post_hoc'] is True
            and p['owner_execution_authorized'] is True and p['source_commit']==args.execution_source,'frozen_owner_authorized_posthoc_source')
        a.check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['alpha']==.5 and p['final_TEST_allowed'] is False
            and len(p['valid_groups'])==len(set(p['valid_groups']))==128 and set(p['runs'])=={'3','4','5'}
            and p['execution_device']=='cpu' and p['cuda_visible_devices']=='' and p['cpu_threads']==8 and p['deadline_seconds']==5400,
            'fixed_arms_alpha_cohort_resources_no_test')
        qualification=p['qualification']
        a.check(qualification['primary']=='benign_mix' and qualification['min_mean_gain_pp']==1
            and all(qualification[k] is True for k in ('require_all_seed_deltas_positive','require_mean_above_all_train_mix',
                'require_mean_above_fraud_mix','require_paired_t95_lower_above_zero','controls_not_candidates')),'strong_frozen_primary_criterion_no_control_promotion')
        a.check(p['expected_if_evaluated']==dict(GEOMETRY_COUNTS,**MODEL_COUNTS,**{name:0 for name in NO_MODEL}),
            'exact31_registered_reference_geometry_and_classifier_counter_values')
        expected_inputs={}
        def retain(ref):
            path=a.file(ref['path'],ref['sha256']);old=expected_inputs.setdefault(ref['path'],ref['sha256'])
            a.check(old==ref['sha256'],'consistent_frozen_input_references');return path
        for name,wanted in p['source_files'].items():
            retain(dict(path=name,sha256=wanted));data=subprocess.check_output(['git','show',args.execution_source+':'+name],cwd=root)
            a.check(hashlib.sha256(data).hexdigest()==wanted,'registered_source_committed_bytes')
        a.check({'course/train_prototype_mix/'+name for name in ('__init__.py','prepare.py','core.py','run.py','synthetic_tests.py')}
            <=set(p['source_files']),'complete_new_package_source_freeze')
        for ref in p['inputs']+p.get('retained_inputs',[]):retain(ref)
        a.check(p['base_experiment_ref']['sha256']==BASE_SHA,'exact_completed_capture_experiment')
        base=a.json(retain(p['base_experiment_ref']))
        a.check(base['source_commit']==BASE_SOURCE and base['execution_device']=='cpu' and base['valid_groups']==p['valid_groups']
            and base['final_test_allowed'] is False,'original_cpu_complete_cohort')
        for name,wanted in base['source_files'].items():a.check(p['source_files'].get(name)==wanted,'all_reused_scientific_sources_unchanged')
        for ref in base['inputs']+base['retained_inputs']:retain(ref)
        for ref in p['reference_data'].values():retain(ref)
        if 'parity_result_ref' in p:
            parity=a.json(retain(p['parity_result_ref']));a.check(parity['status']=='completed' and parity['parity_passed'] is True,'declared_prior_parity_rationale')
        snapshot=p['budget_snapshot'];ledger_bytes=subprocess.check_output(['git','show',args.execution_source+':evidence/research_cost_ledger.json'],cwd=root)
        a.check(hashlib.sha256(ledger_bytes).hexdigest()==snapshot['ledger_sha256_at_registered_source']
            and snapshot['new_phase_GPU_hours']==0 and snapshot['charged']+snapshot['reserve']<=snapshot['limit'],'source_bound_resource_snapshot')
        a.close(snapshot['remaining'],snapshot['limit']-snapshot['charged']-snapshot['reserve'],'resource_remaining_recount')
        runs={}
        for seed in SEEDS:
            spec=p['runs'][str(seed)];run=a.json(retain(spec['result_ref']));rows=keyed(a,a.rows(retain(spec['groups_ref'])),p['valid_groups'],'full_source128_group_cover')
            a.check(run['status']=='completed' and run['seed']==seed and run['groups']==128 and run['source_commit']==BASE_SOURCE
                and run['protocol_sha256']==BASE_SHA and run['groups_jsonl_sha256']==spec['groups_ref']['sha256'],'original_source_run_binding')
            exports={v['group']:v for v in run['exports']};captures={v['group']:v for v in spec['captures']};directory=a.path(spec['result_ref']['path']).parent
            a.check(len(exports)==len(run['exports'])==len(captures)==len(spec['captures'])==128 and set(exports)==set(captures)==set(rows),'source384_capture_cover')
            for group,ref in captures.items():
                path=retain(ref);a.check(ref['sha256']==exports[group]['sha256'] and rows[group]['graph_capture']==exports[group]
                    and path==(directory/exports[group]['path']).resolve(),'source_capture_exact_original_binding')
            runs[seed]=dict(rows=rows,captures=captures)
        receipt=a.json(stage/'registered.json');r=a.json(stage/'result.json');provenance=dict(source_commit=args.execution_source,protocol_sha256=args.protocol_sha256)
        a.check(r['status']=='completed' and r['provenance']==receipt['provenance']==provenance,'successful_producer_execution_provenance')
        a.check(receipt['own_source_sha256']==p['source_files']['course/train_prototype_mix/run.py'] and receipt['alpha']==.5
            and tuple(receipt['arms'])==ARMS and receipt['base_experiment_ref']==p['base_experiment_ref'],'registered_executed_source_and_arms')
        for name,wanted in r['output_hashes'].items():a.file(stage/name,wanted)
        a.check(a.json(stage/'input_hashes.json')==receipt['source_and_input_hashes']==expected_inputs,'all_producer_source_input_hashes_reconciled')
        refmeta=a.json(stage/'reference.json');refseal=a.json(stage/'reference_sealed.json')
        a.check(refmeta['status']=='reference_sealed_before_classifier' and refmeta['provenance']==provenance and refmeta['alpha']==.5
            and refmeta['no_model_outcome_reference_selection'] is True and refseal['status']=='sealed','reference_scope_and_seal')
        reference_path=a.file(stage/'reference.pt',refseal['reference_pt_sha256']);a.file(stage/'reference.json',refseal['reference_json_sha256'])
        a.check(refmeta['artifact']==dict(path='reference.pt',sha256=refseal['reference_pt_sha256'])
            and refmeta['content_hashes']==refseal['content_hashes']==r['reference_content_hashes']
            and r['reference_pt_sha256']==refseal['reference_pt_sha256'],'all_reference_tensor_and_metadata_bindings')
        marker=a.json(stage/'geometry_sealed.json',r['geometry_marker_sha256'])
        a.check(marker['status']=='all_feature_proposals_sealed_before_any_classifier' and marker['provenance']==provenance
            and marker['source_captures']==384 and marker['variants']==marker['canonical_sequences']==1536
            and marker['fixed_tokens']==list(FIXED_TOKENS),'complete_reference_feature_geometry_model_gate')
        a.file(stage/'reference_sealed.json',marker['reference_seal_sha256'])
        geometry_rows=a.rows(stage/'geometry.jsonl');records={(v['seed'],v['group']):v for v in geometry_rows}
        a.check(len(records)==len(geometry_rows)==r['prepared_cases']==384
            and set(records)=={(s,g) for s in SEEDS for g in p['valid_groups']},'exact_all384_prepared_cases')
        a.file(stage/'geometry.jsonl',marker['geometry_jsonl_sha256']);exports={(v['seed'],v['group']):v for v in marker['artifacts']}
        a.check(len(exports)==len(marker['artifacts'])==384 and set(exports)==set(records),'all384_new_artifact_refs')
        for key,row in records.items():
            a.check(row['artifact']['path']=='geometry/seed%d_group%d.pt'%key
                and exports[key]==dict(seed=key[0],group=key[1],**row['artifact']),'strict_new_geometry_artifact_paths')
            a.file(stage/row['artifact']['path'],row['artifact']['sha256'])
        write(out/'authorized_input_manifest.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())])
        a.check(os.environ.get('CUDA_VISIBLE_DEVICES')=='','explicit_CPU_audit_visibility')
        import dgl
        import numpy as np
        import torch
        a.check(torch.__version__=='1.13.1+cu117' and np.__version__=='1.24.4' and dgl.__version__.startswith('1.1.2'),'locked_library_runtime')
        a.check(not torch.cuda.is_available() and torch.cuda.device_count()==torch._C._cuda_getDeviceCount()==0,'zero_torch_native_CUDA_visibility')
        torch.set_num_threads(8);reference=torch.load(reference_path,map_location='cpu');counters['reference_tensor_loads']+=1
        for name in REFERENCE_NAMES:a.check(name in reference and torch.is_tensor(reference[name]) and reference[name].device.type=='cpu','reference_tensor_schema_CPU')
        a.check({name:tensor_hash(reference[name]) for name in REFERENCE_NAMES}==refseal['content_hashes'],'actual_reference_tensor_content_hashes')
        minimum,maximum,original,reference_report=independent_reference(a,p,reference,torch,np,dgl,counters)
        a.check(refmeta['counts']==r['reference_counts']==reference_report['counts'],'independent_reference_population_counts')
        a.check(refmeta['labels_read_scope']==reference['labels_read_scope'] and refmeta['reference_scope']==reference['reference_scope'],
            'prototype_selector_scope_receipts_consistent')
        movable=maximum>minimum;totals_geometry=[];identity_counts=collections.Counter()
        for seed in SEEDS:
            nodes=edges=occurrences=0;unique=set()
            for group in p['valid_groups']:
                row=records[seed,group];saved=runs[seed]['rows'][group];oldref=runs[seed]['captures'][group]
                source_path=a.file(oldref['path'],oldref['sha256']);artifact_path=a.file(stage/row['artifact']['path'],row['artifact']['sha256'])
                old=torch.load(source_path,map_location='cpu');counters['base_capture_tensor_loads']+=1
                new=torch.load(artifact_path,map_location='cpu');counters['new_artifact_tensor_loads']+=1
                a.check(set(old)=={'capture','features'} and set(new)=={'capture','features','canonical_sequences'}
                    and set(new['features'])==set(new['canonical_sequences'])==set(ARMS),'strict_base_and_new_artifact_schemas')
                cap=old['capture'];a.check(set(cap)==set(new['capture'])=={'x_atk','adj_attack','target_ids','candidate_ids'},'strict_capture_schema')
                hashes={name:tensor_hash(value) for name,value in cap.items()}
                a.check(hashes==row['immutable_capture_content_hashes'] and hashes=={name:tensor_hash(value) for name,value in new['capture'].items()},'all_original_capture_shape_dtype_bits_unchanged')
                x0=cap['x_atk'];adj=cap['adj_attack'];targets=cap['target_ids'].tolist();candidates=cap['candidate_ids'].tolist();n=len(x0);t=len(targets);c=len(candidates)
                a.check(n>0 and t>0 and x0.dtype==adj.dtype==torch.float32 and tuple(x0.shape)==(n,768)
                    and tuple(adj.shape)==(n,t+c+n) and bool(torch.isfinite(x0).all()) and bool(torch.isfinite(adj).all())
                    and bool(((adj==0)|(adj==1)).all()) and cap['target_ids'].dtype==cap['candidate_ids'].dtype==torch.int64,'original_capture_valid_shapes_and_binary_edges')
                a.check(len(set(targets+candidates))==t+c and all(0<=v<original for v in targets+candidates)
                    and targets==row['node_ids']==saved['node_ids'] and row['labels']==saved['labels']
                    and all(y==1 for y in row['labels']) and len(row['labels'])==t==row['targets'],'original_complete_target_candidate_identity')
                legal=torch.tril(torch.ones_like(adj,dtype=torch.bool),diagonal=t+c-1)
                a.check(not bool((adj[~legal]!=0).any()) and bool((adj[:,:t].sum(1)>=1).all())
                    and n==row['node_budget']==saved['node_budget'] and int(adj.sum())==row['edge_budget']==saved['edge_budget'],
                    'unchanged_legal_hard_graph_and_actual_physical_budget')
                a.check(row['source_capture_sha256']==oldref['sha256'] and row['reference_pt_sha256']==refseal['reference_pt_sha256']
                    and hashes['adj_attack']==saved['adjacency_content_sha256'] and hashes['x_atk']==saved['feature_statistics']['baseline']['content_sha256']
                    and tensor_hash(old['features']['baseline'])==hashes['x_atk'],'original_baseline_and_reference_bindings')
                physical=[int(adj[i].sum())+int(adj[i+1:,t+c+i].sum()) for i in range(n)]
                violation=float(torch.maximum((minimum-x0).clamp_min(0),(x0-maximum).clamp_min(0)).max())
                a.check(violation<=1e-6 and row['baseline_box_max_violation']==violation and row['baseline_exact_box']==(violation==0),
                    'original_baseline_box_tolerance_preserved_not_silently_clipped')
                a.check(row['alpha']==.5 and row['zero_width_dimensions']==int((~movable).sum()),'fixed_alpha_zero_width_count')
                for arm in ARMS:
                    actual=new['features'][arm];sequence=new['canonical_sequences'][arm];meta=row['variants'][arm]
                    if arm=='baseline':expected=x0
                    else:
                        key={'benign_mix':'benign_mean','all_train_mix':'all_train_mean','fraud_mix':'fraud_mean'}[arm]
                        expected=x0*.5+reference[key].unsqueeze(0)*.5
                        expected=torch.maximum(minimum,torch.minimum(maximum,expected));expected=torch.where(movable.unsqueeze(0),expected,x0)
                    a.check(actual.device.type=='cpu' and actual.dtype==torch.float32 and bool(torch.isfinite(actual).all())
                        and tensor_hash(actual)==tensor_hash(expected)==meta['feature_content_sha256'],'independent_exact_stored_prototype_mix_F32_projection')
                    a.check(tensor_hash(actual[:,~movable])==tensor_hash(x0[:,~movable]),'all_zero_width_baseline_bits_exact')
                    if arm!='baseline':a.check(bool(((actual>=minimum)&(actual<=maximum)).all()),'proposed_features_exact_original_raw_box')
                    oldbytes=x0.contiguous().view(torch.uint8).reshape(n,768,4);newbytes=actual.contiguous().view(torch.uint8).reshape(n,768,4)
                    changed=(oldbytes!=newbytes).any(-1)
                    identity=dict(exact_equals_baseline=not bool(changed.any()),changed_rows=int(changed.any(1).sum()),changed_coordinates=int(changed.sum()),
                        rows=n,coordinates=x0.numel(),comparison='Exact logical row-major float32 bytes')
                    a.check(meta['identity']==identity,'actual_feature_change_identity_counts')
                    identity_counts[arm+'_identity_cases']+=int(identity['exact_equals_baseline'])
                    delta=(actual-x0).double();a.close(meta['raw_displacement_linf'],float(delta.abs().max()),'actual_raw_Linf_movement');a.close(meta['raw_displacement_l2'],float(delta.norm()),'actual_raw_L2_movement')
                    a.check(meta['actual_injected_nodes']==n and meta['actual_injected_nonself_undirected_edges']==int(adj.sum())
                        and meta['physical_injected_degrees']==physical,'every_injected_degree_and_edge_cost_fixed')
                    a.check(sequence.dtype==torch.float32 and tuple(sequence.shape)==(t,7,768) and bool(torch.isfinite(sequence).all())
                        and tensor_hash(sequence)==meta['canonical_sequence_content_sha256'],'sealed_canonical_input_shape_dtype_hash')
                    a.check(meta['unchanged_tokens0_1_2_4_5_exact'] is True and tensor_hash(sequence[:,FIXED_TOKENS])
                        ==tensor_hash(new['canonical_sequences']['baseline'][:,FIXED_TOKENS]),'actual_fixed_SELF_and_known_class_token_bits')
                nodes+=n;edges+=int(adj.sum());occurrences+=t;unique.update(targets)
                del old,new,cap,x0,adj,actual,expected,sequence
            a.check((nodes,edges,occurrences,len(unique))==(7776,358491,1091,374),'all128_scenario_costs_and_complete_denominators')
            totals_geometry.extend(dict(seed=seed,arm=arm,groups=128,target_occurrences=1091,unique_targets=374,summed_injected_nodes=nodes,summed_nonself_undirected_edges=edges) for arm in ARMS)
        a.check(totals_geometry==marker['whole_seed_geometry_totals'],'all12_geometry_budget_summaries')
        all_counts=dict(GEOMETRY_COUNTS,**MODEL_COUNTS,**{name:0 for name in NO_MODEL})
        a.check(r['counters']==all_counts,'exact_all_reference_geometry_and1664_classifier_counters')
        a.check(refseal['counters_before_models']=={name:(1 if name in ('reference_selections','reference_tensor_saves') else 0) for name in all_counts},'reference_seal_precedes_any_classifier')
        a.check(marker['counters_before_models']=={name:GEOMETRY_COUNTS.get(name,0) for name in all_counts},'all1536_variant_seal_precedes_any_classifier')
        a.check(r['all_geometry_prepared_before_classifier'] is True and r['all_reference_and_feature_proposals_sealed_before_models'] is True
            and r['classifier_loading_after_geometry_seal'] is True,'producer_temporal_model_gate_receipts')
        a.check(r['immutable_content_hashes_retained'] is True and r['immutable_content_hashes']['before']==r['immutable_content_hashes']['after']
            and marker['original_content_hashes']=={k:r['immutable_content_hashes']['before'][k] for k in ('features','feat_sum','degrees')},'immutable_original_model_feature_cache_hash_receipts')
        a.check(r['GPU_hours']==0 and r['cuda_visibility']=='' and r['torch_cuda_available'] is False
            and r['torch_visible_gpu_count']==r['native_visible_gpu_count']==0 and r['final_TEST_used'] is False
            and r['experimental_API_calls']==0 and r['process_cpu_seconds']>=0 and 0<r['wall_seconds']<=5415,'actual_CPU_clocks_zero_GPU_API_TEST')
        clean=keyed(a,a.rows(stage/'clean_predictions.jsonl'),p['valid_groups'],'all128_fresh_clean_predictions');clean_pred={}
        for group,row in clean.items():
            a.check(row['node_ids']==records[3,group]['node_ids'] and row['labels']==records[3,group]['labels'],'fresh_clean_target_identity')
            for detector in ('gcn','gaga'):clean_pred[group,detector]=classify(a,row[detector+'_logits'],len(row['labels']))
        summaries={(v['seed'],v['arm']):v for v in r['summaries']};a.check(len(summaries)==len(r['summaries'])==12,'all12_seed_arm_summaries')
        metrics=[];outputs={}
        for seed in SEEDS:
            for arm in ARMS:
                rows=keyed(a,a.rows(stage/('%s_seed%d_predictions.jsonl'%(arm,seed))),p['valid_groups'],'all1536_fresh_variant_prediction_rows')
                totals={d:collections.Counter() for d in ('gcn','gaga')};macro={d:[] for d in ('gcn','gaga')}
                for group,row in rows.items():
                    geometry=records[seed,group];meta=geometry['variants'][arm];n=geometry['targets'];labels=geometry['labels'];hashes=geometry['immutable_capture_content_hashes']
                    a.check(row['seed']==seed and row['arm']==arm and row['node_ids']==geometry['node_ids'] and row['labels']==labels and row['alpha']==.5,'fresh_prediction_complete_identity_alpha')
                    a.check(row['actual_injected_nodes']==row['node_budget']==geometry['node_budget'] and row['actual_injected_edges']==row['edge_budget']==geometry['edge_budget']
                        and row['source_capture_sha256']==geometry['source_capture_sha256'] and row['prepared_artifact_sha256']==geometry['artifact']['sha256']
                        and row['reference_pt_sha256']==refseal['reference_pt_sha256'] and row['original_adjacency_content_sha256']==hashes['adj_attack']
                        and row['target_order_content_sha256']==hashes['target_ids'] and row['candidate_order_content_sha256']==hashes['candidate_ids']
                        and row['injected_feature_content_sha256']==meta['feature_content_sha256'] and row['canonical_sequence_content_sha256']==meta['canonical_sequence_content_sha256'],
                        'prediction_exact_sealed_reference_graph_feature_and_cost_binding')
                    for detector in ('gcn','gaga'):
                        count=transitions(labels,clean_pred[group,detector],classify(a,row[detector+'_logits'],n));a.check(row[detector+'_transitions']==count,'fresh_raw_argmax_transition_recount')
                        totals[detector].update(count);macro[detector].append(100*count['attacked_wrong']/n)
                summary=summaries[seed,arm];a.check(summary['groups']==128 and summary['unique_targets']==374,'summary_fixed_group_unique_denominators')
                for detector in ('gcn','gaga'):
                    total={name:totals[detector][name] for name in FIELDS};mr=100*total['attacked_wrong']/1091
                    a.check(total['occurrences']==1091 and total==summary['totals'][detector],'raw_complete_summary_counts');a.close(summary[detector+'_mr_percent'],mr,'raw_MR_recount')
                    metrics.append(dict(seed=seed,arm=arm,detector=detector,**total,mr_percent=mr,group_macro_mr_percent=statistics.fmean(macro[detector])))
                outputs[seed,arm]=rows
        lookup={(v['seed'],v['arm'],v['detector']):v for v in metrics};contrasts={};t95=math.sqrt(2)*.95/math.sqrt(1-.95**2)
        pairs=[(arm,'baseline') for arm in ARMS[1:]]+[('benign_mix','all_train_mix'),('benign_mix','fraud_mix')]
        for arm,control in pairs:
            for detector in ('gcn','gaga'):
                differences=[lookup[s,arm,detector]['mr_percent']-lookup[s,control,detector]['mr_percent'] for s in SEEDS]
                mean=statistics.fmean(differences);sd=statistics.stdev(differences);half=t95*sd/math.sqrt(3);paired=collections.Counter()
                for seed in SEEDS:
                    for group in p['valid_groups']:
                        first=outputs[seed,control][group];second=outputs[seed,arm][group];labels=first['labels']
                        before=classify(a,first[detector+'_logits'],len(labels));after=classify(a,second[detector+'_logits'],len(labels))
                        delta=sum(x!=y for x,y in zip(after,labels))-sum(x!=y for x,y in zip(before,labels));paired['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']+=1
                        for y,c,x,z in zip(labels,clean_pred[group,detector],before,after):
                            paired['occurrences']+=1;paired['gained_wrong']+=x==y and z!=y;paired['lost_wrong']+=x!=y and z==y
                            if c==y:paired['new_success_gained']+=x==y and z!=y;paired['baseline_new_success_lost']+=x!=y and z==y
                            else:paired['additional_recovery']+=x!=y and z==y;paired['previous_recovery_lost']+=x==y and z!=y
                a.check(paired['occurrences']==3273 and sum(paired[k] for k in ('group_wins','group_ties','group_losses'))==384,'full_correlated_paired_occurrence_case_counts')
                contrasts[arm+'-minus-'+control+'_'+detector]=dict(seed_differences_pp=differences,mean_difference_pp=mean,sample_sd_pp=sd,
                    paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts=dict(paired))
        primary=contrasts['benign_mix-minus-baseline_gaga'];control_advantages={arm:contrasts['benign_mix-minus-'+arm+'_gaga']['mean_difference_pp'] for arm in ('all_train_mix','fraud_mix')}
        criterion=dict(mean_delta_pp=primary['mean_difference_pp'],all_three_seed_deltas_positive=all(v>0 for v in primary['seed_differences_pp']),
            mean_advantages_over_controls_pp=control_advantages,paired_t95_lower_pp=primary['paired_t95_pp'][0],
            criterion_met=primary['mean_difference_pp']>=1 and all(v>0 for v in primary['seed_differences_pp']) and all(v>0 for v in control_advantages.values()) and primary['paired_t95_pp'][0]>0,
            controls_not_candidates=True,final_TEST_allowed=False)
        expected_outputs={'registered.json','reference.pt','reference.json','reference_sealed.json','geometry.jsonl','geometry_sealed.json','clean_predictions.jsonl','input_hashes.json'}|{'%s_seed%d_predictions.jsonl'%(arm,s) for arm in ARMS for s in SEEDS}
        a.check(set(r['output_hashes'])==expected_outputs,'all20_reference_geometry_and_prediction_outputs_hash_bound')
        a.check(counters['original_DGL_graph_loads']==1 and counters['partition_ID_array_loads']==3 and counters['TRAIN_label_entries_indexed']==6595
            and counters['reference_tensor_loads']==1 and counters['base_capture_tensor_loads']==counters['new_artifact_tensor_loads']==384
            and all(counters[k]==0 for k in ('nonTRAIN_label_entries_indexed','classifier_weight_deserializations','model_constructions','model_forwards','autograd_calls','optimizer_updates','API_calls','TEST_prediction_reads','GPU_hours')),
            'only_authorized_dataset_TRAIN_and769_tensor_reads_no_models')
        result.update(status='passed',provenance=provenance,independent_reference=reference_report,geometry_totals=totals_geometry,
            feature_identity_counts=dict(identity_counts),metrics=metrics,paired_contrasts=contrasts,primary_development_criterion=criterion,
            producer_counters=r['counters'],producer_cpu_seconds=r['process_cpu_seconds'],producer_wall_seconds=r['wall_seconds'],
            immutable_content_hash_receipts=r['immutable_content_hashes'],audit_runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,dgl=dgl.__version__),
            limitations=['POST HOC new development experiment after previous failures; no control becomes the selected primary after outcomes.',
                'Reference pool labels and every prototype coordinate are independently verified from original TRAIN only, using compensated fsum means. One-float32ULP reduction-order allowance is registered; exact output-bit agreement is reported separately.',
                'DGL deserialization materializes the global label field. The audit indexes only6595TRAIN labels then discards it; held-out ID arrays serve partition checks, not held-out truth or TEST prediction reads.',
                'The producer original DataHandler separately materializes whole-graph label/class-count metadata. TRAIN-only mean selection does not mean the whole producer loader never accesses such metadata; this audit independently checks the actual reference pools and values.',
                'All384BASEand384NEWcapture artifacts plus reference.pt are CPU-deserialized. No classifier weight, model or inference is loaded/run; no producer method function is imported.',
                'Original-original arc identity, full canonical sequence construction and temporal seal-before-classifier order are frozen-source/producer assertions; saved token0/1/2/4/5 bits and feature/adjacency tensors are independently checked.',
                'Numerical embedding-space prototypes/mixes are not demonstrated article text, semantic validity, camouflage, novelty or course completion.',
                '3273 repeated target occurrences across3seeds are correlated. Conditional n3 paired intervals do not establish population generalization or fresh-teacher uncertainty. FullTEST/tag gates remain separate.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(audit_operation_counters=counters,check_counts=dict(a.counts),byte_hashed_files=len(a.files),process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'input_hashes.json',[dict(path=n,sha256=h) for n,h in sorted(a.files.items())]);write(out/'result.json',result)
        print(json.dumps(dict(status=result['status'],output=str(out),checks=sum(a.counts.values()),CPU_seconds=result['process_cpu_seconds'],wall_seconds=result['wall_seconds'])),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
