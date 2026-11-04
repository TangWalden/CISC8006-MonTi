"""Independent saved-record audit of the post-hoc feature parity diagnostic.

Standard library only. No ML imports, tensor/checkpoint/dataset loading, model
construction/inference, gradients, updates, network/API calls or new science.
Freeze before diagnostic output reads. Producer-only cache/sequence arithmetic
and tensor immutability receipts are explicitly distinguished from raw-logit
and endpoint arithmetic independently recounted here. Agents are not humans.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import struct
import subprocess
import time
import traceback

ARMS=('baseline','gaga_pgd')
SEEDS=(3,4,5)
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
SCHEMA='monti_feature_parity_diagnostic_v1'
EXPECTED=dict(capture_tensor_loads=384,hard_graph_reconstructions=384,affine_builders=384,
    canonical_sequence_updates=768,fresh_teacher_completed_forwards=768,stored_affine_comparisons=768,
    fresh_teacher_forward_dispatches=768,guard_alias_checks=384,data_load_dispatches=1,data_loads_completed=1,
    teacher_load_dispatches=1,teacher_loads_completed=1,new_affine_model_forwards=0,
    generator_model_constructions=0,generator_forwards=0,GCN_model_constructions=0,GCN_forwards=0,
    official_victim_model_constructions=0,official_victim_forwards=0,optimizer_steps=0,autograd_calls=0,API_calls=0,TEST_queries=0)
U32=2.0**-24


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


class Audit:
    def __init__(self,root):self.root=root;self.files={};self.counts=collections.Counter()
    def check(self,value,name):
        self.counts[name]+=1
        if not value:raise AssertionError('Parity saved audit failed: '+name)
    def path(self,name,directory=False):
        raw=pathlib.Path(name);p=raw.resolve() if raw.is_absolute() else (self.root/raw).resolve()
        self.check(self.root in p.parents and (p.is_dir() if directory else p.is_file()),'repository_input_boundary')
        return p
    def file(self,name,wanted=None):
        p=self.path(name);actual=sha(p);key=p.relative_to(self.root).as_posix()
        self.check(self.files.setdefault(key,actual)==actual,'bytes_unchanged_during_audit')
        if wanted is not None:self.check(actual==wanted,'frozen_input_byte_hash')
        return p
    def json(self,name,wanted=None):return json.loads(self.file(name,wanted).read_text(encoding='utf-8'))
    def rows(self,name):
        with self.file(name).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
    def close(self,x,y,name,atol=1e-10,rtol=1e-9):
        self.check(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*abs(y),name)
    def reduction(self,measured,values,name):
        # Explicit conservative float32 reduction/function-rounding envelope.
        # This tolerance does NOT alter the diagnostic's elementwise parity
        # bounds, class predictions, sign counts or violation membership.
        n=len(values);mean=statistics.fmean(values)
        bound=2e-6+8*(n+2)*U32*max(1.,statistics.fmean(abs(v) for v in values))
        self.close(measured,mean,name,atol=bound,rtol=0)


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def softplus(x):return x if x>20 else max(x,0.)+math.log1p(math.exp(-abs(x)))
def sign(x):return 1 if x>0 else -1 if x<0 else 0


def logits(a,values,n):
    a.check(len(values)==n,'raw_logit_denominator')
    for z in values:a.check(len(z)==2 and all(type(v) in (int,float) and math.isfinite(v) and f32(v)==v for v in z),'finite_float32_binary_logits')
    prediction=[0 if z[0]>=z[1] else 1 for z in values]
    margin=[f32(z[0]-z[1]) for z in values]
    loss=[f32(softplus(f32(z[1]-z[0]))) for z in values]
    return dict(prediction=prediction,margin=margin,loss=loss)


def check_metrics(a,row,computed,labels):
    a.check(row['occurrences']==len(labels) and row['wrong']==sum(x!=y for x,y in zip(computed['prediction'],labels)),
        'raw_argmax_wrong_count_recount')
    a.reduction(row['mean_benign_minus_fraud_margin'],computed['margin'],'raw_mean_margin_recount')
    a.reduction(row['mean_softplus'],computed['loss'],'raw_mean_softplus_recount')


def numeric_comparison(a,actual,expected,metadata):
    error=[];fraction=[];passed=True
    for left,right in zip(actual,expected):
        for x,y in zip(left,right):
            difference=abs(f32(x-y))
            tolerance=f32(f32(2e-4)+f32(f32(2e-4)*abs(y)))
            error.append(difference);fraction.append(f32(difference/tolerance));passed=passed and difference<=tolerance
    a.check(metadata['shape_matches'] is True and metadata['passed']==passed,'raw_elementwise_logit_tolerance_recount')
    a.close(metadata['max_absolute_error'],max(error),'raw_maximum_logit_error_recount',atol=1e-10,rtol=1e-9)
    a.close(metadata['max_tolerance_fraction'],max(fraction),'raw_maximum_logit_tolerance_fraction',atol=2e-6,rtol=2e-6)
    return passed


def index(a,rows,groups,name):
    indexed={row['group']:row for row in rows}
    a.check(len(indexed)==len(rows)==128 and set(indexed)==set(groups),name)
    return indexed


def register(a,p,source,protocol_sha):
    a.check(p['schema']==SCHEMA and p['status']=='frozen' and p['post_hoc'] is True
        and p['owner_execution_authorized'] is True,'frozen_explicit_posthoc_scope')
    a.check(p['source_commit']==source and p['base_experiment_ref']['sha256']==BASE_SHA,'exact_new_and_completed_base_sources')
    a.check(tuple(p['seeds'])==SEEDS and tuple(p['arms'])==ARMS and set(p['runs'])=={'3','4','5'}
        and len(p['valid_groups'])==len(set(p['valid_groups']))==128,'fixed_diagnostic_cohort')
    a.check(p['expected_counts']==EXPECTED and p['deadline_seconds']==1800 and p['cpu_threads']==8
        and p['execution_device']=='cpu' and p['cuda_visible_devices']==''
        and p['final_TEST_allowed'] is False and p['new_affine_model_forwards']==0,'bounded_no_new_victim_or_affine_model_calls')
    a.check(p['sequence_tolerances']==dict(atol=2e-5,rtol=2e-5)
        and p['logit_tolerances']==dict(atol=2e-4,rtol=2e-4),'original_numeric_tolerances')
    expected_files={}
    def retain(ref):
        a.file(ref['path'],ref['sha256']);old=expected_files.setdefault(ref['path'],ref['sha256'])
        a.check(old==ref['sha256'],'no_conflicting_input_references')
    for name,wanted in p['source_files'].items():
        retain(dict(path=name,sha256=wanted))
        data=subprocess.check_output(['git','show',source+':'+name],cwd=a.root)
        a.check(hashlib.sha256(data).hexdigest()==wanted,'registered_source_matches_committed_bytes')
    required={'course/feature_parity_check/'+name for name in ('__init__.py','prepare.py','run.py')}
    a.check(required<=set(p['source_files']),'complete_new_diagnostic_source_inventory')
    for row in p['inputs']:retain(row)
    retain(p['base_experiment_ref']);base=a.json(p['base_experiment_ref']['path'],BASE_SHA)
    a.check(base['source_commit']==BASE_SOURCE and base['valid_groups']==p['valid_groups'] and base['execution_device']=='cpu'
        and base['final_test_allowed'] is False,'completed_base_identity_cpu_no_test')
    for name,wanted in base['source_files'].items():a.check(p['source_files'].get(name)==wanted,'reused_science_bytes_unchanged')
    for row in base['inputs']+base['retained_inputs']:retain(row)
    a.check(p['teacher_ref']==base['teacher_ref'],'same_original_full_train_teacher')
    retain(p['teacher_ref']);teacher=a.json(p['teacher_ref']['path'],p['teacher_ref']['sha256'])
    a.check(teacher['status']=='completed' and teacher['official_victim_loaded'] is False
        and teacher['original_VALID_used'] is False and teacher['TEST_used'] is False,'own_teacher_scope_reconciled')
    runs={};unique=None
    for seed in SEEDS:
        spec=p['runs'][str(seed)]
        for key in ('result_ref','groups_ref'):retain(spec[key])
        result=a.json(spec['result_ref']['path']);rows=index(a,a.rows(spec['groups_ref']['path']),p['valid_groups'],'all_saved_refinement_groups')
        a.check(result['status']=='completed' and result['seed']==seed and result['groups']==128
            and result['source_commit']==BASE_SOURCE and result['protocol_sha256']==BASE_SHA
            and result['groups_jsonl_sha256']==spec['groups_ref']['sha256']
            and result['teacher_ref']==p['teacher_ref'] and result['teacher_state_sha256']==teacher['final_state_sha256'],
            'saved_refinement_provenance_and_teacher')
        a.check([row['group'] for row in spec['captures']]==p['valid_groups'],'ordered_128_capture_reference_cover')
        exports={row['group']:row for row in result['exports']};directory=a.path(spec['result_ref']['path']).parent
        a.check(len(exports)==len(result['exports'])==128 and set(exports)==set(rows),'original_128_capture_cover')
        for capture in spec['captures']:
            retain(capture);original=exports[capture['group']]
            a.check(rows[capture['group']]['graph_capture']==original and capture['sha256']==original['sha256']
                and a.path(capture['path'])==(directory/original['path']).resolve(),'all384_capture_bindings')
        a.check(set(spec['victim_prediction_refs'])==set(ARMS),'two_saved_victim_endpoints_only')
        victims={}
        for arm in ARMS:
            retain(spec['victim_prediction_refs'][arm]);victims[arm]=index(a,a.rows(spec['victim_prediction_refs'][arm]['path']),p['valid_groups'],'saved_victim_full_128_cover')
        ids={node for row in rows.values() for node in row['node_ids']}
        a.check(len(ids)==374 and sum(row['targets'] for row in rows.values())==1091,'unique_and_occurrence_denominators')
        if unique is None:unique=ids
        else:a.check(ids==unique,'same_unique_target_cohort_all_seeds')
        runs[seed]=dict(rows=rows,victims=victims)
    return base,teacher,runs,expected_files,dict(source_commit=source,protocol_sha256=protocol_sha)


def cache_checks(a,checks):
    a.check(checks['unique_VALID_targets']==374 and checks['masked_self_count_each_hop']==1,'producer_cache_scope_374_self_each_hop')
    n=checks['original_adjacency_asymmetric_matrix_entries']
    a.check(type(n) is int and n>=0 and checks['original_adjacency_symmetric']==(n==0),'producer_symmetry_metadata_consistency')
    flags=('original_adjacency_binary','original_adjacency_symmetric','original_adjacency_zero_diagonal',
        'original_adj2_binary','original_adj2_zero_diagonal','original_adj2_selected_support_matches')
    for name in flags:a.check(type(checks[name]) is bool,'producer_topology_check_boolean')
    degrees=checks['unknown_degree_checks'];a.check([row['token'] for row in degrees]==[3,6],'both_unknown_hop_denominator_checks')
    for row in degrees:
        count=row['mismatched_targets'];error=row['maximum_absolute_count_error']
        a.check(type(count) is int and 0<=count<=374 and math.isfinite(error) and error>=0
            and row['exact_match']==(count==0 and error==0),'producer_denominator_metadata_consistency')
    return all(checks[name] for name in flags) and all(row['exact_match'] for row in degrees)


def endpoint_changes(a,record,before,after,labels):
    margins=[f32(y-x) for x,y in zip(before['margin'],after['margin'])]
    losses=[f32(y-x) for x,y in zip(before['loss'],after['loss'])]
    a.check(record['benign_margin_increases']==sum(y>x for x,y in zip(before['margin'],after['margin']))
        and record['benign_margin_decreases']==sum(y<x for x,y in zip(before['margin'],after['margin']))
        and record['wrong_count_change']==sum(x!=y for x,y in zip(after['prediction'],labels))-sum(x!=y for x,y in zip(before['prediction'],labels)),
        'two_endpoint_margin_signs_and_class_changes')
    a.check(record['per_target_benign_margin_change']==margins and len(record['per_target_softplus_change'])==len(labels),
        'per_target_margin_changes_recount')
    for measured,wanted,x,y in zip(record['per_target_softplus_change'],losses,before['loss'],after['loss']):
        a.close(measured,wanted,'per_target_softplus_change_recount',atol=2e-6+16*U32*max(1.,abs(x),abs(y)),rtol=2e-6)
    a.reduction(record['mean_benign_margin_change'],margins,'two_endpoint_mean_margin_recount')
    a.reduction(record['mean_softplus_change'],losses,'two_endpoint_mean_loss_recount')
    return margins


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--execution-source',required=True)
    parser.add_argument('--stage',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--audit-source-sha256',required=True);args=parser.parse_args()
    root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
    out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh saved audit output must remain in ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);start,cpu=time.monotonic(),time.process_time();a=Audit(root)
    result=dict(status='failed',schema='independent_feature_parity_saved_audit_v1',post_hoc=True,
        independent_of_diagnostic_implementation_authorship=True,human_verification=False,
        ML_imports=0,tensor_deserializations=0,dataset_loads=0,model_calls=0,optimizer_updates=0,API_calls=0,GPU_hours=0)
    try:
        own=sha(__file__);a.check(own==args.audit_source_sha256,'externally_frozen_saved_auditor')
        write(out/'registered.json',dict(audit_source_sha256=own,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            execution_source=args.execution_source,protocol_sha256=args.protocol_sha256,new_diagnostic_output_reads_started=False,
            scope='Saved JSON/raw logits and opaque source/input/capture bytes only; no tensor/weight/dataset loads or model calls',
            numeric_reduction_bound='2e-6 + 8*(n+2)*2^-24*max(1,mean(abs(values))); unchanged elementwise tolerance and exact class/sign counts'))
        p=a.json(args.protocol,args.protocol_sha256);stage=a.path(args.stage,directory=True)
        base,teacher,runs,expected_files,provenance=register(a,p,args.execution_source,args.protocol_sha256)
        receipt=a.json(stage/'registered.json');r=a.json(stage/'result.json')
        a.check(receipt['provenance']==r['provenance']==provenance and r['status']=='completed','completed_diagnostic_provenance')
        a.check(receipt['own_source_sha256']==p['source_files']['course/feature_parity_check/run.py'],
            'diagnostic_registered_executed_source_hash')
        a.check(receipt['base_experiment_ref']==p['base_experiment_ref'] and receipt['expected_captures']==384
            and receipt['expected_canonical_updates']==receipt['expected_fresh_own_teacher_forwards']==768
            and receipt['new_affine_model_forwards']==0,'diagnostic_registered_384_768_counts')
        for name,wanted in r['output_hashes'].items():a.file(stage/name,wanted)
        a.check(set(r['output_hashes'])=={'registered.json','original_topology_and_cache_checks.json','comparisons.jsonl','endpoint_joins.jsonl','violations.json','input_hashes.json'},'complete_output_hash_manifest')
        measured_inputs=a.json(stage/'input_hashes.json')
        a.check(measured_inputs==receipt['source_and_input_hashes']==expected_files
            and r['input_files_hashed']==len(expected_files),'all_source_and_input_refs_reconciled')
        a.check(r['counters']==EXPECTED and r['official_victim_loaded'] is False and r['GPU_hours']==0
            and r['cuda_visibility']=='' and r['torch_cuda_available'] is False
            and r['torch_visible_gpu_count']==r['native_visible_gpu_count']==0,'exact_counts_cpu_no_new_official_calls')
        a.check(r['immutable_teacher_original_features_caches'] is True and r['teacher_state_sha256']==teacher['final_state_sha256'],
            'same_frozen_own_teacher_receipt')
        hashes=r['immutable_content_hashes'];a.check(set(hashes)=={'before','after'} and hashes['before']==hashes['after']
            and set(hashes['before'])=={'teacher','features','feat_sum','degrees'}
            and hashes['before']['teacher']==teacher['final_state_sha256'],'producer_immutable_content_hash_receipts')
        for value in hashes['before'].values():a.check(len(value)==64 and all(c in '0123456789abcdef' for c in value),'immutable_content_digest_format')
        a.check(r['process_cpu_seconds']>=0 and 0<r['wall_seconds']<=1815,'actual_cpu_wall_clocks_no_gpu_charge')
        checks=a.json(stage/'original_topology_and_cache_checks.json')
        a.check(checks==r['original_topology_and_cache_checks'],'cache_check_record_identity')
        expected_violations=[]
        if not cache_checks(a,checks):expected_violations.append(dict(kind='original_topology_or_unknown_cache_assumption',detail=checks))
        rows=a.rows(stage/'comparisons.jsonl');joins=a.rows(stage/'endpoint_joins.jsonl')
        a.check(len(rows)==r['completed_comparison_rows']==r['registered_comparison_rows']==768
            and len(joins)==r['completed_endpoint_joins']==384,'all768_comparisons_384_joins_no_drops')
        indexed={(row['seed'],row['group'],row['arm']):row for row in rows}
        joined={(row['seed'],row['group']):row for row in joins}
        expected_keys={(seed,group,arm) for seed in SEEDS for group in p['valid_groups'] for arm in ARMS}
        a.check(len(indexed)==768 and set(indexed)==expected_keys and len(joined)==384
            and set(joined)=={(seed,group) for seed in SEEDS for group in p['valid_groups']},'exact_complete_case_endpoint_cover')
        numeric={};group_summaries=[];category=collections.Counter();directions=collections.Counter();max_changed_margin=0.
        for seed in SEEDS:
            for group in p['valid_groups']:
                saved=runs[seed]['rows'][group];labels=saved['labels'];ids=saved['node_ids'];n=len(ids)
                a.check(n==saved['targets']==len(labels) and n>0 and len(set(ids))==n and all(y==1 for y in labels),'stored_fraud_target_occurrences')
                base_gc=[0 if z[0]>=z[1] else 1 for z in saved['final_own_logits']['baseline']['gcn']]
                final_gc=[0 if z[0]>=z[1] else 1 for z in saved['final_own_logits']['gaga_pgd']['gcn']]
                accepted=all(x!=0 or y==0 for x,y in zip(base_gc,final_gc));alias='gaga_pgd' if accepted else 'baseline'
                logical_guard_ok=(accepted==saved['guard']['accepted'] and saved['final_own_logits']['gaga_guarded']==saved['final_own_logits'][alias]
                    and saved['feature_statistics']['gaga_guarded']['content_sha256']==saved['feature_statistics'][alias]['content_sha256'])
                guard_ok=indexed[seed,group,'baseline']['guarded_alias_matches']
                a.check(type(guard_ok) is bool and (not guard_ok or logical_guard_ok),
                    'producer_physical_guard_report_has_consistent_saved_alias_metadata')
                if not guard_ok:expected_violations.append(dict(kind='saved_guard_alias',seed=seed,group=group))
                for arm in ARMS:
                    row=indexed[seed,group,arm];victim=runs[seed]['victims'][arm][group]
                    a.check(row['targets']==n and row['node_ids']==victim['node_ids']==ids and row['labels']==victim['labels']==labels
                        and victim['seed']==seed and victim['arm']==arm,'comparison_original_target_identity_order')
                    a.check(row['capture_sha256']==victim['common_capture_sha256']==saved['graph_capture']['sha256']
                        and row['feature_content_sha256']==victim['injected_feature_content_sha256']==saved['feature_statistics'][arm]['content_sha256']
                        and victim['hard_adjacency_content_sha256']==saved['adjacency_content_sha256'],'same_capture_features_and_saved_victim_binding')
                    a.check(row['stored_affine_teacher_logits']==saved['final_own_logits'][arm]['gaga_teacher']
                        and row['saved_victim_logits']==victim['gaga_logits'],'saved_logit_vectors_exact_input_identity')
                    current=logits(a,row['canonical_teacher_logits'],n);stored=logits(a,row['stored_affine_teacher_logits'],n);v=logits(a,row['saved_victim_logits'],n)
                    for field,value in (('teacher_canonical_metrics',current),('teacher_saved_affine_metrics',stored),('saved_victim_metrics',v)):
                        check_metrics(a,row[field],value,labels)
                    a.close(row['teacher_mean_loss_difference'],row['teacher_canonical_metrics']['mean_softplus']-row['teacher_saved_affine_metrics']['mean_softplus'],'stored_mean_loss_difference')
                    a.close(row['teacher_mean_margin_difference'],row['teacher_canonical_metrics']['mean_benign_minus_fraud_margin']-row['teacher_saved_affine_metrics']['mean_benign_minus_fraud_margin'],'stored_mean_margin_difference')
                    passed=numeric_comparison(a,row['canonical_teacher_logits'],row['stored_affine_teacher_logits'],row['teacher_logits'])
                    sequence=row['sequence'];a.check(sequence['shape_matches'] is True and sequence['max_absolute_error'] is not None
                        and sequence['max_tolerance_fraction'] is not None and sequence['max_absolute_error']>=0
                        and math.isfinite(sequence['max_absolute_error']) and math.isfinite(sequence['max_tolerance_fraction'])
                        and sequence['max_tolerance_fraction']>=0
                        and (sequence['max_tolerance_fraction']<=1+2e-7 if sequence['passed'] else sequence['max_tolerance_fraction']>=1-2e-7),
                        'producer_sequence_tolerance_metadata_consistency')
                    changed=[i for i,(x,y) in enumerate(zip(current['prediction'],stored['prediction'])) if x!=y]
                    a.check(row['teacher_class_differences']==len(changed),'raw_argmax_identity_recount')
                    a.check(row['guarded_alias_matches']==guard_ok,'stored_guard_alias_receipt_consistency')
                    if not sequence['passed']:expected_violations.append(dict(kind='affine_sequence_parity',seed=seed,group=group,arm=arm,detail=sequence))
                    if not passed:expected_violations.append(dict(kind='canonical_teacher_vs_saved_affine_logits',seed=seed,group=group,arm=arm,detail=row['teacher_logits']))
                    if changed:
                        detail=dict(count=len(changed),canonical_changed_max_absolute_margin=max(abs(current['margin'][i]) for i in changed),
                            stored_changed_max_absolute_margin=max(abs(stored['margin'][i]) for i in changed))
                        kind='within_tolerance_teacher_argmax_flip' if passed else 'canonical_teacher_vs_saved_affine_class'
                        expected_violations.append(dict(kind=kind,seed=seed,group=group,arm=arm,detail=detail))
                        category[kind+'_comparisons']+=1;category[kind+'_occurrences']+=len(changed)
                        max_changed_margin=max(max_changed_margin,detail['canonical_changed_max_absolute_margin'],detail['stored_changed_max_absolute_margin'])
                    numeric[seed,group,arm]=dict(current=current,stored=stored,victim=v)
                join=joined[seed,group];a.check(join['targets']==n,'endpoint_join_denominator')
                base_n=numeric[seed,group,'baseline'];final_n=numeric[seed,group,'gaga_pgd']
                own_delta=endpoint_changes(a,join['canonical_own_teacher'],base_n['current'],final_n['current'],labels)
                victim_delta=endpoint_changes(a,join['saved_victim'],base_n['victim'],final_n['victim'],labels)
                for x,y in zip(own_delta,victim_delta):directions[str((sign(x),sign(y)))]+=1
            for arm in ARMS:
                selected=[indexed[seed,group,arm] for group in p['valid_groups']]
                summary=dict(seed=seed,arm=arm,graphs=128,occurrences=sum(row['targets'] for row in selected),
                    sequence_parity_failures=sum(not row['sequence']['passed'] for row in selected),
                    logit_parity_failures=sum(not row['teacher_logits']['passed'] for row in selected),
                    class_differences=sum(row['teacher_class_differences'] for row in selected),
                    maximum_sequence_absolute_error=max(row['sequence']['max_absolute_error'] for row in selected),
                    maximum_teacher_logit_absolute_error=max(row['teacher_logits']['max_absolute_error'] for row in selected),
                    canonical_teacher_wrong=sum(row['teacher_canonical_metrics']['wrong'] for row in selected),
                    saved_affine_teacher_wrong=sum(row['teacher_saved_affine_metrics']['wrong'] for row in selected),
                    saved_victim_wrong=sum(row['saved_victim_metrics']['wrong'] for row in selected))
                a.check(summary['occurrences']==1091,'all128_groups1091_occurrences_per_seed_arm');group_summaries.append(summary)
        violations=a.json(stage/'violations.json');a.check(violations==expected_violations and len(violations)==r['violation_count'],'all_violations_retained_and_correctly_categorized')
        a.check(r['summaries']==group_summaries and r['unique_VALID_targets_checked']==374,'all_six_raw_summary_recounts')
        a.check(r['parity_passed']==(not violations)
            and r['sequence_numeric_parity_passed']==all(row['sequence']['passed'] for row in rows)
            and r['teacher_numeric_logit_parity_passed']==all(row['teacher_logits']['passed'] for row in rows)
            and r['teacher_argmax_identity']==all(row['teacher_class_differences']==0 for row in rows)
            and r['guard_alias_integrity']==all(row['guarded_alias_matches'] for row in rows),'separate_parity_categories_not_collapsed')
        a.check(sum(directions.values())==3273,'signed_endpoint_join_correlated_denominator')
        result.update(status='passed',provenance=provenance,producer_parity_passed=r['parity_passed'],
            independent_raw_logit_and_endpoint_recount=True,comparisons=768,endpoint_joins=384,
            unique_targets=374,correlated_endpoint_occurrences=3273,summaries=group_summaries,
            violation_counts=dict(collections.Counter(row['kind'] for row in violations)),
            class_flip_categories=dict(category),maximum_changed_absolute_margin=max_changed_margin,
            signed_two_endpoint_margin_counts=dict(directions),producer_original_topology_cache_checks=checks,
            immutable_content_hash_receipts=hashes,producer_cpu_seconds=r['process_cpu_seconds'],producer_wall_seconds=r['wall_seconds'],
            limitations=['POST HOC after known negative development outcomes; not preregistered efficacy or a new method winner.',
                'No sequence/cache tensor, graph, weight or gradient replay. Sequence maxima, cache support/denominator checks and immutable tensor hashes are producer measurements/receipts, not independently remeasured here.',
                'Official-victim logits are immutable saved values only, with no new victim/model queries.',
                'Own-teacher/victim scales are separate and uncalibrated. Signed finite two-endpoint changes are not derivatives, slopes, causal directions or comparable gradient magnitudes.',
                '3273 repeated memberships across three seeds are correlated and do not support IID occurrence uncertainty.',
                'This agent audit is not human technical verification, efficacy/novelty confirmation or final TEST evaluation.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(check_counts=dict(a.counts),byte_hashed_files=len(a.files),process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start)
        write(out/'input_hashes.json',[dict(path=name,sha256=value) for name,value in sorted(a.files.items())])
        write(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
