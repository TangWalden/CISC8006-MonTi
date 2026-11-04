"""Independent saved-JSON/byte audit for fixed-graph feature refinement.

No ML imports, tensor deserialization, dataset/graph loads, model construction,
inference, updates or API calls. This auditor did not author the implementation.
Freeze this file before reading the new outputs. Stored labels/state/counter
declarations are reconciled, not independently inferred from opaque weights.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time
import traceback

ARMS=('baseline','gcn_pgd','gaga_pgd','gaga_guarded','random')
SEEDS=(3,4,5)
SOURCE='f3892f06335e4d91813cc537089694df3ec54ce5'
PREPARATION='c3794d4f8ee11458f6c405b574911059b4113feb6898509613c1727cd2fd4967'
SECOND_SOURCE='db48fd8f441b0ab8a9d36647b4dc4b4f99c77d02'
SECOND_PREPARATION='359bef0232c5eed07d741754db8453cf3af175298ff39fef528e92d751359b84'
SECOND_REFINEMENT='8dd7408cefa68b840a56ee6ee6d5fe1e6fa6bf8c58180202680539152061a6a2'
SECOND_FAILURE='ae104e9db4030892af8600aa61841f2cb21946b9552099a79f116d07c1e7a0a3'
SECOND_PROGRESS='6b42ab941996fb8c697d0f80fc238eec40f9275abfe7346e015f63055fed2012'
INITIAL='99a6aaec50775a52130b40cec9a0af76f3da60863d2211c5b286c3b8e9144f38'
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
PAIR_FIELDS=('occurrences','baseline_wrong','candidate_wrong','shared_new_wrong',
             'baseline_new_success_lost','new_success_gained','additional_recovery','previous_recovery_lost')


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


class Audit:
    def __init__(self,root):
        self.root=root;self.files={};self.checks=collections.Counter();self.warnings=[]
    def check(self,condition,name):
        self.checks[name]+=1
        if not condition:raise AssertionError('Audit failed: '+name)
    def path(self,name,directory=False):
        raw=pathlib.Path(name);p=raw.resolve() if raw.is_absolute() else (self.root/raw).resolve()
        self.check(self.root in p.parents,'repository_path_boundary')
        self.check(p.is_dir() if directory else p.is_file(),'input_exists')
        return p
    def bytes(self,name,wanted=None):
        p=self.path(name);actual=sha(p);relative=p.relative_to(self.root).as_posix()
        self.check(self.files.setdefault(relative,actual)==actual,'input_unchanged_during_audit')
        if wanted is not None:self.check(actual==wanted,'recorded_byte_hash')
        return p
    def json(self,name,wanted=None):return json.loads(self.bytes(name,wanted).read_text(encoding='utf-8'))
    def jsonl(self,name):
        with self.bytes(name).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
    def close(self,actual,expected,name,atol=1e-8,rtol=1e-7):
        self.check(math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected)<=atol+rtol*abs(expected),name)


def classify(a,values,n=None):
    if n is not None:a.check(len(values)==n,'logit_denominator')
    for row in values:a.check(len(row)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in row),'finite_binary_logits')
    return [0 if row[0]>=row[1] else 1 for row in values]


def softplus(value):return max(value,0.)+math.log1p(math.exp(-abs(value)))


def finite_difference_recount(a,row,h=None):
    automatic=row['autograd'];numerical=row['central_difference']
    a.check(math.isfinite(automatic) and math.isfinite(numerical),'finite_functional_derivatives')
    if h is not None:
        a.close(numerical,(row['plus_value']-row['minus_value'])/(2*h),
            'finite_difference_plus_minus_recount',atol=1e-12,rtol=1e-10)
    error=abs(numerical-automatic);tolerance=.001+.03*max(abs(numerical),abs(automatic))
    a.close(row['absolute_error'],error,'finite_difference_error_recount',atol=1e-12,rtol=1e-10)
    a.close(row['tolerance'],tolerance,'unchanged_finite_difference_tolerance',atol=1e-12,rtol=1e-10)
    passed=error<=tolerance
    a.check(row['passed']==passed,'finite_difference_pass_recount')
    return passed


def fixed_gcn_preflight(a,report):
    grid=[.001,.0005,.00025,.000125,.0000625]
    a.check(report['status']=='passed' and report['actual_refinement_changed'] is False
        and report['finite_difference_grid']==grid and report['finite_difference_atol']==.001
        and report['finite_difference_rtol']==.03 and report['zero_width_gradients_exact_zero'] is True,
        'fixed_operator_fd_amendment_no_refinement_change')
    identities=report['frozen_sampled_block_identity']
    a.check(len(identities)==2 and report['frozen_block_identity_retained'] is True,'fixed_sampled_block_order_retained')
    for block in identities:
        a.check(block['source_nodes']>=block['destination_nodes']>0 and block['arcs']>0,'sampled_block_count_ranges')
        for name in ('source_nid_sha256','destination_nid_sha256','original_eid_sha256','local_arc_order_sha256'):
            value=block[name];a.check(len(value)==64 and all(c in '0123456789abcdef' for c in value),'fixed_block_identity_digest')
    a.check(len(report['float32_center_repeats'])==2,'two_fixed_float32_repeat_calls')
    for index,repeat in enumerate(report['float32_center_repeats'],1):
        a.check(repeat['repeat']==index and repeat['logits']['passed'] is True
            and 0<=repeat['logits']['max_tolerance_fraction']<=1
            and math.isfinite(repeat['raw_margin_difference']) and repeat['raw_margin_difference']>=0
            and math.isfinite(repeat['softplus_difference']) and repeat['softplus_difference']>=0,
            'fixed_float32_center_repeat_parity')
    comparisons=report['float32_vs_float64_gradients']
    a.check(set(comparisons)=={'raw_margin','actual_softplus'},'both_objective_float32_float64_checks')
    for name,comparison in comparisons.items():
        a.check(comparison['passed'] is True and comparison['failing_coordinates']==0
            and 0<=comparison['max_tolerance_fraction']<=1
            and all(math.isfinite(comparison[k]) and comparison[k]>=0 for k in
                ('max_absolute_error','float32_gradient_l2','float64_gradient_l2')),
            'all_coordinate_gradient_comparison_receipt')
        if name=='raw_margin':a.check(comparison['float32_gradient_l2']>0 and comparison['float64_gradient_l2']>0,'nonzero_fixed_gcn_feature_gradient')
    centers=report['center_leaky_relu_sign_patterns']
    a.check(len(centers)==2 and [row['layer'] for row in centers]==[1,2],'two_registered_leaky_relu_centers')
    for center in centers:
        a.check(center['elements']>0 and 0<=center['positive_count']<=center['elements']
            and len(center['sha256'])==64,'center_activation_pattern_metadata')
    coordinates=report['coordinates']
    a.check(len(coordinates)==4 and len({row['flat_coordinate'] for row in coordinates})==4,'four_fixed_gcn_gradient_coordinates')
    order=[(-abs(row['actual_float32_raw_margin_gradient']),row['flat_coordinate']) for row in coordinates]
    a.check(order==sorted(order),'fixed_coordinate_gradient_magnitude_order')
    all_pass=True
    for coordinate in coordinates:
        a.check(coordinate['flat_coordinate']>=0 and coordinate['actual_float32_raw_margin_gradient']!=0
            and [row['h'] for row in coordinate['grid']]==grid,'fixed_full_five_step_grid_retained')
        passes=[]
        for probe in coordinate['grid']:
            a.check(set(probe['objectives'])=={'raw_margin','actual_softplus'},'both_objectives_each_grid_step')
            numerical_pass=[finite_difference_recount(a,probe['objectives'][name],probe['h'])
                for name in ('raw_margin','actual_softplus')]
            raw=probe['objectives']['raw_margin']['autograd'];actual=coordinate['actual_float32_raw_margin_gradient']
            a.check(abs(raw-actual)<=.001+.03*max(abs(raw),abs(actual)),'selected_coordinate_float32_float64_gradient')
            changes=probe['leaky_relu_branch_changes'];a.check(len(changes)==2,'two_activation_patterns_each_step')
            stable=True
            for index,change in enumerate(changes):
                center=centers[index]
                a.check(change['layer']==center['layer'] and change['center_sha256']==center['sha256'],
                    'finite_difference_pattern_center_binding')
                for prefix in ('plus','minus'):
                    changed=change[prefix+'_changed'];same=change[prefix+'_sha256']==center['sha256']
                    a.check(type(changed) is int and 0<=changed<=center['elements']
                        and (changed==0)==same,'activation_change_count_and_digest_consistency')
                    stable=stable and changed==0
            a.check(probe['stable_branches']==stable,'branch_stability_recount')
            passes.append(stable and all(numerical_pass))
        finest=all(passes[-2:]);all_pass=all_pass and finest
        a.check(coordinate['finest_two_passed']==finest,'finest_two_step_criterion_recount')
    a.check(all_pass and report['finest_two_all_coordinates_passed']==all_pass,'finest_two_all_coordinates_pass')


def transitions(labels,clean,prediction):
    return dict(occurrences=len(labels),clean_wrong=sum(x!=y for x,y in zip(clean,labels)),
        attacked_wrong=sum(x!=y for x,y in zip(prediction,labels)),
        new_wrong=sum(c==y and p!=y for c,p,y in zip(clean,prediction,labels)),
        recoveries=sum(c!=y and p==y for c,p,y in zip(clean,prediction,labels)))


def keyed(a,rows,groups,name):
    indexed={row['group']:row for row in rows}
    a.check(len(indexed)==len(rows)==len(groups) and set(indexed)==set(groups),name)
    return indexed


def case_seed(seed,phase,epoch,group):
    payload='%d|%s|%d|%d|dual-surrogate-v1'%(seed,phase,epoch,group)
    return int(hashlib.sha256(payload.encode('ascii')).hexdigest()[:8],16)


def pair_counts(labels,clean,baseline,candidate):
    c=collections.Counter(occurrences=len(labels))
    for y,cc,ap,bp in zip(labels,clean,baseline,candidate):
        aw,bw,cw=ap!=y,bp!=y,cc!=y
        c['baseline_wrong']+=aw;c['candidate_wrong']+=bw
        if cw:
            c['additional_recovery']+=aw and not bw;c['previous_recovery_lost']+=bw and not aw
        else:
            c['shared_new_wrong']+=aw and bw;c['baseline_new_success_lost']+=aw and not bw
            c['new_success_gained']+=bw and not aw
    c['delta_wrong']=c['candidate_wrong']-c['baseline_wrong']
    return c


def construction_lineage(a,p):
    """Reconcile both pre-VALID failures and immutable teacher exactly once."""
    a.check(p['teacher_training_source']==SOURCE and p['teacher_preparation_protocol_sha256']==PREPARATION,
        'preserved_original_teacher_lineage')
    a.check(p['reuse_teacher_directory']==p['teacher_directory'],'exact_successful_teacher_reused')
    active=p;lineage=[]
    for expected_previous in (SECOND_SOURCE,SOURCE):
        amendment=active['construction_amendment']
        a.check(amendment['previous_source']==expected_previous,'recursive_previous_source')
        ref=active['prior_progress_ref'];prior=a.json(ref['path'],ref['sha256'])
        old_stage=a.path(pathlib.Path(ref['path']).parent.parent,directory=True)
        previous=a.json(old_stage/'refinement_protocol_v1.json',amendment['previous_protocol_sha256'])
        expected_preparation=SECOND_PREPARATION if expected_previous==SECOND_SOURCE else PREPARATION
        old_preparation=a.json(old_stage/'preparation_protocol_v1.json',expected_preparation)
        a.check(previous['source_commit']==old_preparation['source_commit']==expected_previous
            and previous['teacher_ref']==p['teacher_ref'],'recursive_previously_bound_teacher')
        a.check(set(previous['source_files'])==set(active['source_files']),'continuation_keeps_source_inventory')
        allowed=({'course/feature_polish/preflight.py','course/feature_polish/prepare_protocol.py','course/FEATURE_POLISH_PROTOCOL_V1.json'}
            if expected_previous==SECOND_SOURCE else {'course/feature_polish/refine.py','course/feature_polish/prepare_protocol.py','course/feature_polish/queue.py'})
        for name,wanted in previous['source_files'].items():
            original=subprocess.check_output(['git','show',expected_previous+':'+name],cwd=a.root)
            a.check(hashlib.sha256(original).hexdigest()==wanted,'previous_source_committed_hash')
            if name not in allowed:a.check(active['source_files'][name]==wanted,'construction_only_source_amendment_scope')
        previous_master=json.loads(subprocess.check_output(['git','show',expected_previous+':course/FEATURE_POLISH_PROTOCOL_V1.json'],cwd=a.root))
        current_master=json.loads(subprocess.check_output(['git','show',active['source_commit']+':course/FEATURE_POLISH_PROTOCOL_V1.json'],cwd=a.root))
        if expected_previous==SECOND_SOURCE:
            previous_master.pop('preflight');current_master.pop('preflight')
        a.check(previous_master==current_master,'scientific_method_controls_seeds_resources_unchanged')
        a.check(active['reuse_execution_resource']=={key:previous[key] for key in
            ('execution_device','cuda_visible_devices','physical_gpu_uuid','resource_selection')},'same_preselected_execution_resource')
        a.check(active['reuse_synthetic_ref']==p['reuse_synthetic_ref'],'original_synthetic_receipt_reused_recursively')
        failed_path=old_stage/'preflight_attempt1/result.json'
        failure=a.json(failed_path,SECOND_FAILURE if expected_previous==SECOND_SOURCE else None)
        a.check(failure['status']=='failed' and failure['official_victim_loaded'] is False,'construction_failure_preserved_no_victim')
        if expected_previous==SECOND_SOURCE:
            a.check(ref['sha256']==SECOND_PROGRESS and amendment['previous_protocol_sha256']==SECOND_REFINEMENT,
                'second_failure_protocol_and_progress_frozen_before_amendment')
            a.check([row['name'] for row in prior['jobs']]==['full_train_teacher','real_train_preflight','real_train_preflight']
                and [row['exit_code']==0 for row in prior['jobs']]==[True,False,False],
                'retained_teacher_and_both_pre_outcome_failures')
            expected=dict(generator_forwards=2,gaga_forward_dispatches=30,gaga_completed_forwards=30,
                gcn_forward_dispatches=20,gcn_completed_forwards=20,canonical_sequence_updates=6,autograd_calls=8,
                finite_difference_gaga_forwards=16,finite_difference_gcn_forwards=14,optimizer_steps=0,
                official_victim_forwards=0,API_calls=0,TEST_queries=0)
            a.check(failure['counts']==expected and [row['group'] for row in failure['cases']]==[263,238],
                'strict_fd_failure_actual_partial_work_not_zero')
            a.check('flat_coordinate' in failure['error'] and '69' in failure['error'] and 'passed' in failure['error'],
                'strict_fd_failure_reason_retained')
            for row in failure['cases']:
                for comparison in row['canonical_affine_comparisons']:
                    a.check(comparison['sequence']['passed'] and comparison['logits']['passed'],
                        'failed_fd_attempt_completed_sequence_checks_retained')
        else:
            a.check([row['name'] for row in prior['jobs']]==['full_train_teacher','real_train_preflight']
                and prior['jobs'][0]['exit_code']==0 and prior['jobs'][1]['exit_code']!=0,
                'original_success_and_zero_forward_failure')
            a.check(failure['counts'] and all(value==0 for value in failure['counts'].values()),
                'failed_cpu_checkpoint_startup_explicit_zero_work_retained')
        lineage.append(dict(source_commit=expected_previous,preparation_sha256=expected_preparation,
            refinement_sha256=amendment['previous_protocol_sha256'],failed_preflight_sha256=sha(failed_path),
            failed_preflight_counts=failure['counts'],prior_job_rows=len(prior['jobs'])))
        active=previous
    a.check('construction_amendment' not in active,'exact_two_preserved_construction_amendments')
    return lineage


def protocols(a,stage,source,preparation_sha):
    preparation=a.json(stage/'preparation_protocol_v1.json',preparation_sha)
    refinement=a.json(stage/'refinement_protocol_v1.json')
    experiment_path=stage/'experiment_protocol_v1.json';p=a.json(experiment_path)
    stripped=dict(p);preflight_ref=stripped.pop('preflight_ref')
    a.check(stripped==refinement,'experiment_adds_only_preflight_reference')
    a.check(preparation['status']=='frozen' and preparation['owner_execution_authorized'] is True,'frozen_owner_authorization')
    a.check(preparation['source_commit']==refinement['source_commit']==p['source_commit']==source,'same_execution_source')
    a.check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS,'fixed_five_arms_three_seeds')
    a.check(p['polish_settings']==dict(steps=20,alpha=.02),'fixed_terminal_twenty_step_rule')
    a.check(p['teacher_device']=='cpu' and p['teacher_settings']==dict(seed=20261103,epochs=38,batch_size=512,
        lr=.001,weight_decay=.0001,train_nodes=6595,reference_initial_state_sha256=INITIAL),'fixed_full_train_teacher')
    a.check(p['phase_gpu_hours']==2 and p['phase_total_wall_hours']==3 and p['cpu_threads']==8,'resource_limits')
    a.check(p['preflight']['max_gaga_forwards']==60 and p['preflight']['max_gcn_forwards']==184
        and p['preflight']['autograd_calls']==24 and p['preflight']['gcn_float32_forwards']==20
        and p['preflight']['gcn_float64_functional_forwards']==164
        and p['preflight']['gcn_functional_float64_grid']==[.001,.0005,.00025,.000125,.0000625],
        'frozen_amended_functional_diagnostic_registration')
    a.check(p['final_test_allowed'] is False and p['LLM_API_calls']==0,'no_final_test_or_api')
    a.check(len(p['train_groups'])==len(set(p['train_groups']))==256 and len(p['valid_groups'])==len(set(p['valid_groups']))==128
        and not set(p['train_groups'])&set(p['valid_groups']),'group_partition_and_counts')
    for name,wanted in p['source_files'].items():
        a.bytes(name,wanted)
        committed=subprocess.check_output(['git','show',source+':'+name],cwd=a.root)
        a.check(hashlib.sha256(committed).hexdigest()==wanted,'source_hash_matches_registered_commit')
    for row in p['inputs']+p.get('retained_inputs',[]):a.bytes(row['path'],row['sha256'])
    a.check(preparation['execution_device'] is None and preparation['cuda_visible_devices']==''
        and preparation['physical_gpu_uuid'] is None,'teacher_protocol_precedes_device_choice')
    selection=p['resource_selection'];metadata=selection['available_metadata']
    a.check(len(metadata)==len(p['allowed_gpus'])==2 and selection['selection_before_baseline_generation'] is True,'resource_metadata_before_generation')
    eligible=[]
    for measured,allowed in zip(metadata,p['allowed_gpus']):
        a.check(all(measured[k]==allowed[k] for k in ('index','uuid')) and type(measured['free_mib']) is int,'registered_gpu_metadata')
        if measured['free_mib']>=8192:eligible.append(allowed)
    if eligible:
        a.check(p['execution_device']=='cuda:0' and p['cuda_visible_devices']==eligible[0]['index']
            and p['physical_gpu_uuid']==eligible[0]['uuid'],'first_eligible_gpu_selected')
    else:a.check(p['execution_device']=='cpu' and p['cuda_visible_devices']=='' and p['physical_gpu_uuid'] is None,'uniform_cpu_fallback')
    construction=construction_lineage(a,p)
    gate=a.json(preflight_ref['path'],preflight_ref['sha256'])
    a.check(gate['status']=='passed' and gate['official_victim_loaded'] is False,'functional_preflight_passed')
    expected=dict(generator_forwards=4,gaga_forward_dispatches=60,gaga_completed_forwards=60,
        gcn_forward_dispatches=184,gcn_completed_forwards=184,canonical_sequence_updates=12,autograd_calls=24,
        finite_difference_gaga_forwards=32,finite_difference_gcn_forwards=160,
        gcn_native_sample_block_calls=8,gcn_fixed_block_sample_calls=4,gcn_fixed_float32_center_forwards=12,
        gcn_float64_center_forwards=4,gcn_float64_finite_difference_forwards=160,gcn_float64_clones=4,
        gcn_fixed_float32_autograd_calls=8,gcn_float64_autograd_calls=8,optimizer_steps=0,
        official_victim_forwards=0,API_calls=0,TEST_queries=0)
    a.check(gate['counts']==expected,'functional_preflight_exact_counters')
    a.check(gate['tolerances']==dict(sequence_atol=2e-5,sequence_rtol=2e-5,logit_atol=.0002,logit_rtol=.0002,
        finite_difference_h=.001,finite_difference_atol=.001,finite_difference_rtol=.03,
        gcn_functional_float64_grid=[.001,.0005,.00025,.000125,.0000625]),'original_tolerances_retained_in_functional_receipt')
    a.check(gate['provenance']==dict(source_commit=source,protocol_sha256=sha(stage/'refinement_protocol_v1.json')),'preflight_protocol_binding')
    a.check([row['group'] for row in gate['cases']]==[263,238,84,125],'preflight_fixed_train_groups')
    for row in gate['cases']:
        a.check(row['actual_injected_nodes']==row['node_budget'] and row['actual_nonself_undirected_edges']==row['edge_budget'],'preflight_actual_budget_counts')
        for comparison in row['canonical_affine_comparisons']:
            a.check(comparison['sequence']['passed'] and comparison['logits']['passed'],'canonical_affine_equivalence_pass')
        a.check(row['hard_gcn_x0_replacement']['passed'],'hard_gcn_feature_replacement_parity')
        gradient=row['gaga_feature_gradient']
        a.check(gradient['raw_margin_gradient_l2']>0 and gradient['zero_width_gradients_exact_zero'],'functional_gaga_feature_gradient_connected')
        a.check(len(gradient['coordinates'])==4 and gradient['finite_difference_h']==.001,'gaga_four_original_fd_coordinates')
        for coordinate in gradient['coordinates']:
            for check in coordinate['checks'].values():a.check(finite_difference_recount(a,check),'registered_gaga_feature_finite_difference')
        fixed_gcn_preflight(a,row['gcn_feature_gradient'])
    return p,preparation,dict(source_commit=source,protocol_sha256=sha(experiment_path)),construction


def teacher(a,stage,p,preparation_sha,source):
    directory=a.path(p['teacher_directory'],directory=True)
    registered=a.json(directory/'registered.json');result=a.json(directory/'result.json')
    manifest=a.json(p['teacher_ref']['path'],p['teacher_ref']['sha256'])
    a.check(result['status']==manifest['status']=='completed','teacher_completed')
    a.check(result['teacher_artifacts_sha256']==p['teacher_ref']['sha256'],'teacher_result_manifest_binding')
    expected_provenance=dict(source_commit=p['teacher_training_source'],protocol_sha256=p['teacher_preparation_protocol_sha256'])
    a.check(registered['provenance']==manifest['provenance']==expected_provenance,'teacher_preparation_provenance')
    a.check(manifest['execution_device']=='cpu' and manifest['settings']==dict(seed=20261103,epochs=38,batch_size=512,
        lr=.001,weight_decay=.0001,train_nodes=6595),'teacher_cpu_fixed_recipe')
    for row in (result,manifest,registered):
        a.check(row['official_victim_loaded'] is False and row['original_VALID_used'] is False and row['TEST_used'] is False,'teacher_declared_label_boundary')
    a.check(manifest['initial_state_sha256']==result['initial_state_sha256']==INITIAL
        and manifest['reference_initial_state_sha256']==INITIAL and manifest['initial_state_matches_reference'] is True,'teacher_same_random_initialization_receipt')
    a.check(manifest['final_state_sha256']==result['final_state_sha256'] and manifest['final_state_sha256']!=INITIAL,'teacher_recorded_state_changed')
    checkpoint=a.bytes(directory/manifest['checkpoint'],manifest['checkpoint_sha256'])
    a.check(result['checkpoint_sha256']==manifest['checkpoint_sha256'] and manifest['checkpoint_sha256']!=p['sealed_victim_sha256'],'fresh_teacher_checkpoint_not_victim_bytes')
    a.check(manifest['counters']==result['counters']==dict(optimizer_steps=494,training_forwards=494,final_train_forwards=13),'teacher_exact_update_forward_counts')
    a.check(manifest['full_train_nodes']==6595 and manifest['final_in_sample_train']['held_out'] is False
        and manifest['final_in_sample_train']['calibrated'] is False,'teacher_train_score_is_in_sample')
    predictions=a.jsonl(directory/'raw/final_in_sample_train_predictions.jsonl')
    a.check(len(predictions)==13,'teacher_thirteen_final_batches')
    final_ids=[];labels=[];raw_logits=[]
    for row in predictions:
        a.check(len(row['node_ids'])==len(row['labels'])==len(row['logits']),'teacher_final_batch_denominator')
        final_ids+=row['node_ids'];labels+=row['labels'];raw_logits+=row['logits']
    a.check(len(final_ids)==len(set(final_ids))==6595 and all(type(x) is int and x>=0 for x in final_ids),'teacher_final_train_id_cover')
    a.check(set(labels)=={0,1} and all(type(x) is int for x in labels),'teacher_final_binary_labels')
    a.check(hashlib.sha256(json.dumps(sorted(final_ids),separators=(',',':')).encode()).hexdigest()==manifest['train_ids_sha256'],'teacher_train_id_digest')
    truth=dict(zip(final_ids,labels));counts=[labels.count(0),labels.count(1)]
    a.check(counts==manifest['train_class_counts'],'teacher_class_counts')
    for measured,n in zip(manifest['class_weights'],counts):a.close(measured,6595/(2*n),'teacher_train_class_weight',atol=1e-6)
    prediction=classify(a,raw_logits,6595);confusion=[[0,0],[0,0]]
    for y,pred in zip(labels,prediction):confusion[y][pred]+=1
    score=manifest['final_in_sample_train']
    a.check(confusion==score['confusion_true_by_prediction'],'teacher_final_confusion_recount')
    a.close(score['accuracy'],sum(x==y for x,y in zip(labels,prediction))/6595,'teacher_final_accuracy_recount')
    losses=[softplus((row[1]-row[0])*(1 if y==0 else -1)) for row,y in zip(raw_logits,labels)]
    a.close(score['unweighted_cross_entropy'],statistics.fmean(losses),'teacher_final_unweighted_loss',atol=1e-5,rtol=1e-5)
    for epoch in range(1,39):
        batches=a.jsonl(directory/'raw'/('train_epoch_%03d.jsonl'%epoch));receipt=a.json(directory/('epoch_%03d.json'%epoch))
        a.check(len(batches)==13 and receipt['batches']==13,'teacher_thirteen_batches_per_epoch')
        positions=[];ids=[]
        for number,row in enumerate(batches,1):
            a.check(row['epoch']==epoch and row['batch']==number and row['targets']==len(row['positions'])==len(row['node_ids']),'teacher_batch_counts')
            a.check(math.isfinite(row['loss']) and row['loss']>=0 and 0<=row['correct']<=row['targets'],'teacher_batch_loss_correct_range')
            a.check(all(final_ids[position]==node for position,node in zip(row['positions'],row['node_ids'])),'teacher_position_identity_mapping')
            positions+=row['positions'];ids+=row['node_ids']
        a.check(sorted(positions)==list(range(6595)) and set(ids)==set(final_ids),'teacher_full_train_cover_every_epoch')
        a.check(receipt['epoch']==epoch and receipt['optimizer_steps']==receipt['training_forwards']==epoch*13
            and receipt['validation_performed'] is False,'teacher_epoch_terminal_counts_no_validation')
        a.close(receipt['mean_batch_loss'],statistics.fmean(row['loss'] for row in batches),'teacher_epoch_mean_batch_loss')
    a.check(manifest['cache']['sequence']['sha256']==sha(a.path(manifest['cache']['sequence']['path'])),'teacher_full_train_cache_binding')
    return dict(epochs=38,train_nodes=6595,optimizer_updates=494,training_forwards=494,final_in_sample_forwards=13,
        stored_truth_recounted=True,independent_dataset_truth_checked=False,teacher_checkpoint_bytes=checkpoint.stat().st_size,
        initial_state_receipt_matches_prior=True,post_refit_score_is_not_held_out=True)


def refinements(a,matrix,p,provenance):
    all_rows={};receipts={};loss_rows=[];feature_stats=[];guard=collections.Counter();counter_totals=collections.Counter()
    for seed in SEEDS:
        directory=matrix/('seed%d'%seed);registered=a.json(directory/'registered.json');result=a.json(directory/'result.json')
        a.check(result['status']=='completed' and result['seed']==seed and result['groups']==128
            and tuple(result['arms'])==ARMS and result['official_victim_loaded'] is False,'refinement_completed_five_arms')
        a.check(registered['provenance']==provenance and registered['seed']==seed and tuple(registered['arms'])==ARMS,'refinement_source_protocol_seed')
        a.check(result['source_commit']==provenance['source_commit'] and result['protocol_sha256']==provenance['protocol_sha256'],'refinement_result_provenance')
        a.check(registered['fixed_graph_all_arms'] is True and registered['official_victim_loaded'] is False and registered['TEST_queries']==0,'refinement_declared_no_victim_boundary')
        expected=dict(baseline_generations=128,gcn_forwards=3072,gaga_teacher_forwards=3072,
            feature_autograd_calls=5120,feature_pgd_steps=5120,generator_optimizer_steps=0,
            teacher_optimizer_steps=0,official_victim_forwards=0,API_calls=0,TEST_queries=0)
        a.check(result['counters']==expected and result['recorded_engine_generator_forwards']==128,'refinement_exact_counters')
        counter_totals.update(result['counters'])
        reference=p['baseline_runs'][str(seed)];baseline=a.json(pathlib.Path(reference['directory'])/'result.json',reference['result_sha256'])
        a.check(baseline['status']=='completed' and baseline['seed']==seed and baseline['arm']=='legacy'
            and baseline['source_mode']=='legacy' and baseline['epochs']==8 and baseline['optimizer_steps']==2048,'retained_legacy_terminal_baseline')
        a.bytes(pathlib.Path(reference['directory'])/'final_model.pt',reference['final_model_sha256'])
        a.check(result['generator_checkpoint_sha256']==registered['generator_checkpoint_sha256']==reference['final_model_sha256']
            and result['generator_state_sha256']==registered['generator_normalized_state_sha256']==baseline['final_model_state_sha256'],'retained_generator_exact_binding')
        a.check(result['teacher_ref']==registered['teacher_ref']==p['teacher_ref']
            and result['teacher_state_sha256']==registered['teacher_state_sha256']
            and result['original_gcn_state_sha256']==registered['original_gcn_state_sha256'],'recorded_frozen_classifier_states')
        own_path=directory/'groups.jsonl';a.bytes(own_path,result['groups_jsonl_sha256'])
        rows=keyed(a,a.jsonl(own_path),p['valid_groups'],'refinement_full_group_cover')
        exports={row['group']:row for row in result['exports']}
        a.check(len(exports)==len(result['exports'])==128 and set(exports)==set(rows),'refinement_capture_cover')
        guard_seed=collections.Counter();node_sum=edge_sum=occurrences=0
        for ordinal,group in enumerate(p['valid_groups'],1):
            row=rows[group];n=row['targets'];node_sum+=row['actual_injected_nodes'];edge_sum+=row['actual_injected_nonself_undirected_edges'];occurrences+=n
            a.check(row['seed']==seed and n==len(row['node_ids'])==len(row['labels']) and all(y==1 for y in row['labels']),'refinement_target_counts_labels')
            a.check(row['node_budget']==row['actual_injected_nodes'] and row['edge_budget']==row['actual_injected_nonself_undirected_edges'],'recorded_hard_budget_exact')
            a.check(row['generation_seed']==case_seed(seed,'validation',8,group)
                and row['random_seed']==case_seed(seed,'feature-polish-random',20,group),'registered_case_and_random_rng_keys')
            a.check(row['graph_capture']==exports[group],'capture_and_row_receipts_equal')
            relative=pathlib.Path(exports[group]['path']);a.check(not relative.is_absolute() and '..' not in relative.parts,'capture_relative_path')
            a.bytes(directory/relative,exports[group]['sha256'])
            a.check(set(row['final_own_logits'])==set(row['final_own_losses'])==set(row['feature_statistics'])==set(ARMS),'refinement_five_feature_logit_branches')
            baseline_pred=classify(a,row['final_own_logits']['baseline']['gcn'],n)
            candidate_pred=classify(a,row['final_own_logits']['gaga_pgd']['gcn'],n)
            protected=sum(value==0 for value in baseline_pred);lost=sum(b==0 and c!=0 for b,c in zip(baseline_pred,candidate_pred))
            accepted=lost==0;g=row['guard'];alias='gaga_pgd' if accepted else 'baseline'
            a.check(g['accepted']==accepted and g['protected_target_count']==protected and g['lost_protected_target_count']==lost
                and g['candidate_benign_count']==sum(value==0 for value in candidate_pred)
                and g['empty_protected_set']==(protected==0),'guard_raw_gcn_decision_recount')
            a.check(g['uses_evaluation_labels'] is False and g['uses_official_victim'] is False and g['step_selection'] is False,'guard_declared_information_boundary')
            a.check(row['final_own_logits']['gaga_guarded']==row['final_own_logits'][alias]
                and row['feature_statistics']['gaga_guarded']['content_sha256']==row['feature_statistics'][alias]['content_sha256'],'guarded_whole_group_alias')
            selected=classify(a,row['final_own_logits']['gaga_guarded']['gcn'],n)
            a.check(all(b!=0 or c==0 for b,c in zip(baseline_pred,selected)),'guarded_recorded_gcn_zeros_preserved')
            guard_seed['accepted' if accepted else 'fallback']+=1;guard_seed['empty_protected_set']+=int(protected==0)
            for arm in ARMS:
                stats=row['feature_statistics'][arm]
                for value in stats.values():
                    if isinstance(value,float):a.check(math.isfinite(value),'finite_feature_statistic')
                a.check(0<=stats['normalized_displacement_linf']<=.40001 and 0<=stats['saturation_fraction']<=1
                    and stats['movable_coordinates']>0,'recorded_box_movement_saturation_ranges')
                if arm=='baseline':a.check(stats['normalized_displacement_linf']==stats['normalized_displacement_l2']==0,'baseline_has_zero_feature_change')
                feature_stats.append(dict(seed=seed,group=group,arm=arm,**stats))
                for role in ('gcn','gaga_teacher'):
                    logits=row['final_own_logits'][arm][role];classify(a,logits,n)
                    measured=statistics.fmean(softplus(x[1]-x[0]) for x in logits)
                    a.close(row['final_own_losses'][arm][role],measured,'own_softplus_raw_logit_recount',atol=1e-5,rtol=1e-5)
                    loss_rows.append(dict(arm=arm,seed=seed,role=role,before=row['final_own_losses']['baseline'][role],after=row['final_own_losses'][arm][role]))
            history=a.json(directory/'histories'/('group_%d.json'%group))
            a.check(history['group']==group and history['terminal_only'] is True,'terminal_only_feature_history')
            for arm in ('gcn_pgd','gaga_pgd'):
                steps=history[arm];a.check([step['step'] for step in steps]==list(range(1,21)),'exact_twenty_feature_steps')
                for step in steps:
                    a.check(all(math.isfinite(step[k]) and step[k]>=0 for k in ('loss_before_update','gradient_linf','gradient_l2','normalized_update_linf'))
                        and 0<=step['normalized_update_linf']<=.020002 and 0<=step['zero_gradient_coordinates']<=row['node_budget']*768,'feature_step_finite_bounds')
            expected_cumulative=dict(expected)
            for name,value in expected_cumulative.items():
                if name in ('baseline_generations',):expected_cumulative[name]=ordinal
                elif name in ('gcn_forwards','gaga_teacher_forwards'):expected_cumulative[name]=ordinal*24
                elif name in ('feature_autograd_calls','feature_pgd_steps'):expected_cumulative[name]=ordinal*40
            a.check(row['counters']==expected_cumulative,'per_group_cumulative_call_counts')
        a.check((node_sum,edge_sum,occurrences)==(7776,358491,1091),'refinement_summed_scenario_budgets_and_occurrences')
        a.check(result['guard_counts']==dict(accepted=guard_seed['accepted'],fallback=guard_seed['fallback'],empty_protected_set=guard_seed['empty_protected_set']),'guard_seed_totals')
        guard.update(guard_seed);all_rows[seed]=rows;receipts[seed]=result
    return all_rows,receipts,loss_rows,feature_stats,dict(guard),dict(counter_totals)


def evaluation(a,stage,p,provenance,own,loss_rows):
    directory=stage/'evaluation_attempt1';result=a.json(directory/'result.json');registered=a.json(directory/'registered.json')
    a.check(result['status']=='completed' and result['provenance']==registered['provenance']==provenance,'completed_official_evaluation_provenance')
    a.check(registered['all3_refinements_complete_before_victim_load'] is True and registered['final_TEST_used'] is False
        and result['final_TEST_used'] is False and result['TEST_queries']==0 and result['experimental_API_calls']==0,'official_evaluation_registered_gate_no_test')
    expected=dict(official_gaga_forwards=2048,original_gcn_clean_forwards=128,attacked_gcn_rows_reused=1920,
        hard_graph_reconstructions=384,canonical_sequence_updates=1920,generator_forwards=0)
    a.check(result['counts']==expected,'official_evaluation_exact_query_and_reuse_counts')
    clean=keyed(a,a.jsonl(directory/'clean_predictions.jsonl'),p['valid_groups'],'clean_group_cover')
    clean_pred={};unique=set();multiplicity=collections.Counter()
    for group,row in clean.items():
        labels=row['labels'];ids=row['node_ids']
        a.check(len(ids)==len(labels)>0 and len(ids)==len(set(ids)) and all(y==1 for y in labels),'clean_fraud_target_denominator')
        unique.update(ids);multiplicity.update(ids)
        for detector in ('gcn','gaga'):clean_pred[group,detector]=classify(a,row[detector+'_logits'],len(labels))
    a.check(len(unique)==374 and sum(len(row['labels']) for row in clean.values())==1091,'clean_unique_occurrence_denominators')
    summaries={(r['arm'],r['seed']):r for r in result['summaries']}
    a.check(len(summaries)==len(result['summaries'])==15 and set(summaries)=={(arm,seed) for arm in ARMS for seed in SEEDS},'fifteen_complete_official_summaries')
    attacked={};metrics=[];same_input_checks=0;max_same_input_logit_difference=0.;same_input_class_differences=0
    for seed in SEEDS:
        for arm in ARMS:
            rows=keyed(a,a.jsonl(directory/('%s_seed%d_predictions.jsonl'%(arm,seed))),p['valid_groups'],'full_official_attacked_group_cover')
            group_receipts=keyed(a,a.json(directory/('%s_seed%d_groups.json'%(arm,seed))),p['valid_groups'],'official_group_receipt_cover')
            sums={name:collections.Counter() for name in ('gcn','gaga')};macro={name:[] for name in ('gcn','gaga')}
            for group,row in rows.items():
                baseline=clean[group];source=own[seed][group];n=len(baseline['labels'])
                a.check(row['arm']==arm and row['seed']==seed and row['node_ids']==baseline['node_ids']==source['node_ids']
                    and row['labels']==baseline['labels']==source['labels'],'clean_refinement_evaluation_identity_order')
                a.check(row['common_capture_sha256']==source['graph_capture']['sha256']
                    and row['hard_adjacency_content_sha256']==source['adjacency_content_sha256']
                    and row['injected_feature_content_sha256']==source['feature_statistics'][arm]['content_sha256'],'shared_graph_and_branch_feature_binding')
                a.check(row['actual_injected_nodes']==row['node_budget']==source['node_budget']
                    and row['actual_injected_edges']==row['edge_budget']==source['edge_budget'],'official_actual_node_edge_budgets')
                a.check(row['gcn_logits_reused'] is True and row['gcn_logits']==source['final_own_logits'][arm]['gcn'],'attacked_gcn_channel_exact_reuse')
                stripped={k:v for k,v in row.items() if k not in ('node_ids','labels','gcn_logits','gaga_logits')}
                a.check(group_receipts[group]==stripped,'official_raw_group_receipt_identity')
                for detector in ('gcn','gaga'):
                    pred=classify(a,row[detector+'_logits'],n);count=transitions(baseline['labels'],clean_pred[group,detector],pred)
                    a.check(row[detector+'_transitions']==count,'official_raw_argmax_transition_recount')
                    sums[detector].update(count);macro[detector].append(100*count['attacked_wrong']/n)
            a.check(sum(row['actual_injected_nodes'] for row in rows.values())==7776
                and sum(row['actual_injected_edges'] for row in rows.values())==358491,'official_summed_separate_scenario_budgets')
            summary=summaries[arm,seed]
            a.check(summary['groups']==128 and summary['injected_nodes']==7776 and summary['injected_edges']==358491
                and summary['stored_gcn_channel'] is True,'official_summary_scope_and_budgets')
            for detector in ('gcn','gaga'):
                totals={key:sums[detector][key] for key in FIELDS};a.check(totals['occurrences']==1091 and totals==summary['totals'][detector],'official_summary_raw_count_identity')
                mr=100*totals['attacked_wrong']/1091;a.close(summary[detector+'_mr_percent'],mr,'official_summary_mr_identity')
                metrics.append(dict(arm=arm,seed=seed,detector=detector,**totals,mr_percent=mr,group_macro_mr_percent=statistics.fmean(macro[detector])))
            attacked[arm,seed]=rows
        for group in p['valid_groups']:
            selected={}
            for arm in ARMS:
                row=attacked[arm,seed][group];key=row['injected_feature_content_sha256']
                if key in selected:
                    other=selected[key];same_input_checks+=1
                    left,right=row['gaga_logits'],other['gaga_logits']
                    max_same_input_logit_difference=max(max_same_input_logit_difference,max(abs(x-y) for arow,brow in zip(left,right) for x,y in zip(arow,brow)))
                    same_input_class_differences+=sum(x!=y for x,y in zip(classify(a,left),classify(a,right)))
                else:selected[key]=row
    if same_input_class_differences:a.warnings.append('Same capture/feature identity produced different official-GAGA classes in repeated calls; this limits interpretation of small gate effects.')
    lookup={(row['arm'],row['seed'],row['detector']):row for row in metrics}
    aggregate={detector:{arm:dict(mean_mr_percent=statistics.fmean(lookup[arm,seed,detector]['mr_percent'] for seed in SEEDS),
        sample_sd_pp=statistics.stdev(lookup[arm,seed,detector]['mr_percent'] for seed in SEEDS),
        mean_group_macro_mr_percent=statistics.fmean(lookup[arm,seed,detector]['group_macro_mr_percent'] for seed in SEEDS),
        pooled_wrong=sum(lookup[arm,seed,detector]['attacked_wrong'] for seed in SEEDS)) for arm in ARMS} for detector in ('gaga','gcn')}
    comparisons={};group_pairs=[]
    critical=math.sqrt(2)*.95/math.sqrt(1-.95**2)
    for arm in ARMS[1:]:
        comparisons[arm]={}
        for detector in ('gaga','gcn'):
            deltas=[];pairs=[];total=collections.Counter()
            for seed in SEEDS:
                delta=lookup[arm,seed,detector]['mr_percent']-lookup['baseline',seed,detector]['mr_percent']
                deltas.append(dict(seed=seed,delta_mr_pp=delta,delta_group_macro_pp=lookup[arm,seed,detector]['group_macro_mr_percent']-lookup['baseline',seed,detector]['group_macro_mr_percent']))
                for group in p['valid_groups']:
                    base=attacked['baseline',seed][group];candidate=attacked[arm,seed][group]
                    a.check(all(base[k]==candidate[k] for k in ('node_ids','labels','node_budget','edge_budget','common_capture_sha256','hard_adjacency_content_sha256')),'paired_common_graph_identity_budget')
                    c=pair_counts(base['labels'],clean_pred[group,detector],classify(a,base[detector+'_logits']),classify(a,candidate[detector+'_logits']))
                    a.check(c['delta_wrong']==c['new_success_gained']-c['baseline_new_success_lost']-c['additional_recovery']+c['previous_recovery_lost'],'paired_success_transition_algebra')
                    c['group_wins' if c['delta_wrong']>0 else 'group_losses' if c['delta_wrong']<0 else 'group_ties']=1
                    total.update(c);pair=dict(arm=arm,detector=detector,seed=seed,group=group,node_budget=base['node_budget'],edge_budget=base['edge_budget'],**dict(c))
                    pairs.append(pair);group_pairs.append(pair)
            values=[row['delta_mr_pp'] for row in deltas];mean=statistics.fmean(values);sd=statistics.stdev(values);half=critical*sd/math.sqrt(3)
            strata=[]
            for regime,n,cases in [('one_node_one_edge',48,12),('one_node_multiple_edges',165,45),('multiple_nodes',3060,327)]:
                def membership(row):return 'multiple_nodes' if row['node_budget']>1 else 'one_node_one_edge' if row['edge_budget']==1 else 'one_node_multiple_edges'
                selected=[row for row in pairs if membership(row)==regime];counts=collections.Counter()
                for row in selected:counts.update({key:row.get(key,0) for key in PAIR_FIELDS+('delta_wrong','group_wins','group_ties','group_losses')})
                a.check(len(selected)==cases and counts['occurrences']==n,'fixed_disjoint_budget_strata')
                strata.append(dict(regime=regime,group_seed_cases=cases,**dict(counts),delta_mr_pp=100*counts['delta_wrong']/n))
            a.check(total['occurrences']==3273 and len(pairs)==384,'paired_correlated_occurrence_denominator')
            comparisons[arm][detector]=dict(paired_seed_differences=deltas,mean_delta_mr_pp=mean,sample_sd_delta_pp=sd,
                paired_t95_pp=[mean-half,mean+half],attack_seeds=3,degrees_of_freedom=2,
                consistent_development_gain=mean>=1 and all(value>0 for value in values),
                paired_transition_counts={key:total[key] for key in PAIR_FIELDS+('delta_wrong','group_wins','group_ties','group_losses')},budget_strata=strata)
    def candidate(arm):
        mean=aggregate['gaga'][arm]['mean_mr_percent'];controls={key:mean-aggregate['gaga'][key]['mean_mr_percent'] for key in ('gcn_pgd','random')}
        own_delta={str(seed):statistics.fmean(row['after']-row['before'] for row in loss_rows if row['arm']==arm and row['seed']==seed and row['role']=='gaga_teacher') for seed in SEEDS}
        consistent=comparisons[arm]['gaga']['consistent_development_gain'];own_improved=all(value<0 for value in own_delta.values());better=all(value>0 for value in controls.values())
        return dict(arm=arm,consistency_criterion_met=consistent,own_gaga_mean_group_loss_delta_by_seed=own_delta,
            own_gaga_objective_improved_each_seed=own_improved,mean_advantage_over_controls_pp=controls,
            better_mean_than_both_controls=better,complete_predeclared_development_rule_met=consistent and own_improved and better)
    return dict(metrics=metrics,aggregate_by_detector=aggregate,comparisons_against_new_baseline=comparisons,
        primary_development_rule=candidate('gaga_guarded'),predeclared_secondary_candidate=candidate('gaga_pgd'),
        secondary_never_replaces_primary=True,groups=128,unique_targets=374,correlated_occurrences_per_arm=3273,
        repeated_targets=sum(value>1 for value in multiplicity.values()),max_group_memberships=max(multiplicity.values()),
        identical_input_official_gaga=dict(comparisons=same_input_checks,max_logit_difference=max_same_input_logit_difference,class_differences=same_input_class_differences)),group_pairs


def queue_cost(a,stage,queue_directory,p,preparation_sha,provenance):
    complete=a.json(stage/'queue_complete.json');mirror=a.json(queue_directory/'queue_complete.json');progress=a.json(queue_directory/'queue_progress.json')
    a.check(complete==mirror and complete['status']=='completed','queue_completion_receipt_identity')
    a.check(complete['scientific_source']==provenance['source_commit'] and complete['preparation_protocol_sha256']==preparation_sha
        and complete['experiment_protocol_sha256']==provenance['protocol_sha256'],'queue_source_protocol_binding')
    a.check(complete['TEST_queries']==complete['experimental_API_calls']==0,'queue_no_test_or_api')
    a.bytes(stage/'evaluation_attempt1/result.json',complete['evaluation_result_sha256'])
    continued='prior_progress_ref' in p
    expected=['full_train_teacher','real_train_preflight','real_train_preflight','real_train_preflight','refine_seed3','refine_seed4','refine_seed5','development_evaluation']
    jobs=complete['jobs'];a.check([row['name'] for row in jobs]==expected and progress['jobs']==jobs,'exclusive_serial_job_order_with_retained_failure')
    if continued:
        prior=a.json(p['prior_progress_ref']['path'],p['prior_progress_ref']['sha256'])
        old_queue=pathlib.Path(p['prior_progress_ref']['path']).parent
        inherited=[dict(row,guardian_path=(old_queue/(row['name']+'_guardian.json')).as_posix())
            if 'guardian_path' not in row else row for row in prior['jobs']]
        a.check(len(inherited)==3 and jobs[:3]==inherited,'prior_success_and_two_failures_not_reexecuted_or_rewritten')
    marker=a.json(stage/'all3_refinements_complete.json');a.check(marker['status']=='completed' and marker['jobs']==jobs[:-1],'all_refinements_marker_precedes_victim')
    registered=a.json(stage/'evaluation_attempt1/registered.json');a.check(registered['marker_sha256']==sha(stage/'all3_refinements_complete.json'),'victim_gate_marker_hash')
    for seed in SEEDS:
        a.check(marker['refinement_results'][str(seed)]==registered['refinement_result_sha256s'][str(seed)]
            ==sha(queue_directory/'refinements'/('seed%d'%seed)/'result.json'),'marker_refinement_result_hashes')
    charge=0.;guardians=[];previous_end=None
    for ordinal,job in enumerate(jobs):
        failed=continued and ordinal in (1,2)
        a.check((job['exit_code']!=0 if failed else job['exit_code']==0) and job['outer_seconds']>0,'job_exit_and_outer_clock')
        g=a.json(job.get('guardian_path') or queue_directory/(job['name']+'_guardian.json'),job['guardian_sha256'])
        a.check(g['status']==('failed' if failed else 'completed') and g['module']==job['module'],'guardian_status_module')
        if not failed:a.check(g.get('exit_code',0)==0,'successful_guardian_exit')
        elif ordinal==1:
            # The module catches its own exception and exits1, so the guardian
            # can lack an error string. Exact original module result is checked
            # in construction_lineage; retain the nonzero guardian status.
            a.check(job['exit_code']!=0,'preserved_cpu_cuda_deserialization_failure')
        else:a.check(job['exit_code']!=0,'preserved_strict_finite_difference_failure')
        cpu=job['name']=='full_train_teacher' or p['execution_device']=='cpu'
        expected_device='cpu' if cpu else 'cuda:0'
        a.check(job['execution_device']==g['execution_device']==expected_device,'single_registered_execution_device')
        a.check(g['torch']=='1.13.1+cu117' and g['elapsed_seconds']>0 and g['process_cpu_seconds']>=0,'locked_torch_and_actual_clocks')
        if cpu:
            a.check(g['numeric_visibility']=='' and g['torch_visible_count']==g['native_visible_count']==0 and g['own_physical_uuids']==[],'cpu_has_no_visible_cuda_devices')
            a.check(job['charged_gpu_wall_hours']==g['charged_gpu_wall_hours']==0,'cpu_gpu_charge_zero')
            cost=0.
        else:
            a.check(g['numeric_visibility']==p['cuda_visible_devices'] and g['torch_visible_count']==g['native_visible_count']==1
                and g['own_physical_uuids']==[p['physical_gpu_uuid']],'own_physical_gpu_verified')
            a.close(g['charged_gpu_wall_hours'],g['elapsed_seconds']/3600,'guardian_gpu_lifetime_charge')
            cost=max(job['outer_seconds']/3600,g['charged_gpu_wall_hours']);a.close(job['charged_gpu_wall_hours'],cost,'maximum_outer_guardian_gpu_charge')
        limit=p['teacher_wall_seconds'] if job['name']=='full_train_teacher' else p['preflight_wall_seconds'] if job['name']=='real_train_preflight' else p['evaluation_wall_seconds'] if job['name']=='development_evaluation' else p['refine_job_wall_seconds']
        a.check(g['limit_seconds']==limit and g['elapsed_seconds']<=limit+15,'registered_job_deadline')
        opened=dt.datetime.fromisoformat(g['started_utc']).timestamp()
        if previous_end is not None and opened<previous_end-1:a.warnings.append('Wall-clock guardian timestamps overlap; monotonic queue lifetimes/source are authoritative.')
        previous_end=opened+g['elapsed_seconds'];charge+=cost
        guardians.append(dict(name=job['name'],status='failed' if failed else 'completed',device=expected_device,
            outer_seconds=job['outer_seconds'],guardian_wall_seconds=g['elapsed_seconds'],process_cpu_seconds=g['process_cpu_seconds'],charged_gpu_wall_hours=cost))
    a.close(complete['charged_gpu_hours'],charge,'complete_gpu_charge_reconciled');a.close(progress['charged_gpu_hours'],charge,'progress_gpu_charge_reconciled')
    a.check(charge<=2 and 0<complete['elapsed_wall_seconds']<=3*3600,'whole_phase_gpu_and_wall_bounds')
    a.close(marker['charged_gpu_hours'],sum(row['charged_gpu_wall_hours'] for row in jobs[:-1]),'pre_evaluation_marker_gpu_charge')
    synthetic_ref=p.get('reuse_synthetic_ref')
    synthetic_path=pathlib.Path(synthetic_ref['path']) if synthetic_ref else queue_directory/'synthetic_receipt.json'
    synthetic=a.json(synthetic_path,synthetic_ref['sha256'] if synthetic_ref else None)
    synthetic_source=p['teacher_training_source'] if continued else p['source_commit']
    a.check(synthetic['status']=='passed' and synthetic['exit_code']==0 and synthetic['GPU_hours']==0
        and synthetic['source_commit']==synthetic_source,'synthetic_cpu_checks_passed')
    a.bytes(synthetic_path.parent/'synthetic_checks.log',synthetic['stdout_sha256'])
    if synthetic_ref:
        reused=a.json(queue_directory/'synthetic_reused.json')
        a.check(reused==dict(reference=synthetic_ref,new_execution=False),'unchanged_synthetic_receipt_reused_without_retest')
    return dict(charged_gpu_hours=charge,phase_gpu_limit=2,total_phase_wall_seconds=complete['elapsed_wall_seconds'],phase_wall_limit_hours=3,
        selected_execution_device=p['execution_device'],teacher_device='cpu',guardians=guardians,
        successful_teacher_fits=1,preserved_failed_zero_forward_cpu_preflights=1,
        preserved_failed_partial_train_finite_difference_preflights=1,
        CPU_process_time_can_exceed_wall_time_with_threads=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--stage',required=True);parser.add_argument('--queue',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--audit-source-sha256',required=True)
    parser.add_argument('--execution-source',required=True);parser.add_argument('--preparation-protocol-sha256',required=True)
    args=parser.parse_args();root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
    out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh audit output must be below ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);wall,cpu=time.monotonic(),time.process_time();a=Audit(root)
    result=dict(status='failed',schema='feature_polish_independent_saved_audit_v2',independent_of_method_implementation_authorship=True,human_verification=False,
        model_calls=0,tensor_deserializations=0,dataset_loads=0,graph_loads=0,API_calls=0,GPU_hours=0)
    try:
        own=sha(__file__);a.check(own==args.audit_source_sha256,'externally_frozen_audit_source')
        write(out/'registered.json',dict(audit_source_sha256=own,expected_execution_source=args.execution_source,
            expected_preparation_sha256=args.preparation_protocol_sha256,registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            new_outcome_reads_started=False,scope='Saved JSON and opaque bytes only; no ML imports or tensor/dataset loads'))
        stage=a.path(args.stage,directory=True);queue=a.path(args.queue,directory=True)
        p,preparation,provenance,construction=protocols(a,stage,args.execution_source,args.preparation_protocol_sha256)
        training=teacher(a,stage,p,args.preparation_protocol_sha256,args.execution_source)
        own_rows,refine_receipts,loss_rows,feature_stats,guards,counters=refinements(a,queue/'refinements',p,provenance)
        measured,pairs=evaluation(a,stage,p,provenance,own_rows,loss_rows)
        cost=queue_cost(a,stage,queue,p,args.preparation_protocol_sha256,provenance)
        write(out/'private_group_pairs.json',pairs)
        result.update(status='passed',provenance=provenance,preserved_construction_attempts=construction,teacher_training=training,evaluation=measured,
            refinement_recorded_counters=counters,guard_counts=guards,cost=cost,
            capture_files_byte_checked=384,refinement_groups=384,feature_branches=1920,
            limitations=['Saved truth/state/counter declarations are reconciled; classifier weights are hashed as opaque bytes, never replayed.',
                'The separate tensor audit is required for an independent check of capture topology/box/alias facts; this JSON audit does not deserialize captures.',
                'No independent human verification, model inference, generalization/novelty proof, or final TEST evaluation is performed.',
                'Three attack seeds and repeated target memberships do not quantify uncertainty over fresh teachers or independent news datasets.',
                'Fixed-block float64 finite differences and activation-pattern checks validate an isolated functional operator; actual refinement remains float32 with its original sampler. Whole-gradient comparisons and frozen weight casts are reconciled from producer receipts, not replayed here.',
                'Primary and prespecified secondary/control contrasts remain separate exploratory development results, not familywise confirmatory evidence.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(check_counts=dict(a.checks),actual_byte_hashed_files=len(a.files),warnings=a.warnings,
            process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'input_hashes.json',[dict(path=name,sha256=value) for name,value in sorted(a.files.items())])
        write(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
