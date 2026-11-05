"""Saved-JSON-only candidate case reporting, source frozen before outcome reads.

Previous studies and the earlier ID-only cohorts are known. This is
producer-author reporting, not an independent method audit or a claim that the
adaptively inspected development project is confirmatorily preregistered.
No tensors, NumPy, original data arrays, checkpoints, models or APIs are used.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import time
import traceback


ARMS = ('baseline', 'balanced_target_fill', 'random_target_fill', 'candidate_rewire')
SEEDS = (3, 4, 5)
DETECTORS = ('gcn', 'gaga')
FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
PAIR_FIELDS = ('occurrences', 'gained_wrong', 'lost_wrong', 'new_success_gained',
    'baseline_new_success_lost', 'additional_recovery', 'previous_recovery_lost',
    'group_wins', 'group_ties', 'group_losses')
PAIRS = tuple((arm, 'baseline') for arm in ARMS[1:])+(
    ('balanced_target_fill', 'candidate_rewire'), ('balanced_target_fill', 'random_target_fill'))
FIXED_EXAMPLE_ORDINALS = (0, 64, 127)
T95_DF2 = math.sqrt(2)*.95/math.sqrt(1-.95**2)



def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def compact_hash(value):
    return hashlib.sha256(json.dumps(value,separators=(',', ':'),ensure_ascii=True).encode('ascii')).hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser()
    for name in ('protocol','stage','audit','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    protocol_path,stage,audit_path,output=(pathlib.Path(getattr(args,name)).resolve()
                                         for name in ('protocol','stage','audit','output'))
    output.mkdir(parents=True,exist_ok=False)
    cpu_started,wall_started=time.process_time(),time.monotonic()
    checks=0;inputs={};cache={}
    receipt=dict(status='failed',source_sha256=sha(__file__),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        producer_author_reporting=True,independent_method_audit=False,human_verification=False,
        reporting_model_queries=0,reporting_tensor_loads=0,reporting_NumPy_imports=0,
        reporting_dataset_loads=0,reporting_checkpoint_loads=0,reporting_API_calls=0,reporting_GPU_hours=0)
    write(output/'registered_reporting_scope.json',dict(source_sha256=receipt['source_sha256'],
        started_utc=receipt['started_utc'],new_saved_JSON_input_reads_started=False,
        protocol_path=str(protocol_path),stage=str(stage),audit_path=str(audit_path),
        fixed_example_ordinals=list(FIXED_EXAMPLE_ORDINALS),
        scope='Saved JSON/JSONL only; no tensors, NumPy, datasets, checkpoints, model calls or APIs. Raw node IDs/logits never emitted. Private group mapping kept separately.',
        attribution='Producer-author reporting, not an independent method audit or human verification. External root registration binds this source SHA before current cohort/model calls.'))

    def require(condition,message):
        nonlocal checks
        checks+=1
        if not condition:raise AssertionError(message)

    def load(path,jsonl=False):
        path=pathlib.Path(path).resolve()
        require(path.suffix in ('.json','.jsonl'),'Reporting reads saved JSON text only.')
        if path not in cache:
            raw=path.read_bytes();inputs[str(path)]=hashlib.sha256(raw).hexdigest();text=raw.decode('utf-8')
            cache[path]=[json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)
        return cache[path]

    def keyed(rows,key,expected=None):
        result={}
        for row in rows:
            identity=key(row);require(identity not in result,'Duplicate saved row identity.');result[identity]=row
        if expected is not None:require(set(result)==set(expected),'Saved cohort differs.')
        return result

    def close(actual,expected,message):
        x,y=float(actual),float(expected)
        require(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-9+1e-8*abs(y),message)

    def argmax(logits,count):
        require(len(logits)==count,'Logit denominator differs.')
        values=[]
        for pair in logits:
            require(len(pair)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in pair),'Invalid binary logits.')
            values.append(0 if pair[0]>=pair[1] else 1)
        return values

    def transition(labels,clean,attacked):
        require(len(labels)==len(clean)==len(attacked),'Transition lengths differ.')
        values=dict(occurrences=len(labels),clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0)
        for y,c,a in zip(labels,clean,attacked):
            values['clean_wrong']+=int(c!=y);values['attacked_wrong']+=int(a!=y)
            values['new_wrong']+=int(c==y and a!=y);values['recoveries']+=int(c!=y and a==y)
        require(values['attacked_wrong']==values['clean_wrong']+values['new_wrong']-values['recoveries'],
                'Transition identity fails.')
        return values

    def paired(labels,clean,reference,candidate):
        values={name:0 for name in PAIR_FIELDS}
        delta=sum(x!=y for x,y in zip(candidate,labels))-sum(x!=y for x,y in zip(reference,labels))
        values['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']=1
        for y,c,old,new in zip(labels,clean,reference,candidate):
            values['occurrences']+=1;values['gained_wrong']+=int(old==y and new!=y);values['lost_wrong']+=int(old!=y and new==y)
            if c==y:
                values['new_success_gained']+=int(old==y and new!=y);values['baseline_new_success_lost']+=int(old!=y and new==y)
            else:
                values['additional_recovery']+=int(old!=y and new==y);values['previous_recovery_lost']+=int(old==y and new!=y)
        require(delta==values['gained_wrong']-values['lost_wrong']==values['new_success_gained']-
                values['baseline_new_success_lost']-values['additional_recovery']+values['previous_recovery_lost'],
                'Paired gain/loss/recovery identity fails.')
        return dict(delta_wrong=delta,delta_mr_pp=100*delta/len(labels),**values)

    def case_key(seed,group):
        return hashlib.sha256(('candidate-case-report-v1|%d|%d'%(seed,group)).encode('ascii')).hexdigest()[:20]

    try:
        p=load(protocol_path);producer=load(stage/'result.json');audit=load(audit_path)
        require(p['schema']=='monti_candidate_target_reallocation_v1' and p['status']=='frozen' and p['post_hoc'] is True,
                'Require the frozen post-hoc candidate registration.')
        require(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['scientific_version']=='candidate-target-reallocation-v1',
                'Fixed arms/seeds/scientific version differ.')
        qualification=p['qualification']
        require(qualification['primary']=='balanced_target_fill' and qualification['min_mean_gain_pp']==1 and
            all(qualification[name] is True for name in ('require_all_seed_deltas_positive','require_paired_t95_lower_above_zero',
                'require_mean_above_candidate_rewire','random_target_comparator_need_not_be_worse','controls_not_candidates')),
            'Frozen role-allocation criterion differs.')
        require(producer['status'] in ('completed','completed_no_headroom') and audit['status']=='passed' and
                audit['schema']=='independent_candidate_target_reallocation_audit_v1',
                'Require a complete/planned-stop producer and passed separate audit.')
        require(producer['provenance']==audit['provenance']==dict(source_commit=p['source_commit'],protocol_sha256=inputs[str(protocol_path)])
                and producer['counters']==audit['producer_counters'] and audit['producer_status']==producer['status'],
                'Producer/audit/protocol provenance or counters differ.')
        require(audit['cohort_metadata']==p['cohort_metadata'],'Audited dynamic cohort metadata differs.')
        for name,wanted in producer['output_hashes'].items():
            require(not pathlib.Path(name).is_absolute() and '..' not in pathlib.Path(name).parts,'Unsafe saved output reference.')
            require(pathlib.Path(name).suffix in ('.json','.jsonl'),'Unexpected binary in producer text-output manifest.')
            load(stage/name,name.endswith('.jsonl'))
            require(inputs[str((stage/name).resolve())]==wanted,'A producer JSON output byte hash differs.')
        registration=load(stage/'registered.json')
        require(registration['valid_groups']==p['valid_groups'] and registration['excluded_valid_groups']==p['excluded_valid_groups']
                and registration['selection']==p['selection'],'Producer registration cohort differs.')
        require(registration['own_source_sha256']==p['source_files']['course/candidate_target_reallocation/run.py'] and
                registration['base_experiment_ref']==p['base_experiment_ref'] and registration['arms']==p['arms'] and
                registration['scientific_version']==p['scientific_version'],'Executed runner source/lineage differs.')
        require(registration['source_and_input_hashes']==load(stage/'input_hashes.json'),'Executed input receipts differ.')
        declared_refs=dict(p['source_files'])
        declared_refs.update(p['excluded_sharpen_source_files'])
        for ref in p['inputs']+p.get('retained_inputs',[]):
            require(declared_refs.setdefault(ref['path'],ref['sha256'])==ref['sha256'],'Conflicting declared input hashes.')
        require(all(registration['source_and_input_hashes'].get(name)==value for name,value in declared_refs.items()),
                'Registered source/input byte references disagree with the executed receipt.')
        groups=list(p['valid_groups']);cm=p['cohort_metadata'];n_occ=cm['target_occurrences_per_seed'];n_unique=cm['unique_targets']
        require(len(groups)==len(set(groups))==128 and not set(groups)&set(p['excluded_valid_groups']),
                'Exactly128 newly selected original VALID contexts must exclude both earlier128-context lists.')
        require(type(n_occ) is int and n_occ>0 and type(n_unique) is int and 0<n_unique<=n_occ,'Invalid dynamic cohort denominators.')
        geometry=keyed(load(stage/'geometry.jsonl',True),lambda row:(row['seed'],row['group']),
                       {(seed,group) for seed in SEEDS for group in groups})
        require(len(geometry)==384,'Retain all384 group-seed geometries.')
        cases={};private_map=[];geometry_totals=[];topology_doses=[];nonidentity={str(seed):0 for seed in SEEDS}
        unread_binary=[];first_members={};role_counts=collections.Counter();all_donor_fractions=[]
        for seed in SEEDS:
            selected=[geometry[seed,group] for group in groups];union=set()
            require(sum(row['targets'] for row in selected)==n_occ,'Dynamic target-occurrence denominator differs.')
            for group,row in zip(groups,selected):
                require(row['targets']==len(row['node_ids'])==len(row['labels'])==cm['group_target_counts'][str(group)]
                        and len(row['node_ids'])==len(set(row['node_ids'])) and all(y==1 for y in row['labels']),
                        'Original complete fraud-target membership differs.')
                require(type(row['node_budget']) is int and type(row['edge_budget']) is int and
                        row['edge_budget']>=row['node_budget']>=1,'Invalid scenario budgets.')
                if group in first_members:require(row['node_ids']==first_members[group],'Original target order differs across seeds.')
                else:first_members[group]=list(row['node_ids'])
                union.update(row['node_ids']);key=case_key(seed,group);transform=row['topology_transform']
                require(all(transform[name] is True for name in ('all_old_target_edges_preserved','injected_block_preserved',
                    'row_and_total_edge_counts_preserved','physical_injected_degrees_preserved','maximum_common_feasible_k'))
                    and all(transform[name] is False for name in ('uses_full_clean_target_degree','labels_read',
                    'model_outputs_used','features_read_or_changed','tunable_fraction')),'Fixed cost-only topology policy differs.')
                plans=transform['row_plan'];require(len(plans)==row['node_budget'],'Complete injection row plan differs.')
                k=sum(value['k'] for value in plans);candidate_before=sum(value['baseline_candidate_count'] for value in plans)
                require(k==transform['total_reallocated_edges'] and candidate_before==transform['candidate_incidence_before'],
                        'Saved role-change counts do not sum.')
                for index,value in enumerate(plans):
                    require(value['row']==index and all(type(value[name]) is int and value[name]>=0 for name in
                        ('k','baseline_candidate_count','missing_target_count','missing_candidate_count')),'Invalid per-row geometry count.')
                    require(value['k']==min(value['baseline_candidate_count'],value['missing_target_count'],value['missing_candidate_count']),
                            'Fixed maximal common feasibility differs.')
                    fraction=value['k']/value['baseline_candidate_count'] if value['baseline_candidate_count'] else None
                    if fraction is None:require(value['candidate_donor_fraction'] is None,'Zero-candidate donor ratio differs.')
                    else:close(fraction,value['candidate_donor_fraction'],'Saved donor fraction differs.')
                    require(len(value['removed_candidate_ids'])==len(value['primary_added_target_ids'])==
                        len(value['random_added_target_ids'])==len(value['control_added_candidate_ids'])==value['k'],
                        'Saved shared-donor/addition counts differ.')
                degrees=transform['target_incoming_injected_degrees_before']
                require(len(degrees)==row['targets'] and all(type(value) is int and value>=0 for value in degrees)
                    and sum(degrees)==transform['total_target_incidence_before'],'Baseline target-incidence counts differ.')
                target_before=sum(degrees);inter_injection=row['edge_budget']-target_before-candidate_before
                require(inter_injection>=0 and len(transform['original_row_edge_counts'])==len(transform['physical_injected_degrees'])==row['node_budget']
                    and sum(transform['original_row_edge_counts'])==transform['total_owned_nonself_undirected_edges']==row['edge_budget']
                    and sum(transform['physical_injected_degrees'])==row['edge_budget']+inter_injection,'Owned/stub/row physical costs do not reconcile.')
                active_rows=sum(value['k']>0 for value in plans);arm_geometry={}
                for arm in ARMS:
                    meta=row['variants'][arm];identity=transform['identity'][arm];after=transform['target_incoming_injected_degrees_after'][arm]
                    require(identity==meta['identity'] and identity['changed_rows']==(0 if arm=='baseline' else active_rows)
                        and identity['changed_coordinates']==(0 if arm=='baseline' else 2*k)
                        and identity['exact_equals_baseline']==(arm=='baseline' or k==0),'Actual topology identity/Hamming metadata differs.')
                    target_gain=k if arm in ('balanced_target_fill','random_target_fill') else 0
                    require(len(after)==len(degrees) and all(type(v) is int and v>=0 for v in after)
                        and sum(after)==transform['total_target_incidence_after'][arm]==target_before+target_gain
                        and transform['candidate_incidence_after'][arm]==candidate_before-target_gain,
                        'Target/candidate role-incidence totals differ.')
                    require(all(old<=new for old,new in zip(degrees,after)) and not transform['previously_covered_target_ids_lost'][arm],
                            'An old target attachment or covered target was lost.')
                    require(meta['self_token0_exact'] is True and len(meta['changed_coordinates_by_token'])==7
                        and meta['changed_coordinates_by_token'][0]==0 and all(type(v) is int and v>=0 for v in meta['changed_coordinates_by_token']),
                        'Native canonical self-token/change counts differ.')
                    require(meta['feature_content_sha256']==row['immutable_capture_content_hashes']['x_atk'] and
                        meta['actual_injected_nodes']==row['node_budget'] and meta['actual_injected_nonself_undirected_edges']==row['edge_budget'] and
                        meta['physical_injected_degrees']==transform['physical_injected_degrees'],'All features/degrees/physical budgets must remain fixed.')
                    covered_before=sum(value>0 for value in degrees);covered_after=sum(value>0 for value in after)
                    require(covered_after-covered_before==len(transform['newly_covered_target_ids'][arm]),'Target-coverage counts differ.')
                    arm_geometry[arm]=dict(identity=dict(identity),target_incidence=sum(after),candidate_incidence=transform['candidate_incidence_after'][arm],
                        injected_injected_owned_edges=inter_injection,covered_targets=covered_after,target_coverage_fraction=covered_after/len(degrees),
                        newly_covered_targets=covered_after-covered_before,target_degree_min=min(after),target_degree_max=max(after),
                        target_degree_mean=statistics.fmean(after),target_degree_histogram=dict(sorted(collections.Counter(after).items())),
                        canonical_changed_coordinates_by_token=list(meta['changed_coordinates_by_token']))
                nonidentity[str(seed)]+=int(not transform['identity']['balanced_target_fill']['exact_equals_baseline'])
                fractions=[value['candidate_donor_fraction'] for value in plans if value['candidate_donor_fraction'] is not None]
                topology=dict(total_reallocated_edges=k,active_injected_rows=active_rows,injected_rows=row['node_budget'],
                    baseline_target_incidence=target_before,baseline_candidate_incidence=candidate_before,
                    candidate_donor_fraction=k/candidate_before if candidate_before else None,
                    mean_per_row_donor_fraction=statistics.fmean(fractions) if fractions else None,
                    maximum_per_row_donor_fraction=max(fractions) if fractions else None,
                    rows_losing_all_candidate_edges=sum(value['k']==value['baseline_candidate_count']>0 for value in plans),
                    primary_equals_random=bool(transform['primary_equals_random']),arms=arm_geometry)
                role_counts.update(dict(baseline_candidate_incidence=candidate_before,reallocated_edges=k,
                    baseline_target_incidence=target_before,primary_target_incidence=target_before+k,
                    primary_candidate_incidence_after=candidate_before-k,candidate_control_incidence_after=candidate_before,
                    all_injected_rows=row['node_budget'],primary_equals_random_cases=int(transform['primary_equals_random'])))
                for plan in plans:
                    if plan['baseline_candidate_count']:
                        all_donor_fractions.append(plan['candidate_donor_fraction']);role_counts['rows_with_candidate_edges']+=1
                        role_counts['rows_losing_all_candidate_neighbors']+=int(plan['k']==plan['baseline_candidate_count'])
                    else:role_counts['rows_without_candidate_edges']+=1
                    role_counts['rows_with_positive_k']+=int(plan['k']>0)
                    role_counts['positive_k_rows_filling_all_missing_targets']+=int(plan['k']>0 and plan['k']==plan['missing_target_count'])
                cases[seed,group]=dict(case_key=key,seed=seed,targets=row['targets'],node_budget=row['node_budget'],edge_budget=row['edge_budget'],
                    topology=topology,outcomes={},contrasts={})
                private_map.append(dict(case_key=key,seed=seed,group=group,private_mapping_only=True))
                unread_binary.append(dict(path=row['artifact']['path'],sha256=row['artifact']['sha256'],read=False))
            require(len(union)==n_unique and compact_hash(sorted(union))==cm['target_ids_sha256'],'Dynamic unique-target denominator/hash differs.')
            for arm in ARMS:
                geometry_totals.append(dict(seed=seed,arm=arm,groups=128,target_occurrences=n_occ,unique_targets=n_unique,
                    summed_injected_nodes=sum(row['node_budget'] for row in selected),summed_nonself_undirected_edges=sum(row['edge_budget'] for row in selected)))
                values=[cases[seed,group]['topology'] for group in groups];arm_values=[value['arms'][arm] for value in values]
                source_candidates=sum(value['baseline_candidate_incidence'] for value in values);total_k=sum(value['total_reallocated_edges'] for value in values)
                topology_doses.append(dict(seed=seed,arm=arm,group_scenarios=128,exact_identity_cases=sum(v['identity']['exact_equals_baseline'] for v in arm_values),
                    reallocated_edges_per_nonbaseline_arm=total_k if arm!='baseline' else 0,
                    baseline_candidate_incidence=source_candidates,shared_donor_fraction=total_k/source_candidates if source_candidates else None,
                    target_incidence=sum(v['target_incidence'] for v in arm_values),candidate_incidence=sum(v['candidate_incidence'] for v in arm_values),
                    covered_target_occurrences=sum(v['covered_targets'] for v in arm_values),newly_covered_target_occurrences=sum(v['newly_covered_targets'] for v in arm_values),
                    group_mean_target_coverage_fraction=statistics.fmean(v['target_coverage_fraction'] for v in arm_values),
                    occurrence_weighted_target_coverage_fraction=sum(v['covered_targets'] for v in arm_values)/n_occ,
                    changed_canonical_coordinates_by_token=[sum(v['canonical_changed_coordinates_by_token'][j] for v in arm_values) for j in range(7)],
                    primary_equals_random_cases=sum(value['primary_equals_random'] for value in values),
                    scope='Counts sum separate128attack scenarios, not one graph; donor/target role allocation changes but physical node/edge budgets and injected degrees remain fixed.'))
        require(len({value['case_key'] for value in cases.values()})==384,'Opaque case-key collision.')
        require(geometry_totals==producer['whole_seed_geometry_totals']==audit['geometry_totals'],'Dynamic scenario resources disagree with audit.')
        require(nonidentity==audit['nonidentity_cases_by_seed'],'Primary topology headroom disagrees with audit.')
        require(all(role_counts[key]==audit['role_allocation_counts'].get(key,0) for key in set(role_counts)|set(audit['role_allocation_counts'])),
                'Saved allocation/donor/active-row counts disagree with the separate tensor audit.')
        donor_summary=dict(rows=len(all_donor_fractions),
            pooled_edge_fraction=role_counts['reallocated_edges']/role_counts['baseline_candidate_incidence'] if role_counts['baseline_candidate_incidence'] else None,
            minimum=min(all_donor_fractions) if all_donor_fractions else None,
            mean=statistics.fmean(all_donor_fractions) if all_donor_fractions else None,
            maximum=max(all_donor_fractions) if all_donor_fractions else None)
        for name,value in donor_summary.items():
            if value is None:require(audit['candidate_donor_fraction_summary'][name] is None,'Empty donor statistic disagrees with audit.')
            else:close(value,audit['candidate_donor_fraction_summary'][name],'Donor statistics disagree with audit.')
        headroom=load(stage/'headroom.json');stop=any(value==0 for value in nonidentity.values())
        require(headroom==producer['headroom'] and headroom['nonidentity_cases_by_seed']==nonidentity and
                headroom['stop_before_evaluation']==stop and headroom['original_generator_calls_already_spent']==384,
                'Planned headroom gate/retained generator costs differ.')
        metrics=[];contrasts={};budgets=[];aggregate=[];criterion=None;posthoc=dict(maximum_gain=None,maximum_loss=None)
        if stop:
            require(producer['status']=='completed_no_headroom' and audit['metrics']==[] and audit['paired_contrasts']=={}
                    and audit['primary_development_criterion'] is None,'Planned no-headroom stop must have no performance conclusion.')
            require(not (stage/'clean_predictions.jsonl').exists() and all(not (stage/('%s_seed%d_predictions.jsonl'%(arm,seed))).exists()
                    for arm in ARMS for seed in SEEDS),'No classifier prediction files are allowed after a headroom stop.')
            for name in ('original_gcn_completed_forwards','official_gaga_completed_forwards','original_gcn_models_loaded','official_gaga_models_loaded'):
                require(producer['counters'][name]==0,'Headroom stop unexpectedly spent a dedicated evaluator call/load.')
            geometry_names=('baseline_engine_load_dispatches','baseline_engines_loaded','dataset_loads','proposal_gcn_models_loaded',
                'generator_models_loaded','generator_forward_dispatches','generator_completed_forwards','baseline_generations',
                'proposal_gcn_forward_attempts','scheduler_restoration_ticks_reported','geometry_cases_prepared',
                'geometry_graph_reconstructions','canonical_sequence_updates','self_token_invariant_checks')
            require(producer['counters']=={name:(value if name in geometry_names else 0) for name,value in p['expected_if_evaluated'].items()},
                    'Planned stop counters must preserve all preparation costs and zero every evaluation/update channel.')
        else:
            require(producer['status']=='completed','Nonzero headroom requires complete fixed evaluation.')
            require(producer['counters']==p['expected_if_evaluated'],'Complete fixed model/canonical counters differ.')
            clean=keyed(load(stage/'clean_predictions.jsonl',True),lambda row:row['group'],groups)
            clean_pred={}
            for group in groups:
                require(clean[group]['node_ids']==geometry[3,group]['node_ids'] and clean[group]['labels']==geometry[3,group]['labels'],
                        'Clean target identity/order differs.')
                for detector in DETECTORS:clean_pred[group,detector]=argmax(clean[group][detector+'_logits'],len(clean[group]['labels']))
            pred={}
            for seed in SEEDS:
                for arm in ARMS:
                    rows=keyed(load(stage/('%s_seed%d_predictions.jsonl'%(arm,seed)),True),lambda row:row['group'],groups)
                    for detector in DETECTORS:
                        total=collections.Counter();macro=[]
                        for group in groups:
                            row,original=rows[group],geometry[seed,group]
                            require(row['seed']==seed and row['arm']==arm and row['node_ids']==original['node_ids']==clean[group]['node_ids']
                                    and row['labels']==original['labels']==clean[group]['labels'],'Attacked target identity/order differs.')
                            predicted=argmax(row[detector+'_logits'],len(row['labels']));pred[seed,arm,group,detector]=predicted
                            values=transition(row['labels'],clean_pred[group,detector],predicted)
                            require(values==row[detector+'_transitions'],'Saved transitions disagree with raw argmax.')
                            cases[seed,group]['outcomes'].setdefault(arm,{})[detector]=values
                            total.update(values);macro.append(100*values['attacked_wrong']/values['occurrences'])
                        require(total['occurrences']==n_occ,'Dynamic per-arm occurrence denominator differs.')
                        metrics.append(dict(seed=seed,arm=arm,detector=detector,**{name:total[name] for name in FIELDS},
                            mr_percent=100*total['attacked_wrong']/n_occ,group_macro_mr_percent=statistics.fmean(macro)))
            lookup={(row['seed'],row['arm'],row['detector']):row for row in metrics}
            audit_metrics=keyed(audit['metrics'],lambda row:(row['seed'],row['arm'],row['detector']),lookup)
            for key,row in lookup.items():
                require(all(row[name]==audit_metrics[key][name] for name in FIELDS),'Metrics disagree with separate audit.')
                for name in ('mr_percent','group_macro_mr_percent'):close(row[name],audit_metrics[key][name],'Micro/macro MR disagrees with audit.')
            for arm,reference in PAIRS:
                contrast=arm+'-minus-'+reference
                for detector in DETECTORS:
                    total=collections.Counter();parts=[]
                    for seed in SEEDS:
                        for group in groups:
                            row=geometry[seed,group]
                            values=paired(row['labels'],clean_pred[group,detector],pred[seed,reference,group,detector],pred[seed,arm,group,detector])
                            total.update({name:values[name] for name in PAIR_FIELDS})
                            cases[seed,group]['contrasts'].setdefault(contrast,{})[detector]=values
                            parts.append(dict(node_budget=row['node_budget'],edge_budget=row['edge_budget'],**values))
                    require(total['occurrences']==3*n_occ and sum(total[name] for name in ('group_wins','group_ties','group_losses'))==384,
                            'Dynamic paired correlated-case denominators differ.')
                    differences=[lookup[seed,arm,detector]['mr_percent']-lookup[seed,reference,detector]['mr_percent'] for seed in SEEDS]
                    mean,sd=statistics.fmean(differences),statistics.stdev(differences);half=T95_DF2*sd/math.sqrt(3)
                    key=contrast+'_'+detector
                    contrasts[key]=dict(seed_differences_pp=differences,mean_difference_pp=mean,sample_sd_pp=sd,
                        paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts={name:total[name] for name in PAIR_FIELDS})
                    checked_audit=audit['paired_contrasts'][key]
                    for name in PAIR_FIELDS:require(total[name]==checked_audit['paired_counts'].get(name,0),'Paired counts disagree with audit.')
                    for name in ('mean_difference_pp','sample_sd_pp'):close(contrasts[key][name],checked_audit[name],'Mean/SD contrast disagrees with audit.')
                    for name in ('seed_differences_pp','paired_t95_pp'):
                        require(len(contrasts[key][name])==len(checked_audit[name]),'Contrast vector length differs.')
                        for actual,wanted in zip(contrasts[key][name],checked_audit[name]):close(actual,wanted,'Analytic df2 conditional contrast disagrees with audit.')
                    for name,condition in (('one_node_one_edge',lambda row:row['node_budget']==row['edge_budget']==1),
                        ('one_node_multiple_edges',lambda row:row['node_budget']==1 and row['edge_budget']>1),
                        ('multiple_nodes',lambda row:row['node_budget']>1)):
                        selected=[row for row in parts if condition(row)];n=sum(row['occurrences'] for row in selected);net=sum(row['delta_wrong'] for row in selected)
                        budgets.append(dict(contrast=contrast,detector=detector,regime=name,group_seed_cases=len(selected),occurrences=n,
                            delta_wrong=net,delta_mr_pp=100*net/n if n else None,
                            group_macro_delta_pp=statistics.fmean(row['delta_mr_pp'] for row in selected) if selected else None,
                            paired_counts={field:sum(row[field] for row in selected) for field in PAIR_FIELDS}))
            for arm in ARMS:
                for detector in DETECTORS:
                    selected=[lookup[seed,arm,detector] for seed in SEEDS]
                    aggregate.append(dict(arm=arm,detector=detector,mean_micro_mr_percent=statistics.fmean(row['mr_percent'] for row in selected),
                        sample_sd_micro_pp=statistics.stdev(row['mr_percent'] for row in selected),
                        mean_group_macro_mr_percent=statistics.fmean(row['group_macro_mr_percent'] for row in selected)))
            primary=contrasts['balanced_target_fill-minus-baseline_gaga'];advantage=contrasts['balanced_target_fill-minus-candidate_rewire_gaga']['mean_difference_pp']
            criterion=dict(primary='balanced_target_fill',mean_delta_pp=primary['mean_difference_pp'],all_three_seed_deltas_positive=all(value>0 for value in primary['seed_differences_pp']),
                paired_t95_lower_pp=primary['paired_t95_pp'][0],mean_advantage_over_candidate_rewire_pp=advantage,
                criterion_met=primary['mean_difference_pp']>=1 and all(value>0 for value in primary['seed_differences_pp']) and
                    primary['paired_t95_pp'][0]>0 and advantage>0,
                random_target_comparator_need_not_be_worse=True,target_priority_superiority_required=False,controls_not_candidates=True,final_TEST_allowed=False)
            audited_criterion=audit['primary_development_criterion']
            for name in ('primary','criterion_met','all_three_seed_deltas_positive','random_target_comparator_need_not_be_worse',
                    'target_priority_superiority_required','controls_not_candidates','final_TEST_allowed'):
                require(criterion[name]==audited_criterion[name],'Fixed primary decision criterion disagrees with audit.')
            for name in ('mean_delta_pp','paired_t95_lower_pp','mean_advantage_over_candidate_rewire_pp'):
                close(criterion[name],audited_criterion[name],'Fixed primary criterion arithmetic disagrees with audit.')
            gain_candidates=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['balanced_target_fill-minus-baseline']['gaga']['delta_wrong']>0]
            loss_candidates=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['balanced_target_fill-minus-baseline']['gaga']['delta_wrong']<0]
            posthoc=dict(maximum_gain=min(gain_candidates,key=lambda value:(-value[2]['contrasts']['balanced_target_fill-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if gain_candidates else None,
                maximum_loss=min(loss_candidates,key=lambda value:(value[2]['contrasts']['balanced_target_fill-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if loss_candidates else None)
        fixed_examples=[dict(selection_ordinal=ordinal,selection='p.valid_groups[ordinal], fixed in this reporting source before current outcome reads.',
                            **cases[seed,groups[ordinal]]) for ordinal in FIXED_EXAMPLE_ORDINALS for seed in SEEDS]
        for key in ('generator_completed_forwards','baseline_generations'):require(producer['counters'][key]==384,'The planned headroom stop does not erase384generator calls.')
        close(producer['process_cpu_seconds'],audit['producer_cpu_seconds'],'Producer CPU receipt disagrees with audit.')
        close(producer['wall_seconds'],audit['producer_wall_seconds'],'Producer wall receipt disagrees with audit.')
        require(producer['GPU_hours']==0 and producer['experimental_API_calls']==0 and producer['final_TEST_used'] is False,
                'Study CPU/API/TEST receipts differ.')
        require(producer['immutable_content_hashes_retained'] is True and
            producer['immutable_content_hashes']['before']==producer['immutable_content_hashes']['after'] and
            audit['immutable_content_hash_receipts']==producer['immutable_content_hashes'],'Saved immutable-data/model receipts disagree.')
        report=dict(status='completed_saved_only_reporting',producer_status=producer['status'],provenance=producer['provenance'],
            cohort_metadata={key:value for key,value in cm.items() if key!='group_target_counts'},correlated_occurrences_all_seeds=3*n_occ,group_seed_cases=384,
            metrics=metrics,arm_aggregates=aggregate,paired_contrasts=contrasts,disjoint_budget_contrasts=budgets,
            geometry_totals=geometry_totals,topology_doses=topology_doses,
            role_allocation_counts=dict(role_counts),candidate_donor_fraction_summary=donor_summary,
            headroom=dict(nonidentity_cases_by_seed=nonidentity,planned_stop_before_evaluation=stop,original_generator_calls_already_spent=384,
                proposal_GCN_loads_already_spent=producer['counters']['proposal_gcn_models_loaded']),
            fixed_ordinal_case_examples=fixed_examples,posthoc_examples=dict(selection='Maximum positive/negative primary-minus-baseline GAGA wrong-count delta across all384 contexts; ties seed ascending then original group ascending. Illustrations only, never selection of a method or new gate.',**posthoc),
            primary_development_criterion=criterion,qualified_candidate='balanced_target_fill_development_candidate_only' if criterion and criterion['criterion_met'] else None,
            qualified_control_winner=None,
            random_target_comparison=(contrasts.get('balanced_target_fill-minus-random_target_fill_gaga')),
            random_target_interpretation='Random target-fill may be equivalent or better; qualification concerns target-role allocation against candidate rewire, never superior target ranking or promotion of a control.',
            study_resources=dict(process_cpu_seconds=producer['process_cpu_seconds'],wall_seconds=producer['wall_seconds'],GPU_hours=producer['GPU_hours'],
                producer_counters=producer['counters'],source='Producer receipts reconciled by separately authored saved audit; CPU and wall durations are distinct.'),
            independent_audit_sha256=inputs[str(audit_path)],unread_binary_artifact_references=len(unread_binary),
            unread_binary_artifact_manifest_sha256=compact_hash(unread_binary),
            reporting_scope='Producer-author saved arithmetic reporting checked against a separately authored audit, not independent method or human verification.',
            limitations=['Previous negative outcomes were known. This reporting source was frozen before reading current model outcomes, not before the full adaptive research history.',
                'Fresh original VALID group contexts exclude both earlier128-context lists but retain overlapping news identities; cohort counts and overlap are copied from the ID-only registration and verified where saved memberships permit.',
                'All target occurrences across384 group-seed contexts are correlated. The nominal n3 paired t95 is conditional on fixed checkpoints, models and adaptively inspected development data.',
                'Physical candidate-to-target role reallocation uses frozen baseline injected-attachment counts, not full clean-graph degrees or model outcomes. It is not a claim of superior ranking, novelty, camouflage, course completion or lower physical cost.',
                'Common feasibility can allocate every missing target slot, making primary and random target-fill algebraically identical. All control outcomes and identity counts remain visible; controls are never promoted.',
                'Reallocation totals concatenate separate attack scenarios, never one combined graph. Exact physical edge/node/degree equality does not imply identical implementation runtime or effects.',
                'Case output contains count-only outcomes and opaque keys; no raw target-node IDs, labels or logits are written. PRIVATE_case_key_map.json alone connects original group IDs.',
                'Only the preselected primary can qualify as a development candidate. Controls never become a winner; final TEST and submission/tag gates remain separate.'])
        all_cases=[cases[seed,group] for seed in SEEDS for group in groups]
        require(len(all_cases)==384,'Retain all384 count-only case rows.')
        write(output/'summary.json',report);write(output/'all_case_counts.json',all_cases);write(output/'PRIVATE_case_key_map.json',private_map)
        receipt.update(status='passed',checks_passed=checks,retained_count_only_cases=384,
            producer_status=producer['status'],dynamic_target_occurrences_per_seed=n_occ,dynamic_unique_targets=n_unique,
            independent_audit_sha256=inputs[str(audit_path)],reporting_source_frozen_before_current_outcome_reads=True,
            output_hashes={name:sha(output/name) for name in ('summary.json','all_case_counts.json','PRIVATE_case_key_map.json')})
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),checks_passed_before_failure=checks)
    finally:
        receipt.update(process_cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.monotonic()-wall_started,saved_JSON_input_files_read=len(inputs))
        write(output/'input_hashes.json',inputs);write(output/'receipt.json',receipt)
        print(json.dumps({key:value for key,value in receipt.items() if key!='traceback'},allow_nan=False),flush=True)
    if receipt['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
