"""Saved-JSON-only sharpen case reporting, source frozen before outcome reads.

Previous studies and the fresh ID-only cohort metadata are known. This is
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


ARMS = ('baseline', 'logit_sharp', 'linear_contrast', 'shuffled_delta')
SEEDS = (3, 4, 5)
DETECTORS = ('gcn', 'gaga')
FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
PAIR_FIELDS = ('occurrences', 'gained_wrong', 'lost_wrong', 'new_success_gained',
    'baseline_new_success_lost', 'additional_recovery', 'previous_recovery_lost',
    'group_wins', 'group_ties', 'group_losses')
PAIRS = tuple((arm, 'baseline') for arm in ARMS[1:])+(
    ('logit_sharp', 'shuffled_delta'), ('logit_sharp', 'linear_contrast'))
FIXED_EXAMPLE_ORDINALS = (0, 64, 127)
T95_DF2 = math.sqrt(2)*.95/math.sqrt(1-.95**2)
COUNT_STATS = ('terminal_lower_endpoint_coordinates', 'terminal_upper_endpoint_coordinates',
               'preclip_below_zero_coordinates', 'preclip_above_one_coordinates')
NORM_STATS = ('preclip_unit_l2_frobenius', 'postclip_unit_l2_frobenius',
              'raw_delta_frobenius', 'raw_delta_linf')


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
        return hashlib.sha256(('sharpen-case-report-v1|%d|%d'%(seed,group)).encode('ascii')).hexdigest()[:20]

    try:
        p=load(protocol_path);producer=load(stage/'result.json');audit=load(audit_path)
        require(p['schema']=='monti_feature_logit_sharpen_v1' and p['status']=='frozen' and p['post_hoc'] is True,
                'Require the frozen post-hoc sharpen registration.')
        require(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['beta']==2 and p['linear_slope']==1.5,
                'Fixed arms/seeds/transforms differ.')
        require(producer['status'] in ('completed','completed_no_headroom') and audit['status']=='passed',
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
        groups=list(p['valid_groups']);cm=p['cohort_metadata'];n_occ=cm['target_occurrences_per_seed'];n_unique=cm['unique_targets']
        require(len(groups)==len(set(groups))==128 and not set(groups)&set(p['excluded_valid_groups']),
                'Exactly128 newly selected original VALID contexts must exclude the earlier128.')
        require(type(n_occ) is int and n_occ>0 and type(n_unique) is int and 0<n_unique<=n_occ,'Invalid dynamic cohort denominators.')
        geometry=keyed(load(stage/'geometry.jsonl',True),lambda row:(row['seed'],row['group']),
                       {(seed,group) for seed in SEEDS for group in groups})
        require(len(geometry)==384,'Retain all384 group-seed geometries.')
        cases={};private_map=[];geometry_totals=[];feature_doses=[];nonidentity={str(seed):0 for seed in SEEDS}
        maximum_preclip_gap=0.;unread_binary=[];first_members={}
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
                union.update(row['node_ids']);key=case_key(seed,group)
                transform=row['feature_transform'];identities=transform['identity'];dose={}
                require(transform['beta']==2 and transform['linear_slope']==1.5 and
                        transform['shuffle_matches_preclip_unit_l2_by_construction'] is True and
                        transform['no_terminal_or_raw_norm_matching_claim'] is True and
                        transform['class_confidence_interpretation'] is False,'Fixed transform interpretation differs.')
                gap=0.
                for arm in ARMS:
                    require(identities[arm]==row['variants'][arm]['identity'] and
                            row['variants'][arm]['unchanged_tokens0_1_2_4_5_exact'] is True,
                            'Feature identity or fixed canonical token metadata differs.')
                    stats=transform['stats'][arm]
                    require(len(stats['rows'])==row['node_budget'],'Per-injected-row feature statistics differ.')
                    for name in NORM_STATS:require(math.isfinite(stats[name]) and stats[name]>=0,'Invalid saved feature-displacement norm.')
                    for name in COUNT_STATS:
                        require(type(stats[name]) is int and stats[name]>=0 and stats[name]==sum(value[name] for value in stats['rows']),
                                'Saved feature clipping/endpoint counts do not sum.')
                    dose[arm]={name:stats[name] for name in NORM_STATS+COUNT_STATS}
                    for value in stats['rows']:
                        for name in ('preclip_unit_l2','postclip_unit_l2','raw_delta_l2','raw_delta_linf'):
                            require(math.isfinite(value[name]) and value[name]>=0,'Invalid per-row norm.')
                for a,b in zip(transform['stats']['logit_sharp']['rows'],transform['stats']['shuffled_delta']['rows']):
                    gap=max(gap,abs(a['preclip_unit_l2']-b['preclip_unit_l2']))
                    close(a['preclip_unit_l2'],b['preclip_unit_l2'],'Primary/shuffle preclip unit displacement norms differ.')
                close(gap,transform['maximum_recorded_preclip_unit_l2_gap'],'Saved maximum preclip norm gap differs.')
                maximum_preclip_gap=max(maximum_preclip_gap,gap)
                nonidentity[str(seed)]+=int(not identities['logit_sharp']['exact_equals_baseline'])
                cases[seed,group]=dict(case_key=key,seed=seed,targets=row['targets'],node_budget=row['node_budget'],edge_budget=row['edge_budget'],
                    feature_changes={arm:dict(identities[arm]) for arm in ARMS},feature_doses=dose,outcomes={},contrasts={})
                private_map.append(dict(case_key=key,seed=seed,group=group,private_mapping_only=True))
                unread_binary.append(dict(path=row['artifact']['path'],sha256=row['artifact']['sha256'],read=False))
            require(len(union)==n_unique and compact_hash(sorted(union))==cm['target_ids_sha256'],'Dynamic unique-target denominator/hash differs.')
            for arm in ARMS:
                geometry_totals.append(dict(seed=seed,arm=arm,groups=128,target_occurrences=n_occ,unique_targets=n_unique,
                    summed_injected_nodes=sum(row['node_budget'] for row in selected),summed_nonself_undirected_edges=sum(row['edge_budget'] for row in selected)))
                identity_rows=[cases[seed,group]['feature_changes'][arm] for group in groups]
                stats=[cases[seed,group]['feature_doses'][arm] for group in groups]
                feature_doses.append(dict(seed=seed,arm=arm,cases=128,injected_node_instances=sum(row['node_budget'] for row in selected),
                    exact_identity_cases=sum(value['exact_equals_baseline'] for value in identity_rows),
                    changed_rows=sum(value['changed_rows'] for value in identity_rows),changed_coordinates=sum(value['changed_coordinates'] for value in identity_rows),
                    combined_preclip_unit_l2=math.sqrt(math.fsum(value['preclip_unit_l2_frobenius']**2 for value in stats)),
                    combined_postclip_unit_l2=math.sqrt(math.fsum(value['postclip_unit_l2_frobenius']**2 for value in stats)),
                    combined_raw_l2=math.sqrt(math.fsum(value['raw_delta_frobenius']**2 for value in stats)),
                    maximum_case_raw_linf=max(value['raw_delta_linf'] for value in stats),
                    saturation_and_preclip_counts={name:sum(value[name] for value in stats) for name in COUNT_STATS},
                    scope='Concatenated injected-feature instances from128 separate attack scenarios, never one combined graph; only preclip unit L2 is matched.'))
        require(len({value['case_key'] for value in cases.values()})==384,'Opaque case-key collision.')
        require(geometry_totals==producer['whole_seed_geometry_totals']==audit['geometry_totals'],'Dynamic physical scenario totals disagree with audit.')
        require(nonidentity==audit['nonidentity_cases_by_seed'],'Primary feature headroom disagrees with audit.')
        close(maximum_preclip_gap,audit['maximum_preclip_unit_L2_gap'],'Maximum saved preclip norm gap disagrees with audit.')
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
        else:
            require(producer['status']=='completed','Nonzero headroom requires complete fixed evaluation.')
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
            primary=contrasts['logit_sharp-minus-baseline_gaga'];advantage=contrasts['logit_sharp-minus-shuffled_delta_gaga']['mean_difference_pp']
            criterion=dict(mean_delta_pp=primary['mean_difference_pp'],all_three_seed_deltas_positive=all(value>0 for value in primary['seed_differences_pp']),
                paired_t95_lower_pp=primary['paired_t95_pp'][0],mean_advantage_over_shuffled_delta_pp=advantage,
                criterion_met=primary['mean_difference_pp']>=1 and all(value>0 for value in primary['seed_differences_pp']) and
                    primary['paired_t95_pp'][0]>0 and advantage>0,
                linear_comparator_need_not_be_worse=True,controls_not_candidates=True,final_TEST_allowed=False)
            audited_criterion=audit['primary_development_criterion']
            for name in ('criterion_met','all_three_seed_deltas_positive','linear_comparator_need_not_be_worse','controls_not_candidates','final_TEST_allowed'):
                require(criterion[name]==audited_criterion[name],'Fixed primary decision criterion disagrees with audit.')
            for name in ('mean_delta_pp','paired_t95_lower_pp','mean_advantage_over_shuffled_delta_pp'):
                close(criterion[name],audited_criterion[name],'Fixed primary criterion arithmetic disagrees with audit.')
            gain_candidates=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['logit_sharp-minus-baseline']['gaga']['delta_wrong']>0]
            loss_candidates=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['logit_sharp-minus-baseline']['gaga']['delta_wrong']<0]
            posthoc=dict(maximum_gain=min(gain_candidates,key=lambda value:(-value[2]['contrasts']['logit_sharp-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if gain_candidates else None,
                maximum_loss=min(loss_candidates,key=lambda value:(value[2]['contrasts']['logit_sharp-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if loss_candidates else None)
        fixed_examples=[dict(selection_ordinal=ordinal,selection='p.valid_groups[ordinal], fixed in this reporting source before current outcome reads.',
                            **cases[seed,groups[ordinal]]) for ordinal in FIXED_EXAMPLE_ORDINALS for seed in SEEDS]
        for key in ('generator_completed_forwards','baseline_generations'):require(producer['counters'][key]==384,'The planned headroom stop does not erase384generator calls.')
        close(producer['process_cpu_seconds'],audit['producer_cpu_seconds'],'Producer CPU receipt disagrees with audit.')
        close(producer['wall_seconds'],audit['producer_wall_seconds'],'Producer wall receipt disagrees with audit.')
        require(producer['GPU_hours']==0 and producer['experimental_API_calls']==0 and producer['final_TEST_used'] is False,
                'Study CPU/API/TEST receipts differ.')
        report=dict(status='completed_saved_only_reporting',producer_status=producer['status'],provenance=producer['provenance'],
            cohort_metadata=cm,correlated_occurrences_all_seeds=3*n_occ,group_seed_cases=384,
            metrics=metrics,arm_aggregates=aggregate,paired_contrasts=contrasts,disjoint_budget_contrasts=budgets,
            geometry_totals=geometry_totals,feature_doses=feature_doses,maximum_preclip_unit_L2_gap=maximum_preclip_gap,
            headroom=dict(nonidentity_cases_by_seed=nonidentity,planned_stop_before_evaluation=stop,original_generator_calls_already_spent=384,
                proposal_GCN_loads_already_spent=producer['counters']['proposal_gcn_models_loaded']),
            fixed_ordinal_case_examples=fixed_examples,posthoc_examples=dict(selection='Maximum positive/negative primary-minus-baseline GAGA wrong-count delta across all384 contexts; ties seed ascending then original group ascending. Illustrations only, never selection of a method or new gate.',**posthoc),
            primary_development_criterion=criterion,qualified_candidate='logit_sharp_development_candidate_only' if criterion and criterion['criterion_met'] else None,
            qualified_control_winner=None,
            linear_comparison=(contrasts.get('logit_sharp-minus-linear_contrast_gaga')),
            linear_interpretation='Linear comparator may be better; qualification never asserts beta2 superiority over it and never promotes a control.',
            study_resources=dict(process_cpu_seconds=producer['process_cpu_seconds'],wall_seconds=producer['wall_seconds'],GPU_hours=producer['GPU_hours'],
                producer_counters=producer['counters'],source='Producer receipts reconciled by separately authored saved audit; CPU and wall durations are distinct.'),
            independent_audit_sha256=inputs[str(audit_path)],unread_binary_references=unread_binary,
            reporting_scope='Producer-author saved arithmetic reporting checked against a separately authored audit, not independent method or human verification.',
            limitations=['Previous negative outcomes were known. This reporting source was frozen before reading current model outcomes, not before the full adaptive research history.',
                'Fresh original VALID group contexts exclude the old128 but retain overlapping news identities; cohort counts and overlap are copied from the ID-only registration and verified where saved memberships permit.',
                'All target occurrences across384 group-seed contexts are correlated. The nominal n3 paired t95 is conditional on fixed checkpoints, models and adaptively inspected development data.',
                'Feature-coordinate odds sharpening is not calibrated class confidence, demonstrated realizable news content, stealth, novelty or course completion.',
                'Shuffle matches only PRECLIP movable-unit displacement norms. Actual postclip unit movement, raw movement and endpoint/clipping counts are reported separately.',
                'Frobenius aggregations concatenate injected-feature instances from separate scenarios; they are not one combined graph or matched physical/query costs.',
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
