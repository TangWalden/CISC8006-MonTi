"""Saved-JSON-only producer-author case reporting, frozen before new outcomes.

Earlier failed studies are known. This reporting freeze is before inspection
of the current prototype study, not preregistration of the whole project or an
independent method audit. No tensors, NumPy, data arrays, models or APIs.
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


ARMS = ('baseline', 'benign_mix', 'all_train_mix', 'fraud_mix')
SEEDS = (3, 4, 5)
DETECTORS = ('gcn', 'gaga')
FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
PAIR_FIELDS = ('occurrences', 'gained_wrong', 'lost_wrong', 'new_success_gained',
    'baseline_new_success_lost', 'additional_recovery', 'previous_recovery_lost',
    'group_wins', 'group_ties', 'group_losses')
PAIRS = (( 'benign_mix', 'baseline'), ('all_train_mix', 'baseline'), ('fraud_mix', 'baseline'),
         ('benign_mix', 'all_train_mix'), ('benign_mix', 'fraud_mix'))
FOLLOWUP_GROUPS = (938, 1001, 1071, 1130)
T95_DF2 = math.sqrt(2)*.95/math.sqrt(1-.95**2)


def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    stage,audit_path,output=pathlib.Path(args.stage).resolve(),pathlib.Path(args.audit).resolve(),pathlib.Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    cpu_started,wall_started=time.process_time(),time.monotonic()
    checks=0;inputs={};cache={}
    receipt=dict(status='failed',source_sha256=sha(__file__),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        producer_author_reporting=True,independent_method_audit=False,human_verification=False,
        model_queries=0,tensor_loads=0,NumPy_imports=0,dataset_loads=0,checkpoint_loads=0,API_calls=0,GPU_hours=0)

    def require(condition,message):
        nonlocal checks
        checks+=1
        if not condition:raise AssertionError(message)

    def load(path,jsonl=False):
        path=pathlib.Path(path).resolve()
        if path not in cache:
            raw=path.read_bytes();inputs[str(path)]=hashlib.sha256(raw).hexdigest()
            text=raw.decode('utf-8')
            cache[path]=[json.loads(line) for line in text.splitlines() if line.strip()] if jsonl else json.loads(text)
        return cache[path]

    def keyed(rows,key,expected=None):
        result={}
        for row in rows:
            identity=key(row);require(identity not in result,'Duplicate saved row identity.');result[identity]=row
        if expected is not None:require(set(result)==set(expected),'Saved cohort differs.')
        return result

    def argmax(logits,count):
        require(len(logits)==count,'Logit denominator differs.')
        values=[]
        for pair in logits:
            require(len(pair)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in pair),'Invalid binary logits.')
            values.append(0 if pair[0]>=pair[1] else 1)
        return values

    def transition(labels,clean,attacked):
        require(len(labels)==len(clean)==len(attacked),'Transition lengths differ.')
        counts=dict(occurrences=len(labels),clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0)
        for y,c,a in zip(labels,clean,attacked):
            counts['clean_wrong']+=int(c!=y);counts['attacked_wrong']+=int(a!=y)
            counts['new_wrong']+=int(c==y and a!=y);counts['recoveries']+=int(c!=y and a==y)
        require(counts['attacked_wrong']==counts['clean_wrong']+counts['new_wrong']-counts['recoveries'],'Transition identity fails.')
        return counts

    def paired(labels,clean,reference,candidate):
        counts={name:0 for name in PAIR_FIELDS}
        delta=sum(x!=y for x,y in zip(candidate,labels))-sum(x!=y for x,y in zip(reference,labels))
        counts['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']=1
        for y,c,old,new in zip(labels,clean,reference,candidate):
            counts['occurrences']+=1;counts['gained_wrong']+=int(old==y and new!=y);counts['lost_wrong']+=int(old!=y and new==y)
            if c==y:
                counts['new_success_gained']+=int(old==y and new!=y);counts['baseline_new_success_lost']+=int(old!=y and new==y)
            else:
                counts['additional_recovery']+=int(old!=y and new==y);counts['previous_recovery_lost']+=int(old==y and new!=y)
        require(delta==counts['gained_wrong']-counts['lost_wrong']==counts['new_success_gained']-counts['baseline_new_success_lost']-counts['additional_recovery']+counts['previous_recovery_lost'],
                'Paired gain/loss/recovery identity fails.')
        return dict(delta_wrong=delta,delta_mr_pp=100*delta/len(labels),**counts)

    def close(actual,expected,message):require(math.isclose(float(actual),float(expected),rel_tol=0,abs_tol=1e-10),message)

    def case_key(seed,group):
        return hashlib.sha256(('prototype-case-report-v1|%d|%d'%(seed,group)).encode('ascii')).hexdigest()[:20]

    try:
        producer=load(stage/'result.json');audit=load(audit_path)
        require(producer['status']=='completed' and audit['status']=='passed','Require complete producer and passed separate audit.')
        require(producer['provenance']==audit['provenance'] and producer['counters']==audit['producer_counters'],'Producer/audit source or counts differ.')
        unread_binary=[]
        for name,wanted in producer['output_hashes'].items():
            if not name.endswith(('.json','.jsonl')):
                unread_binary.append(dict(path=name,sha256=wanted,read=False));continue
            load(stage/name,name.endswith('.jsonl'))
            require(inputs[str((stage/name).resolve())]==wanted,'A producer JSON output byte hash differs.')
        geometry=keyed(load(stage/'geometry.jsonl',True),lambda row:(row['seed'],row['group']))
        groups=sorted(group for seed,group in geometry if seed==3)
        require(len(groups)==128 and len(geometry)==384 and set(geometry)=={(seed,group) for seed in SEEDS for group in groups},'Keep exactly all384cases.')
        clean=keyed(load(stage/'clean_predictions.jsonl',True),lambda row:row['group'],groups)
        clean_pred={(group,detector):argmax(clean[group][detector+'_logits'],len(clean[group]['labels'])) for group in groups for detector in DETECTORS}
        outputs={};pred={};metrics=[];cases={};private_map=[]
        for seed in SEEDS:
            selected=[geometry[seed,group] for group in groups]
            require(sum(row['targets'] for row in selected)==1091 and len({node for row in selected for node in row['node_ids']})==374,'Target occurrence/unique denominators differ.')
            require(sum(row['node_budget'] for row in selected)==7776 and sum(row['edge_budget'] for row in selected)==358491,'Physical scenario costs differ.')
            for group in groups:
                row=geometry[seed,group];key=case_key(seed,group)
                cases[seed,group]=dict(case_key=key,seed=seed,targets=row['targets'],node_budget=row['node_budget'],edge_budget=row['edge_budget'],
                    alpha=row['alpha'],outcomes={},contrasts={},feature_changes={arm:dict(row['variants'][arm]['identity']) for arm in ARMS})
                require(row['alpha']==.5 and all(row['variants'][arm]['unchanged_tokens0_1_2_4_5_exact'] for arm in ARMS),'Fixed alpha/token assertions differ.')
                private_map.append(dict(case_key=key,seed=seed,group=group,private_mapping_only=True))
            for arm in ARMS:
                rows=keyed(load(stage/('%s_seed%d_predictions.jsonl'%(arm,seed)),True),lambda row:row['group'],groups)
                outputs[seed,arm]=rows
                for detector in DETECTORS:
                    total=collections.Counter();macro=[]
                    for group in groups:
                        row,original,c=rows[group],geometry[seed,group],clean[group]
                        require(row['seed']==seed and row['arm']==arm and row['node_ids']==original['node_ids']==c['node_ids'] and row['labels']==original['labels']==c['labels'],
                                'Prediction targets/order/labels differ.')
                        p=argmax(row[detector+'_logits'],len(row['labels']));pred[seed,arm,group,detector]=p
                        values=transition(row['labels'],clean_pred[group,detector],p)
                        require(values==row[detector+'_transitions'],'Saved transition fields differ from raw argmax.')
                        cases[seed,group]['outcomes'].setdefault(arm,{})[detector]=values
                        total.update(values);macro.append(100*values['attacked_wrong']/values['occurrences'])
                    require(total['occurrences']==1091,'Per-arm occurrence denominator differs.')
                    metrics.append(dict(seed=seed,arm=arm,detector=detector,**{name:total[name] for name in FIELDS},
                        mr_percent=100*total['attacked_wrong']/1091,group_macro_mr_percent=statistics.fmean(macro)))
        require(len({row['case_key'] for row in cases.values()})==384,'Opaque case-key collision.')
        lookup={(row['seed'],row['arm'],row['detector']):row for row in metrics}
        audit_metrics=keyed(audit['metrics'],lambda row:(row['seed'],row['arm'],row['detector']),lookup)
        for key,row in lookup.items():
            reference=audit_metrics[key]
            require(all(row[name]==reference[name] for name in FIELDS),'Counts disagree with separate audit.')
            close(row['mr_percent'],reference['mr_percent'],'Micro MR disagrees with audit.')
            close(row['group_macro_mr_percent'],reference['group_macro_mr_percent'],'Macro MR disagrees with audit.')
        contrasts={};budgets=[]
        for arm,reference in PAIRS:
            contrast=arm+'-minus-'+reference
            for detector in DETECTORS:
                total=collections.Counter();parts=[]
                for seed in SEEDS:
                    for group in groups:
                        labels=geometry[seed,group]['labels']
                        values=paired(labels,clean_pred[group,detector],pred[seed,reference,group,detector],pred[seed,arm,group,detector])
                        total.update({name:values[name] for name in PAIR_FIELDS})
                        cases[seed,group]['contrasts'].setdefault(contrast,{})[detector]=values
                        parts.append(dict(node_budget=geometry[seed,group]['node_budget'],edge_budget=geometry[seed,group]['edge_budget'],**values))
                require(total['occurrences']==3273 and sum(total[name] for name in ('group_wins','group_ties','group_losses'))==384,'Paired complete-case denominators differ.')
                differences=[lookup[seed,arm,detector]['mr_percent']-lookup[seed,reference,detector]['mr_percent'] for seed in SEEDS]
                mean,sd=statistics.fmean(differences),statistics.stdev(differences);half=T95_DF2*sd/math.sqrt(3)
                key=contrast+'_'+detector
                contrasts[key]=dict(seed_differences_pp=differences,mean_difference_pp=mean,sample_sd_pp=sd,
                    paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts={name:total[name] for name in PAIR_FIELDS})
                checked_audit=audit['paired_contrasts'][key]
                for name in PAIR_FIELDS:require(total[name]==checked_audit['paired_counts'].get(name,0),'Paired count disagrees with audit.')
                for actual,wanted in zip(differences,checked_audit['seed_differences_pp']):close(actual,wanted,'Seed contrast disagrees with audit.')
                close(mean,checked_audit['mean_difference_pp'],'Mean contrast disagrees with audit.')
                for actual,wanted in zip(contrasts[key]['paired_t95_pp'],checked_audit['paired_t95_pp']):close(actual,wanted,'Analytic df2 conditional t interval disagrees with audit.')
                for name,condition in (('one_node_one_edge',lambda row:row['node_budget']==row['edge_budget']==1),
                    ('one_node_multiple_edges',lambda row:row['node_budget']==1 and row['edge_budget']>1),
                    ('multiple_nodes',lambda row:row['node_budget']>1)):
                    selected=[row for row in parts if condition(row)];n=sum(row['occurrences'] for row in selected);net=sum(row['delta_wrong'] for row in selected)
                    budgets.append(dict(contrast=contrast,detector=detector,regime=name,group_seed_cases=len(selected),occurrences=n,
                        delta_wrong=net,delta_mr_pp=100*net/n if n else None,
                        group_macro_delta_pp=statistics.fmean(row['delta_mr_pp'] for row in selected) if selected else None,
                        paired_counts={name:sum(row[name] for row in selected) for name in PAIR_FIELDS}))
        aggregate=[]
        for arm in ARMS:
            for detector in DETECTORS:
                selected=[lookup[seed,arm,detector] for seed in SEEDS]
                aggregate.append(dict(arm=arm,detector=detector,mean_micro_mr_percent=statistics.fmean(row['mr_percent'] for row in selected),
                    sample_sd_micro_pp=statistics.stdev(row['mr_percent'] for row in selected),
                    mean_group_macro_mr_percent=statistics.fmean(row['group_macro_mr_percent'] for row in selected)))
        primary=contrasts['benign_mix-minus-baseline_gaga']
        advantages={arm:contrasts['benign_mix-minus-'+arm+'_gaga']['mean_difference_pp'] for arm in ('all_train_mix','fraud_mix')}
        qualified=primary['mean_difference_pp']>=1 and all(value>0 for value in primary['seed_differences_pp']) and all(value>0 for value in advantages.values()) and primary['paired_t95_pp'][0]>0
        require(qualified==audit['primary_development_criterion']['criterion_met'],'Fixed primary criterion differs from independent audit.')
        fixed_examples=[]
        for group in FOLLOWUP_GROUPS:
            for seed in SEEDS:
                if (seed,group) in cases:
                    fixed_examples.append(dict(prior_group=group,selection='Fixed in this reporting source before this phase outcomes; previously selected in earlier analyses.',**cases[seed,group]))
                else:fixed_examples.append(dict(prior_group=group,seed=seed,present_in_fixed_cohort=False))
        gains=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['benign_mix-minus-baseline']['gaga']['delta_wrong']>0]
        losses=[(seed,group,row) for (seed,group),row in cases.items() if row['contrasts']['benign_mix-minus-baseline']['gaga']['delta_wrong']<0]
        gain=min(gains,key=lambda value:(-value[2]['contrasts']['benign_mix-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if gains else None
        loss=min(losses,key=lambda value:(value[2]['contrasts']['benign_mix-minus-baseline']['gaga']['delta_wrong'],value[0],value[1]))[2] if losses else None
        feature_counts=[]
        for seed in SEEDS:
            for arm in ARMS:
                identity=[cases[seed,group]['feature_changes'][arm] for group in groups]
                feature_counts.append(dict(seed=seed,arm=arm,cases=128,
                    exact_identity_cases=sum(row['exact_equals_baseline'] for row in identity),
                    changed_rows=sum(row['changed_rows'] for row in identity),changed_coordinates=sum(row['changed_coordinates'] for row in identity)))
        for arm in ARMS:
            require(sum(row['exact_identity_cases'] for row in feature_counts if row['arm']==arm)==audit['feature_identity_counts'][arm+'_identity_cases'],
                    'Exact feature-identity case counts disagree with the separate tensor audit.')
        report=dict(status='completed_saved_only_reporting',provenance=producer['provenance'],metrics=metrics,arm_aggregates=aggregate,
            paired_contrasts=contrasts,disjoint_budget_contrasts=budgets,feature_identity_counts=feature_counts,
            fixed_prior_case_followups=fixed_examples,
            posthoc_examples=dict(selection='Maximum positive/negative benign-minus-baseline GAGA wrong-count delta over all384 cases; ties seed ascending then original group ascending. Illustrations only, never method selection.',
                maximum_gain=gain,maximum_loss=loss),
            primary_development_criterion=dict(criterion_met=qualified,mean_delta_pp=primary['mean_difference_pp'],
                all_three_seed_deltas_positive=all(value>0 for value in primary['seed_differences_pp']),
                mean_advantages_over_controls_pp=advantages,paired_t95_lower_pp=primary['paired_t95_pp'][0],
                controls_not_candidates=True,final_TEST_allowed=False),
            qualified_candidate='benign_mix_development_candidate_only' if qualified else None,qualified_control_winner=None,
            independent_audit_sha256=inputs[str(audit_path)],unread_binary_references=unread_binary,
            reporting_scope='Producer-author arithmetic reporting checked against separately authored saved audit; not independent method verification.',
            limitations=['Previous negative studies were known; current reporting rules/source were frozen before these current outcomes were inspected.',
                'Reporting freeze does not turn the repeatedly inspected development study into confirmatory preregistration.',
                'All3273 target occurrences/384group-seed contexts are correlated; t95 is conditional on three generator seeds and fixed models/cohort.',
                'Group-equal MR and budget strata are descriptive secondary summaries; primary criterion is unchanged.',
                'Paired field baseline_new_success_lost denotes the contrast reference arm, including a control when used as reference.',
                'Numeric prototypes/mixes are not demonstrated realizable article text, camouflage, novelty or course completion.',
                'All-case rows use opaque keys. The original group-key map is PRIVATE; no target-node IDs, labels or raw logits are written here.'])
        all_cases=[cases[seed,group] for seed in SEEDS for group in groups]
        require(len(all_cases)==384,'Retain all384 count-only case rows.')
        write(output/'summary.json',report);write(output/'all_case_counts.json',all_cases)
        write(output/'PRIVATE_case_key_map.json',private_map)
        receipt.update(status='passed',checks_passed=checks,retained_count_only_cases=384,
            independent_audit_sha256=inputs[str(audit_path)],reporting_source_frozen_before_current_outcome_reads=True,
            output_hashes={name:sha(output/name) for name in ('summary.json','all_case_counts.json','PRIVATE_case_key_map.json')})
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),checks_passed_before_failure=checks)
    finally:
        receipt.update(process_cpu_seconds=time.process_time()-cpu_started,wall_seconds=time.monotonic()-wall_started,
            saved_JSON_input_files_read=len(inputs))
        write(output/'input_hashes.json',inputs);write(output/'receipt.json',receipt)
        print(json.dumps({key:value for key,value in receipt.items() if key!='traceback'},allow_nan=False),flush=True)
    if receipt['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
