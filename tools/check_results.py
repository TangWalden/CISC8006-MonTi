"""Check compact saved-count tables; no model, tensor or raw-logit replay."""
import argparse
import collections
import csv
import hashlib
import json
import math
import pathlib
import re
import statistics


ROOT=pathlib.Path(__file__).resolve().parents[1]
ARMS=('baseline','balanced_target_fill','random_target_fill','candidate_rewire')
SEEDS=(3,4,5)
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
SNAPSHOT='ca093d8b12ea153a076b02bbdbc4ede58508560c'
SCIENTIFIC_SOURCE='c3238ee62272d39f0337828718a3560abe101a00'
PROTOCOL_SHA256='24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=pathlib.Path,default=ROOT/'results')
    args=parser.parse_args();folder=args.results.resolve();checks=0
    def require(value,message):
        nonlocal checks
        checks+=1
        if not value:raise AssertionError(message)
    def close(actual,wanted,message):
        x,y=float(actual),float(wanted)
        require(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=1e-9+1e-10*abs(y),message)
    def load(name):return json.loads((folder/name).read_text(encoding='utf-8'))
    def rows(name):
        with (folder/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))
    provenance=load('provenance.json');latest=load('latest.json');audit=load('audit.json')
    require(provenance['record_snapshot']==SNAPSHOT and provenance['scientific_execution_source']==SCIENTIFIC_SOURCE
        and provenance['original_execution_protocol_sha256']==PROTOCOL_SHA256,'Wrong immutable scientific snapshot/protocol.')
    expected_files={'latest.json','seed_metrics.csv','case_counts.csv','previous_studies.csv','audit.json'}
    require(set(provenance['compact_files'])==expected_files,'Compact file inventory differs.')
    for name,record in provenance['compact_files'].items():
        require(hashlib.sha256((folder/name).read_bytes()).hexdigest()==record['sha256'],'Compact file byte hash differs: '+name)
    old_audit='evidence/candidate_target_reallocation_v1/independent_saved_tensor_audit_v1.json'
    require(provenance['compact_files']['audit.json']['sha256']==provenance['source_json_records'][old_audit]['sha256'],
        'The copied independent audit must retain its original bytes.')
    require(audit['status']=='passed' and audit['provenance']==dict(source_commit=SCIENTIFIC_SOURCE,protocol_sha256=PROTOCOL_SHA256),
        'Original independent audit is not a passed audit of this study.')
    core=ROOT/provenance['method_core']['current_path']
    require(core.is_file() and hashlib.sha256(core.read_bytes()).hexdigest()==provenance['method_core']['sha256'],
        'Published mathematical core differs from the frozen source bytes.')
    raw_cases=rows('case_counts.csv');raw_metrics=rows('seed_metrics.csv')
    require(len(raw_cases)==1536 and len(raw_metrics)==24,'Retain1536arm-case rows and24seed/detector metrics.')
    cases={}
    for value in raw_cases:
        require(set(value).isdisjoint({'node_ids','group','labels','logits'}),'Raw identifiers/logits are not part of compact cases.')
        require(re.fullmatch('[0-9a-f]{20}',value['opaque_case_key']) is not None,'Malformed opaque case key.')
        row={key:(value[key] if key in ('opaque_case_key','arm') else int(value[key])) for key in value}
        key=(row['seed'],row['opaque_case_key'],row['arm'])
        require(key not in cases and row['seed'] in SEEDS and row['arm'] in ARMS,'Duplicate or invalid arm/seed case.')
        require(row['targets']>0 and row['edge_budget']>=row['node_budget']>=1,'Invalid target or physical scenario cost.')
        require(all(v>=0 for k,v in row.items() if isinstance(v,int)),'Negative saved count.')
        for detector in ('gcn','gaga'):
            n=row['targets'];cw=row[detector+'_clean_wrong'];aw=row[detector+'_attacked_wrong']
            nw=row[detector+'_new_wrong'];rec=row[detector+'_recoveries']
            require(0<=cw<=n and 0<=aw<=n and nw<=n-cw and rec<=cw and aw==cw+nw-rec,'Invalid clean/new/recovery identity.')
        require(row['identity'] in (0,1) and row['primary_random_topologies_equal'] in (0,1),'Invalid identity flag.')
        require(row['changed_adjacency_coordinates']==2*row['k'] and row['identity']==int(row['k']==0)
            and row['changed_injected_rows']<=row['node_budget'] and row['rows_losing_all_candidate_neighbors']<=row['changed_injected_rows'],
            'Topology identity/Hamming/candidate-loss counts differ.')
        require(row['covered_targets']<=row['targets'] and row['target_injected_attachment_incidence']>=row['node_budget']
            and row['target_injected_attachment_incidence']+row['candidate_attachment_incidence']+row['injected_injected_owned_edges']==row['edge_budget'],
            'Physical edge-role counts do not sum.')
        cases[key]=row
    case_keys={(s,k) for s,k,a in cases}
    require(len(case_keys)==384,'Retain all384group-seed cases.')
    paired=collections.Counter();wins=ties=losses=0;seed_differences=[]
    metrics={};source_metrics={(r['seed'],r['arm'],r['detector']):r for r in audit['metrics']}
    source_geometry={(r['seed'],r['arm']):r for r in audit['geometry_totals']}
    for value in raw_metrics:
        key=(int(value['seed']),value['arm'],value['detector'])
        require(key not in metrics and key in source_metrics,'Duplicate or invalid compact seed metric.')
        metrics[key]={k:(int(v) if k in ('seed',)+FIELDS else float(v) if k in ('mr_percent','group_macro_mr_percent') else v) for k,v in value.items()}
    require(len(metrics)==24,'Incomplete seed metrics.')
    for seed in SEEDS:
        keys=[key for s,key in sorted(case_keys) if s==seed]
        require(len(keys)==128,'Each seed must retain128complete groups.')
        for key in keys:
            variants=[cases[seed,key,a] for a in ARMS];base=variants[0]
            require(set(a for s,k,a in cases if s==seed and k==key)==set(ARMS),'Missing case arm.')
            for row in variants:
                require(all(row[name]==base[name] for name in ('targets','node_budget','edge_budget','gcn_clean_wrong','gaga_clean_wrong','injected_injected_owned_edges','primary_random_topologies_equal')),
                    'The comparison changes a common cohort/cost/clean channel.')
                target_gain=row['k'] if row['arm'] in ('balanced_target_fill','random_target_fill') else 0
                require(row['target_injected_attachment_incidence']==base['target_injected_attachment_incidence']+target_gain
                    and row['candidate_attachment_incidence']==base['candidate_attachment_incidence']-target_gain,'Wrong candidate-to-target role movement.')
                gained=row['gaga_gained_wrong_vs_baseline'];lost=row['gaga_lost_wrong_vs_baseline']
                delta=row['gaga_attacked_wrong']-base['gaga_attacked_wrong']
                require(gained<=row['targets']-base['gaga_attacked_wrong'] and lost<=base['gaga_attacked_wrong'] and delta==gained-lost,
                    'Saved paired gained/lost counts do not match attacked counts.')
                require(delta==row['gaga_new_success_gained_vs_baseline']-row['gaga_baseline_new_success_lost']
                    -row['gaga_additional_recovery_vs_baseline']+row['gaga_previous_recovery_lost'],
                    'Saved paired success/recovery decomposition differs.')
            require(all(v['k']==variants[1]['k'] and v['rows_losing_all_candidate_neighbors']==variants[1]['rows_losing_all_candidate_neighbors'] for v in variants[1:]),
                'All nonbaseline controls must use the common feasible donor count.')
            primary=variants[1];delta=primary['gaga_attacked_wrong']-base['gaga_attacked_wrong']
            wins+=int(delta>0);ties+=int(delta==0);losses+=int(delta<0)
            paired['gained_wrong']+=primary['gaga_gained_wrong_vs_baseline'];paired['lost_wrong']+=primary['gaga_lost_wrong_vs_baseline']
        for arm in ARMS:
            selected=[cases[seed,key,arm] for key in keys];geometry=source_geometry[seed,arm]
            require(sum(r['targets'] for r in selected)==1016,'Complete occurrence denominator must remain1016per seed/arm.')
            require(sum(r['node_budget'] for r in selected)==geometry['summed_injected_nodes'] and
                sum(r['edge_budget'] for r in selected)==geometry['summed_nonself_undirected_edges'],'Saved physical totals differ from the original audit.')
            for detector in ('gcn','gaga'):
                counts=dict(occurrences=sum(r['targets'] for r in selected))
                counts.update({field:sum(r[detector+'_'+field] for r in selected) for field in FIELDS[1:]})
                mr=100*counts['attacked_wrong']/counts['occurrences']
                macro=statistics.fmean(100*r[detector+'_attacked_wrong']/r['targets'] for r in selected)
                recorded=metrics[seed,arm,detector];original=source_metrics[seed,arm,detector]
                require(all(counts[f]==recorded[f]==original[f] for f in FIELDS),'Seed counts differ from compact metrics/original audit.')
                for field,actual in (('mr_percent',mr),('group_macro_mr_percent',macro)):
                    close(actual,recorded[field],'Recounted micro/macro MR differs.');close(actual,original[field],'MR differs from the original audit.')
        seed_differences.append(metrics[seed,'balanced_target_fill','gaga']['mr_percent']-metrics[seed,'baseline','gaga']['mr_percent'])
    require((paired['gained_wrong'],paired['lost_wrong'],wins,ties,losses)==(184,17,112,267,5),
        'All positive AND negative primary cases must be retained.')
    cohort=latest['cohort']
    require(cohort['seeds']==list(SEEDS) and cohort['groups_per_seed']==128 and cohort['group_seed_cases']==384
        and cohort['target_occurrences_per_seed']==1016 and cohort['unique_targets_per_seed']==audit['cohort_metadata']['unique_targets']==373
        and cohort['correlated_occurrences_all_seeds']==3048,'Recorded cohort scope differs; unique targets are inherited audited metadata, not re-deduplicated here.')
    for arm in ARMS:
        close(statistics.fmean(metrics[s,arm,'gaga']['mr_percent'] for s in SEEDS),latest['gaga_mean_mr_percent'][arm],'Mean GAGA MR differs.')
    for detector in ('gcn','gaga'):
        close(100*metrics[3,'baseline',detector]['clean_wrong']/1016,latest['clean_mr_percent'][detector],'Clean MR uses a different denominator.')
    mean=statistics.fmean(seed_differences);sd=statistics.stdev(seed_differences)
    half=(math.sqrt(2)*.95/math.sqrt(1-.95**2))*sd/math.sqrt(3);interval=[mean-half,mean+half]
    recorded=latest['primary_result'];original=audit['paired_contrasts']['balanced_target_fill-minus-baseline_gaga']
    close(mean,recorded['mean_gain_pp'],'Primary mean gain differs.')
    for actual,wanted in zip(seed_differences,recorded['seed_gains_pp']):close(actual,wanted,'Primary seed gain differs.')
    for actual,wanted,old in zip(interval,recorded['nominal_paired_t95_pp'],original['paired_t95_pp']):
        close(actual,wanted,'Analytic df2 interval differs.');close(actual,old,'Interval differs from original audit.')
    control_mean=statistics.fmean(metrics[s,'balanced_target_fill','gaga']['mr_percent']-metrics[s,'candidate_rewire','gaga']['mr_percent'] for s in SEEDS)
    close(control_mean,recorded['mean_advantage_over_candidate_rewire_pp'],'Matched candidate-control contrast differs.')
    criterion=mean>=1 and all(v>0 for v in seed_differences) and interval[0]>0 and control_mean>0
    require(criterion is False and recorded['criterion_met'] is False and audit['primary_development_criterion']['criterion_met'] is False
        and recorded['controls_promoted'] is False and recorded['qualified_candidate'] is None,'Frozen criterion/control policy was changed.')
    require(recorded['paired_counts']==dict(gained_wrong=184,lost_wrong=17,net_wrong=167,group_wins=112,group_ties=267,group_losses=5),
        'Published pair totals differ.')
    require(latest['costs']['GPU_hours']==0 and latest['costs']['TEST_queries']==0 and latest['new_experiments_or_queries']==0,
        'Curation must not invent GPU/TEST/new research work.')
    for key in ('process_cpu_seconds','wall_seconds'):
        close(latest['costs'][key],audit['producer_'+('cpu_seconds' if key=='process_cpu_seconds' else 'wall_seconds')],'Original runtime receipt differs.')
    history=rows('previous_studies.csv');require(bool(history),'Compact history is missing.')
    for row in history:
        close(float(row['method_mr_percent'])-float(row['baseline_mr_percent']),row['delta_pp'],'A historical within-phase delta differs.')
        require(row['criterion_met']=='false' and row['source_path'] in provenance['source_json_records']
            and '/blob/'+SNAPSHOT+'/' in row['source_url'],'Historical scope/source labels differ.')
    print(json.dumps(dict(status='passed_saved_count_curation_checks',checks=checks,case_rows=1536,retained_group_seed_cases=384,
        seed_metric_rows=24,primary_mean_gain_pp=mean,nominal_paired_t95_pp=interval,criterion_met=criterion,
        independent_raw_argmax_replay=False,human_verification=False,new_model_or_dataset_calls=0)))


if __name__=='__main__':main()
