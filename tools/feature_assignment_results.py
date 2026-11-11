"""Export/check four compact saved-count tables; no neural or dataset imports."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import pathlib
import re
import statistics

ARMS=('baseline','relation_assignment','gcn_assignment','matched_random_permutation')
SEEDS=(3,4,5)
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
CHANGES=('raw_moved_positions','final_moved_positions','final_changed_feature_rows','final_canonical_changed_coordinates')
CASE_COLUMNS=('opaque_case_key','seed','arm','targets','node_budget','edge_budget','accepted')+CHANGES+tuple(
    detector+'_'+field for detector in ('gaga','gcn') for field in FIELDS[1:])
METRIC_COLUMNS=('seed','arm','detector')+FIELDS+('mr_percent','group_macro_mr_percent','unique_targets')
QUALIFICATION=dict(primary='relation_assignment',mean_gain_at_least_pp=1,all_three_seed_deltas_positive=True,
    paired_t95_lower_positive=True,mean_above_both_controls=True,controls_not_candidates=True)
FILES=('summary.json','seed_metrics.csv','case_counts.csv')


def require(value,message):
    if not value:raise AssertionError(message)


def digest(raw):return hashlib.sha256(raw).hexdigest()
def load(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def csv_rows(path):
    with pathlib.Path(path).open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
def encoded(value):return (json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode()
def csv_bytes(columns,rows):
    handle=io.StringIO(newline='');writer=csv.DictWriter(handle,fieldnames=columns,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return handle.getvalue().encode()
def close(actual,wanted):
    require(math.isfinite(float(actual)) and math.isfinite(float(wanted))
        and abs(float(actual)-float(wanted))<=1e-9+1e-10*abs(float(wanted)),'Saved-count arithmetic differs.')
def interval(values):
    mean=statistics.fmean(values);sd=statistics.stdev(values)
    half=(math.sqrt(2)*.95/math.sqrt(1-.95**2))*sd/math.sqrt(3)
    return dict(seed_differences_pp=values,mean_difference_pp=mean,sample_sd_pp=sd,
        paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2)


def parse_cases(raw):
    rows=[];seen=set()
    for record in raw:
        require(set(record)==set(CASE_COLUMNS),'Compact case columns differ.')
        row={k:(v if k in ('opaque_case_key','arm') else int(v)) for k,v in record.items()}
        require(re.fullmatch('[0-9a-f]{20}',row['opaque_case_key']) is not None,'Malformed opaque case key.')
        key=(row['seed'],row['opaque_case_key'],row['arm'])
        require(key not in seen and row['seed'] in SEEDS and row['arm'] in ARMS,'Duplicate/unknown case arm.')
        require(all(v>=0 for v in row.values() if isinstance(v,int)),'Negative case count.')
        require(row['targets']>0 and row['edge_budget']>=row['node_budget']>=1,'Invalid scenario cost.')
        require(row['accepted'] in (0,1),'Invalid acceptance flag.')
        require(row['final_changed_feature_rows']<=row['final_moved_positions']<=row['node_budget']
            and row['raw_moved_positions']<=row['node_budget'],'Invalid moved-row counts.')
        if row['arm']=='baseline':
            require(row['accepted']==0 and not any(row[k] for k in CHANGES),'Baseline must be unchanged/unfiltered.')
        elif not row['accepted']:
            require(not any(row[k] for k in CHANGES[1:]),'Rejected proposal must retain baseline features/input.')
        else:require(row['final_moved_positions']==row['raw_moved_positions'],'Accepted permutation differs from raw proposal.')
        for detector in ('gaga','gcn'):
            n=row['targets'];cw=row[detector+'_clean_wrong'];aw=row[detector+'_attacked_wrong']
            nw=row[detector+'_new_wrong'];rec=row[detector+'_recoveries']
            require(0<=cw<=n and 0<=aw<=n and nw<=n-cw and rec<=cw and aw==cw+nw-rec,
                'Clean/new/recovery identity or denominator differs.')
        seen.add(key);rows.append(row)
    require(len(rows)==1536 and len({(r['seed'],r['opaque_case_key']) for r in rows})==384,
        'Retain all384 cases and1536 arm-case rows.')
    return rows


def recount(rows,cohort):
    require(cohort['groups_per_seed']==128 and cohort['correlated_occurrences_all_seeds']==3*cohort['target_occurrences_per_seed'],
        'Cohort scope differs.')
    cases={(r['seed'],r['opaque_case_key'],r['arm']):r for r in rows};metrics=[];lookup={}
    for seed in SEEDS:
        keys=sorted({r['opaque_case_key'] for r in rows if r['seed']==seed})
        require(len(keys)==128,'Each seed retains128 cases.')
        for key in keys:
            base=cases[seed,key,'baseline']
            for arm in ARMS:
                row=cases[seed,key,arm]
                require(all(row[k]==base[k] for k in ('targets','node_budget','edge_budget','gaga_clean_wrong','gcn_clean_wrong')),
                    'An arm changes cohort, clean channel or physical cost.')
        for arm in ARMS:
            selected=[cases[seed,key,arm] for key in keys];n=sum(r['targets'] for r in selected)
            require(n==cohort['target_occurrences_per_seed'],'Incomplete occurrence denominator.')
            for detector in ('gaga','gcn'):
                value=dict(seed=seed,arm=arm,detector=detector,occurrences=n,unique_targets=cohort['unique_targets_per_seed'])
                value.update({f:sum(r[detector+'_'+f] for r in selected) for f in FIELDS[1:]})
                value.update(mr_percent=100*value['attacked_wrong']/n,
                    group_macro_mr_percent=statistics.fmean(100*r[detector+'_attacked_wrong']/r['targets'] for r in selected))
                metrics.append(value);lookup[seed,arm,detector]=value
    contrasts={}
    pairs=[(a,'baseline') for a in ARMS[1:]]+[('relation_assignment',a) for a in ARMS[2:]]
    for arm,control in pairs:
        for detector in ('gaga','gcn'):
            values=[lookup[s,arm,detector]['mr_percent']-lookup[s,control,detector]['mr_percent'] for s in SEEDS]
            value=interval(values);counts=collections.Counter({key:0 for key in
                ('group_wins','group_losses','group_ties','net_wrong','occurrences','net_new_wrong','net_recoveries')})
            for seed,key in sorted({(r['seed'],r['opaque_case_key']) for r in rows}):
                after,before=cases[seed,key,arm],cases[seed,key,control]
                delta=after[detector+'_attacked_wrong']-before[detector+'_attacked_wrong']
                counts['group_wins' if delta>0 else 'group_losses' if delta<0 else 'group_ties']+=1
                counts['net_wrong']+=delta;counts['occurrences']+=after['targets']
                counts['net_new_wrong']+=after[detector+'_new_wrong']-before[detector+'_new_wrong']
                counts['net_recoveries']+=after[detector+'_recoveries']-before[detector+'_recoveries']
            require(counts['net_wrong']==counts['net_new_wrong']-counts['net_recoveries'],'Paired count identity differs.')
            value['recounted_paired_counts']=dict(counts);contrasts[arm+'-minus-'+control+'_'+detector]=value
    primary=contrasts['relation_assignment-minus-baseline_gaga']
    controls={a:contrasts['relation_assignment-minus-'+a+'_gaga']['mean_difference_pp'] for a in ARMS[2:]}
    criterion=dict(registered=QUALIFICATION,mean_gain_pp=primary['mean_difference_pp'],
        all_three_seed_deltas_positive=all(v>0 for v in primary['seed_differences_pp']),
        nominal_paired_t95_pp=primary['paired_t95_pp'],mean_advantages_over_controls_pp=controls,
        criterion_met=bool(primary['mean_difference_pp']>=1 and all(v>0 for v in primary['seed_differences_pp'])
            and primary['paired_t95_pp'][0]>0 and all(v>0 for v in controls.values())),
        controls_promoted=False,final_TEST_allowed=False,established_control_superiority_claimed=False,
        limitation='Control advantages are means; qualification does not establish positive control-comparison confidence intervals.')
    counts={}
    for arm in ARMS:
        subset=[r for r in rows if r['arm']==arm]
        counts[arm]=dict(accepted=sum(r['accepted'] for r in subset),
            rejected=None if arm=='baseline' else sum(not r['accepted'] for r in subset),
            raw_nonidentity_cases=sum(r['raw_moved_positions']>0 for r in subset),
            final_nonidentity_cases=sum(r['final_moved_positions']>0 for r in subset),
            changed_feature_cases=sum(r['final_changed_feature_rows']>0 for r in subset))
    return metrics,contrasts,criterion,counts


def export(args):
    audit_path=pathlib.Path(args.audit);protocol_path=pathlib.Path(args.protocol);main_path=pathlib.Path(args.main_result)
    audit=load(audit_path);p=load(protocol_path);main=load(main_path);protocol_sha=digest(protocol_path.read_bytes())
    provenance=dict(study_source_commit=p['source_manifest']['source_commit'],runtime_commit=p['runtime_commit'],protocol_sha256=protocol_sha)
    require(audit['status']=='passed' and audit['schema']=='independent_feature_assignment_saved_tensor_audit_v1'
        and audit['provenance']==provenance and main['status']=='completed' and main['provenance']==provenance,
        'A completed source-bound study and passed matching audit are required.')
    require(p['schema']=='monti_feature_assignment_validation_v1' and p['status']=='frozen' and p['post_hoc'] is True
        and tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['qualification']==QUALIFICATION and p['final_TEST_allowed'] is False,
        'Registered scope/qualification differs.')
    folder=audit_path.parent;csv_path=folder/'all_case_counts.csv';hashes_path=folder/'input_hashes.json';scope_path=folder/'registered_scope.json'
    require(digest(csv_path.read_bytes())==audit['case_counts_csv_sha256'],'Audited case CSV bytes differ.')
    hashes=load(hashes_path);scope=load(scope_path);main_sha=digest(main_path.read_bytes())
    candidates=[r['sha256'] for r in hashes if r['path'].replace('\\','/').endswith('/main_attempt1/result.json')]
    require(candidates==[main_sha] and scope['protocol_sha256']==protocol_sha
        and re.fullmatch('[0-9a-f]{64}',args.audit_source_sha256) is not None
        and scope['audit_source_sha256']==args.audit_source_sha256,
        'Original producer/auditor input hash binding differs.')
    raw=csv_rows(csv_path);converted=[];case_groups={};groups=set(p['valid_groups'])
    for value in raw:
        seed,group=int(value['seed']),int(value['group'])
        require(group in groups and seed in SEEDS,'Case lies outside frozen cohort.')
        key=digest(json.dumps([protocol_sha,seed,group,'feature-assignment-publiccase-v1'],separators=(',',':')).encode())[:20]
        require(case_groups.setdefault((seed,key),group)==group,'Opaque case-key collision.')
        require(value['accepted'] in ('True','False'),'Unexpected audit acceptance serialization.')
        row={k:value[k] for k in CASE_COLUMNS if k not in ('opaque_case_key','accepted')}
        row.update(opaque_case_key=key,accepted=int(value['accepted']=='True'));converted.append(row)
    rows=parse_cases(converted);metrics,contrasts,criterion,counts=recount(rows,audit['cohort'])
    originals={(r['seed'],r['arm'],r['detector']):r for r in audit['metrics']}
    require(len(originals)==len(audit['metrics'])==24,'Audited seed metrics incomplete.')
    for row in metrics:
        old=originals[row['seed'],row['arm'],row['detector']]
        require(all(row[f]==old[f] for f in FIELDS+('unique_targets',)),'Case counts disagree with raw-argmax audit.')
        for field in ('mr_percent','group_macro_mr_percent'):close(row[field],old[field])
    for key,value in contrasts.items():
        old=audit['paired_contrasts'][key]
        for field in ('mean_difference_pp','sample_sd_pp'):close(value[field],old[field])
        for field in ('seed_differences_pp','paired_t95_pp'):
            for x,y in zip(value[field],old[field]):close(x,y)
        require(all(value['recounted_paired_counts'][f]==old['paired_counts'].get(f,0)
            for f in ('occurrences','group_wins','group_ties','group_losses')),'Audited paired case counts differ.')
    require(criterion['criterion_met']==audit['primary_qualification']['criterion_met']
        and audit['qualified_candidate']==('relation_assignment' if criterion['criterion_met'] else None),
        'Registered qualification recount differs.')
    for arm in ARMS[1:]:
        require(all(counts[arm][kind]==audit['acceptance_counts'][arm+'_'+kind] for kind in ('accepted','rejected')),
            'Audited acceptance counts differ.')
    require(main['counters']==audit['producer_counters'] and all(main['counters'][k]==0 for k in ('optimizer_updates','API_calls','TEST_queries')),
        'Producer operation accounting differs.')
    summary=dict(schema='monti_feature_assignment_compact_v1',method='Own-GAGA-informed bijective injected-feature assignment',
        split='GossipCop-S split0 development VALID; final TEST closed',
        baseline='Frozen MonTi followed by the existing random-target-fill topology; fresh cohort, not the earlier study paired control',
        metric='MR includes ALL fraud-target occurrences and clean errors; higher means a stronger attack',
        cohort=audit['cohort'],mean_mr_percent={d:{a:statistics.fmean(r['mr_percent'] for r in metrics if r['arm']==a and r['detector']==d) for a in ARMS} for d in ('gaga','gcn')},
        paired_contrasts=contrasts,primary_qualification=criterion,acceptance_and_change_counts=counts,
        physical_cost_per_seed=audit['physical_cost_per_seed'],
        costs=dict(main_process_cpu_seconds=main['process_cpu_seconds'],main_wall_seconds=main['wall_seconds'],
            phase_wall_seconds=audit['accounting']['phase_wall_seconds'],GPU_hours=audit['accounting']['charged_GPU_hours'],
            main_counters=main['counters'],preflight_counters=audit['preflight']['counters'],
            GCN_attacked_channel='Cached own-GCN logits; not independent fresh neural replay'),
        interpretation=['Three generator seeds, repeated targets and adaptively inspected VALID contexts; conditional df2 uncertainty, not IID or untouched confirmation.',
            'The linear solver optimizes saved first-order costs, not the nonlinear victim. The own-GCN acceptance guard does not protect official GAGA.',
            'Raw cycle matching does not match accepted displacement or prove agent, stealth, novelty or control superiority.'],
        new_model_calls_by_exporter=0,human_verification=False)
    outputs={'summary.json':encoded(summary),'seed_metrics.csv':csv_bytes(METRIC_COLUMNS,metrics),'case_counts.csv':csv_bytes(CASE_COLUMNS,rows)}
    public_provenance=dict(schema='monti_feature_assignment_compact_provenance_v1',**provenance,
        exporter_source_sha256=digest(pathlib.Path(__file__).read_bytes()),
        exporter_scope='Saved-count report authored after the scientific model-run source; not part of that earlier source archive',
        original_audit_result_sha256=digest(audit_path.read_bytes()),original_main_result_sha256=main_sha,
        original_audited_case_csv_sha256=audit['case_counts_csv_sha256'],original_audit_input_hashes_sha256=digest(hashes_path.read_bytes()),
        original_audit_scope_sha256=digest(scope_path.read_bytes()),audit_source_sha256=scope['audit_source_sha256'],
        audit_status='passed',audit_independent_of_method_author=audit['independent_of_method_implementation_authorship'],human_verification=False,
        compact_files={name:digest(raw) for name,raw in outputs.items()},
        scope='Derived saved-count tables, not original executed JSON or a new independent model/tensor audit. No raw target/group IDs, logits or private absolute paths are exported.')
    outputs['provenance.json']=encoded(public_provenance)
    destination=pathlib.Path(args.output);destination.mkdir(parents=True,exist_ok=False)
    for name,raw in outputs.items():
        with (destination/name).open('xb') as handle:handle.write(raw)
    print(json.dumps(dict(status='exported_saved_counts',files=4,case_rows=1536,seed_metrics=24,new_model_queries=0)))


def check(args):
    folder=pathlib.Path(args.results);p=load(folder/'provenance.json');summary=load(folder/'summary.json')
    require(p['schema']=='monti_feature_assignment_compact_provenance_v1' and p['audit_status']=='passed'
        and set(p['compact_files'])==set(FILES),'Compact provenance differs.')
    for name,wanted in p['compact_files'].items():require(digest((folder/name).read_bytes())==wanted,'Compact file hash differs.')
    metrics,contrasts,criterion,counts=recount(parse_cases(csv_rows(folder/'case_counts.csv')),summary['cohort'])
    require(summary['paired_contrasts']==contrasts and summary['primary_qualification']==criterion
        and summary['acceptance_and_change_counts']==counts,'Compact summary arithmetic differs.')
    saved=csv_rows(folder/'seed_metrics.csv')
    require(len(saved)==len(metrics)==24,'Incomplete compact seed metrics.')
    for row,wanted in zip(saved,metrics):
        require(set(row)==set(METRIC_COLUMNS),'Seed metric columns differ.')
        for field in METRIC_COLUMNS:
            if field in ('arm','detector'):require(row[field]==wanted[field],'Seed metric order differs.')
            else:close(row[field],wanted[field])
    for detector in ('gaga','gcn'):
        for arm in ARMS:close(summary['mean_mr_percent'][detector][arm],statistics.fmean(r['mr_percent'] for r in metrics if r['detector']==detector and r['arm']==arm))
    require(summary['new_model_calls_by_exporter']==0 and summary['human_verification'] is False
        and all(summary['costs']['main_counters'][k]==0 for k in ('optimizer_updates','API_calls','TEST_queries')),
        'Export must not invent model/training/API/TEST work.')
    print(json.dumps(dict(status='passed_saved_count_compact_checks',case_rows=1536,group_seed_cases=384,
        criterion_met=criterion['criterion_met'],new_model_queries=0,independent_raw_argmax_replay=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    one=sub.add_parser('export')
    for field in ('audit','protocol','main-result','audit-source-sha256','output'):one.add_argument('--'+field,required=True)
    two=sub.add_parser('check');two.add_argument('--results',required=True)
    args=parser.parse_args();(export if args.command=='export' else check)(args)


if __name__=='__main__':main()
