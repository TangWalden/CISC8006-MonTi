"""Export compact publication tables from immutable saved Git records.

Standard library only. This does not load datasets, models, checkpoints, tensors
or raw predictions. The original independent audit is copied without changes.
"""
import argparse
import csv
import hashlib
import io
import json
import pathlib
import statistics
import subprocess


ROOT=pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT='ca093d8b12ea153a076b02bbdbc4ede58508560c'
SCIENTIFIC_SOURCE='c3238ee62272d39f0337828718a3560abe101a00'
PROTOCOL_SHA256='24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14'
REPOSITORY='https://github.com/TangWalden/CISC8006-MonTi'
PREFIX='evidence/candidate_target_reallocation_v1/'
ARMS=('baseline','balanced_target_fill','random_target_fill','candidate_rewire')
SEEDS=(3,4,5)
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
SEED_COLUMNS=('detector','seed','arm')+FIELDS+('mr_percent','group_macro_mr_percent')
CASE_COLUMNS=('opaque_case_key','seed','arm','targets','node_budget','edge_budget')+tuple(
    detector+'_'+field for detector in ('gcn','gaga') for field in FIELDS[1:])+(
    'gaga_gained_wrong_vs_baseline','gaga_lost_wrong_vs_baseline',
    'gaga_new_success_gained_vs_baseline','gaga_baseline_new_success_lost',
    'gaga_additional_recovery_vs_baseline','gaga_previous_recovery_lost',
    'k','identity','changed_injected_rows','changed_adjacency_coordinates',
    'rows_losing_all_candidate_neighbors','target_injected_attachment_incidence',
    'candidate_attachment_incidence','injected_injected_owned_edges','covered_targets',
    'primary_random_topologies_equal')
HISTORY_COLUMNS=('stage','detector','split','baseline_arm','method_arm',
    'baseline_mr_percent','method_mr_percent','delta_pp','criterion_met','conclusion','source_path','source_url')


def digest(raw):return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False)+'\n').encode('utf-8')


def csv_bytes(columns,rows):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
    writer.writeheader();writer.writerows(rows);return stream.getvalue().encode('utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,default=ROOT/'results')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    sources={};cache={}
    def saved(path):
        if path not in cache:
            raw=subprocess.check_output(['git','show',SNAPSHOT+':'+path],cwd=ROOT)
            sources[path]=dict(sha256=digest(raw),url=REPOSITORY+'/blob/'+SNAPSHOT+'/'+path)
            cache[path]=raw
        return json.loads(cache[path])
    summary=saved(PREFIX+'summary.json');cases=saved(PREFIX+'all_case_counts.json')
    trace=saved(PREFIX+'execution_trace.json');outer=saved(PREFIX+'outer_receipt.json')
    audit=saved(PREFIX+'independent_saved_tensor_audit_v1.json')
    protocol=saved(PREFIX+'execution_protocol.json')
    if (summary['provenance']['source_commit']!=SCIENTIFIC_SOURCE or
            digest(cache[PREFIX+'execution_protocol.json'])!=PROTOCOL_SHA256 or audit['status']!='passed'):
        raise RuntimeError('Immutable source records do not match the completed study.')
    aggregate={(v['arm'],v['detector']):v for v in summary['arm_aggregates']}
    pair=summary['paired_contrasts']['balanced_target_fill-minus-baseline_gaga']
    counts=pair['paired_counts'];cohort=summary['cohort_metadata'];cost=summary['study_resources']
    primary=summary['primary_development_criterion']
    clean={d:next(v['clean_wrong'] for v in summary['metrics'] if v['detector']==d) for d in ('gcn','gaga')}
    latest=dict(
        schema='monti_compact_saved_results_v1',method='Candidate-to-target edge-budget reallocation',
        split='GossipCop-S, split0, development VALID; no new full TEST',
        metric_definition='MR=100*wrong fraud-target predictions/ALL target occurrences, including clean errors; higher MR is a stronger attack. Clean MR uses the same denominator before injection.',
        cohort=dict(seeds=list(SEEDS),groups_per_seed=128,group_seed_cases=384,
            target_occurrences_per_seed=cohort['target_occurrences_per_seed'],unique_targets_per_seed=cohort['unique_targets'],
            correlated_occurrences_all_seeds=summary['correlated_occurrences_all_seeds']),
        gaga_mean_mr_percent={arm:aggregate[arm,'gaga']['mean_micro_mr_percent'] for arm in ARMS},
        clean_mr_percent={d:100*clean[d]/cohort['target_occurrences_per_seed'] for d in ('gcn','gaga')},
        primary_result=dict(arm='balanced_target_fill',mean_gain_pp=pair['mean_difference_pp'],
            seed_gains_pp=pair['seed_differences_pp'],nominal_paired_t95_pp=pair['paired_t95_pp'],
            paired_counts=dict(gained_wrong=counts['gained_wrong'],lost_wrong=counts['lost_wrong'],
                net_wrong=counts['gained_wrong']-counts['lost_wrong'],group_wins=counts['group_wins'],
                group_ties=counts['group_ties'],group_losses=counts['group_losses']),
            mean_advantage_over_candidate_rewire_pp=primary['mean_advantage_over_candidate_rewire_pp'],
            criterion_met=primary['criterion_met'],failed_condition='Nominal paired-t95 lower endpoint is not above zero',
            random_target_fill_need_not_be_worse=True,controls_promoted=False,qualified_candidate=None),
        physical_cost_per_arm_seed=dict(summed_injected_node_instances=summary['geometry_totals'][0]['summed_injected_nodes'],
            summed_nonself_undirected_edges=summary['geometry_totals'][0]['summed_nonself_undirected_edges'],
            scope='Sums128 separate scenarios; all arms preserve physical injected degrees and budgets'),
        costs=dict(process_cpu_seconds=cost['process_cpu_seconds'],wall_seconds=cost['wall_seconds'],
            outer_seconds=outer['outer_seconds'],GPU_hours=cost['GPU_hours'],generator_forwards=384,
            fresh_GCN_forwards=1664,fresh_official_GAGA_forwards=1664,optimizer_updates=0,autograd_calls=0,
            API_calls=0,TEST_queries=0),
        interpretation=['Observed gains in all three seeds; frozen stability criterion failed. No selected candidate.',
            'The n=3/df=2 interval is conditional on fixed models and correlated, adaptively inspected development contexts.',
            'Random target filling is slightly higher; no superior-priority, novelty, camouflage, human-verification or course-completion claim.'],
        historical_formal_TEST_reference='Separate earlier reproduction: evidence/reproduction_summary.json; not the paired control of this development cohort',
        new_experiments_or_queries=0)
    seed_rows=[{name:row[name] for name in SEED_COLUMNS} for row in summary['metrics']]
    case_rows=[]
    for case in cases:
        for arm in ARMS:
            topology=case['topology'];arm_topology=topology['arms'][arm]
            contrast=({name:0 for name in ('gained_wrong','lost_wrong','new_success_gained',
                'baseline_new_success_lost','additional_recovery','previous_recovery_lost')} if arm=='baseline'
                else case['contrasts'][arm+'-minus-baseline']['gaga'])
            row=dict(opaque_case_key=case['case_key'],seed=case['seed'],arm=arm,targets=case['targets'],
                node_budget=case['node_budget'],edge_budget=case['edge_budget'])
            for detector in ('gcn','gaga'):
                for field in FIELDS[1:]:row[detector+'_'+field]=case['outcomes'][arm][detector][field]
            row.update(gaga_gained_wrong_vs_baseline=contrast['gained_wrong'],gaga_lost_wrong_vs_baseline=contrast['lost_wrong'],
                gaga_new_success_gained_vs_baseline=contrast['new_success_gained'],gaga_baseline_new_success_lost=contrast['baseline_new_success_lost'],
                gaga_additional_recovery_vs_baseline=contrast['additional_recovery'],gaga_previous_recovery_lost=contrast['previous_recovery_lost'],
                k=0 if arm=='baseline' else topology['total_reallocated_edges'],identity=int(arm_topology['identity']['exact_equals_baseline']),
                changed_injected_rows=arm_topology['identity']['changed_rows'],changed_adjacency_coordinates=arm_topology['identity']['changed_coordinates'],
                rows_losing_all_candidate_neighbors=0 if arm=='baseline' else topology['rows_losing_all_candidate_edges'],
                target_injected_attachment_incidence=arm_topology['target_incidence'],candidate_attachment_incidence=arm_topology['candidate_incidence'],
                injected_injected_owned_edges=arm_topology['injected_injected_owned_edges'],covered_targets=arm_topology['covered_targets'],
                primary_random_topologies_equal=int(topology['primary_equals_random']))
            case_rows.append(row)
    previous=[]
    def previous_row(stage,path,detector,baseline,method,before,after,scope,conclusion):
        previous.append(dict(stage=stage,detector=detector,split=scope,baseline_arm=baseline,method_arm=method,
            baseline_mr_percent=before,method_mr_percent=after,delta_pp=after-before,
            criterion_met='false',conclusion=conclusion,source_path=path,source_url=sources[path]['url']))
    path='evidence/new_methods_20261103/final_saved_recount.json';v=saved(path)['alignment']['mean_mr_percent']
    for arm in ('always','gated'):
        previous_row('Relationship alignment',path,'GCN','off',arm,v['off'],v[arm],
            'VALID: selected checkpoints,128groups/1091occurrences per seed','Lower attack MR than its own OFF control; not selected')
    path='evidence/dual_surrogate_v1/results.json';v=saved(path)['summaries']
    means={arm:statistics.fmean(r['gaga_mr_percent'] for r in v if r['arm']==arm) for arm in ('gcn_only','gcn_gaga')}
    previous_row('Dual-surrogate feedback',path,'GAGA','gcn_only','gcn_gaga',means['gcn_only'],means['gcn_gaga'],
        'VALID: terminal8,128groups/1091occurrences per seed','Lower in all three seeds; no positive improvement')
    path='evidence/source_role_fix_v1/saved_summary.json';v=saved(path)['gaga_aggregate']
    previous_row('Score-source conformance',path,'GAGA','legacy','source_fixed',v['legacy']['mean_mr_percent'],v['source_fixed']['mean_mr_percent'],
        'VALID: six terminal8models,128groups/1091occurrences per seed','Mixed seed directions; paper conformance is not an efficacy gain')
    path='evidence/feature_polish_v1/results_summary.json';v=saved(path)['aggregate_by_detector']['gaga']
    for arm in ('gaga_pgd','gaga_guarded'):
        previous_row('Fixed-graph feature PGD',path,'GAGA','baseline',arm,v['baseline']['mean_mr_percent'],v[arm]['mean_mr_percent'],
            'VALID: fixed captured graphs,128groups/1091occurrences per seed','Lower in every seed; no selected improvement')
    path='evidence/target_degree_balance_v1/summary.json';v={(r['arm'],r['detector']):r for r in saved(path)['arm_aggregates']}
    previous_row('Target attachment balancing',path,'GAGA','baseline','balanced_target',v['baseline','gaga']['mean_micro_mr_percent'],v['balanced_target','gaga']['mean_micro_mr_percent'],
        'VALID: fixed captured graphs,128groups/1091occurrences per seed','Positive mean; one seed declines and matched random is higher; criterion failed')
    path='evidence/train_prototype_mix_v1/result.json';v=saved(path)['summaries']
    means={arm:statistics.fmean(r['gaga_mr_percent'] for r in v if r['arm']==arm) for arm in ('baseline','benign_mix')}
    previous_row('Fixed TRAIN prototype mix',path,'GAGA','baseline','benign_mix',means['baseline'],means['benign_mix'],
        'VALID: fixed0.5mix,128groups/1091occurrences per seed','No qualified improvement; retain all controls in the source snapshot')
    path='evidence/feature_logit_sharpen_v1/summary.json';v={(r['arm'],r['detector']):r for r in saved(path)['arm_aggregates']}
    previous_row('Fixed feature sharpening',path,'GAGA','baseline','logit_sharp',v['baseline','gaga']['mean_micro_mr_percent'],v['logit_sharp','gaga']['mean_micro_mr_percent'],
        'VALID: separate ID-only128contexts/1102occurrences per seed','Small positive mean; fixed+1pp and positive-CI conditions failed')
    formal=saved('evidence/reproduction_summary.json')
    core_path='course/candidate_target_reallocation/core.py'
    core=subprocess.check_output(['git','show',SNAPSHOT+':'+core_path],cwd=ROOT)
    outputs={'latest.json':json_bytes(latest),'seed_metrics.csv':csv_bytes(SEED_COLUMNS,seed_rows),
        'case_counts.csv':csv_bytes(CASE_COLUMNS,case_rows),'previous_studies.csv':csv_bytes(HISTORY_COLUMNS,previous),
        'audit.json':cache[PREFIX+'independent_saved_tensor_audit_v1.json']}
    provenance=dict(schema='monti_compact_saved_publication_provenance_v1',record_snapshot=SNAPSHOT,
        scientific_execution_source=SCIENTIFIC_SOURCE,original_execution_protocol_sha256=PROTOCOL_SHA256,
        original_execution_protocol_url=sources[PREFIX+'execution_protocol.json']['url'],
        method_core=dict(original_path=core_path,current_path='methods/target_reallocation.py',sha256=digest(core),
            url=REPOSITORY+'/blob/'+SCIENTIFIC_SOURCE+'/'+core_path,copy_policy='Exact bytes; unchanged mathematical core'),
        source_json_records=sources,
        compact_files={name:dict(sha256=digest(raw),kind='Exact original audit bytes' if name=='audit.json' else 'Derived saved-count publication table') for name,raw in outputs.items()},
        formal_reproduction_context=dict(source='evidence/reproduction_summary.json',separate_from_current_development=True,
            learned_mean_mr_percent=formal['learned_mean_mr_percent'],random_mean_mr_percent=formal['random_mean_mr_percent'],
            paired_mean_difference_pp=formal['paired_mean_difference_pp'],paired_t95_pp=formal['paired_95ci_pp'],conclusion=formal['conclusion']),
        interpretation='New compact curation manifest, not an original executed experiment manifest. Paths in the copied audit refer to the immutable historical snapshot/private artifacts.',
        previous_studies_policy='Each row compares only its own phase/cohort/control. Absolute MR values between rows are not paired comparisons; the list is a selected compact history, not an exhaustive review.',
        verification_scope='Saved-count arithmetic and publication-byte checks only, not a new raw-logit argmax recount, model experiment, independent method audit or human verification.',
        new_model_queries=0,new_dataset_or_tensor_loads=0,new_training_or_API_calls=0,
        reproducibility='Exporter reads only git show of the fixed snapshot; fetch full history if a shallow clone lacks that object. Private data/checkpoints are required to reproduce the historical full ML run.')
    outputs['provenance.json']=json_bytes(provenance)
    for name,raw in outputs.items():
        temporary=output/(name+'.tmp');temporary.write_bytes(raw);temporary.replace(output/name)
    print(json.dumps(dict(status='exported_saved_counts',snapshot=SNAPSHOT,seed_metric_rows=len(seed_rows),
        arm_case_rows=len(case_rows),retained_cases=len(cases),previous_study_rows=len(previous),new_queries=0)))


if __name__=='__main__':main()
