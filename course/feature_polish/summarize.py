"""Saved-JSON-only five-arm paired reporting; no ML imports or tensor loads."""
import argparse
import collections
import json
import math
import pathlib
import statistics
import time

from course.feature_polish.common import ARMS,sha256,write_json


SEEDS=(3,4,5)
FIELDS=('occurrences','clean_wrong','attacked_wrong','new_wrong','recoveries')
PAIR_FIELDS=('occurrences','baseline_wrong','candidate_wrong','shared_new_wrong',
    'baseline_new_success_lost','new_success_gained','additional_recovery','previous_recovery_lost')


def classify(values):
    if not values or any(len(row)!=2 or not all(math.isfinite(float(x)) for x in row) for row in values):
        raise RuntimeError('Finite two-class raw logits are required.')
    return [0 if row[0]>=row[1] else 1 for row in values]


def transitions(labels,clean,attacked):
    if not len(labels)==len(clean)==len(attacked):raise RuntimeError('Raw prediction denominator differs.')
    return dict(occurrences=len(labels),clean_wrong=sum(a!=y for a,y in zip(clean,labels)),
        attacked_wrong=sum(b!=y for b,y in zip(attacked,labels)),
        new_wrong=sum(a==y and b!=y for a,b,y in zip(clean,attacked,labels)),
        recoveries=sum(a!=y and b==y for a,b,y in zip(clean,attacked,labels)))


def paired_counter(labels,clean,baseline,candidate):
    result={key:0 for key in PAIR_FIELDS}
    for y,cc,ac,bc in zip(labels,clean,baseline,candidate):
        cw,aw,bw=cc!=y,ac!=y,bc!=y
        result['occurrences']+=1;result['baseline_wrong']+=aw;result['candidate_wrong']+=bw
        if not cw:
            result['shared_new_wrong']+=aw and bw
            result['baseline_new_success_lost']+=aw and not bw
            result['new_success_gained']+=bw and not aw
        else:
            result['additional_recovery']+=aw and not bw
            result['previous_recovery_lost']+=bw and not aw
    delta=result['candidate_wrong']-result['baseline_wrong']
    if delta!=result['new_success_gained']-result['baseline_new_success_lost']-result['additional_recovery']+result['previous_recovery_lost']:
        raise RuntimeError('Paired transition identity failed.')
    return dict(result,delta_wrong=delta)


def regime(row):
    return ('one_node_one_edge' if row['edge_budget']==1 else 'one_node_multiple_edges') if row['node_budget']==1 else 'multiple_nodes'


def interval(values):
    mean=statistics.mean(values);sd=statistics.stdev(values)
    half=4.302652729911275*sd/math.sqrt(3)
    return dict(mean_delta_mr_pp=mean,sample_sd_delta_pp=sd,paired_t95_pp=[mean-half,mean+half],
        attack_seeds=3,degrees_of_freedom=2,consistent_development_gain=mean>=1 and all(x>0 for x in values))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evaluation',required=True)
    parser.add_argument('--matrix',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();cpu,wall=time.process_time(),time.monotonic()
    evaluation,matrix=pathlib.Path(args.evaluation).resolve(),pathlib.Path(args.matrix).resolve()
    output=pathlib.Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=[];seen_paths=set()
    def remember(path):
        path=path.resolve()
        if path not in seen_paths:
            inputs.append(dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size));seen_paths.add(path)
    def read_json(path):
        remember(path);return json.loads(path.read_text(encoding='utf-8'))
    def read_rows(path):
        remember(path)
        with path.open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
    def indexed(rows):
        result={row['group']:row for row in rows}
        if len(result)!=len(rows) or len(result)!=128:raise RuntimeError('Duplicate/incomplete fixed128group cohort.')
        return result
    result=read_json(evaluation/'result.json')
    if result['status']!='completed' or len(result['summaries'])!=15:
        raise RuntimeError('All five arms/three seeds must complete before reporting.')
    provenance=result['provenance']
    clean=indexed(read_rows(evaluation/'clean_predictions.jsonl'))
    unique={node for row in clean.values() for node in row['node_ids']}
    if len(unique)!=374 or sum(len(row['labels']) for row in clean.values())!=1091:
        raise RuntimeError('Clean occurrence/unique target denominators differ.')
    clean_predictions={}
    for group,row in clean.items():
        if len(row['node_ids'])!=len(row['labels']) or any(y!=1 for y in row['labels']):
            raise RuntimeError('The registered fraud-target cohort changed.')
        clean_predictions[group]={name:classify(row[name+'_logits']) for name in ('gaga','gcn')}
    evaluation_summaries={(row['arm'],row['seed']):row for row in result['summaries']}
    if set(evaluation_summaries)!={(arm,seed) for arm in ARMS for seed in SEEDS}:
        raise RuntimeError('Evaluation arm/seed coverage differs.')
    metrics=[];all_attacks={};own_rows=[];branch_statistics=[];own_losses=[];guard_totals=collections.Counter()
    refinement_counter_totals=collections.Counter();captures_checked=0
    for seed in SEEDS:
        directory=matrix/('seed%d'%seed)
        receipt=read_json(directory/'result.json');registered=read_json(directory/'registered.json')
        if (receipt['status']!='completed' or receipt['seed']!=seed or receipt['groups']!=128 or
                registered['provenance']!=provenance or tuple(registered['arms'])!=ARMS):
            raise RuntimeError('A complete source/protocol-bound refinement job is required.')
        path=directory/'groups.jsonl'
        if sha256(path)!=receipt['groups_jsonl_sha256']:raise RuntimeError('Refinement own-output rows changed.')
        own=indexed(read_rows(path))
        if set(own)!=set(clean):raise RuntimeError('Refinement cohort differs from clean evaluation.')
        refinement_counter_totals.update(receipt['counters'])
        if len(receipt['exports'])!=128:raise RuntimeError('A refinement capture list is incomplete.')
        for export in receipt['exports']:
            relative=pathlib.Path(export['path'])
            if relative.is_absolute() or '..' in relative.parts:raise RuntimeError('Unsafe private capture path.')
            capture=directory/relative
            remember(capture)
            if sha256(capture)!=export['sha256']:raise RuntimeError('A fixed graph/branch-feature tensor file changed.')
            captures_checked+=1
        for group,row in own.items():
            if (row['node_ids']!=clean[group]['node_ids'] or row['labels']!=clean[group]['labels'] or
                    row['targets']!=len(row['labels']) or row['seed']!=seed):
                raise RuntimeError('Refinement target identities/order/labels differ.')
            if set(row['final_own_logits'])!=set(ARMS) or set(row['feature_statistics'])!=set(ARMS):
                raise RuntimeError('A branch lacks retained own logits or feature statistics.')
            bp=classify(row['final_own_logits']['baseline']['gcn'])
            cp=classify(row['final_own_logits']['gaga_pgd']['gcn'])
            accepted=all(a!=0 or b==0 for a,b in zip(bp,cp))
            if accepted!=row['guard']['accepted']:raise RuntimeError('Whole-group GCN-preservation gate does not recount.')
            alias='gaga_pgd' if accepted else 'baseline'
            if (row['final_own_logits']['gaga_guarded']!=row['final_own_logits'][alias] or
                    row['feature_statistics']['gaga_guarded']['content_sha256']!=row['feature_statistics'][alias]['content_sha256']):
                raise RuntimeError('Guarded final predictions/features differ from the chosen whole-group alias.')
            if any(a==0 and b!=0 for a,b in zip(bp,classify(row['final_own_logits']['gaga_guarded']['gcn']))):
                raise RuntimeError('Guarded arm loses a protected original-GCN class0 prediction.')
            guard_totals['groups']+=1;guard_totals['accepted' if accepted else 'fallback']+=1
            guard_totals['empty_protected_set']+=int(row['guard']['empty_protected_set'])
            for arm in ARMS:
                statistics_row=row['feature_statistics'][arm]
                for key,value in statistics_row.items():
                    if isinstance(value,float) and not math.isfinite(value):raise RuntimeError('Nonfinite retained feature statistic.')
                branch_statistics.append(dict(arm=arm,seed=seed,**statistics_row))
                for teacher in ('gcn','gaga_teacher'):
                    raw=row['final_own_logits'][arm][teacher];classify(raw)
                    differences=[values[1]-values[0] for values in raw]
                    value=statistics.mean(max(x,0)+math.log1p(math.exp(-abs(x))) for x in differences)
                    if not math.isclose(value,row['final_own_losses'][arm][teacher],rel_tol=1e-5,abs_tol=1e-5):
                        raise RuntimeError('Retained own-teacher Softplus does not reconcile with raw logits.')
                    own_losses.append(dict(arm=arm,seed=seed,teacher=teacher,group=group,targets=row['targets'],
                        before=row['final_own_losses']['baseline'][teacher],after=row['final_own_losses'][arm][teacher]))
            own_rows.append(row)
        for arm in ARMS:
            attacks=indexed(read_rows(evaluation/('%s_seed%d_predictions.jsonl'%(arm,seed))))
            if set(attacks)!=set(clean):raise RuntimeError('An evaluation arm drops a case.')
            all_attacks[arm,seed]=attacks
            nodes=sum(row['actual_injected_nodes'] for row in attacks.values())
            edges=sum(row['actual_injected_edges'] for row in attacks.values())
            if (nodes,edges)!=(7776,358491):raise RuntimeError('Actual summed physical budgets differ.')
            for detector in ('gaga','gcn'):
                totals=collections.Counter();group_rates=[]
                for group,row in attacks.items():
                    base=clean[group];own_row=own[group]
                    if (row['node_ids']!=base['node_ids'] or row['labels']!=base['labels'] or
                            row['common_capture_sha256']!=own_row['graph_capture']['sha256'] or
                            row['injected_feature_content_sha256']!=own_row['feature_statistics'][arm]['content_sha256'] or
                            row['hard_adjacency_content_sha256']!=own_row['adjacency_content_sha256']):
                        raise RuntimeError('Clean/attack/feature/shared-graph identities differ.')
                    if row['gcn_logits']!=own_row['final_own_logits'][arm]['gcn'] or row['gcn_logits_reused'] is not True:
                        raise RuntimeError('Stored attacked original-GCN channel differs from exact refinement output.')
                    pred=classify(row[detector+'_logits'])
                    values=transitions(base['labels'],clean_predictions[group][detector],pred)
                    if values!=row[detector+'_transitions']:raise RuntimeError('Raw argmax transition recount differs.')
                    totals.update(values);group_rates.append(100*values['attacked_wrong']/values['occurrences'])
                if totals['occurrences']!=1091 or dict(totals)!=evaluation_summaries[arm,seed]['totals'][detector]:
                    raise RuntimeError('Raw recount differs from recorded evaluation totals.')
                mr=100*totals['attacked_wrong']/totals['occurrences']
                if not math.isclose(mr,evaluation_summaries[arm,seed][detector+'_mr_percent'],abs_tol=1e-10):
                    raise RuntimeError('Raw MR differs from evaluation receipt.')
                metrics.append(dict(arm=arm,seed=seed,detector=detector,**{key:totals[key] for key in FIELDS},
                    mr_percent=mr,group_macro_mr_percent=statistics.mean(group_rates),unique_original_targets=374,
                    summed_injected_nodes=nodes,summed_injected_nonself_undirected_edges=edges,
                    attacked_gcn_channel_reused=(detector=='gcn')))
    group_pairs=[];comparisons={}
    metric_lookup={(r['arm'],r['seed'],r['detector']):r for r in metrics}
    for arm in ARMS[1:]:
        comparisons[arm]={}
        for detector in ('gaga','gcn'):
            totals=collections.Counter();pairs=[];seeds=[]
            for seed in SEEDS:
                a=metric_lookup['baseline',seed,detector];b=metric_lookup[arm,seed,detector]
                seeds.append(dict(seed=seed,delta_mr_pp=b['mr_percent']-a['mr_percent'],
                    delta_group_macro_pp=b['group_macro_mr_percent']-a['group_macro_mr_percent']))
                for group in sorted(clean):
                    baseline,candidate=all_attacks['baseline',seed][group],all_attacks[arm,seed][group]
                    for key in ('node_ids','labels','node_budget','edge_budget','common_capture_sha256','hard_adjacency_content_sha256'):
                        if baseline[key]!=candidate[key]:raise RuntimeError('Paired attack graph/identity/budget differs.')
                    values=paired_counter(clean[group]['labels'],clean_predictions[group][detector],
                        classify(baseline[detector+'_logits']),classify(candidate[detector+'_logits']))
                    totals.update(values);totals['group_wins' if values['delta_wrong']>0 else 'group_losses' if values['delta_wrong']<0 else 'group_ties']+=1
                    pair=dict(arm=arm,detector=detector,seed=seed,group=group,targets=len(baseline['labels']),
                        node_budget=baseline['node_budget'],edge_budget=baseline['edge_budget'],**values)
                    pairs.append(pair);group_pairs.append(pair)
            strata=[]
            for name,expected in [('one_node_one_edge',(12,48)),('one_node_multiple_edges',(45,165)),('multiple_nodes',(327,3060))]:
                selected=[row for row in pairs if regime(row)==name];counter=collections.Counter()
                for row in selected:
                    counter.update({key:row[key] for key in PAIR_FIELDS+('delta_wrong',)})
                    counter['group_wins' if row['delta_wrong']>0 else 'group_losses' if row['delta_wrong']<0 else 'group_ties']+=1
                if (len(selected),counter['occurrences'])!=expected:raise RuntimeError('Disjoint paired budget stratum differs.')
                strata.append(dict(regime=name,group_seed_cases=len(selected),**{key:counter[key] for key in PAIR_FIELDS+('delta_wrong','group_wins','group_ties','group_losses')},
                    delta_mr_pp=100*counter['delta_wrong']/counter['occurrences']))
            comparisons[arm][detector]=dict(paired_seed_differences=seeds,**interval([r['delta_mr_pp'] for r in seeds]),
                paired_transition_counts={key:totals[key] for key in PAIR_FIELDS+('delta_wrong','group_wins','group_ties','group_losses')},
                budget_strata=strata)
    aggregate={}
    for detector in ('gaga','gcn'):
        aggregate[detector]={}
        for arm in ARMS:
            selected=[metric_lookup[arm,seed,detector] for seed in SEEDS]
            aggregate[detector][arm]=dict(mean_mr_percent=statistics.mean(row['mr_percent'] for row in selected),
                sample_sd_pp=statistics.stdev(row['mr_percent'] for row in selected),
                mean_group_macro_mr_percent=statistics.mean(row['group_macro_mr_percent'] for row in selected),
                pooled_micro_mr_percent=100*sum(row['attacked_wrong'] for row in selected)/3273,
                pooled_wrong=sum(row['attacked_wrong'] for row in selected),correlated_occurrences=3273,
                unique_original_targets_across_seeds=374)
    def candidate_rule(arm):
        comparison=comparisons[arm]['gaga'];mean=aggregate['gaga'][arm]['mean_mr_percent']
        controls={control:mean-aggregate['gaga'][control]['mean_mr_percent'] for control in ('gcn_pgd','random')}
        teacher_deltas={str(seed):statistics.mean(row['after']-row['before'] for row in own_losses if row['arm']==arm and row['seed']==seed and row['teacher']=='gaga_teacher') for seed in SEEDS}
        teacher_improved=all(value<0 for value in teacher_deltas.values())
        return dict(arm=arm,consistency_criterion_met=comparison['consistent_development_gain'],
            own_gaga_mean_group_loss_delta_by_seed=teacher_deltas,own_gaga_objective_improved_each_seed=teacher_improved,
            mean_advantage_over_controls_pp=controls,better_mean_than_both_controls=all(value>0 for value in controls.values()),
            complete_predeclared_development_rule_met=comparison['consistent_development_gain'] and teacher_improved and all(value>0 for value in controls.values()))
    primary=candidate_rule('gaga_guarded');secondary=candidate_rule('gaga_pgd')
    secondary.update(predeclared_secondary=True,consider_only_if_primary_rule_failed=not primary['complete_predeclared_development_rule_met'],
        does_not_replace_primary_result=True)
    own_aggregate=[];feature_aggregate=[]
    for arm in ARMS:
        for teacher in ('gcn','gaga_teacher'):
            selected=[row for row in own_losses if row['arm']==arm and row['teacher']==teacher]
            own_aggregate.append(dict(arm=arm,teacher=teacher,groups=len(selected),
                mean_group_loss_before=statistics.mean(row['before'] for row in selected),
                mean_group_loss_after=statistics.mean(row['after'] for row in selected),
                mean_group_loss_delta=statistics.mean(row['after']-row['before'] for row in selected),
                lower_loss_groups=sum(row['after']<row['before'] for row in selected),
                equal_loss_groups=sum(row['after']==row['before'] for row in selected),
                higher_loss_groups=sum(row['after']>row['before'] for row in selected)))
        selected=[row for row in branch_statistics if row['arm']==arm]
        feature_aggregate.append(dict(arm=arm,groups=len(selected),
            mean_normalized_displacement_linf=statistics.mean(row['normalized_displacement_linf'] for row in selected),
            max_normalized_displacement_linf=max(row['normalized_displacement_linf'] for row in selected),
            mean_normalized_displacement_l2=statistics.mean(row['normalized_displacement_l2'] for row in selected),
            mean_group_saturation_fraction=statistics.mean(row['saturation_fraction'] for row in selected),
            movable_coordinate_weighted_saturation_fraction=sum(row['saturation_fraction']*row['movable_coordinates'] for row in selected)/sum(row['movable_coordinates'] for row in selected),
            zero_displacement_groups=sum(row['normalized_displacement_linf']==0 for row in selected)))
    public=dict(status='completed_saved_json_refinement_summary',provenance=provenance,arms=list(ARMS),metrics=metrics,
        aggregate_by_detector=aggregate,comparisons_against_new_baseline=comparisons,
        primary_development_rule=primary,predeclared_secondary_candidate=secondary,
        primary_result_replaced_by_secondary=False,own_teacher_loss_summaries=own_aggregate,
        feature_displacement_and_saturation=feature_aggregate,guard_counts=dict(guard_totals),
        refinement_recorded_counter_totals=dict(refinement_counter_totals),evaluation_recorded_counts=result['counts'],
        fixed_graph_captures_byte_checked=captures_checked,
        denominators=dict(original_unique_targets=374,groups_per_seed=128,occurrences_per_arm_seed=1091,
            correlated_occurrences_per_arm_across_seeds=3273,shared_graphs_across_all_five_arms=384),
        limitations=['Development only on repeatedly inspected fixed cohort, not final TEST.',
            'Intervals are over three attack seeds conditional on the fixed full-TRAIN teacher, original GCN/victim, generators and cohort; repeated news memberships are correlated.',
            'Baselines are newly generated on this phase\'s selected execution device; historical52.7%is not a paired baseline.',
            'Attacked original-GCN logits are reused exact refinement records, not independently replayed predictions.',
            'Official GAGA uses fresh canonical hard-graph sequences; surrogate optimization loss reduction is not a guarantee of transfer.',
            'Equal physical node/edge budgets do not imply equal queries, computation or terminal feature movement; random is a fixed-direction control.',
            'The guarded arm protects original-GCN class0 decisions only and does not guarantee official-victim preservation or gains.',
            'Primary and secondary/control comparisons are descriptive development analyses; no familywise confirmatory significance or first-novelty claim.',
            'A positive secondary arm does not replace the predeclared primary result or authorize final TEST, extra seeds or tuning.'],
        model_queries=0,tensor_deserializations=0,API_calls=0,GPU_hours=0,
        process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
    write_json(output/'summary.json',public,exclusive=True)
    write_json(output/'private_group_pairs.json',group_pairs,exclusive=True)
    write_json(output/'input_hashes.json',inputs,exclusive=True)
    write_json(output/'receipt.json',dict(status='completed',source_sha256=sha256(__file__),input_files=len(inputs),
        outputs={name:sha256(output/name) for name in ('summary.json','private_group_pairs.json','input_hashes.json')},
        model_queries=0,tensor_deserializations=0,API_calls=0,GPU_hours=0,
        process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall),exclusive=True)
    print(json.dumps(dict(status=public['status'],primary=primary,secondary=secondary,output=str(output)),allow_nan=False))


if __name__=='__main__':main()
