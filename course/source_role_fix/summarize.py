"""Saved JSON-only paired development reporting; never loads tensors/models."""
import argparse
import collections
import json
import math
import pathlib
import statistics
import time

from course.source_role_fix.common import sha256, write_json


def rows(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', required=True)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cpu, wall = time.process_time(), time.monotonic()
    evaluation, matrix = pathlib.Path(args.evaluation), pathlib.Path(args.matrix)
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result = json.loads((evaluation/'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'completed':
        raise RuntimeError('Incomplete development evaluation.')
    inputs = []
    def read(path):
        inputs.append(dict(path=str(path), sha256=sha256(path)))
        return rows(path)
    clean = {r['group']: r for r in read(evaluation/'clean_predictions.jsonl')}
    metrics, paired, group_pairs, budget_pairs = [], [], [], []
    totals = collections.Counter()
    for seed in (3, 4, 5):
        arm_rows, initial = {}, []
        for arm in ('legacy', 'source_fixed'):
            directory = matrix/('%s_seed%d' % (arm, seed))
            train = json.loads((directory/'result.json').read_text(encoding='utf-8'))
            if train['status'] != 'completed' or train['optimizer_steps'] != 2048:
                raise RuntimeError('Incomplete fixed-epoch training.')
            initial.append(train['initial_model_state_sha256'])
            arm_rows[arm] = {r['group']: r for r in read(evaluation/('%s_seed%d_predictions.jsonl' % (arm, seed)))}
            if set(arm_rows[arm]) != set(clean) or len(clean) != 128:
                raise RuntimeError('Group denominator mismatch.')
            count = sum(len(r['labels']) for r in arm_rows[arm].values())
            unique = {n for r in arm_rows[arm].values() for n in r['node_ids']}
            if count != 1091 or len(unique) != 374:
                raise RuntimeError('Occurrence/unique-target denominator mismatch.')
            nodes = sum(r['actual_injected_nodes'] for r in arm_rows[arm].values())
            edges = sum(r['actual_injected_edges'] for r in arm_rows[arm].values())
            if (nodes, edges) != (7776, 358491):
                raise RuntimeError('Actual budget totals differ.')
            for detector in ('gaga', 'gcn'):
                for group, attacked in arm_rows[arm].items():
                    baseline = clean[group]
                    if attacked['node_ids'] != baseline['node_ids'] or attacked['labels'] != baseline['labels']:
                        raise RuntimeError('Raw clean/attack target identities differ.')
                    def classify(values):
                        if any(len(v)!=2 or not all(math.isfinite(x) for x in v) for v in values):
                            raise RuntimeError('Nonfinite or malformed raw logits.')
                        return [0 if v[0]>=v[1] else 1 for v in values]
                    clean_pred=classify(baseline[detector+'_logits'])
                    attacked_pred=classify(attacked[detector+'_logits'])
                    labels=baseline['labels']
                    if len(clean_pred)!=len(labels) or len(attacked_pred)!=len(labels):
                        raise RuntimeError('Raw logit denominator differs.')
                    raw=dict(occurrences=len(labels),
                        clean_wrong=sum(a!=y for a,y in zip(clean_pred,labels)),
                        attacked_wrong=sum(a!=y for a,y in zip(attacked_pred,labels)),
                        new_wrong=sum(a==y and b!=y for a,b,y in zip(clean_pred,attacked_pred,labels)),
                        recoveries=sum(a!=y and b==y for a,b,y in zip(clean_pred,attacked_pred,labels)))
                    if raw!=attacked[detector+'_transitions']:
                        raise RuntimeError('Saved transition metrics disagree with raw argmax.')
                channel = [r[detector+'_transitions'] for r in arm_rows[arm].values()]
                summed = {k: sum(v[k] for v in channel) for k in ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')}
                metrics.append(dict(arm=arm, seed=seed, detector=detector, **summed,
                    mr_percent=100*summed['attacked_wrong']/count,
                    group_macro_mr_percent=statistics.mean(100*v['attacked_wrong']/v['occurrences'] for v in channel),
                    unique_targets=374, summed_injected_nodes=nodes, summed_injected_nonself_undirected_edges=edges))
        if len(set(initial)) != 1:
            raise RuntimeError('Paired initialization differs.')
        for group in sorted(clean):
            a, b, c = arm_rows['legacy'][group], arm_rows['source_fixed'][group], clean[group]
            for name in ('node_ids', 'labels', 'node_budget', 'edge_budget'):
                if a[name] != b[name]:
                    raise RuntimeError('Paired targets/order/labels/budgets differ.')
            if a['node_ids'] != c['node_ids'] or a['labels'] != c['labels']:
                raise RuntimeError('Clean/attack identity mismatch.')
            def prediction(row):
                return [0 if x[0] >= x[1] else 1 for x in row['gaga_logits']]
            ap, bp, cp = prediction(a), prediction(b), prediction(c)
            counter = collections.Counter()
            for label, ac, bc, cc in zip(c['labels'], ap, bp, cp):
                aw, bw, cw = ac != label, bc != label, cc != label
                counter['occurrences'] += 1
                counter['legacy_wrong'] += aw
                counter['fixed_wrong'] += bw
                if not cw:
                    counter['shared_new_wrong'] += aw and bw
                    counter['legacy_new_success_lost'] += aw and not bw
                    counter['new_success_gained'] += bw and not aw
                else:
                    counter['additional_recovery'] += aw and not bw
                    counter['previous_recovery_lost'] += bw and not aw
            delta = counter['fixed_wrong']-counter['legacy_wrong']
            if delta != counter['new_success_gained']-counter['legacy_new_success_lost']-counter['additional_recovery']+counter['previous_recovery_lost']:
                raise RuntimeError('Paired transition identity failed.')
            totals.update(counter)
            totals['group_wins' if delta > 0 else 'group_losses' if delta < 0 else 'group_ties'] += 1
            # Group identifiers are retained privately for auditable examples.
            group_pairs.append(dict(seed=seed, group=group, targets=len(c['labels']),
                node_budget=a['node_budget'], edge_budget=a['edge_budget'], delta_wrong=delta, **counter))
        am = next(r for r in metrics if r['arm']=='legacy' and r['seed']==seed and r['detector']=='gaga')
        bm = next(r for r in metrics if r['arm']=='source_fixed' and r['seed']==seed and r['detector']=='gaga')
        paired.append(dict(seed=seed, delta_mr_pp=bm['mr_percent']-am['mr_percent'],
            delta_group_macro_pp=bm['group_macro_mr_percent']-am['group_macro_mr_percent']))
    for name, condition in (
        ('one_node_one_edge', lambda r:r['node_budget']==r['edge_budget']==1),
        ('one_node_multiple_edges', lambda r:r['node_budget']==1 and r['edge_budget']>1),
        ('multiple_nodes', lambda r:r['node_budget']>1)):
        selected=[r for r in group_pairs if condition(r)]
        occurrences=sum(r['occurrences'] for r in selected)
        net=sum(r['delta_wrong'] for r in selected)
        budget_pairs.append(dict(regime=name, group_seed_cases=len(selected), occurrences=occurrences,
            delta_wrong=net, delta_mr_pp=100*net/occurrences))
    values=[r['delta_mr_pp'] for r in paired]
    mean, sd=statistics.mean(values), statistics.stdev(values)
    half=4.302652729911275*sd/math.sqrt(3)
    aggregate={arm:dict(mean_mr_percent=statistics.mean(r['mr_percent'] for r in metrics if r['arm']==arm and r['detector']=='gaga'),
        sample_sd_pp=statistics.stdev(r['mr_percent'] for r in metrics if r['arm']==arm and r['detector']=='gaga')) for arm in ('legacy','source_fixed')}
    public=dict(status='completed_saved_record_summary',metrics=metrics,paired_seed_differences=paired,
        gaga_aggregate=aggregate,mean_paired_delta_mr_pp=mean,paired_t95_pp=[mean-half,mean+half],
        attack_seeds=3,degrees_of_freedom=2,consistent_development_gain=(mean>=1 and all(v>0 for v in values)),
        paired_transition_counts=dict(totals),budget_regimes=budget_pairs,
        limitations=['Repeatedly inspected fixed development cohort; not final TEST.',
            '3273 target occurrences are correlated; interval is over three attack seeds conditional on fixed models/cohort.',
            'Scoring-role conformance is not proof of novelty or a cause of the earlier dual-surrogate failure.',
            'No follow-up seeds/hyperparameter search is authorized by positive or negative outcomes.'],
        process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
        model_queries=0,tensor_loads=0,API_calls=0,GPU_hours=0)
    write_json(output/'summary.json',public,exclusive=True)
    write_json(output/'private_group_pairs.json',group_pairs,exclusive=True)
    write_json(output/'input_hashes.json',inputs,exclusive=True)
    print(json.dumps(public,allow_nan=False))


if __name__ == '__main__':
    main()
