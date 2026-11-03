"""POST-HOC saved-JSON-only paired analysis; never imports ML or loads tensors."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

ARMS = ('gcn_only', 'gaga_only', 'two_gcn', 'gcn_gaga')
SEEDS = (3, 4, 5)
SIZE_BINS = ('1-3', '4-6', '7-12', '13+')
DEFINITION = {
    'schema': 'monti_posthoc_paired_outcomes_v1',
    'post_hoc': True,
    'known_before_registration': 'The complete phase was negative; aggregate arm outcomes were already known.',
    'allowed_inputs': 'Thirteen saved clean/attacked prediction JSONLs, frozen execution protocol and saved evaluation counts only.',
    'primary_detector': 'gaga',
    'secondary_detector': 'gcn',
    'contrasts': ['gcn_gaga minus ' + a for a in ARMS[:-1]],
    'pair_key': ['seed', 'group', 'node_id'],
    'group_key': ['seed', 'group'],
    'primary_denominator': 'All 384 group-seed cases and 3273 target occurrences per contrast; no filtering or replacement of original MR.',
    'group_size_bins': ['1-3', '4-6', '7-12', '13+'],
    'group_size_bin_source': 'Historical pilot TRAIN-derived cutoffs [4,7,13] in evidence/results_first_diagnosis_v1/bin_boundaries.json; original bisect_right semantics: size<4, 4<=size<7, 7<=size<13, size>=13.',
    'disjoint_budget_strata': ['nb=eb=1', 'nb=1 and eb>1', 'nb>1'],
    'budget_dimensions': ['node_budget=edge_budget=1 versus other', 'edge_budget=1 versus other', 'node_budget=1 versus other'],
    'strata': 'Separate dimension tables plus size x exact-one-node-one-edge; overlapping dimensions must not be summed together.',
    'lost_existing_wrong': 'reference prediction wrong and joint correct; split by clean-correct versus clean-wrong.',
    'gained_wrong': 'reference prediction correct and joint wrong; split by clean-correct versus clean-wrong.',
    'new_error_set_intersection': 'Both reference and joint wrong among clean-correct target occurrences.',
    'new_error_set_difference': 'Gross lost = reference new errors minus intersection; gross gained = joint new errors minus intersection.',
    'group_win': 'Joint has more wrong occurrences than reference; tie equal; loss fewer.',
    'repetition': 'Count each occurrence within its separately attacked graph. Report node membership/exposure histogram, not a unique-node MR.',
    'representatives': [
        'largest gross clean-correct lost-success count, then smallest group ID, then seed',
        'largest net loss among exact node_budget=edge_budget=1 cases, then smallest group ID, then seed',
        'largest clean-wrong recovery contribution, then smallest group ID, then seed',
        'largest positive net joint gain, then smallest group ID, then seed'],
    'margin': 'logit_benign minus logit_fraud; positive is wrong for fraud-labelled targets; exact stored logits only.',
    'limits': ['Different learned checkpoints and generated graphs; matching seeds/budgets does not equate weights/features/graphs/RNG streams.',
               'Exploratory associations and representative cases are not causal proof or a training-example selection.',
               'No saved fresh-GAGA teacher VALID predictions exist in these inputs; held-out teacher quality or causal teacher-victim decision alignment cannot be diagnosed.',
               'No new inference, training, API, tensor/dataset/checkpoint loading or GPU work.'],
    'maximum_cpu_seconds': 300,
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reject_duplicates(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key')
        out[key] = value
    return out


def decode(raw):
    return json.loads(raw, object_pairs_hook=reject_duplicates,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Unexpected empty output table')
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def integer(value):
    if type(value) is not int:
        raise ValueError('Expected a strict integer')
    return value


def size_bin(n):
    if 1 <= n < 4:
        return '1-3'
    if 4 <= n < 7:
        return '4-6'
    if 7 <= n < 13:
        return '7-12'
    if n >= 13:
        return '13+'
    raise ValueError('Empty target group')


def stats(values):
    return {'n': len(values), 'min': min(values) if values else None,
            'median': statistics.median(values) if values else None,
            'max': max(values) if values else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--counts', required=True)
    ap.add_argument('--registration', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    wall_start, cpu_start = time.monotonic(), time.process_time()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'failed', 'post_hoc_saved_only': True,
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'command': sys.argv, 'inputs': [], 'new_model_queries': 0,
               'new_training_updates': 0, 'LLM_API_calls': 0, 'GPU_hours': 0,
               'source_sha256': sha(pathlib.Path(__file__).read_bytes())}

    def check_cpu():
        if time.process_time() - cpu_start > 300:
            raise TimeoutError('Registered CPU limit reached')

    def read(path):
        check_cpu()
        raw = pathlib.Path(path).read_bytes()
        receipt['inputs'].append({'name': pathlib.Path(path).name,
                                  'bytes': len(raw), 'sha256': sha(raw)})
        return raw

    def load_rows(path):
        result = {}
        for line in read(path).splitlines():
            check_cpu()
            row = decode(line)
            group = integer(row['group'])
            ids, labels = row['node_ids'], row['labels']
            if group in result or len(ids) != len(labels) or len(set(ids)) != len(ids) or not ids:
                raise ValueError('Malformed or repeated group/node rows')
            for node in ids:
                integer(node)
            if any(type(y) is not int or y != 1 for y in labels):
                raise ValueError('Expected all fraud-labelled targets')
            for detector in ('gaga', 'gcn'):
                logits = row[detector + '_logits']
                if len(logits) != len(ids) or any(len(z) != 2 or any(type(v) not in (int, float) or not math.isfinite(v) for v in z) for z in logits):
                    raise ValueError('Invalid saved two-class logits')
            result[group] = row
        return result

    try:
        reg = decode(pathlib.Path(args.registration).read_bytes())
        if reg['source_sha256'] != receipt['source_sha256'] or reg['definition'] != DEFINITION:
            raise ValueError('Source/definition differs from pre-read registration')
        receipt['registration_sha256'] = sha(pathlib.Path(args.registration).read_bytes())
        protocol = decode(read(args.protocol))
        counts = decode(read(args.counts))
        if protocol['seeds'] != list(SEEDS) or set(protocol['arms']) != set(ARMS):
            raise ValueError('Frozen protocol arms/seeds differ')
        if counts['status'] != 'completed' or counts['test_queries'] != 0 or counts['counts'] != {'gcn_base': 1664, 'gaga_victim': 1664, 'generations': 1536}:
            raise ValueError('Saved completed evaluation counts differ')
        groups = sorted(protocol['valid_groups'])
        if len(groups) != 128 or len(set(groups)) != 128:
            raise ValueError('Frozen group count differs')
        raw_dir = pathlib.Path(args.raw_dir)
        clean = load_rows(raw_dir / 'clean_predictions.jsonl')
        if sorted(clean) != groups or sum(len(r['node_ids']) for r in clean.values()) != 1091:
            raise ValueError('Clean complete-cohort denominator differs')
        attacked = {}
        saved_summaries = {(r['arm'], r['seed']): r for r in counts['summaries']}
        if len(saved_summaries) != 12 or len(counts['summaries']) != 12:
            raise ValueError('Expected twelve unique saved arm/seed summaries')
        for seed in SEEDS:
            for arm in ARMS:
                rows = load_rows(raw_dir / ('%s_seed%d_predictions.jsonl' % (arm, seed)))
                if sorted(rows) != groups:
                    raise ValueError('Arm group keys differ')
                for group in groups:
                    if any(rows[group][k] != clean[group][k] for k in ('node_ids', 'labels')):
                        raise ValueError('Matched target order/label differs')
                    for key in ('node_budget', 'edge_budget', 'actual_injected_nodes', 'actual_injected_edges'):
                        if integer(rows[group][key]) < 1:
                            raise ValueError('Non-positive injection budget/count')
                    if rows[group]['node_budget'] != rows[group]['actual_injected_nodes'] or rows[group]['edge_budget'] != rows[group]['actual_injected_edges']:
                        raise ValueError('Actual budgets differ from declared caps')
                attacked[arm, seed] = rows
                saved = saved_summaries[arm, seed]
                for detector in ('gaga', 'gcn'):
                    totals = dict(occurrences=0, clean_wrong=0, attacked_wrong=0, new_wrong=0, recoveries=0)
                    for group, row in rows.items():
                        for zc, za in zip(clean[group][detector+'_logits'], row[detector+'_logits']):
                            wc, wa = not(zc[1] > zc[0]), not(za[1] > za[0])
                            totals['occurrences'] += 1
                            totals['clean_wrong'] += int(wc)
                            totals['attacked_wrong'] += int(wa)
                            totals['new_wrong'] += int(not wc and wa)
                            totals['recoveries'] += int(wc and not wa)
                    if totals != saved['totals'][detector]:
                        raise ValueError('Raw transitions differ from saved arm/seed summary')
                    if abs(100*totals['attacked_wrong']/totals['occurrences'] - saved[detector+'_mr_percent']) > 1e-10:
                        raise ValueError('Raw MR differs from saved arm/seed summary')
        for seed in SEEDS:
            for group in groups:
                rows = [attacked[a, seed][group] for a in ARMS]
                if any(any(r[k] != rows[0][k] for k in ('node_budget', 'edge_budget', 'actual_injected_nodes', 'actual_injected_edges')) for r in rows):
                    raise ValueError('Paired actual budget caps differ')
        occurrence_rows, group_rows = [], []
        for reference in ARMS[:-1]:
            for detector in ('gaga', 'gcn'):
                for seed in SEEDS:
                    for group in groups:
                        base, joint = attacked[reference, seed][group], attacked['gcn_gaga', seed][group]
                        local = []
                        for index, node in enumerate(base['node_ids']):
                            zc, zb, zj = (r[detector + '_logits'][index] for r in (clean[group], base, joint))
                            # Python max preserves class zero on an exact argmax tie, as saved torch argmax does.
                            pc, pb, pj = [int(z[1] > z[0]) for z in (zc, zb, zj)]
                            wc, wb, wj = pc != 1, pb != 1, pj != 1
                            row = dict(reference=reference, detector=detector, seed=seed,
                                       private_group_id=group, private_node_id=node,
                                       group_size=len(base['node_ids']), size_bin=size_bin(len(base['node_ids'])),
                                       node_budget=base['node_budget'], edge_budget=base['edge_budget'],
                                       clean_wrong=int(wc), reference_wrong=int(wb), joint_wrong=int(wj),
                                       lost_existing_wrong=int(wb and not wj), gained_wrong=int(not wb and wj),
                                       both_wrong=int(wb and wj), both_correct=int(not wb and not wj),
                                       clean_margin=zc[0]-zc[1], reference_margin=zb[0]-zb[1], joint_margin=zj[0]-zj[1])
                            occurrence_rows.append(row)
                            local.append(row)
                        group_rows.append(dict(reference=reference, detector=detector, seed=seed, private_group_id=group,
                                               group_size=len(local), size_bin=local[0]['size_bin'], node_budget=base['node_budget'], edge_budget=base['edge_budget'],
                                               reference_wrong=sum(r['reference_wrong'] for r in local),
                                               joint_wrong=sum(r['joint_wrong'] for r in local),
                                               lost_existing_wrong=sum(r['lost_existing_wrong'] for r in local),
                                               gained_wrong=sum(r['gained_wrong'] for r in local),
                                               lost_clean_correct_success=sum(r['lost_existing_wrong'] for r in local if not r['clean_wrong']),
                                               gained_clean_correct_success=sum(r['gained_wrong'] for r in local if not r['clean_wrong']),
                                               clean_wrong_recovery_contribution=sum(r['lost_existing_wrong'] for r in local if r['clean_wrong'])))
        def aggregate(rows):
            cases = collections.defaultdict(list)
            for r in rows:
                cases[r['seed'], r['private_group_id']].append(r)
            keys = ('reference_wrong', 'joint_wrong', 'lost_existing_wrong', 'gained_wrong', 'both_wrong', 'both_correct')
            result = {k: sum(r[k] for r in rows) for k in keys}
            result.update(groups=len(cases), occurrences=len(rows),
                          group_wins=sum(sum(r['joint_wrong']-r['reference_wrong'] for r in a)>0 for a in cases.values()),
                          group_ties=sum(sum(r['joint_wrong']-r['reference_wrong'] for r in a)==0 for a in cases.values()),
                          group_losses=sum(sum(r['joint_wrong']-r['reference_wrong'] for r in a)<0 for a in cases.values()))
            result['net_joint_minus_reference_wrong'] = result['joint_wrong']-result['reference_wrong']
            result['net_joint_minus_reference_pp'] = 100*result['net_joint_minus_reference_wrong']/len(rows) if rows else None
            result['reference_occurrence_mr_percent'] = 100*result['reference_wrong']/len(rows) if rows else None
            result['joint_occurrence_mr_percent'] = 100*result['joint_wrong']/len(rows) if rows else None
            result['reference_group_macro_mr_percent'] = statistics.mean(100*sum(r['reference_wrong'] for r in a)/len(a) for a in cases.values()) if cases else None
            result['joint_group_macro_mr_percent'] = statistics.mean(100*sum(r['joint_wrong'] for r in a)/len(a) for a in cases.values()) if cases else None
            for clean_wrong, name in ((0, 'clean_correct'), (1, 'clean_wrong')):
                subset=[r for r in rows if r['clean_wrong']==clean_wrong]
                result[name+'_occurrences']=len(subset)
                for key in keys:
                    result[name+'_'+key]=sum(r[key] for r in subset)
            if result['lost_existing_wrong']-result['gained_wrong'] != -result['net_joint_minus_reference_wrong']:
                raise ValueError('Transition decomposition failed')
            return result
        tables, summaries, representative_details, representative_public = [], {}, [], []
        for reference in ARMS[:-1]:
            summaries[reference] = {}
            for detector in ('gaga', 'gcn'):
                rows=[r for r in occurrence_rows if r['reference']==reference and r['detector']==detector]
                summaries[reference][detector] = aggregate(rows)
                predicates=[('all','all',lambda r:True)]
                for v in SIZE_BINS:
                    predicates.append(('group_size',v,lambda r,v=v:r['size_bin']==v))
                for value, predicate in [('nb=eb=1', lambda r:r['node_budget']==r['edge_budget']==1),
                                         ('nb=1 and eb>1',lambda r:r['node_budget']==1 and r['edge_budget']>1),
                                         ('nb>1',lambda r:r['node_budget']>1)]:
                    predicates.append(('budget_disjoint',value,predicate))
                for dimension, test in [('exact_nb1_eb1',lambda r:r['node_budget']==r['edge_budget']==1),('edge_budget1',lambda r:r['edge_budget']==1),('node_budget1',lambda r:r['node_budget']==1)]:
                    for flag in (True,False):
                        predicates.append((dimension,str(flag).lower(),lambda r,test=test,flag=flag:test(r)==flag))
                for flag in (0,1):
                    predicates.append(('clean_status','already_wrong' if flag else 'correct',lambda r,flag=flag:r['clean_wrong']==flag))
                for v in SIZE_BINS:
                    for flag in (True,False):
                        predicates.append(('size_x_exact_nb1_eb1',v+'|'+str(flag).lower(),lambda r,v=v,flag=flag:r['size_bin']==v and (r['node_budget']==r['edge_budget']==1)==flag))
                for seed in ('all',)+SEEDS:
                    for dimension, value, predicate in predicates:
                        check_cpu()
                        selected=[r for r in rows if (seed=='all' or r['seed']==seed) and predicate(r)]
                        tables.append(dict(reference=reference,detector=detector,seed=seed,dimension=dimension,stratum=value,**aggregate(selected)))
                if detector!='gaga':
                    continue
                candidates=[r for r in group_rows if r['reference']==reference and r['detector']==detector]
                rules=[('largest_clean_correct_lost_success',lambda r:r['lost_clean_correct_success']>0,lambda r:r['lost_clean_correct_success']),
                       ('largest_exact_one_edge_net_loss',lambda r:r['node_budget']==r['edge_budget']==1 and r['reference_wrong']>r['joint_wrong'],lambda r:r['reference_wrong']-r['joint_wrong']),
                       ('largest_clean_wrong_recovery_contribution',lambda r:r['clean_wrong_recovery_contribution']>0,lambda r:r['clean_wrong_recovery_contribution']),
                       ('largest_joint_net_gain_counterexample',lambda r:r['joint_wrong']>r['reference_wrong'],lambda r:r['joint_wrong']-r['reference_wrong'])]
                for rule, predicate, score in rules:
                    eligible=[r for r in candidates if predicate(r)]
                    if not eligible:
                        representative_public.append(dict(reference=reference,rule=rule,available=False))
                        continue
                    chosen=sorted(eligible,key=lambda r:(-score(r),r['private_group_id'],r['seed']))[0]
                    detail=[r for r in rows if r['seed']==chosen['seed'] and r['private_group_id']==chosen['private_group_id']]
                    representative_details.extend(dict(rule=rule,**r) for r in detail)
                    public={k:v for k,v in chosen.items() if k!='private_group_id'}
                    public.update(rule=rule,available=True,clean_correct_lost_reference_margins=stats([r['reference_margin'] for r in detail if r['lost_existing_wrong'] and not r['clean_wrong']]),
                                  clean_correct_lost_joint_margins=stats([r['joint_margin'] for r in detail if r['lost_existing_wrong'] and not r['clean_wrong']]))
                    representative_public.append(public)
        membership=collections.defaultdict(set)
        for group,row in clean.items():
            for node in row['node_ids']:
                membership[node].add(group)
        if len(membership)!=374:
            raise ValueError('Unique target-node count differs')
        node_rows=[]
        for reference in ARMS[:-1]:
            by_node=collections.defaultdict(list)
            for row in occurrence_rows:
                if row['reference']==reference and row['detector']=='gaga':by_node[row['private_node_id']].append(row)
            for node,rows in sorted(by_node.items()):
                node_rows.append(dict(reference=reference,private_node_id=node,group_memberships=len(membership[node]),**aggregate(rows)))
        summary=dict(status='completed',definition=DEFINITION,comparisons=summaries,
                     repetition=dict(unique_target_nodes=len(membership),groups=128,occurrences_per_seed=1091,
                                     occurrences_over_three_seeds=3273,group_membership_histogram=dict(sorted(collections.Counter(len(v) for v in membership.values()).items())),
                                     unique_node_MR_computed=False),representative_cases=representative_public,
                     original_primary_metric_changed=False,all_denominators_retained=True)
        write_csv(out/'paired_occurrences_PRIVATE.csv',occurrence_rows)
        write_csv(out/'paired_groups_PRIVATE.csv',group_rows)
        write_csv(out/'all_strata.csv',tables)
        write_csv(out/'node_repetition_PRIVATE.csv',node_rows)
        write_csv(out/'representative_details_PRIVATE.csv',representative_details)
        write_json(out/'summary_PUBLIC_REVIEW.json',summary)
        receipt['outputs']=[dict(name=p.name,sha256=sha(p.read_bytes()),bytes=p.stat().st_size) for p in sorted(out.iterdir())]
        receipt['status']='SAVED_PAIRED_OUTCOME_ANALYSIS_COMPLETE'
        receipt['group_seed_cases_per_contrast']=384
        receipt['target_occurrences_per_contrast']=3273
        receipt['raw_prediction_files']=13
    except BaseException as error:
        receipt['error_type']=type(error).__name__
        receipt['error']=str(error)
        raise
    finally:
        receipt['process_cpu_seconds']=time.process_time()-cpu_start
        receipt['wall_seconds']=time.monotonic()-wall_start
        receipt['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out/'receipt.json',receipt)


if __name__=='__main__':
    main()
