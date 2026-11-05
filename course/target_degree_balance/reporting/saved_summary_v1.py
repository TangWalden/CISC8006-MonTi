"""Frozen-before-outcomes, saved-JSON-only target-balance reporting.

No tensor/NumPy/dataset/checkpoint/graph/model/network operations. The author
implemented the producer wrapper; this is arithmetic reporting crosschecked
against a separately authored audit, not another independent implementation.
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


ARMS = ('baseline', 'balanced_target', 'matched_random')
SEEDS = (3, 4, 5)
DETECTORS = ('gcn', 'gaga')
FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
PAIR_FIELDS = ('occurrences', 'gained_wrong', 'lost_wrong', 'new_success_gained',
               'baseline_new_success_lost', 'additional_recovery', 'previous_recovery_lost',
               'group_wins', 'group_ties', 'group_losses')
CONTRASTS = (('balanced_target', 'baseline'), ('matched_random', 'baseline'),
             ('balanced_target', 'matched_random'))
T95_DF2 = 4.302652729911275
FOLLOWUP_GROUPS = (1071, 1001)


def file_sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    stage, audit_path, output = pathlib.Path(args.stage).resolve(), pathlib.Path(args.audit).resolve(), pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpu, wall = time.process_time(), time.monotonic()
    inputs, loaded, checks = {}, {}, 0
    receipt = dict(status='failed', source_sha256=file_sha(__file__),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_inputs='Completed saved producer/audit JSON only',
        model_queries=0, tensor_loads=0, NumPy_imports=0, dataset_loads=0,
        checkpoint_loads=0, API_calls=0, GPU_hours=0, independent_of_producer=False,
        human_verification=False)

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:raise AssertionError(message)

    def read(path, kind='json'):
        path = pathlib.Path(path).resolve()
        if path not in loaded:
            raw = path.read_bytes()
            inputs[str(path)] = hashlib.sha256(raw).hexdigest()
            text = raw.decode('utf-8')
            loaded[path] = ([json.loads(line) for line in text.splitlines() if line.strip()]
                            if kind == 'jsonl' else json.loads(text))
        return loaded[path]

    def indexed(values, key, expected=None):
        result = {}
        for value in values:
            identity = key(value)
            require(identity not in result, 'Duplicate saved row identity.')
            result[identity] = value
        if expected is not None:require(set(result) == set(expected), 'Saved row cohort differs.')
        return result

    def predictions(logits, count):
        require(len(logits) == count, 'Logit denominator differs.')
        result = []
        for pair in logits:
            require(len(pair) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in pair),
                    'Malformed or nonfinite binary logits.')
            result.append(0 if pair[0] >= pair[1] else 1)
        return result

    def transitions(labels, clean, attacked):
        require(len(labels) == len(clean) == len(attacked), 'Transition target denominator differs.')
        result = dict(occurrences=len(labels), clean_wrong=0, attacked_wrong=0, new_wrong=0, recoveries=0)
        for label, c, a in zip(labels, clean, attacked):
            result['clean_wrong'] += int(c != label)
            result['attacked_wrong'] += int(a != label)
            result['new_wrong'] += int(c == label and a != label)
            result['recoveries'] += int(c != label and a == label)
        require(result['attacked_wrong'] == result['clean_wrong']+result['new_wrong']-result['recoveries'],
                'Within-arm transition identity fails.')
        return result

    def paired(labels, clean, reference, candidate):
        values = {name:0 for name in PAIR_FIELDS}
        delta = sum(value != label for value, label in zip(candidate, labels))-sum(value != label for value, label in zip(reference, labels))
        values['group_wins' if delta > 0 else 'group_losses' if delta < 0 else 'group_ties'] = 1
        for label, c, old, new in zip(labels, clean, reference, candidate):
            values['occurrences'] += 1
            values['gained_wrong'] += int(old == label and new != label)
            values['lost_wrong'] += int(old != label and new == label)
            if c == label:
                values['new_success_gained'] += int(old == label and new != label)
                values['baseline_new_success_lost'] += int(old != label and new == label)
            else:
                values['additional_recovery'] += int(old != label and new == label)
                values['previous_recovery_lost'] += int(old == label and new != label)
        require(delta == values['gained_wrong']-values['lost_wrong'] ==
                values['new_success_gained']-values['baseline_new_success_lost']-values['additional_recovery']+values['previous_recovery_lost'],
                'Paired new-success/recovery accounting fails.')
        return dict(delta_wrong=delta, delta_mr_pp=100*delta/len(labels), **values)

    def close(actual, expected, name):
        require(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-10), name)

    try:
        producer = read(stage/'result.json')
        audit = read(audit_path)
        require(audit['status'] == 'passed', 'Separate audit has not passed.')
        require(producer['status'] in ('completed', 'completed_no_headroom')
                and audit['producer_status'] == producer['status'] and audit['provenance'] == producer['provenance'],
                'Producer/audit completion or provenance differs.')
        require(audit['producer_counters'] == producer['counters'], 'Producer/auditor operation counts differ.')
        for name, wanted in producer['output_hashes'].items():
            value = read(stage/name, 'jsonl' if name.endswith('.jsonl') else 'json')
            require(inputs[str((stage/name).resolve())] == wanted, 'A completed producer output hash differs.')
        geometry_values = read(stage/'geometry.jsonl', 'jsonl')
        require(len(geometry_values) == 384, 'Expected all384 saved geometry cases.')
        geometry = indexed(geometry_values, lambda row:(row['seed'], row['group']))
        groups = sorted(row['group'] for row in geometry_values if row['seed'] == 3)
        require(len(groups) == 128 and set(geometry) == {(seed, group) for seed in SEEDS for group in groups},
                'The complete3seed/128group geometry cohort differs.')
        geometry_cases, geometry_aggregates = {}, []
        nonidentity = {}
        for seed in SEEDS:
            selected = [geometry[seed, group] for group in groups]
            require(sum(row['targets'] for row in selected) == 1091, 'Per-seed1091 target-occurrence denominator differs.')
            require(len({node for row in selected for node in row['node_ids']}) == 374, 'Per-seed374unique targets differ.')
            nonidentity[str(seed)] = sum(not row['geometry']['balanced_equals_baseline'] for row in selected)
            totals = collections.Counter()
            for row in selected:
                g = row['geometry']
                before, after = g['target_incoming_injected_degrees_before'], g['target_incoming_injected_degrees_after']
                require(sum(before) == g['total_target_incidence'], 'Target incidence recount differs.')
                require(sum(x*x for x in before) == g['sum_squared_target_degree_before'], 'Original Phi recount differs.')
                k = sum(plan['final_k'] for plan in g['row_plan'])
                require(k == g['final_rewired_edges'] and len(g['row_plan']) == row['node_budget'], 'Final row-turnover k recount differs.')
                for plan in g['row_plan']:
                    require(all(len(plan[name]) == plan['final_k'] for name in
                        ('balanced_removed_target_ids','balanced_added_target_ids','random_removed_target_ids','random_added_target_ids')),
                        'Balanced/random final row turnover matching differs.')
                old_coverage = sum(value > 0 for value in before)
                coverage = {arm:sum(value > 0 for value in after[arm]) for arm in ARMS}
                phi = {arm:sum(value*value for value in after[arm]) for arm in ARMS}
                require(phi['balanced_target'] == g['sum_squared_target_degree_balanced'] and phi['balanced_target'] <= phi['baseline'],
                        'Balanced structural Phi/monotonicity recount differs.')
                require(all(sum(after[arm]) == sum(before) for arm in ARMS), 'Target-incidence totals changed.')
                values = dict(nonidentity_balanced=not g['balanced_equals_baseline'],
                    nonidentity_random=any(a != b for a,b in zip(before, after['matched_random'])) or k > 0,
                    final_k_sum=k, final_changed_target_bits=2*k,
                    rows_with_positive_final_k=sum(plan['final_k'] > 0 for plan in g['row_plan']),
                    balancing_iterations=g['balancing_iterations'],
                    total_target_incidence=g['total_target_incidence'],
                    phi_before=phi['baseline'], phi_balanced=phi['balanced_target'], phi_random=phi['matched_random'],
                    phi_reduction_balanced=phi['baseline']-phi['balanced_target'],
                    covered_target_occurrences_before=old_coverage,
                    covered_target_occurrences_after=coverage,
                    newly_covered_target_occurrences={arm:len(g['newly_covered_target_ids'][arm]) for arm in ARMS},
                    previously_covered_target_occurrences_lost={arm:len(g['previously_covered_target_ids_lost'][arm]) for arm in ARMS})
                require(values['previously_covered_target_occurrences_lost']['balanced_target'] == 0, 'Balancing uncovered an old covered target.')
                geometry_cases[seed, row['group']] = values
                for name in ('final_k_sum','final_changed_target_bits','rows_with_positive_final_k','balancing_iterations',
                             'total_target_incidence','phi_before','phi_balanced','phi_random','phi_reduction_balanced','covered_target_occurrences_before'):
                    totals[name] += values[name]
            geometry_aggregates.append(dict(seed=seed,cases=128,nonidentity_balanced_cases=nonidentity[str(seed)],
                summed_injected_nodes=sum(row['node_budget'] for row in selected),
                summed_nonself_undirected_edges=sum(row['edge_budget'] for row in selected),
                **dict(totals),
                covered_target_occurrences_after={arm:sum(geometry_cases[seed, group]['covered_target_occurrences_after'][arm] for group in groups) for arm in ARMS},
                newly_covered_target_occurrences={arm:sum(geometry_cases[seed, group]['newly_covered_target_occurrences'][arm] for group in groups) for arm in ARMS},
                previously_covered_target_occurrences_lost={arm:sum(geometry_cases[seed, group]['previously_covered_target_occurrences_lost'][arm] for group in groups) for arm in ARMS}))
        require(nonidentity == producer['balanced_nonidentity_cases_by_seed'] == audit['headroom_nonidentity'], 'Structural nonidentity counts differ from producer/audit.')
        base_summary = dict(status='completed_saved_only_summary', producer_status=producer['status'],
            provenance=producer['provenance'], independent_audit_sha256=inputs[str(audit_path)],
            geometry_by_seed=geometry_aggregates, model_queries=0, tensor_loads=0, API_calls=0, GPU_hours=0,
            control_selection_allowed=False, qualified_control_winner=None,
            interpretation=['Private reporting source frozen before this target-balance run outcomes were inspected.',
                'Study is still post-hoc exploratory after earlier negative studies; this reporting freeze is not an efficacy preregistration.',
                'All3273 target occurrences/384group-seed scenarios are correlated; scenario totals are not one combined graph.',
                'Maximum-gain/loss examples below are explicitly post-hoc illustrations, never gate/method selection.',
                'Matched random is a control and cannot be promoted as a qualified method winner.',
                'Primary balanced-target criterion and final TEST/tag checkpoints remain separate.'])
        all_cases = [dict(seed=seed,group=group,targets=geometry[seed,group]['targets'],
            node_budget=geometry[seed,group]['node_budget'],edge_budget=geometry[seed,group]['edge_budget'],
            geometry=geometry_cases[seed,group]) for seed in SEEDS for group in groups]
        if producer['status'] == 'completed_no_headroom':
            require(any(value == 0 for value in nonidentity.values()), 'No-headroom status lacks its structural condition.')
            require(not (stage/'clean_predictions.jsonl').exists(), 'No-headroom study unexpectedly queried classifiers.')
            base_summary.update(primary_balanced_criterion_met=False, qualification=None,
                stop_reason='No structural headroom in at least one complete seed; no classifier performance was measured.')
        else:
            clean = indexed(read(stage/'clean_predictions.jsonl','jsonl'), lambda row:row['group'], groups)
            clean_pred = {(group,detector):predictions(clean[group][detector+'_logits'],len(clean[group]['labels']))
                          for group in groups for detector in DETECTORS}
            outputs, output_pred, metrics = {}, {}, []
            for seed in SEEDS:
                for arm in ARMS:
                    rows = indexed(read(stage/('%s_seed%d_predictions.jsonl'%(arm,seed)),'jsonl'), lambda row:row['group'], groups)
                    outputs[seed,arm] = rows
                    for detector in DETECTORS:
                        total = collections.Counter();macro=[]
                        for group in groups:
                            row, identity, original_clean = rows[group], geometry[seed,group], clean[group]
                            require(row['seed']==seed and row['arm']==arm and row['node_ids']==identity['node_ids']==original_clean['node_ids']
                                    and row['labels']==identity['labels']==original_clean['labels'], 'Fresh prediction identity/order differs.')
                            predicted = predictions(row[detector+'_logits'],len(row['labels']))
                            output_pred[seed,arm,group,detector] = predicted
                            values = transitions(row['labels'],clean_pred[group,detector],predicted)
                            require(values == row[detector+'_transitions'], 'Saved transition field does not recount from raw argmax.')
                            total.update(values);macro.append(100*values['attacked_wrong']/values['occurrences'])
                        require(total['occurrences'] == 1091, 'Fixed per-run occurrence denominator differs.')
                        metrics.append(dict(seed=seed,arm=arm,detector=detector,
                            **{name:total[name] for name in FIELDS},mr_percent=100*total['attacked_wrong']/1091,
                            group_macro_mr_percent=statistics.mean(macro)))
            lookup = {(row['seed'],row['arm'],row['detector']):row for row in metrics}
            audit_metrics = indexed(audit['metrics'],lambda row:(row['seed'],row['arm'],row['detector']),lookup)
            for key,row in lookup.items():
                reference = audit_metrics[key]
                require(all(row[name]==reference[name] for name in FIELDS), 'Raw totals disagree with independent audit.')
                close(row['mr_percent'],reference['mr_percent'],'Micro MR disagrees with independent audit.')
                close(row['group_macro_mr_percent'],reference['group_macro_mr_percent'],'Group macro MR disagrees with independent audit.')
            contrasts, budget_rows = {}, []
            by_case = {(row['seed'],row['group']):row for row in all_cases}
            for seed in SEEDS:
                for group in groups:
                    case_record = by_case[seed,group]
                    case_record['outcomes'] = {arm:{detector:dict(outputs[seed,arm][group][detector+'_transitions']) for detector in DETECTORS} for arm in ARMS}
                    case_record['contrasts'] = {}
            for candidate,reference in CONTRASTS:
                contrast_name = candidate+'-minus-'+reference
                for detector in DETECTORS:
                    total = collections.Counter();parts=[]
                    for seed in SEEDS:
                        for group in groups:
                            labels = geometry[seed,group]['labels']
                            value = paired(labels,clean_pred[group,detector],output_pred[seed,reference,group,detector],output_pred[seed,candidate,group,detector])
                            total.update({name:value[name] for name in PAIR_FIELDS})
                            by_case[seed,group]['contrasts'].setdefault(contrast_name,{})[detector] = value
                            parts.append(dict(seed=seed,group=group,node_budget=geometry[seed,group]['node_budget'],
                                edge_budget=geometry[seed,group]['edge_budget'],**value))
                    require(total['occurrences']==3273 and sum(total[name] for name in ('group_wins','group_ties','group_losses'))==384,
                            'Complete paired denominator differs.')
                    values = [lookup[seed,candidate,detector]['mr_percent']-lookup[seed,reference,detector]['mr_percent'] for seed in SEEDS]
                    mean,sd = statistics.mean(values),statistics.stdev(values)
                    half = T95_DF2*sd/math.sqrt(3)
                    key = contrast_name+'_'+detector
                    contrasts[key] = dict(seed_differences_pp=values,mean_difference_pp=mean,sample_sd_pp=sd,
                        paired_t95_pp=[mean-half,mean+half],attack_seeds=3,df=2,paired_counts={name:total[name] for name in PAIR_FIELDS})
                    checked_audit = audit['paired_contrasts'][key]
                    for name in PAIR_FIELDS:require(total[name]==checked_audit['paired_counts'].get(name,0),'Paired count disagrees with independent audit.')
                    for measured,expected in zip(values,checked_audit['seed_differences_pp']):close(measured,expected,'Paired seed difference disagrees with audit.')
                    close(mean,checked_audit['mean_difference_pp'],'Paired mean disagrees with audit.')
                    for measured,expected in zip(contrasts[key]['paired_t95_pp'],checked_audit['paired_t95_pp']):close(measured,expected,'Conditional t interval disagrees with audit.')
                    for name,condition in (
                        ('one_node_one_edge',lambda row:row['node_budget']==row['edge_budget']==1),
                        ('one_node_multiple_edges',lambda row:row['node_budget']==1 and row['edge_budget']>1),
                        ('multiple_nodes',lambda row:row['node_budget']>1)):
                        selected = [row for row in parts if condition(row)]
                        occurrence = sum(row['occurrences'] for row in selected)
                        delta = sum(row['delta_wrong'] for row in selected)
                        budget_rows.append(dict(contrast=contrast_name,detector=detector,regime=name,
                            group_seed_cases=len(selected),occurrences=occurrence,delta_wrong=delta,
                            delta_mr_pp=100*delta/occurrence if occurrence else None,
                            group_macro_delta_pp=statistics.mean(row['delta_mr_pp'] for row in selected) if selected else None,
                            paired_counts={name:sum(row[name] for row in selected) for name in PAIR_FIELDS}))
            aggregate = []
            for arm in ARMS:
                for detector in DETECTORS:
                    selected = [lookup[seed,arm,detector] for seed in SEEDS]
                    aggregate.append(dict(arm=arm,detector=detector,
                        mean_micro_mr_percent=statistics.mean(row['mr_percent'] for row in selected),
                        sample_sd_micro_pp=statistics.stdev(row['mr_percent'] for row in selected),
                        mean_group_macro_mr_percent=statistics.mean(row['group_macro_mr_percent'] for row in selected),
                        sample_sd_group_macro_pp=statistics.stdev(row['group_macro_mr_percent'] for row in selected)))
            primary = contrasts['balanced_target-minus-baseline_gaga']
            advantage = contrasts['balanced_target-minus-matched_random_gaga']['mean_difference_pp']
            criterion = primary['mean_difference_pp']>=1 and all(value>0 for value in primary['seed_differences_pp']) and advantage>0
            require(criterion == audit['primary_development_criterion']['criterion_met'],'Registered primary criterion differs from audit.')
            followups = []
            for group in FOLLOWUP_GROUPS:
                for seed in SEEDS:
                    if (seed,group) in by_case:
                        followups.append(dict(selection='Group frozen in this reporting plan before current outcomes; originally outcome-selected in prior source-role study.',
                            **by_case[seed,group]))
            eligible = [row for row in all_cases if row['contrasts']['balanced_target-minus-baseline']['gaga']['delta_wrong']>0]
            lost = [row for row in all_cases if row['contrasts']['balanced_target-minus-baseline']['gaga']['delta_wrong']<0]
            maximum_gain = min(eligible,key=lambda row:(-row['contrasts']['balanced_target-minus-baseline']['gaga']['delta_wrong'],row['seed'],row['group'])) if eligible else None
            maximum_loss = min(lost,key=lambda row:(row['contrasts']['balanced_target-minus-baseline']['gaga']['delta_wrong'],row['seed'],row['group'])) if lost else None
            base_summary.update(metrics=metrics,arm_aggregates=aggregate,paired_contrasts=contrasts,
                disjoint_budget_contrasts=budget_rows,primary_balanced_criterion_met=criterion,
                qualification='balanced_target_development_candidate_only' if criterion else None,
                primary_definition='balanced_target official-GAGA mean gain>=1pp, all3seed gains>0, mean above matched_random; no control promotion.',
                fixed_prior_case_followups=followups,
                posthoc_examples=dict(selection='Among all384 cases, largest positive/negative balanced-minus-baseline GAGA wrong-count delta, ties ascending seed then group; no case is removed.',
                    maximum_gain=maximum_gain,maximum_loss=maximum_loss))
        require(len(all_cases)==384,'All384 count-only case records must be retained.')
        for row in all_cases:
            require(not any(name in row for name in ('node_ids','labels','gcn_logits','gaga_logits')), 'Raw node/logit fields escaped count-only reporting.')
        write_json(output/'summary.json',base_summary)
        write_json(output/'all_case_counts.json',all_cases)
        receipt.update(status='passed',checks_passed=checks,retained_count_only_cases=384,
            output_hashes={name:file_sha(output/name) for name in ('summary.json','all_case_counts.json')},
            source_frozen_before_outcome_read=True)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),checks_passed_before_failure=checks)
    finally:
        receipt.update(process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
                       saved_input_files_read=len(inputs))
        write_json(output/'input_hashes.json',inputs)
        write_json(output/'receipt.json',receipt)
        print(json.dumps({key:value for key,value in receipt.items() if key not in ('traceback',)},allow_nan=False))
    if receipt['status']!='passed':raise SystemExit(1)


def write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
