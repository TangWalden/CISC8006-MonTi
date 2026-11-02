"""Recount retained owner-method pilot outputs; standard library, no model calls.

This is a post-hoc reporting program, not a new experiment registration. Raw
identities, cards, labels and API replies remain private; only aggregates and
their input hashes are emitted. All nine registered jobs must have finished.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def metrics(rows, prediction):
    benign = [r for r in rows if r['label'] == 0]
    fraud = [r for r in rows if r['label'] == 1]
    assert benign and fraud and len(benign) + len(fraud) == len(rows)
    return dict(cases=len(rows), benign=len(benign), fraud=len(fraud),
                correct=sum(bool(prediction(r)) == bool(r['label']) for r in rows),
                accuracy=sum(bool(prediction(r)) == bool(r['label']) for r in rows)/len(rows),
                benign_false_positive_rate=sum(bool(prediction(r)) for r in benign)/len(benign),
                fraud_recall=sum(bool(prediction(r)) for r in fraud)/len(fraud))


def alignment(stage):
    queue = read(stage/'queue_complete.json')
    assert queue['status'] == 'completed' and len(queue['jobs']) == 9
    expected = {(a, s) for a in ('off', 'always', 'gated') for s in (3, 4, 5)}
    assert {(r['arm'], r['seed']) for r in queue['jobs']} == expected
    runs, hashes = [], {}
    for job in queue['jobs']:
        assert job['exit_code'] == 0
        name = '%s_seed%d' % (job['arm'], job['seed'])
        folder = stage/name
        result = read(folder/'result.json')
        assert result['status'] == 'completed' and result['test_used'] is False
        assert result['gaga_forwards'] == 0 and result['victim_loaded'] is False
        epoch = result['best_epoch']
        groups = read(folder/('validation_epoch_%03d_groups.json' % epoch))
        assert len(groups) == 128
        occurrences = sum(r['target_occurrences'] for r in groups)
        errors = sum(r['incorrect'] for r in groups)
        assert occurrences == 1091
        assert errors == result['best_validation_counts']['incorrect']
        for row in groups:
            assert row['node_budget'] == row['injected_nodes']
            assert row['edge_budget'] == row['actual_undirected_injected_edges']
            assert 0 <= row['outside_tolerance_nodes'] <= row['injected_nodes']
        injected = sum(r['injected_nodes'] for r in groups)
        outside = sum(r['outside_tolerance_nodes'] for r in groups)
        runs.append(dict(arm=job['arm'], seed=job['seed'], best_epoch=epoch,
            completed_epochs=result['completed_epochs'], incorrect=errors,
            occurrences=occurrences, mr_percent=100*errors/occurrences,
            mean_of_group_mean_distances=statistics.mean(r['mean_distance'] for r in groups),
            injected_nodes=injected, outside_tolerance_nodes=outside,
            outside_tolerance_fraction=outside/injected,
            active_penalty_nodes=sum(r['active_nodes'] for r in groups),
            actual_nonself_undirected_injected_edges=sum(r['actual_undirected_injected_edges'] for r in groups),
            fallback_nodes=sum(r['fallback_nodes'] for r in groups),
            optimizer_steps=result['optimizer_steps'], gcn_forwards=result['gcn_forwards'],
            generator_forwards=result['generator_forwards'],
            charged_gpu_wall_hours=job['charged_gpu_wall_hours'],
            checkpoint_sha256=result['checkpoint_sha256'], reference_sha256=result['reference_sha256'],
            raw_selected_prediction_sha256=result['best_validation_counts']['raw_prediction_sha256']))
        for file in (folder/'result.json', folder/'registered.json',
                     folder/('validation_epoch_%03d_groups.json' % epoch),
                     stage/(name+'_guardian.json')):
            hashes[str(file.relative_to(stage))] = sha(file)
    by_key = {(r['arm'], r['seed']):r for r in runs}
    contrasts = {}
    for arm in ('always', 'gated'):
        differences = [by_key[arm,s]['mr_percent']-by_key['off',s]['mr_percent'] for s in (3,4,5)]
        contrasts[arm+'_minus_off'] = dict(seed_order=[3,4,5], paired_mr_difference_pp=differences,
            mean_difference_pp=statistics.mean(differences),
            interpretation='Descriptive selected-checkpoint validation contrast; no winner or significance rule.')
    mean = {arm: statistics.mean(by_key[arm,s]['mr_percent'] for s in (3,4,5))
            for arm in ('off','always','gated')}
    return dict(runs=runs, mean_mr_percent=mean, contrasts=contrasts,
        charged_with_self_gpu_hours=queue['charged_with_self_gpu_hours'],
        alignment_gpu_hours=sum(r['charged_gpu_wall_hours'] for r in runs),
        optimizer_steps=sum(r['optimizer_steps'] for r in runs),
        gcn_forwards=sum(r['gcn_forwards'] for r in runs),
        generator_forwards=sum(r['generator_forwards'] for r in runs),
        input_sha256=hashes, queue_complete_sha256=sha(stage/'queue_complete.json'),
        interpretation=['Higher MR means a stronger attack against the GCN surrogate on the fixed128 validation groups.',
            'Distance and outside-tolerance rate use the optimized two-moment signature, not an independent camouflage detector.',
            'OFF active count is zero by policy; ALWAYS active count equals all injected nodes. These are not outlier rates.',
            'Same physical caps do not imply the same compute or realized edge count across methods.',
            'Original score role/candidate/tie/budget conventions are retained; no final GAGA/test evaluation.'])


def fusion(root):
    primary = lines(root/'api_attempt1/case_outcomes.jsonl')
    extension = lines(root/'text_learning_attempt1/extension_outcomes.jsonl')
    numeric_rows = lines(root/'numeric_gate_grid/outcomes.jsonl')
    numeric = {r['case_id']:r for r in numeric_rows}
    provenance = {r['case_id']:r for r in lines(root/'probe32/provenance_private.jsonl')}
    assert len(primary) == len(numeric_rows) == len(numeric) == len(provenance) == 32
    assert len(extension) == 20
    assert {r['case_id'] for r in primary} == set(numeric) == set(provenance)
    for row in primary + extension:
        assert row['label'] == numeric[row['case_id']]['label'] == provenance[row['case_id']]['label']
    methods = {key:metrics(primary, lambda r,k=key:r['predictions'][k])
               for key in ('self_only','graph_only','static_half','template','generated')}
    methods['numeric_neural_gate_grid'] = metrics(primary, lambda r:numeric[r['case_id']]['predicted_fraud'])
    comparison = {key:metrics(extension,lambda r,k=key:r[k]) for key in (
        'predicted_fraud','primary_template_prediction','primary_generated_prediction',
        'self_only_prediction','graph_only_prediction','static_half_prediction')}
    comparison['numeric_neural_gate_grid'] = metrics(extension,lambda r:numeric[r['case_id']]['predicted_fraud'])
    pairwise = {key:sum(bool(r['predictions'][key]) != bool(r['predictions']['graph_only']) for r in primary)
                for key in ('self_only','static_half','template','generated')}
    pairwise['numeric_neural_gate_grid'] = sum(bool(numeric[r['case_id']]['predicted_fraud']) !=
                                              bool(r['predictions']['graph_only']) for r in primary)
    condition = {}
    for arm in ('clean','off','always','gated'):
        subset = [r for r in primary if provenance[r['case_id']]['arm'] == arm]
        assert len(subset) == 8
        condition[arm] = {key:metrics(subset,lambda r,k=key:r['predictions'][k])
                          for key in ('self_only','graph_only','static_half','template','generated')}
        condition[arm]['numeric_neural_gate_grid'] = metrics(subset,lambda r:numeric[r['case_id']]['predicted_fraud'])
    clean = {provenance[r['case_id']]['node_id']:r for r in primary
             if provenance[r['case_id']]['arm'] == 'clean'}
    transitions = {}
    for arm in ('off','always','gated'):
        pairs = [(clean[provenance[r['case_id']]['node_id']],r) for r in primary
                 if provenance[r['case_id']]['arm'] == arm]
        assert len(pairs) == 8
        transitions[arm] = {method:dict(
            class_changes=sum(x['predictions'][method] != y['predictions'][method] for x,y in pairs),
            new_errors=sum(x['predictions'][method] == bool(x['label']) and
                           y['predictions'][method] != bool(y['label']) for x,y in pairs),
            recoveries=sum(x['predictions'][method] != bool(x['label']) and
                           y['predictions'][method] == bool(y['label']) for x,y in pairs))
            for method in ('graph_only','static_half','template','generated')}
    ledger = lines(root/'api_attempt1/api_attempts.jsonl')
    protocol_sha = sha(root/'api_protocol_v1.json')
    assert len(ledger) <= 120 and all(r['protocol_sha256'] == protocol_sha for r in ledger)
    prompt = sum(r['charged_prompt_tokens'] for r in ledger)
    completion = sum(r['charged_completion_tokens'] for r in ledger)
    cost = sum(r['charged_usd_upper_bound'] for r in ledger)
    assert prompt <= 200000 and completion <= 64000 and cost <= .5
    parse = lambda value:dt.datetime.fromisoformat(value.replace('Z','+00:00'))
    latencies = [(parse(r['completed_utc'])-parse(r['started_utc'])).total_seconds() for r in ledger]
    aggregate = dict(primary_32=methods, selected_twenty_case_learning_comparison=comparison,
        primary_prediction_disagreements_vs_graph=pairwise,
        learning_prediction_disagreements_vs_zero_shot=sum(r['predicted_fraud'] != r['primary_generated_prediction'] for r in extension),
        descriptive_condition_breakdown=condition,
        posthoc_clean_attack_class_transitions=transitions,
        unique_original_nodes=len({r['node_id'] for r in provenance.values()}),
        context_count=len({r['context_sha256'] for r in primary}),
        branch_disagreements=sum(r['predictions']['self_only'] != r['predictions']['graph_only'] for r in primary),
        joint_errors=sum(bool(r['predictions']['self_only']) != bool(r['label']) and
                         bool(r['predictions']['graph_only']) != bool(r['label']) for r in primary),
        api=dict(physical_calls=len(ledger), status_counts=dict(collections.Counter(r['status'] for r in ledger)),
            verified_usage_records=sum(bool(r['usage_verified']) for r in ledger),
            charged_prompt_tokens=prompt, charged_completion_tokens=completion,
            usd_upper_bound=cost, response_models=sorted({r['model'] for r in ledger}),
            observed_system_fingerprints=sorted({str(r['response'].get('system_fingerprint')) for r in ledger}),
            started_utc=min(r['started_utc'] for r in ledger), completed_utc=max(r['completed_utc'] for r in ledger),
            summed_physical_request_seconds=sum(latencies), mean_physical_request_seconds=statistics.mean(latencies),
            max_physical_request_seconds=max(latencies),
            price_interpretation='Registered peak/cache-miss upper bound, not the provider invoice.'),
        protocol_sha256=protocol_sha, input_sha256={str(p.relative_to(root)):sha(p) for p in (
            root/'api_protocol_v1.json', root/'api_attempt1/api_attempts.jsonl',
            root/'api_attempt1/summary.json', root/'api_attempt1/case_outcomes.jsonl',
            root/'text_learning_attempt1/extension_summary.json', root/'text_learning_attempt1/extension_outcomes.jsonl',
            root/'numeric_gate_grid/outcomes.jsonl', root/'numeric_gate_grid/summary.json',
            root/'numeric_gate_continuous/summary.json',root/'probe32/provenance_private.jsonl')},
        limitations=['Numeric-only pilot: no audited article text mapping; text generation adds no new semantic case evidence.',
            'Four supervised TRAIN demonstrations provide in-context learning; no LLM weight update or trained text student.',
            'Only eight original nodes under repeated clean/attack contexts;32case rows are not32independent news items.',
            'Original GAGA TRAIN-meta scores are in-sample, NOT out-of-fold. Both branches are uncalibrated.',
            'Transfers from GCN-optimized attacks are not adaptive attacks against the entire API pipeline.',
            'Retained clean/attack class transitions are post-hoc diagnostics; a probe without induced graph-branch errors cannot establish protective benefit.',
            'Joint same-side errors cannot be corrected by fixed-threshold convex fusion.',
            'No superiority, calibrated rationale truth, full-test robustness or course approval follows.'])
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alignment', required=True, type=pathlib.Path)
    parser.add_argument('--private-root', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    report = dict(schema='monti_owner_methods_saved_recount_v1',
        status='completed_saved_output_recount', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        inference_calls=0, optimizer_steps=0, api_calls=0, gpu_hours=0,
        alignment=alignment(args.alignment), fusion=fusion(args.private_root))
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'], output_sha256=sha(args.output))))


if __name__ == '__main__':
    main()
