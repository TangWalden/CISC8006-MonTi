"""Export a completed, doubly audited downstream pilot without model imports.

Only completed JSON metadata and the two passed audit reports are read. The
export contains no individual node IDs, raw logits, feature tensors or model
checkpoints. It is a derived publication, not an end-to-end rerun bundle.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import time


SOURCE = '9e013cb182b5af71a9d0b719b5a939deda3c88e2'
RUNTIME = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned')
SEEDS = (3, 4, 5)
DETECTORS = ('gcn', 'gaga')
TRANSITIONS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
ROLE_FIELDS = ('target_edges', 'candidate_edges', 'injected_to_injected_owned_edges',
               'total_owned_undirected_edges', 'target_fraction', 'candidate_fraction',
               'injected_to_injected_fraction', 'injected_physical_degree_min',
               'injected_physical_degree_max', 'injected_physical_degree_sum')
QUOTA_FIELDS = ('ideal_target_quota', 'ideal_candidate_quota', 'lower_target_quota',
                'upper_target_quota', 'target_quota', 'candidate_quota',
                'requested_target_fraction', 'target_fraction', 'candidate_fraction',
                'projection_applied', 'exact_half', 'odd_budget', 'target_capacity_limited',
                'mandatory_or_candidate_capacity_limited')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def key(seed, group):
    return hashlib.sha256(json.dumps([seed, group, 'downstream_tensor_audit_v1'])
                          .encode('ascii')).hexdigest()[:20]


def close(left, right):
    require(type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9),
            'A compact value differs from the audited saved record.')


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def csv_bytes(columns, rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def optional_costs(run, protocol, provenance, result):
    """Keep active-process time distinct from whole-phase time including waits."""
    costs = dict(main_child_wall_seconds=result['wall_seconds'],
                 main_process_cpu_seconds=result['process_cpu_seconds'],
                 GPU_hours=result['GPU_hours'],
                 scope='Main child only; full guardian/preflight costs require their adjacent receipts.',
                 setup_attempts=protocol.get('setup_amendments', []))
    inputs = {}
    guardians = {}
    for phase in ('preflight', 'run'):
        path = run.parent / (phase + '_guardian.json')
        if path.is_file():
            item = read(path)
            require(item['phase'] == phase and item['status'] == 'completed'
                    and item['exit_code'] == 0 and item['protocol_sha256'] == provenance['protocol_sha256']
                    and item['whole_phase_GPU_hours'] == 0,
                    'A supplied phase guardian is incomplete or belongs to another protocol.')
            guardians[phase] = item
            inputs[str(path)] = sha(path)
    preflight_path = run.parent / 'preflight_attempt1' / 'result.json'
    preflight = None
    if preflight_path.is_file():
        preflight = read(preflight_path)
        require(preflight.get('status') == 'passed' and preflight.get('provenance') == provenance,
                'The adjacent preflight is not a passed same-protocol receipt.')
        inputs[str(preflight_path)] = sha(preflight_path)
        costs['preflight_child_wall_seconds'] = preflight['wall_seconds']
        costs['preflight_process_cpu_seconds'] = preflight['process_cpu_seconds']
        costs['preflight_counters'] = preflight['counters']
    prior = protocol.get('setup_amendments', [])
    prior_guardian = sum(row['prior_guardian_wall_seconds'] for row in prior)
    costs['prior_failed_startup_guardian_seconds'] = prior_guardian
    costs['prior_failed_startup_process_cpu_seconds'] = sum(
        row['prior_process_cpu_seconds'] or 0 for row in prior)
    if set(guardians) == {'preflight', 'run'} and preflight is not None:
        require(guardians['run']['child_result_sha256'] == sha(run / 'result.json')
                and guardians['preflight']['child_result_sha256'] == sha(preflight_path),
                'A guardian result binding differs.')
        whole = guardians['run']['whole_phase_wall_seconds_including_review_wait']
        costs.update(
            scope='All recorded startup attempts and completed preflight/main guardians; review wait is reported separately from active process time.',
            completed_attempt_whole_phase_wall_seconds_including_review_wait=whole,
            all_attempt_elapsed_seconds_including_completed_attempt_review_wait=whole + prior_guardian,
            active_guardian_wall_seconds=sum(row['guardian_wall_seconds'] for row in guardians.values()) + prior_guardian,
            all_recorded_child_process_cpu_seconds=result['process_cpu_seconds']
                + preflight['process_cpu_seconds'] + costs['prior_failed_startup_process_cpu_seconds'],
        )
    return costs, inputs


def build_exports(run_root, protocol_path, logit_audit_path, tensor_audit_path):
    run = Path(run_root).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    logit_audit_path = Path(logit_audit_path).resolve(strict=True)
    tensor_audit_path = Path(tensor_audit_path).resolve(strict=True)
    result = read(run / 'result.json')
    require(result.get('status') == 'completed', 'Do not export partial or failed experiment output.')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'monti_downstream_cooperation_v1' and protocol['status'] == 'frozen'
            and protocol['owner_execution_authorized'] is True and protocol['final_TEST_allowed'] is False
            and protocol['source_manifest']['source_commit'] == SOURCE
            and protocol['runtime_commit'] == RUNTIME and tuple(protocol['arms']) == ARMS
            and tuple(protocol['seeds']) == SEEDS, 'The executed study identity or boundaries differ.')
    provenance = dict(source_commit=SOURCE, runtime_commit=RUNTIME, protocol_sha256=sha(protocol_path))
    require(result['provenance'] == provenance, 'Completed source/protocol binding differs.')
    logit = read(logit_audit_path)
    tensor = read(tensor_audit_path)
    require(logit.get('status') == tensor.get('status') == 'passed', 'Both independent offline audits must pass.')
    require(logit['schema'] == 'monti_downstream_saved_logit_recount_v1'
            and tensor['schema'] == 'monti_downstream_saved_tensor_geometry_audit_v1', 'Audit schemas differ.')
    recounted = logit['recount']
    require(recounted['provenance'] == tensor['provenance'] == provenance, 'Audit source/protocol binding differs.')
    inputs = {str(protocol_path): sha(protocol_path),
              str(run / 'result.json'): sha(run / 'result.json'),
              str(logit_audit_path): sha(logit_audit_path), str(tensor_audit_path): sha(tensor_audit_path)}
    require(logit['input_file_hashes']['EXPLICIT_PROTOCOL'] == provenance['protocol_sha256']
            and tensor['inputs']['protocol_sha256'] == provenance['protocol_sha256'], 'An audit used another protocol.')
    tensor_inputs = dict(result='completed_result_sha256', upstream_cache_manifest='upstream_cache_manifest_sha256',
                         sealed_before_victim='sealed_plan_manifest_sha256')
    for name, tensor_field in tensor_inputs.items():
        path = run / (name + '.json')
        actual = sha(path)
        require(logit['input_file_hashes'][name + '.json'] == actual
                and tensor['inputs'][tensor_field] == actual, 'An audit is stale for ' + name)
        inputs[str(path)] = actual
    summary = read(run / 'summary.json')
    require(summary.get('status') == 'completed'
            and all(name in result and result[name] == value for name, value in summary.items()),
            'Completed summary and result copies differ.')
    require(logit['input_file_hashes']['summary.json'] == sha(run / 'summary.json'), 'The audited summary changed.')
    inputs[str(run / 'summary.json')] = sha(run / 'summary.json')

    groups = protocol['valid_groups']
    require(len(groups) == len(set(groups)) == 64 and len(protocol['train_groups']) == 128,
            'Frozen group counts differ.')
    cohort = protocol['cohort']
    require(cohort['valid_target_occurrences'] == recounted['occurrence_denominator_per_arm_seed'] == 571
            and cohort['unique_valid_targets'] == recounted['unique_valid_targets'] == 240
            and cohort['valid_target_overlap_with_previous_contexts'] == 233,
            'Registered development denominator or repeated-target scope differs.')
    expected_keys = {(seed, group, arm) for seed in SEEDS for group in groups for arm in ARMS}
    cases = result['all_case_counts']
    by_key = {(row['seed'], row['group'], row['arm']): row for row in cases}
    require(len(cases) == len(by_key) == 768 and set(by_key) == expected_keys, 'Incomplete or duplicated result cases.')
    roles = {(row['seed'], row['opaque_case_key']): row for row in tensor['validation_role_counts']}
    require(len(tensor['validation_role_counts']) == len(roles) == 192
            and set(roles) == {(seed, key(seed, group)) for seed in SEEDS for group in groups},
            'Geometry audit coverage does not match the exported cohort.')
    compact_cases = []
    quotas = []
    for seed in SEEDS:
        for group in groups:
            opaque = key(seed, group)
            geometry = roles[seed, opaque]
            q = geometry['planned_role_allocation']
            quota_row = dict(opaque_case_key=opaque, seed=seed, targets=geometry['targets'],
                             candidates=geometry['candidates'], node_budget=geometry['injected_nodes'],
                             edge_budget=q['edge_budget'])
            quota_row.update({name: q[name] for name in QUOTA_FIELDS})
            for arm in ARMS:
                row = by_key[seed, group, arm]
                actual = geometry['arms'][arm]
                require(row['quota'] == q and row['targets'] == geometry['targets']
                        and row['actual_nodes'] == geometry['injected_nodes']
                        and row['actual_edges'] == actual['total_owned_undirected_edges'] == q['edge_budget'],
                        'Result and independently recounted geometry differ.')
                require(actual['target_edges'] + actual['candidate_edges']
                        + actual['injected_to_injected_owned_edges'] == q['edge_budget'], 'Actual role counts do not sum to the total.')
                if arm != 'native_monti':
                    require(actual['target_edges'] == q['target_quota'] and actual['candidate_edges'] == q['candidate_quota']
                            and actual['injected_to_injected_owned_edges'] == 0, 'Quota arm geometry differs.')
                compact = dict(opaque_case_key=opaque, seed=seed, arm=arm, targets=row['targets'],
                               candidates=geometry['candidates'], node_budget=row['actual_nodes'],
                               edge_budget=q['edge_budget'], actual_nodes=row['actual_nodes'], actual_edges=row['actual_edges'])
                compact.update({'planned_' + name: q[name] for name in QUOTA_FIELDS})
                compact.update({'actual_' + name: actual[name] for name in ROLE_FIELDS})
                for detector in DETECTORS:
                    transitions = row[detector]
                    require(set(transitions) == set(TRANSITIONS) and transitions['occurrences'] == row['targets']
                            and all(type(value) is int and 0 <= value <= row['targets'] for value in transitions.values())
                            and transitions['attacked_wrong'] == transitions['clean_wrong']
                            + transitions['new_wrong'] - transitions['recoveries'], 'Invalid saved transition counts.')
                    compact.update({detector + '_' + name: transitions[name] for name in TRANSITIONS})
                compact_cases.append(compact)
                for name in ROLE_FIELDS:
                    quota_row[arm + '_actual_' + name] = actual[name]
            quotas.append(quota_row)

    metrics = []
    metric_keys = {(row['seed'], row['arm']): row for row in result['seed_metrics']}
    require(len(metric_keys) == len(result['seed_metrics']) == 12, 'Seed metrics differ.')
    for seed in SEEDS:
        for arm in ARMS:
            selected = [row for row in compact_cases if row['seed'] == seed and row['arm'] == arm]
            n = sum(row['targets'] for row in selected)
            require(len(selected) == 64 and n == 571, 'A compact denominator was filtered.')
            item = dict(seed=seed, arm=arm, groups=64, occurrences=n)
            for detector in DETECTORS:
                for name in TRANSITIONS[1:]:
                    item[detector + '_' + name] = sum(row[detector + '_' + name] for row in selected)
                item[detector + '_mr_percent'] = 100 * item[detector + '_attacked_wrong'] / n
                close(item[detector + '_mr_percent'], metric_keys[seed, arm][detector + '_mr_percent'])
            metrics.append(item)
    means = {detector: {arm: statistics.mean(row[detector + '_mr_percent'] for row in metrics if row['arm'] == arm)
                        for arm in ARMS} for detector in DETECTORS}
    for arm in ARMS:
        close(means['gaga'][arm], result['gaga_mean_mr_percent'][arm])
        close(means['gaga'][arm], recounted['gaga_mean_mr_percent'][arm])
    contrasts = result['contrasts']
    require(set(contrasts) == set(recounted['contrasts']) == set(ARMS[:-1]), 'Contrast inventory differs.')
    for control in ARMS[:-1]:
        left, right = contrasts[control], recounted['contrasts'][control]
        require(len(left['seed_differences_pp']) == len(right['seed_differences_pp']) == 3
                and len(left['nominal_paired_t95_pp']) == len(right['nominal_paired_t95_pp']) == 2,
                'Contrast vector length differs.')
        for name in ('seed_differences_pp', 'nominal_paired_t95_pp'):
            for saved, checked in zip(left[name], right[name]):
                close(saved, checked)
        close(left['mean_pp'], right['mean_pp'])
    require(result['qualification_met'] is recounted['qualification_met']
            and result['controls_promoted'] is False and result['full_TEST_evaluated'] is False,
            'Audited contrast, qualification or final-evaluation scope differs.')
    require(result['GPU_hours'] == 0 and result['counters'] == recounted['producer_logical_counters']
            and result['counters']['API_calls'] == result['counters']['TEST_queries'] == 0,
            'Execution cost or query counters differ.')
    require(len(result['downstream']) == 3
            and all(row['parameter_count'] == 151872 for row in result['downstream']), 'Downstream parameter accounting differs.')
    costs, extra_inputs = optional_costs(run, protocol, provenance, result)
    inputs.update(extra_inputs)
    latest = dict(
        schema='monti_downstream_cooperation_compact_v1', method='Fixed upstream MonTi with a trained conditional candidate-edge scorer',
        study='GossipCop-S split0 development VALID; projected quotas, not an exact-half experiment',
        provenance=provenance,
        metric_definition='MR=100*incorrect fraud-target predictions/all target occurrences, including clean errors; higher is a stronger attack.',
        arms=list(ARMS), seeds=list(SEEDS),
        cohort=dict(train_groups=128, validation_groups_per_seed=64, occurrences_per_arm_seed=571,
                    pooled_correlated_occurrences_per_arm=1713, unique_validation_targets=240,
                    validation_targets_seen_in_earlier_contexts=233,
                    interpretation='The group contexts are unused, but most target identities have appeared in earlier development groups; pooled occurrences are correlated.'),
        mean_mr_percent=means, gaga_mean_mr_percent=means['gaga'], contrasts=contrasts,
        primary='quota_learned', qualification_met=result['qualification_met'],
        qualified_candidate='quota_learned' if result['qualification_met'] else None,
        controls_promoted=False, full_TEST_evaluated=False,
        quota_summary=tensor['quota_summary'], counters=result['counters'], costs=costs,
        downstream=result['downstream'],
        audit_scopes=dict(
            logits='Passed independent saved-logit prediction/count/aggregate recount and declared file-byte bindings; no tensor deserialization or new inference.',
            geometry='Passed independent deserialization of own saved captures/plans, with tensor content hashes, IDs, quotas and owned-adjacency geometry; no original dataset, model checkpoint or executed-graph inspection.',
            human_verification=False),
        limits=[
            'Three training seeds on a fixed adaptively developed validation setup; the nominal paired interval is not an independent population guarantee.',
            'The trained scorer adds 151872 parameters. No equal-capacity or action-hidden training ablation isolates the contribution of seeing upstream decisions.',
            'Quota arms can change per-auxiliary-node degrees and remove native injected-to-injected edges. Their gain over native MonTi does not isolate downstream learning.',
            'The three quota arms share fixed upstream features, target attachments, candidate lists, total node/edge budgets and projected role quotas.',
            'The downstream observes an attachment mean and count, not a lossless encoding of every upstream decision.',
            'Training uses a biased selected-edge straight-through surrogate, not the exact derivative of discrete topology.',
            'No real fraud supply chain, novelty, human technical review, course approval or final TEST improvement is established.',
        ],
    )
    outputs = dict(
        **{'latest.json': json_bytes(latest),
           'seed_metrics.csv': csv_bytes(list(metrics[0]), metrics),
           'case_counts.csv': csv_bytes(list(compact_cases[0]), compact_cases),
           'quota_by_case.csv': csv_bytes(list(quotas[0]), quotas)},
        **{'audit_saved.json': logit_audit_path.read_bytes(), 'audit_tensors.json': tensor_audit_path.read_bytes()})
    publication = dict(
        schema='monti_downstream_compact_publication_provenance_v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_execution_source_commit=SOURCE, scientific_runtime_commit=RUNTIME,
        scientific_protocol_sha256=provenance['protocol_sha256'],
        original_run_root=str(run), original_protocol_path=str(protocol_path),
        input_json_file_sha256=inputs, exporter_source_sha256=sha(__file__),
        output_sha256={name: hashlib.sha256(value).hexdigest() for name, value in outputs.items()},
        row_counts=dict(seed_metrics=12, case_counts=768, quota_by_case=192),
        audit_copy_policy='The two passed audit JSONs are copied byte-for-byte; their original scope and limitations are unchanged.',
        opaque_key_scheme='SHA256 of JSON[seed, original group key, downstream_tensor_audit_v1], first 20 hex digits; individual node IDs are omitted.',
        reproducibility='Derived saved-record publication. Original data, checkpoints and ignored run artifacts remain private; this is not a fresh-clone end-to-end execution bundle.',
        export_new_model_queries=0, export_tensor_deserializations=0, export_optimizer_updates=0,
        export_GPU_hours=0, export_API_calls=0, export_TEST_queries=0,
    )
    try:
        publication['exporter_publication_source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        publication['exporter_publication_source_commit'] = None
    outputs['provenance.json'] = json_bytes(publication)
    return outputs, latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'logit-audit', 'tensor-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    start = time.perf_counter()
    output = Path(args.output).resolve()
    require(output.parent.is_dir() and not output.exists(), 'The parent must exist and the output directory must be new.')
    files, latest = build_exports(args.run_root, args.protocol, args.logit_audit, args.tensor_audit)
    output.mkdir(exist_ok=False)
    for name, raw in files.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    print(json.dumps(dict(status='exported_completed_audited_records', output=str(output),
                          files=list(files), qualification_met=latest['qualification_met'],
                          wall_seconds=time.perf_counter() - start, model_queries=0, GPU_hours=0), sort_keys=True))


if __name__ == '__main__':
    main()
