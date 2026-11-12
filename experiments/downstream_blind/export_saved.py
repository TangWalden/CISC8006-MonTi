"""Publish a completed action-blind ablation from audited saved records only.

This standard-library exporter reads JSON metadata, not tensors or models.
Compact tables omit individual node IDs, raw predictions, features and weights.
The two separately authored audit reports are retained byte-for-byte.
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


SOURCE = 'b7809f420fc1b5453e748a9fd19c52234541cfd8'
PROTOCOL_SHA = '8bafbaf6c7590cd97f492f9d5edca90848193c76e5e3da0c4fdf5d8399500fd8'
RUNTIME = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
PARENT_SOURCE = '9e013cb182b5af71a9d0b719b5a939deda3c88e2'
PARENT_PROTOCOL_SHA = '5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'
SEEDS = (3, 4, 5)
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned', 'quota_blind')
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
ALGEBRA_FIELDS = ('all_targets_attached', 'exact_same_mean', 'exact_same_log_count',
                  'exact_same_summary', 'maximum_mean_difference', 'maximum_log_count_difference')


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
    # Preserve the parent's opaque key, so reused geometry has a literal join.
    return hashlib.sha256(json.dumps([seed, group, 'downstream_tensor_audit_v1'])
                          .encode('ascii')).hexdigest()[:20]


def blind_key(seed, group):
    return hashlib.sha256(json.dumps([seed, group, 'downstream_blind_tensor_audit_v1'])
                          .encode('ascii')).hexdigest()[:20]


def close(left, right):
    require(type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9),
            'A saved numerical value differs from the independently audited value.')


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + '\n').encode('utf-8')


def csv_bytes(rows):
    require(bool(rows), 'An exported table must not be empty.')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n', extrasaction='raise')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def parent_file(protocol, field, inputs):
    ref = protocol[field]
    relative = Path(ref['path'])
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe parent metadata path.')
    root = Path(protocol['runtime_root']).resolve(strict=True)
    path = (root / relative).resolve(strict=True)
    require(root in path.parents and sha(path) == ref['sha256'], 'A frozen parent metadata file changed.')
    inputs[str(path)] = ref['sha256']
    return path


def costs(run, protocol, provenance, result, logit, tensor, inputs):
    guardians = {}
    for phase in ('preflight', 'run'):
        path = run.parent / (phase + '_guardian.json')
        item = read(path)
        require(item['phase'] == phase and item['status'] == 'completed' and item['exit_code'] == 0
                and item['protocol_sha256'] == provenance['protocol_sha256']
                and item['whole_phase_GPU_hours'] == 0,
                'A completed same-protocol guardian is required for every executed phase.')
        guardians[phase] = item
        inputs[str(path)] = sha(path)
    preflight_path = run.parent / 'preflight_attempt1' / 'result.json'
    preflight = read(preflight_path)
    require(preflight['status'] == 'passed' and preflight['provenance'] == provenance
            and preflight['GPU_hours'] == 0, 'The preflight identity or completion differs.')
    require(guardians['preflight']['child_result_sha256'] == sha(preflight_path)
            and guardians['run']['child_result_sha256'] == sha(run / 'result.json'),
            'A guardian child-result byte binding differs.')
    inputs[str(preflight_path)] = sha(preflight_path)
    amendments = protocol.get('setup_amendments', [])
    failed_guardian = sum(row['prior_guardian_wall_seconds'] for row in amendments)
    failed_cpu = sum(row.get('prior_process_cpu_seconds') or 0 for row in amendments)
    return dict(
        scope='New blind phase only. Reused parent models and predictions retain their separately charged historical costs. Process CPU, active guardian wall time and phase elapsed time are distinct.',
        GPU_hours=0,
        main_child_wall_seconds=result['wall_seconds'],
        main_process_cpu_seconds=result['process_cpu_seconds'],
        preflight_child_wall_seconds=preflight['wall_seconds'],
        preflight_process_cpu_seconds=preflight['process_cpu_seconds'],
        preflight_real_data_counters=preflight['counters'],
        prior_failed_startup_guardian_seconds=failed_guardian,
        prior_failed_startup_process_cpu_seconds=failed_cpu,
        retained_startup_amendments=amendments,
        active_guardian_wall_seconds=sum(row['guardian_wall_seconds'] for row in guardians.values()) + failed_guardian,
        completed_attempt_whole_phase_wall_seconds_including_review_wait=guardians['run']['whole_phase_wall_seconds_including_review_wait'],
        all_attempt_elapsed_seconds_including_completed_attempt_review_wait=guardians['run']['whole_phase_wall_seconds_including_review_wait'] + failed_guardian,
        recorded_child_process_cpu_seconds=result['process_cpu_seconds'] + preflight['process_cpu_seconds'] + failed_cpu,
        logit_audit_wall_seconds=logit['wall_seconds'],
        logit_audit_process_cpu_seconds=logit.get('process_cpu_seconds'),
        tensor_audit_wall_seconds=tensor['wall_seconds'],
        tensor_audit_process_cpu_seconds=tensor.get('process_cpu_seconds'),
        preparation_cost_scope='Initial source preparation, static review and synthetic-only work outside the executed child receipts are not included in main-process CPU.',
    )


def build_exports(run_root, protocol_path, logit_audit_path, tensor_audit_path):
    run = Path(run_root).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)
    logit_path = Path(logit_audit_path).resolve(strict=True)
    tensor_path = Path(tensor_audit_path).resolve(strict=True)
    result = read(run / 'result.json')
    require(result.get('status') == 'completed', 'Partial or failed runs cannot be published as completed results.')
    protocol = read(protocol_path)
    require(protocol['schema'] == 'monti_downstream_blind_v1' and protocol['status'] == 'frozen'
            and protocol['owner_execution_authorized'] is True and protocol['post_hoc'] is True
            and protocol['final_TEST_allowed'] is False and protocol['runtime_commit'] == RUNTIME
            and protocol['source_manifest']['source_commit'] == SOURCE
            and tuple(protocol['arms']) == ARMS and tuple(protocol['seeds']) == SEEDS
            and sha(protocol_path) == PROTOCOL_SHA,
            'The executed action-blind study identity or information boundary differs.')
    provenance = dict(source_commit=SOURCE, runtime_commit=RUNTIME, protocol_sha256=PROTOCOL_SHA,
                      parent_protocol_sha256=PARENT_PROTOCOL_SHA)
    require(result['provenance'] == provenance, 'Completed scientific provenance differs.')
    logit, tensor = read(logit_path), read(tensor_path)
    require(logit.get('status') == tensor.get('status') == 'passed', 'Both separate saved-record audits must pass.')
    require(logit['schema'] == 'monti_downstream_blind_saved_logit_audit_v1'
            and tensor['schema'] == 'monti_downstream_blind_saved_tensor_geometry_audit_v1',
            'An unexpected audit interface was supplied.')
    recount = logit['recount']
    require(recount['provenance'] == tensor['provenance'] == provenance, 'Audit provenance differs.')
    inputs = {str(protocol_path): sha(protocol_path), str(run / 'result.json'): sha(run / 'result.json'),
              str(logit_path): sha(logit_path), str(tensor_path): sha(tensor_path)}
    require(logit['input_file_hashes'][str(protocol_path)] == PROTOCOL_SHA
            and tensor['inputs']['protocol_sha256'] == PROTOCOL_SHA, 'The audits used another protocol.')
    for name, field in (('result', 'completed_result_sha256'),
                        ('sealed_before_victim', 'new_seal_sha256'),
                        ('conditioning_scope', 'conditioning_scope_sha256')):
        path = run / (name + '.json')
        actual = sha(path)
        require(logit['input_file_hashes'][str(path)] == actual
                and tensor['inputs'][field] == actual, 'An audit is stale for ' + name)
        inputs[str(path)] = actual
    summary = read(run / 'summary.json')
    require(summary.get('status') == 'completed'
            and all(name in result and result[name] == value for name, value in summary.items()),
            'Completed summary/result copies differ.')
    require(logit['input_file_hashes'][str(run / 'summary.json')] == sha(run / 'summary.json'), 'The audited summary changed.')
    inputs[str(run / 'summary.json')] = sha(run / 'summary.json')

    old_result_path = parent_file(protocol, 'parent_result', inputs)
    old_geometry_path = parent_file(protocol, 'parent_tensor_audit', inputs)
    parent = read(old_result_path)
    old_geometry = read(old_geometry_path)
    parent_provenance = dict(source_commit=PARENT_SOURCE, runtime_commit=RUNTIME, protocol_sha256=PARENT_PROTOCOL_SHA)
    require(parent['status'] == 'completed' and parent['provenance'] == parent_provenance
            and old_geometry['status'] == 'passed' and old_geometry['provenance'] == parent_provenance,
            'Reused parent predictions or geometry have another scientific identity.')
    require(result['prior_qualification_unchanged'] is True
            and result['reused_comparator_cases'] == 768 and result['reused_clean_groups'] == 64
            and result['new_blind_cases'] == 192 and result['full_TEST_evaluated'] is False,
            'Reused/new outcome accounting or final-evaluation scope differs.')
    groups = protocol['valid_groups']
    require(len(groups) == len(set(groups)) == 64 and len(protocol['train_groups']) == 128
            and protocol['cohort']['valid_target_occurrences'] == 571
            and protocol['cohort']['unique_valid_targets'] == 240
            and protocol['cohort']['valid_target_overlap_with_previous_contexts'] == 233,
            'The fixed development cohort differs.')
    expected = {(seed, group, arm) for seed in SEEDS for group in groups for arm in ARMS}
    by_key = {(row['seed'], row['group'], row['arm']): row for row in result['all_case_counts']}
    require(len(by_key) == len(result['all_case_counts']) == 960 and set(by_key) == expected,
            'Missing or duplicated seed/group/arm cases.')
    old_by_key = {(row['seed'], row['group'], row['arm']): row for row in parent['all_case_counts']}
    require(len(old_by_key) == 768 and all(by_key[k] == row for k, row in old_by_key.items()),
            'An old comparator was changed instead of reused.')
    old_roles = {(row['seed'], row['opaque_case_key']): row for row in old_geometry['validation_role_counts']}
    new_roles = {(row['seed'], row['opaque_case_key']): row for row in tensor['validation_geometry']}
    expected_roles = {(seed, key(seed, group)) for seed in SEEDS for group in groups}
    expected_blind_roles = {(seed, blind_key(seed, group)) for seed in SEEDS for group in groups}
    require(len(old_roles) == len(new_roles) == 192 and set(old_roles) == expected_roles
            and set(new_roles) == expected_blind_roles,
            'Audited geometry does not cover every registered case.')
    compact_cases, quotas = [], []
    for seed in SEEDS:
        for group in groups:
            opaque = key(seed, group)
            original, new = old_roles[seed, opaque], new_roles[seed, blind_key(seed, group)]
            quota = original['planned_role_allocation']
            require(new['planned_quota'] == quota and new['targets'] == original['targets']
                    and new['candidates'] == original['candidates']
                    and new['injected_nodes'] == original['injected_nodes']
                    and new['upstream_feature_sha256'] == original['frozen_upstream_feature_sha256']
                    and new['upstream_target_block_sha256'] == original['frozen_upstream_target_block_sha256'],
                    'Blind geometry changed frozen case dimensions or upstream bindings.')
            geometry = dict(original['arms'])
            geometry['quota_blind'] = new['actual_roles']
            quota_row = dict(opaque_case_key=opaque, seed=seed, targets=original['targets'],
                             candidates=original['candidates'], node_budget=original['injected_nodes'],
                             edge_budget=quota['edge_budget'])
            quota_row.update({name: quota[name] for name in QUOTA_FIELDS})
            for arm in ARMS:
                row, actual = by_key[seed, group, arm], geometry[arm]
                require(row['quota'] == quota and row['targets'] == original['targets']
                        and row['actual_nodes'] == original['injected_nodes']
                        and row['actual_edges'] == actual['total_owned_undirected_edges'] == quota['edge_budget'],
                        'Saved counts and audited geometry disagree.')
                require(actual['target_edges'] + actual['candidate_edges']
                        + actual['injected_to_injected_owned_edges'] == quota['edge_budget'], 'Actual owned edge roles do not sum to total.')
                if arm != 'native_monti':
                    require(actual['target_edges'] == quota['target_quota']
                            and actual['candidate_edges'] == quota['candidate_quota']
                            and actual['injected_to_injected_owned_edges'] == 0, 'Quota-arm realized geometry differs.')
                compact = dict(opaque_case_key=opaque, seed=seed, arm=arm,
                               measurement_origin='new_blind' if arm == 'quota_blind' else 'reused_parent',
                               targets=row['targets'], candidates=original['candidates'],
                               node_budget=row['actual_nodes'], edge_budget=quota['edge_budget'],
                               actual_nodes=row['actual_nodes'], actual_edges=row['actual_edges'])
                compact.update({'planned_' + name: quota[name] for name in QUOTA_FIELDS})
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
    saved_metrics = {(row['seed'], row['arm']): row for row in result['seed_metrics']}
    require(len(saved_metrics) == len(result['seed_metrics']) == 15, 'Seed metric inventory differs.')
    for seed in SEEDS:
        for arm in ARMS:
            selected = [row for row in compact_cases if row['seed'] == seed and row['arm'] == arm]
            require(len(selected) == 64 and sum(row['targets'] for row in selected) == 571, 'A denominator was filtered.')
            item = dict(seed=seed, arm=arm, measurement_origin='new_blind' if arm == 'quota_blind' else 'reused_parent',
                        groups=64, occurrences=571)
            for detector in DETECTORS:
                for name in TRANSITIONS[1:]:
                    item[detector + '_' + name] = sum(row[detector + '_' + name] for row in selected)
                    require(item[detector + '_' + name] == saved_metrics[seed, arm][detector + '_' + name],
                            'An exported aggregate differs from the saved seed metric.')
                item[detector + '_mr_percent'] = 100 * item[detector + '_attacked_wrong'] / 571
                close(item[detector + '_mr_percent'], saved_metrics[seed, arm][detector + '_mr_percent'])
            metrics.append(item)
    means = {d: {a: statistics.mean(row[d + '_mr_percent'] for row in metrics if row['arm'] == a)
                 for a in ARMS} for d in DETECTORS}
    for detector in DETECTORS:
        for arm in ARMS:
            close(means[detector][arm], result['mean_mr_percent'][detector][arm])
            close(means[detector][arm], recount['mean_mr_percent'][detector][arm])
    require(set(result['contrasts']) == set(recount['contrasts'])
            == {'visible_minus_blind', 'visible_minus_random', 'blind_minus_random'}, 'Contrast inventory differs.')
    for name, left in result['contrasts'].items():
        right = recount['contrasts'][name]
        for field in ('seed_differences_pp', 'nominal_paired_t95_pp'):
            require(len(left[field]) == len(right[field]) == (3 if field == 'seed_differences_pp' else 2),
                    'Contrast vector length differs.')
            for saved, checked in zip(left[field], right[field]):
                close(saved, checked)
        close(left['mean_pp'], right['mean_pp'])
    require(result['direct_action_summary_criterion_met'] is recount['direct_action_summary_criterion_met'],
            'The separate mechanistic screen differs from the independently recounted screen.')
    require(result['GPU_hours'] == 0 and result['counters'] == recount['producer_logical_counters']
            and result['counters']['API_calls'] == result['counters']['TEST_queries'] == 0
            and result['counters']['upstream_generations'] == result['counters']['upstream_optimizer_updates'] == 0,
            'New compute/query counters differ.')
    require(len(result['downstream']) == 3
            and {row['seed'] for row in result['downstream']} == set(SEEDS)
            and all(row['parameter_count'] == 151872 and row['epochs'] == 4
                    and row['matches_visible_initial_state'] is True for row in result['downstream']),
            'Matched downstream architecture or initialization accounting differs.')

    algebra = read(run / 'conditioning_scope.json')
    require(algebra['provenance'] == provenance and len(algebra['cases']) == 576, 'Saved algebra identity or coverage differs.')
    tensor_algebra = {(row['seed'], row['phase'], row['group']): row for row in tensor['conditioning_cases']}
    require(len(tensor_algebra) == len(tensor['conditioning_cases']) == 576,
            'The tensor audit lacks a complete independent conditioning recount.')
    expected_algebra = {(seed, phase, group) for seed in SEEDS
                        for phase, selected in (('train', protocol['train_groups']), ('validation', groups)) for group in selected}
    seen, conditioning = set(), []
    for row in algebra['cases']:
        case = row['seed'], row['phase'], row['group']
        require(case not in seen and case in expected_algebra, 'Duplicated or unregistered conditioning-algebra row.')
        require(tensor_algebra.get(case) == row, 'Conditioning-algebra output differs from the tensor audit.')
        seen.add(case)
        item = dict(opaque_case_key=key(row['seed'], row['group']), seed=row['seed'], phase=row['phase'])
        for name in ALGEBRA_FIELDS:
            value = row[name]
            if name.startswith('maximum_'):
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Nonfinite or negative algebra discrepancy.')
            else:
                require(type(value) is bool, 'Algebra flag must be Boolean.')
            item[name] = value
        require(item['exact_same_summary'] is (item['exact_same_mean'] and item['exact_same_log_count']),
                'Algebra equality flags disagree.')
        conditioning.append(item)
    require(seen == expected_algebra, 'Incomplete conditioning-algebra cohort.')
    cost = costs(run, protocol, provenance, result, logit, tensor, inputs)
    latest = dict(
        schema='monti_downstream_blind_compact_v1', provenance=provenance,
        method='Action-summary ablation of fixed-upstream downstream training',
        study='Post-hoc GossipCop-S split0 development comparison on the already inspected parent cohort.',
        arms=list(ARMS), seeds=list(SEEDS), new_arm='quota_blind',
        measurement_origins=dict(reused_parent_arms=list(ARMS[:-1]), new_blind_arm='quota_blind',
                                 reused_parent_cases=768, reused_clean_groups=64, new_blind_cases=192),
        metric_definition='MR=100*incorrect fraud-target predictions/all target occurrences, including clean errors; higher denotes a stronger attack.',
        cohort=dict(train_groups=128, validation_groups_per_seed=64, occurrences_per_arm_seed=571,
                    pooled_correlated_occurrences_per_arm=1713, unique_validation_targets=240,
                    validation_targets_seen_in_earlier_contexts=233, current_validation_outcomes_previously_inspected=True),
        mean_mr_percent=means, gaga_mean_mr_percent=means['gaga'],
        contrasts=result['contrasts'], primary='visible_minus_blind',
        direct_action_summary_criterion_met=result['direct_action_summary_criterion_met'],
        previous_visible_pilot_qualification_met=parent['qualification_met'], prior_qualification_unchanged=True,
        controls_promoted=False, full_TEST_evaluated=False,
        information_limit=protocol['information_limit'], conditioning_census=tensor['conditioning_census'],
        topology_identity_counts=tensor['topology_identity_counts'], counters=result['counters'], costs=cost,
        downstream=result['downstream'],
        audit_scopes=dict(
            logits='Separately authored saved-logit recount and file-byte bindings; no new inference.',
            geometry='Own saved cache/plan deserialization and geometry/summary algebra; executed graph internals and seal chronology are not independently proven.',
            reused_geometry='Old four-arm actual role counts are reused from the exact hash-bound passed parent geometry audit.',
            human_verification=False),
        limits=[
            'The direct attached-target mean and row-count summary are removed; common generated features/context and TRAIN feedback can still indirectly encode upstream information.',
            'Both trained policies have 151872 parameters, identical initial states, training order, epochs and optimization settings; only the conditioning summary changes.',
            'The full-target-block cases remove no direct summary information; separately trained policy equality does not follow from equality of those case inputs.',
            'Only three training seeds and adaptively inspected development groups are used; nominal intervals are not independent population guarantees.',
            'Passing the direct-summary screen does not itself establish superiority over fixed random candidate connections.',
            'No upstream adaptation, real fraud supply chain, novelty, human verification, course approval or final TEST benefit is established.',
        ])
    outputs = {
        'latest.json': json_bytes(latest), 'seed_metrics.csv': csv_bytes(metrics),
        'case_counts.csv': csv_bytes(compact_cases), 'quota_by_case.csv': csv_bytes(quotas),
        'conditioning_scope.csv': csv_bytes(conditioning),
        'audit_saved.json': logit_path.read_bytes(), 'audit_tensors.json': tensor_path.read_bytes(),
    }
    publication = dict(
        schema='monti_downstream_blind_compact_publication_provenance_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scientific_execution_source_commit=SOURCE, scientific_runtime_commit=RUNTIME,
        scientific_protocol_sha256=PROTOCOL_SHA, parent_scientific_source_commit=PARENT_SOURCE,
        parent_scientific_protocol_sha256=PARENT_PROTOCOL_SHA,
        original_run_root=str(run), original_protocol_path=str(protocol_path),
        input_json_file_sha256=inputs, exporter_source_sha256=sha(__file__),
        exporter_source_binding='The exact script SHA256 and separately retained post-run source manifest bind this exporter. The ambient Git commit, when available, is context only.',
        output_sha256={name: hashlib.sha256(value).hexdigest() for name, value in outputs.items()},
        row_counts=dict(seed_metrics=15, case_counts=960, quota_by_case=192, conditioning_scope=576),
        audit_copy_policy='Both passed audit reports are copied byte-for-byte, retaining their original limitations.',
        opaque_key_scheme='Parent-compatible SHA256 of JSON[seed, group, downstream_tensor_audit_v1], first20hex. Node IDs are omitted from compact tables.',
        reproducibility='Saved-record publication; original private assets and ignored execution artifacts are required for a full model rerun.',
        export_model_queries=0, export_tensor_deserializations=0, export_optimizer_updates=0,
        export_GPU_hours=0, export_API_calls=0, export_TEST_queries=0)
    try:
        publication['exporter_ambient_repository_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[2], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        publication['exporter_ambient_repository_commit'] = None
    outputs['provenance.json'] = json_bytes(publication)
    return outputs, latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'logit-audit', 'tensor-audit', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    require(output.parent.is_dir() and not output.exists(), 'An existing parent and a new exclusive export directory are required.')
    started = time.perf_counter()
    files, latest = build_exports(args.run_root, args.protocol, args.logit_audit, args.tensor_audit)
    output.mkdir(exist_ok=False)
    for name, raw in files.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    print(json.dumps(dict(status='exported_completed_doubly_audited_ablation', output=str(output),
                          files=list(files), direct_action_summary_criterion_met=latest['direct_action_summary_criterion_met'],
                          wall_seconds=time.perf_counter() - started, model_queries=0, GPU_hours=0), sort_keys=True))


if __name__ == '__main__':
    main()
