"""Independently recount completed action-blind saved logits and declarations.

This post-execution auditor imports only the standard library. It hashes own
saved tensor/checkpoint files but does not deserialize them or instantiate a
model. A separate saved-tensor audit is needed to inspect actual geometry.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
import traceback


SOURCE = 'b7809f420fc1b5453e748a9fd19c52234541cfd8'
PARENT_SOURCE = '9e013cb182b5af71a9d0b719b5a939deda3c88e2'
RUNTIME = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
PARENT_PROTOCOL = '5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'
SEEDS = (3, 4, 5)
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned', 'quota_blind')
DETECTORS = ('gcn', 'gaga')
TRANSITION_FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(left, right):
    require(type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, abs_tol=1e-9, rel_tol=1e-10),
            'A saved numeric value differs from the independent recount.')


def logits_to_classes(logits, count):
    require(isinstance(logits, list) and len(logits) == count, 'Saved logit denominator differs.')
    result = []
    for values in logits:
        require(isinstance(values, list) and len(values) == 2
                and all(type(value) in (int, float) and math.isfinite(value) for value in values),
                'Saved binary-class logits are missing or nonfinite.')
        # Torch argmax selects the first index on an exact tie.
        result.append(0 if values[0] >= values[1] else 1)
    return result


def transitions(clean, attacked):
    require(len(clean) == len(attacked), 'Saved clean/attacked occurrence counts differ.')
    return dict(occurrences=len(clean), clean_wrong=sum(value != 1 for value in clean),
                attacked_wrong=sum(value != 1 for value in attacked),
                new_wrong=sum(a == 1 and b != 1 for a, b in zip(clean, attacked)),
                recoveries=sum(a != 1 and b == 1 for a, b in zip(clean, attacked)))


def paired(values):
    require(len(values) == 3 and all(math.isfinite(value) for value in values),
            'Exactly three finite seed differences are required.')
    mean = statistics.mean(values)
    radius = 4.302652729911275 * statistics.stdev(values) / math.sqrt(3)
    return dict(seed_differences_pp=values, mean_pp=mean,
                nominal_paired_t95_pp=[mean - radius, mean + radius])


def projected_quota(quota, target_count):
    t, c, b, e = (quota[name] for name in ('target_count', 'candidate_count', 'injected_count', 'edge_budget'))
    require(all(type(value) is int for value in (t, c, b, e)) and t == target_count
            and t > 0 and 0 <= c <= 128 and b > 0 and e > 0, 'Invalid quota metadata.')
    lo, hi, ideal = max(b, e - b * c), min(b * t, e), (e + 1) // 2
    require(lo <= hi, 'The declared external quota is infeasible.')
    target = min(hi, max(lo, ideal))
    expected = dict(ideal_target_quota=ideal, ideal_candidate_quota=e - ideal,
                    lower_target_quota=lo, upper_target_quota=hi, target_quota=target,
                    candidate_quota=e - target, projection_applied=target != ideal,
                    exact_half=2 * target == e, odd_budget=bool(e % 2),
                    target_capacity_limited=ideal > hi,
                    mandatory_or_candidate_capacity_limited=ideal < lo,
                    injected_to_injected_quota=0, total_budget_preserved=True)
    require(all(quota.get(name) == value for name, value in expected.items()), 'Quota projection declarations differ.')
    close(quota['target_fraction'], target / e)
    close(quota['candidate_fraction'], (e - target) / e)
    close(quota['requested_target_fraction'], .5)
    return expected


def audit(run_root, protocol_path, input_hashes):
    run = Path(run_root).resolve(strict=True)
    protocol_path = Path(protocol_path).resolve(strict=True)

    def bound(root, name):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe artifact path.')
        path = (root / relative).resolve(strict=True)
        require(root in path.parents and path.is_file(), 'Artifact is missing or outside its frozen root.')
        return path

    def bytes_hash(path, expected=None):
        path = Path(path).resolve(strict=True)
        value = sha(path)
        require(expected is None or value == expected, 'Frozen file bytes differ: ' + str(path))
        input_hashes[str(path)] = value
        return value

    def read_path(path):
        raw = Path(path).read_bytes()
        input_hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8'))

    def rows_path(path):
        raw = Path(path).read_bytes()
        input_hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]

    p = read_path(protocol_path)
    require(p.get('schema') == 'monti_downstream_blind_v1' and p.get('status') == 'frozen'
            and p.get('owner_execution_authorized') is True and p.get('final_TEST_allowed') is False
            and p['source_manifest']['source_commit'] == SOURCE and p['runtime_commit'] == RUNTIME
            and tuple(p['seeds']) == SEEDS and tuple(p['arms']) == ARMS and p['new_arm'] == 'quota_blind',
            'Frozen action-blind protocol/source identity differs.')
    runtime = Path(p['runtime_root']).resolve(strict=True)
    provenance = dict(source_commit=SOURCE, runtime_commit=RUNTIME,
                      protocol_sha256=bytes_hash(protocol_path), parent_protocol_sha256=PARENT_PROTOCOL)
    completed = read_path(bound(run, 'result.json'))
    require(completed.get('status') == 'completed' and completed.get('provenance') == provenance,
            'Only a completed same-protocol run may be audited; partial output is rejected.')
    source_root = Path(p['source_root']).resolve(strict=True)
    source_files = p['source_manifest']['files']
    require(source_files and all(name.startswith(('methods/', 'experiments/', 'tests/'))
                                 for name in source_files), 'Study source inventory differs.')
    for name, wanted in source_files.items():
        bytes_hash(bound(source_root, name), wanted)
    summary = read_path(bound(run, 'summary.json'))
    require('all_case_counts' not in summary and summary.get('status') == 'completed'
            and all(name in completed and completed[name] == value for name, value in summary.items()),
            'Completed summary/result copies differ.')
    parent_refs = {}
    for ref in p['parent_refs']:
        require(ref['root'] in ('asset', 'runtime'), 'Unknown parent reference root.')
        path = bound(runtime, ref['path'])
        require(ref['path'] not in parent_refs, 'Duplicate parent file reference.')
        parent_refs[ref['path']] = bytes_hash(path, ref['sha256'])
    for name in ('parent_protocol', 'parent_result', 'parent_cache_manifest', 'parent_seal',
                 'parent_raw_evaluation', 'parent_run_guardian', 'parent_logit_audit', 'parent_tensor_audit'):
        ref = p[name]
        require(parent_refs.get(ref['path']) == ref['sha256'], 'Named parent reference is missing from frozen inventory.')
    require(p['parent_protocol']['sha256'] == PARENT_PROTOCOL, 'Wrong parent protocol.')
    parent_protocol = read_path(bound(runtime, p['parent_protocol']['path']))
    parent_result = read_path(bound(runtime, p['parent_result']['path']))
    old_provenance = dict(source_commit=PARENT_SOURCE, runtime_commit=RUNTIME, protocol_sha256=PARENT_PROTOCOL)
    require(parent_protocol['source_manifest']['source_commit'] == PARENT_SOURCE
            and parent_result.get('status') == 'completed' and parent_result['provenance'] == old_provenance,
            'Completed visible parent source/protocol binding differs.')
    parent_guardian = read_path(bound(runtime, p['parent_run_guardian']['path']))
    parent_logit_audit = read_path(bound(runtime, p['parent_logit_audit']['path']))
    parent_tensor_audit = read_path(bound(runtime, p['parent_tensor_audit']['path']))
    require(parent_guardian['status'] == 'completed' and parent_guardian['exit_code'] == 0
            and parent_guardian['child_result_sha256'] == p['parent_result']['sha256']
            and parent_guardian['protocol_sha256'] == PARENT_PROTOCOL
            and parent_guardian['whole_phase_GPU_hours'] == 0
            and parent_logit_audit['status'] == parent_tensor_audit['status'] == 'passed'
            and parent_logit_audit['recount']['provenance'] == parent_tensor_audit['provenance'] == old_provenance,
            'Parent guardian or independent audit receipt differs.')
    for name in ('train_groups', 'valid_groups', 'train_members', 'valid_members', 'cohort', 'selection'):
        require(p[name] == parent_protocol[name], 'Parent metadata changed: ' + name)
    for name in ('downstream_epochs', 'hidden_dimension', 'learning_rate', 'weight_decay'):
        require(p['settings'][name] == parent_protocol['settings'][name], 'A matched training setting changed.')
    require(p['settings']['action_conditioning'] == 'hidden' and p['settings']['gradient_clip_norm'] == 5.0
            and p['settings']['initialization_seed_base'] == 20261111,
            'The registered action-summary intervention or training initialization differs.')
    groups, train = p['valid_groups'], p['train_groups']
    require(len(groups) == len(set(groups)) == 64 and len(train) == len(set(train)) == 128
            and not set(groups) & set(train), 'Registered cohort differs.')
    members = {int(group): values for group, values in p['valid_members'].items()}
    require(set(members) == set(groups) and all(values and len(values) == len(set(values)) for values in members.values()),
            'Registered target membership differs.')
    denominator = sum(map(len, members.values()))
    unique = len({node for values in members.values() for node in values})
    require(denominator == p['cohort']['valid_target_occurrences'] == 571
            and unique == p['cohort']['unique_valid_targets'] == 240, 'Complete target denominator differs.')
    parent_cache = read_path(bound(runtime, p['parent_cache_manifest']['path']))
    parent_seal = read_path(bound(runtime, p['parent_seal']['path']))
    require(parent_cache['provenance'] == parent_seal['provenance'] == old_provenance
            and parent_seal['upstream_cache_sha256'] == p['parent_cache_manifest']['sha256']
            and parent_result['pre_victim_seal_sha256'] == p['parent_seal']['sha256'], 'Parent seal bindings differ.')
    old_root = Path(p['parent_run_root']).resolve(strict=True)
    require(runtime in old_root.parents, 'Parent run is outside the runtime root.')
    caches = {}
    for row in parent_cache['cases']:
        cache_key = row['seed'], row['phase'], row['group']
        require(cache_key not in caches, 'Duplicate parent cache.')
        filename = 's%d_%s_g%d.pt' % cache_key
        require(row['path'] == filename, 'Parent cache path is misbound.')
        path = bound(old_root, 'cache/' + filename)
        require(bytes_hash(path, row['sha256']) == parent_refs[path.relative_to(runtime).as_posix()],
                'Parent cache is not bound by the new registration.')
        selected = p['train_members'] if row['phase'] == 'train' else p['valid_members']
        require(row['targets'] == len(selected[str(row['group'])]), 'Parent cache target count differs.')
        projected_quota(row['quota'], row['targets'])
        caches[cache_key] = row
    expected_caches = {(seed, phase, group) for seed in SEEDS
                       for phase, cohort in (('train', train), ('validation', groups)) for group in cohort}
    require(len(caches) == 576 and set(caches) == expected_caches, 'The full 576-parent-cache coverage differs.')
    old_plans = {}
    for row in parent_seal['plans']:
        plan_key = row['seed'], row['group']
        require(plan_key not in old_plans and row['path'] == 's%d_g%d.pt' % plan_key, 'Duplicate/misbound parent plan.')
        path = bound(old_root, 'plans/' + row['path'])
        require(bytes_hash(path, row['sha256']) == parent_refs[path.relative_to(runtime).as_posix()],
                'Parent plan is not bound by the new registration.')
        old_plans[plan_key] = row
    require(set(old_plans) == {(seed, group) for seed in SEEDS for group in groups}, 'Parent plan coverage differs.')
    seal = read_path(bound(run, 'sealed_before_victim.json'))
    require(seal['provenance'] == provenance and seal['fitting_finished'] is True
            and seal['official_GAGA_loaded'] is False
            and bytes_hash(run / 'sealed_before_victim.json') == completed['pre_victim_seal_sha256'],
            'Blind final-plan seal differs.')
    require(seal['downstream'] == completed['downstream'] and len(seal['downstream']) == 3,
            'Blind terminal-model records differ.')
    visible_models = {row['seed']: row for row in parent_result['downstream']}
    blind_models = {row['seed']: row for row in completed['downstream']}
    require(set(blind_models) == set(visible_models) == set(SEEDS), 'Terminal seed model inventory differs.')
    for seed in SEEDS:
        blind, visible = blind_models[seed], visible_models[seed]
        require(blind['epochs'] == 4 and blind['parameter_count'] == visible['parameter_count'] == 151872
                and blind['initial_state_sha256'] == visible['initial_state_sha256']
                and blind['matches_visible_initial_state'] is True
                and blind['initial_state_sha256'] != blind['final_state_sha256']
                and blind['optimizer_updates'] == visible['optimizer_updates'] > 0
                and blind['nonzero_gradient_cases'] > 0, 'Matched initial-state or training declarations differ.')
        require(seal['checkpoint_hashes'][str(seed)] == blind['final_checkpoint_sha256'], 'Blind checkpoint declarations differ.')
        bytes_hash(bound(run, 'downstream/s%d_final.pt' % seed), blind['final_checkpoint_sha256'])
        parent_path = bound(old_root, 'downstream/s%d_final.pt' % seed)
        require(bytes_hash(parent_path, visible['final_checkpoint_sha256']) == parent_refs[parent_path.relative_to(runtime).as_posix()],
                'A visible parent checkpoint is not bound by registration.')
    plans = {}
    for row in seal['plans']:
        plan_key = row['seed'], row['group']
        require(plan_key not in plans and row['path'] == 's%d_g%d.pt' % plan_key, 'Duplicate/misbound blind plan.')
        cache = caches[plan_key[0], 'validation', plan_key[1]]
        require(row['quota'] == cache['quota'] and row['upstream_feature_sha256'] == cache['upstream_feature_sha256']
                and row['target_block_sha256'] == cache['upstream_target_block_sha256'], 'Blind plan changed its cached upstream declarations.')
        bytes_hash(bound(run, 'plans/' + row['path']), row['sha256'])
        plans[plan_key] = row
    require(set(plans) == set(old_plans), 'Exactly 192 blind plans are required.')

    # Recompute predictions from every original and new saved logit row.
    old_rows = rows_path(bound(runtime, p['parent_raw_evaluation']['path']))
    new_rows = rows_path(bound(run, 'raw/evaluation.jsonl'))
    require(len(old_rows) == 832 and len(new_rows) == 192, 'Expected 832 parent plus 192 new saved rows.')
    clean, attacked = {}, {}
    for is_new, rows in ((False, old_rows), (True, new_rows)):
        for row in rows:
            group = row['group']
            require(group in members and row['node_ids'] == members[group], 'Saved node order or membership differs.')
            if row.get('stage') == 'clean':
                require(not is_new and group not in clean, 'New or duplicated clean prediction query.')
                clean[group] = {name: logits_to_classes(row[name + '_logits'], len(members[group])) for name in DETECTORS}
            else:
                record_key = row['seed'], group, row['arm']
                require(record_key not in attacked and (row['arm'] == 'quota_blind') is is_new,
                        'A saved arm is duplicated or attributed to the wrong execution.')
                attacked[record_key] = row
    require(set(clean) == set(groups) and set(attacked) == {(seed, group, arm) for seed in SEEDS for group in groups for arm in ARMS},
            'The five-arm evaluation Cartesian cohort differs.')
    independently_counted = {}
    for (seed, group, arm), row in attacked.items():
        cache = caches[seed, 'validation', group]
        q = cache['quota']
        require(row['targets'] == len(members[group]) and row['quota'] == q
                and row['actual_nodes'] == q['injected_count'] and row['actual_edges'] == q['edge_budget']
                and row['feature_sha256'] == cache['upstream_feature_sha256'], 'Saved arm denominator, quota, feature or cost declaration differs.')
        topology = plans[seed, group]['topology_sha256'] if arm == 'quota_blind' else old_plans[seed, group]['topology_sha256'][arm]
        require(row['topology_sha256'] == topology, 'Saved prediction/topology declaration differs.')
        for detector in DETECTORS:
            expected = transitions(clean[group][detector], logits_to_classes(row[detector + '_logits'], row['targets']))
            require(row[detector] == expected, 'Declared transitions differ from saved logits.')
        independently_counted[seed, group, arm] = {name: value for name, value in row.items()
            if name not in ('node_ids', 'gcn_logits', 'gaga_logits')}
    declared = {(row['seed'], row['group'], row['arm']): row for row in completed['all_case_counts']}
    require(len(completed['all_case_counts']) == len(declared) == 960 and declared == independently_counted,
            'The 960 declared case-count rows differ from independent prediction recount.')
    old_declared = {(row['seed'], row['group'], row['arm']): row for row in parent_result['all_case_counts']}
    require(len(old_declared) == 768 and all(declared[key] == value for key, value in old_declared.items()),
            'A reused comparator case changed.')

    # Audit the registered order and the update/no-choice count from TRAIN logs.
    training = rows_path(bound(run, 'raw/training.jsonl'))
    require(len(training) == 1536, 'Four complete 128-group epochs for three seeds are required.')
    offset = 0
    updates = nonzero = skipped = 0
    for seed in SEEDS:
        seed_updates = seed_nonzero = 0
        for epoch in range(1, 5):
            order = list(train)
            random.Random(seed * 1000 + epoch).shuffle(order)
            losses = []
            epoch_skipped = 0
            for group in order:
                row = training[offset]
                offset += 1
                require((row['seed'], row['epoch'], row['group']) == (seed, epoch, group), 'Matched training order changed.')
                q = caches[seed, 'train', group]['quota']
                no_choice = q['candidate_quota'] in (0, q['injected_count'] * q['candidate_count'])
                require(type(row['optimizer_update']) is bool and row['optimizer_update'] is not no_choice,
                        'TRAIN update/no-choice status differs from the fixed quota.')
                if no_choice:
                    skipped += 1
                    epoch_skipped += 1
                else:
                    require(math.isfinite(row['loss']) and row['loss'] >= 0
                            and math.isfinite(row['gradient_l2']) and row['gradient_l2'] >= 0,
                            'A declared training value is nonfinite.')
                    updates += 1
                    seed_updates += 1
                    nonzero += row['gradient_l2'] > 0
                    seed_nonzero += row['gradient_l2'] > 0
                    losses.append(row['loss'])
            epoch_row = read_path(bound(run, 'downstream/s%d_epoch%d.json' % (seed, epoch)))
            require(epoch_row['seed'] == seed and epoch_row['epoch'] == epoch and epoch_row['groups'] == 128
                    and epoch_row['no_choice_groups'] == epoch_skipped
                    and epoch_row['cumulative_optimizer_updates'] == seed_updates
                    and epoch_row['validation_performed'] is False, 'An epoch receipt differs from TRAIN logs.')
            close(epoch_row['mean_loss'], statistics.mean(losses))
        require(blind_models[seed]['optimizer_updates'] == seed_updates
                and blind_models[seed]['nonzero_gradient_cases'] == seed_nonzero,
                'Terminal training counters differ from saved TRAIN logs.')
    scope = read_path(bound(run, 'conditioning_scope.json'))
    require(scope['provenance'] == provenance and bytes_hash(run / 'conditioning_scope.json') == completed['conditioning_scope_sha256'],
            'Conditioning-scope byte binding differs.')
    algebra = {(row['seed'], row['phase'], row['group']): row for row in scope['cases']}
    require(len(scope['cases']) == len(algebra) == 576 and set(algebra) == expected_caches,
            'Conditioning-scope cohort differs.')
    algebra_summary = []
    for seed in SEEDS:
        for phase, cohort in (('train', train), ('validation', groups)):
            full = same = 0
            for group in cohort:
                row = algebra[seed, phase, group]
                q = caches[seed, phase, group]['quota']
                expected_full = q['target_quota'] == q['injected_count'] * q['target_count']
                require(row['all_targets_attached'] is expected_full
                        and type(row['exact_same_mean']) is bool and type(row['exact_same_log_count']) is bool
                        and row['exact_same_summary'] is (row['exact_same_mean'] and row['exact_same_log_count']),
                        'Conditioning equality flags are inconsistent.')
                for equality, difference in (('exact_same_mean', 'maximum_mean_difference'),
                                             ('exact_same_log_count', 'maximum_log_count_difference')):
                    require(math.isfinite(row[difference]) and row[difference] >= 0
                            and row[equality] is (row[difference] == 0), 'Conditioning discrepancy flags differ.')
                require(not expected_full or row['exact_same_summary'], 'A full target block must match the blind summary exactly.')
                full += expected_full
                same += row['exact_same_summary']
            require(full == (84 if phase == 'train' else 50), 'Known full-target-block metadata count differs.')
            algebra_summary.append(dict(seed=seed, phase=phase, cases=len(cohort),
                                        full_target_block_cases=full, declared_exact_same_summary_cases=same))
    metrics = []
    for seed in SEEDS:
        for arm in ARMS:
            selected = [declared[seed, group, arm] for group in groups]
            require(sum(row['targets'] for row in selected) == denominator, 'An evaluation denominator was filtered.')
            metric = dict(seed=seed, arm=arm, groups=64, occurrences=denominator)
            for detector in DETECTORS:
                for name in TRANSITION_FIELDS[1:]:
                    metric[detector + '_' + name] = sum(row[detector][name] for row in selected)
                metric[detector + '_mr_percent'] = 100 * metric[detector + '_attacked_wrong'] / denominator
            metrics.append(metric)
    require(len(completed['seed_metrics']) == 15, 'Fifteen seed/arm rows are required.')
    for expected, row in zip(metrics, completed['seed_metrics']):
        require(set(row) == set(expected), 'Metric fields differ.')
        for name, value in expected.items():
            close(row[name], value) if name.endswith('_mr_percent') else require(row[name] == value, 'Metric counts/order differ.')
    means = {detector: {arm: statistics.mean(row[detector + '_mr_percent'] for row in metrics if row['arm'] == arm)
                        for arm in ARMS} for detector in DETECTORS}
    for detector in DETECTORS:
        for arm in ARMS:
            close(means[detector][arm], completed['mean_mr_percent'][detector][arm])
    comparisons = {}
    for name, left, right in (('visible_minus_blind', 'quota_learned', 'quota_blind'),
                              ('visible_minus_random', 'quota_learned', 'quota_random'),
                              ('blind_minus_random', 'quota_blind', 'quota_random')):
        diffs = [next(row['gaga_mr_percent'] for row in metrics if row['seed'] == seed and row['arm'] == left)
                 - next(row['gaga_mr_percent'] for row in metrics if row['seed'] == seed and row['arm'] == right)
                 for seed in SEEDS]
        comparison = paired(diffs)
        saved = completed['contrasts'][name]
        require(len(saved['seed_differences_pp']) == 3 and len(saved['nominal_paired_t95_pp']) == 2, 'Contrast shapes differ.')
        for field in ('seed_differences_pp', 'nominal_paired_t95_pp'):
            for expected, value in zip(comparison[field], saved[field]):
                close(expected, value)
        close(comparison['mean_pp'], saved['mean_pp'])
        comparisons[name] = comparison
    primary = comparisons['visible_minus_blind']
    criterion = all(value > 0 for value in primary['seed_differences_pp']) and primary['nominal_paired_t95_pp'][0] > 0
    require(completed['direct_action_summary_criterion_met'] is criterion and completed['full_TEST_evaluated'] is False
            and completed['prior_qualification_unchanged'] is True and completed['frozen_models_unchanged'] is True
            and completed['reused_comparator_cases'] == 768 and completed['reused_clean_groups'] == 64
            and completed['new_blind_cases'] == 192, 'Decision, freeze or reuse declarations differ.')
    c = completed['counters']
    require(c['upstream_generations'] == c['upstream_optimizer_updates'] == c['API_calls'] == c['TEST_queries'] == completed['GPU_hours'] == 0
            and c['training_GCN_forwards'] == c['autograd_calls'] == c['downstream_optimizer_updates'] == updates
            and c['evaluation_GCN_forwards'] == c['official_GAGA_forwards'] == 192
            and c['degenerate_training_cases'] == skipped and updates + skipped == 1536
            and c['summary_algebra_cases'] == 576, 'Logical execution counters differ.')
    return dict(provenance=provenance, parent_provenance=old_provenance,
                parent_saved_logit_rows=832, new_saved_logit_rows=192, clean_groups=64,
                reused_comparator_cases=768, new_blind_cases=192, all_attacked_cases=960,
                occurrence_denominator_per_arm_seed=denominator, unique_validation_targets=unique,
                seed_metrics=metrics, mean_mr_percent=means, contrasts=comparisons,
                direct_action_summary_criterion_met=criterion, producer_logical_counters=c,
                conditioning_declaration_summary=algebra_summary,
                study_source_files_hashed=len(source_files),
                parent_cache_files_hashed=576, parent_plan_files_hashed=192,
                parent_visible_checkpoint_files_hashed=3, blind_plan_files_hashed=192,
                blind_checkpoint_files_hashed=3,
                declared_initial_state_matches_visible=True,
                initial_state_scope='Declaration equality with the completed visible parent; parameter states are not deserialized or independently rehashed.',
                geometry_scope='File-byte bindings and declared content hashes/quotas only. Actual tensor values, physical graph construction and conditional-input algebra require the separate geometry audit.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    started, cpu_started = time.perf_counter(), time.process_time()
    output = Path(args.output)
    require(not output.exists(), 'Audit output already exists; preserve every prior attempt.')
    inputs = {}
    report = dict(schema='monti_downstream_blind_saved_logit_audit_v1', status='failed',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  auditor_source_sha256=sha(__file__), post_execution_source=True,
                  scope='Independent saved-logit argmax, transition, TRAIN-order and five-arm statistical recount with declared artifact byte bindings.',
                  limitations=[
                      'Original fraud label1 is inherited from the registered task; dataset labels are not reopened.',
                      'Tensor/checkpoint file bytes are hashed but not deserialized; declared content and initial-state hashes are not independently recomputed.',
                      'Declared conditioning equality flags are checked for consistency, not recomputed from raw features in this standard-library audit.',
                      'Seal byte bindings do not establish chronology or absence of unrecorded model calls; this is not human verification.',
                      'The cohort is adaptively inspected and overlapping target occurrences are correlated; the nominal seed interval is not independent population confirmation.',
                  ])
    failure = None
    try:
        report['recount'] = audit(args.run_root, args.protocol, inputs)
        report['status'] = 'passed'
    except BaseException as error:
        failure = error
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        report['traceback'] = traceback.format_exc()
    finally:
        report.update(input_file_hashes=inputs, wall_seconds=time.perf_counter() - started,
                      process_cpu_seconds=time.process_time() - cpu_started,
                      model_queries=0, model_instances=0, tensor_deserializations=0,
                      checkpoint_deserializations=0, optimizer_updates=0,
                      API_calls=0, TEST_queries=0, GPU_hours=0)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps({name: report[name] for name in ('status', 'wall_seconds', 'model_queries', 'GPU_hours')}, sort_keys=True))
    if failure is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
