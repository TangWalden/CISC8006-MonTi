"""Independently recount the action-blind control from saved own tensors.

Only the fixed parent's upstream captures and this control's final adjacency
plans are deserialized. No original dataset, classifier, generator or trained parameter
checkpoint is loaded. The audit checks stored bindings and geometry, not an
executed DGL graph, model predictions, gradients or seal chronology.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import time
import traceback


ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned')
SEEDS = (3, 4, 5)
ORIGINAL_NODES = 16488
SCHEMA = 'monti_downstream_blind_saved_tensor_geometry_audit_v1'
BLIND_SOURCE = 'b7809f420fc1b5453e748a9fd19c52234541cfd8'
PARENT_PROTOCOL_SHA = '5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tensor_hash(value):
    """Independent implementation of the registered dtype/shape/bytes hash."""
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode('ascii')
    return hashlib.sha256(header + b'\0' + value.numpy().tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def bound_file(directory, name, wanted):
    relative = pathlib.PurePosixPath(name)
    require(len(relative.parts) == 1 and not relative.is_absolute()
            and relative.name not in ('', '.', '..'), 'Unsafe manifest file name.')
    candidate = directory / relative.name
    require(candidate.is_file() and not candidate.is_symlink(), 'Manifest tensor file missing or linked.')
    require(file_hash(candidate) == wanted, 'Manifest tensor file byte hash changed: ' + name)
    return candidate


def tensor_tree(value, torch, counters):
    if isinstance(value, torch.Tensor):
        counters['saved_tensor_objects_checked'] += 1
        require(value.device.type == 'cpu', 'A saved tensor is not on CPU.')
        require(not value.requires_grad, 'A supposedly frozen saved tensor requires gradients.')
        require(bool(torch.isfinite(value).all()), 'A saved tensor contains nonfinite values.')
    elif isinstance(value, dict):
        for child in value.values():
            tensor_tree(child, torch, counters)
    elif isinstance(value, (list, tuple)):
        for child in value:
            tensor_tree(child, torch, counters)


def expected_quota(targets, candidates, injected, edges):
    """Derive the integer projection without importing the producer's rule."""
    require(all(type(value) is int for value in (targets, candidates, injected, edges)),
            'Quota dimensions must be Python integers.')
    require(targets >= 1 and candidates >= 0 and injected >= 1 and edges >= 1,
            'Invalid quota dimensions.')
    lower = max(injected, edges - injected * candidates)
    upper = min(injected * targets, edges)
    require(lower <= upper, 'The declared full-budget external allocation is infeasible.')
    ideal = (edges + 1) // 2
    target = min(upper, max(lower, ideal))
    return dict(target_count=targets, candidate_count=candidates, injected_count=injected,
                edge_budget=edges, ideal_target_quota=ideal,
                ideal_candidate_quota=edges - ideal,
                lower_target_quota=lower, upper_target_quota=upper,
                target_quota=target, candidate_quota=edges - target,
                target_fraction=target / edges, candidate_fraction=(edges - target) / edges,
                requested_target_fraction=0.5, projection_applied=target != ideal,
                exact_half=2 * target == edges, odd_budget=bool(edges % 2),
                target_capacity_limited=ideal > upper,
                mandatory_or_candidate_capacity_limited=ideal < lower,
                injected_to_injected_quota=0, total_budget_preserved=True)


def require_hard(value, shape, torch, message):
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape),
            message + ' shape differs.')
    require(bool(torch.isfinite(value).all()) and bool(((value == 0) | (value == 1)).all()),
            message + ' is not finite exact binary.')


def role_counts(matrix, targets, candidates, injected, edges, torch, quota=None):
    require_hard(matrix, (injected, targets + candidates + injected), torch, 'Owned adjacency')
    target_block = matrix[:, :targets]
    candidate_block = matrix[:, targets:targets + candidates]
    injected_block = matrix[:, targets + candidates:]
    require(bool((target_block.sum(1) >= 1).all()), 'An auxiliary row has no target connection.')
    require(not bool(torch.triu(injected_block, diagonal=0).any()),
            'Injected block violates strict lower-triangular ownership.')
    target_edges = int(target_block.sum())
    candidate_edges = int(candidate_block.sum())
    injected_edges = int(injected_block.sum())
    require(target_edges + candidate_edges + injected_edges == edges,
            'Owned adjacency does not preserve the complete edge budget.')
    degrees = matrix.long().sum(1) + injected_block.long().sum(0)
    # Original endpoints are disjoint from physical injection IDs. Binary cells
    # and lower-triangular ownership establish unique nonself physical pairs.
    require(int(degrees.sum()) == target_edges + candidate_edges + 2 * injected_edges,
            'Physical injected-degree incidence does not reconcile.')
    if quota is not None:
        require(injected_edges == 0, 'A quota arm introduced an injected-to-injected edge.')
        require(target_edges == quota['target_quota'] and candidate_edges == quota['candidate_quota'],
                'A quota arm does not meet the independently projected role quotas.')
    return dict(target_edges=target_edges, candidate_edges=candidate_edges,
                injected_to_injected_owned_edges=injected_edges,
                total_owned_undirected_edges=edges, target_fraction=target_edges / edges,
                candidate_fraction=candidate_edges / edges,
                injected_to_injected_fraction=injected_edges / edges,
                injected_physical_degree_min=int(degrees.min()),
                injected_physical_degree_max=int(degrees.max()),
                injected_physical_degree_sum=int(degrees.sum()),
                injected_physical_degrees_sha256=tensor_hash(degrees),
                physical_degree_scope='Computed from saved owned adjacency; excludes self-loops, not an executed-graph inspection.')


def assert_ids(value, expected, torch, label):
    require(isinstance(value, torch.Tensor) and value.ndim == 1
            and value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
            label + ' must be a one-dimensional integer tensor.')
    actual = value.tolist()
    require(actual == expected, label + ' differs from the registered ordering.')
    require(len(actual) == len(set(actual)), label + ' contains duplicate node IDs.')
    return actual


def check_cache(case, row, protocol, torch, counters):
    seed, phase, group = row['seed'], row['phase'], row['group']
    require(case['seed'] == seed and case['group'] == group, 'Cache seed/group identity differs.')
    members = protocol['train_members'] if phase == 'train' else protocol['valid_members']
    wanted_ids = members[str(group)]
    ids = assert_ids(case['ids'], wanted_ids, torch, 'Target IDs')
    candidates = case['candidate_ids']
    require(isinstance(candidates, torch.Tensor) and candidates.ndim == 1
            and candidates.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8),
            'Candidate IDs must be a one-dimensional integer tensor.')
    cids = candidates.tolist()
    require(len(cids) == len(set(cids)) and not set(ids) & set(cids),
            'Target/candidate identities overlap or contain duplicates.')
    require(all(type(value) is int and 0 <= value < ORIGINAL_NODES for value in ids + cids),
            'An original endpoint is outside the original node range.')
    source_ids = case['source_ids']
    require(isinstance(source_ids, list) and source_ids
            and source_ids == list(range(ORIGINAL_NODES, ORIGINAL_NODES + len(source_ids))),
            'Physical injected IDs are not the registered contiguous suffix.')
    require(all(type(value) is int for value in source_ids), 'Injected IDs must be Python integers.')
    b, t, c = len(source_ids), len(ids), len(cids)
    q = case['quota']
    expected = expected_quota(t, c, b, q['edge_budget'])
    require(q == expected and row['quota'] == expected, 'Quota metadata differs from the independent projection.')
    require(row['targets'] == t, 'Manifest target count differs.')
    require(tuple(case['x_aux'].shape) == (b, 768)
            and tuple(case['x_targets'].shape) == (t, 768)
            and tuple(case['x_candidates'].shape) == (c, 768)
            and tuple(case['h_candidates'].shape) == (c, 64), 'Saved feature/context dimensions differ.')
    require(all(case[name].dtype == torch.float32
                for name in ('x_aux', 'x_targets', 'x_candidates', 'h_candidates')),
            'A saved raw feature/context dtype differs from the registered float32 runtime.')
    tensor_tree(case, torch, counters)
    features_hash = tensor_hash(case['x_aux'])
    targets_hash = tensor_hash(case['target_adjacency'])
    require(features_hash == case['upstream_feature_sha256'] == row['upstream_feature_sha256'],
            'Cached upstream feature content hash differs.')
    require(targets_hash == case['upstream_target_block_sha256'] == row['upstream_target_block_sha256'],
            'Cached upstream target-block content hash differs.')
    require_hard(case['target_adjacency'], (b, t), torch, 'Cached upstream target block')
    require(int(case['target_adjacency'].sum()) == q['target_quota']
            and bool((case['target_adjacency'].sum(1) >= 1).all()),
            'Cached fixed target block violates its projected quota or mandatory attachment.')
    for name in ('original_candidates', 'random_candidates'):
        require_hard(case[name], (b, c), torch, name)
        require(int(case[name].sum()) == q['candidate_quota'], 'Cached candidate control quota differs.')
    native_counts = role_counts(case['native_adjacency'], t, c, b, q['edge_budget'], torch)
    return dict(seed=seed, phase=phase, group=group, targets=t, candidates=c, injected_nodes=b,
                quota=expected, upstream_feature_sha256=features_hash,
                upstream_target_block_sha256=targets_hash, native_roles=native_counts)


def relative_file(root, name):
    relative = pathlib.PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Frozen reference is not a safe relative path.')
    path = (root / pathlib.Path(*relative.parts)).resolve()
    require(root in path.parents and path.is_file() and not path.is_symlink(),
            'Frozen reference is missing, linked or outside its root.')
    return path


def exact_reference(runtime, reference):
    path = relative_file(runtime, reference['path'])
    require(file_hash(path) == reference['sha256'], 'A frozen parent byte reference changed: ' + reference['path'])
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    out = pathlib.Path(args.output).resolve()
    require(not out.exists(), 'Audit output exists; no overwrite is permitted.')
    run = pathlib.Path(args.run_root).resolve()
    result = dict(schema=SCHEMA, status='failed', active_file=None,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  counters=dict(parent_declared_files_hashed=0, parent_cache_files_deserialized=0,
                                blind_plan_files_deserialized=0, saved_tensor_objects_checked=0,
                                blind_checkpoint_files_byte_hashed=0, conditioning_algebra_cases=0,
                                classifier_or_generator_instances=0, model_forwards=0,
                                optimizer_updates=0, original_dataset_loads=0,
                                parameter_checkpoint_deserializations=0, TEST_queries=0, API_calls=0),
                  GPU_hours=0, conditioning_cases=[], validation_geometry=[])
    error = None
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'An explicitly CPU-only audit is required.')
        p = read_json(args.protocol)
        require(p.get('schema') == 'monti_downstream_blind_v1' and p.get('status') == 'frozen'
                and p.get('owner_execution_authorized') is True and p.get('final_TEST_allowed') is False
                and tuple(p['seeds']) == SEEDS and p['new_arm'] == 'quota_blind'
                and p['settings']['action_conditioning'] == 'hidden', 'Wrong frozen action-blind protocol.')
        require(len(p['train_groups']) == len(set(p['train_groups'])) == 128
                and len(p['valid_groups']) == len(set(p['valid_groups'])) == 64
                and not set(p['train_groups']) & set(p['valid_groups']), 'Wrong registered cohort.')
        require(p['source_manifest']['source_commit'] == BLIND_SOURCE
                and p['parent_protocol']['sha256'] == PARENT_PROTOCOL_SHA,
                'The specific new scientific source or fixed parent protocol differs.')
        provenance = dict(source_commit=p['source_manifest']['source_commit'], runtime_commit=p['runtime_commit'],
                          protocol_sha256=file_hash(args.protocol), parent_protocol_sha256=PARENT_PROTOCOL_SHA)
        completed_path = run / 'result.json'
        completed = read_json(completed_path)
        require(completed.get('status') == 'completed' and completed.get('provenance') == provenance,
                'Only a completed same-protocol run may be audited; no partial outputs.')
        runtime = pathlib.Path(p['runtime_root']).resolve()
        require(pathlib.Path(p['asset_root']).resolve() == runtime, 'Runtime/asset binding differs.')
        seen_refs = {}
        for reference in p['parent_refs']:
            previous = seen_refs.setdefault(reference['path'], reference['sha256'])
            require(previous == reference['sha256'], 'Conflicting frozen parent byte identities.')
            exact_reference(runtime, reference)
            result['counters']['parent_declared_files_hashed'] += 1
        named = {name: exact_reference(runtime, p[name]) for name in
                 ('parent_protocol', 'parent_result', 'parent_cache_manifest', 'parent_seal')}
        previous_protocol = read_json(named['parent_protocol'])
        previous_completed = read_json(named['parent_result'])
        parent_provenance = p['parent_provenance']
        require(previous_completed['status'] == 'completed'
                and previous_completed['provenance'] == parent_provenance,
                'The completed parent binding differs.')
        for name in ('train_groups', 'valid_groups', 'train_members', 'valid_members'):
            require(p[name] == previous_protocol[name], 'An inherited cohort or membership changed: ' + name)
        parent_run = pathlib.Path(p['parent_run_root']).resolve()
        require(parent_run == named['parent_result'].parent and runtime in parent_run.parents,
                'Parent run-root binding differs.')
        cache_manifest = read_json(named['parent_cache_manifest'])
        parent_seal = read_json(named['parent_seal'])
        require(cache_manifest['provenance'] == parent_seal['provenance'] == parent_provenance,
                'Parent cache or seal provenance differs.')
        require(parent_seal['upstream_cache_sha256'] == file_hash(named['parent_cache_manifest'])
                and previous_completed['pre_victim_seal_sha256'] == file_hash(named['parent_seal']),
                'Parent manifest/seal byte binding differs.')
        seal_path, scope_path = run / 'sealed_before_victim.json', run / 'conditioning_scope.json'
        seal, declared_scope = read_json(seal_path), read_json(scope_path)
        require(seal['provenance'] == declared_scope['provenance'] == provenance,
                'New seal or conditioning-scope provenance differs.')
        require(file_hash(seal_path) == completed['pre_victim_seal_sha256']
                and file_hash(scope_path) == completed['conditioning_scope_sha256'],
                'New result-to-seal or conditioning file binding differs.')
        rows, plans = cache_manifest['cases'], seal['plans']
        expected_cache = {(seed, phase, group) for seed in SEEDS
                          for phase, groups in (('train', p['train_groups']), ('validation', p['valid_groups']))
                          for group in groups}
        cache_keys = [(row['seed'], row['phase'], row['group']) for row in rows]
        require(len(cache_keys) == len(set(cache_keys)) == 576 and set(cache_keys) == expected_cache,
                'The exact 576 parent cache inventory differs.')
        expected_plans = {(seed, group) for seed in SEEDS for group in p['valid_groups']}
        plan_keys = [(row['seed'], row['group']) for row in plans]
        require(len(plan_keys) == len(set(plan_keys)) == 192 and set(plan_keys) == expected_plans,
                'The exact 192 blind-plan inventory differs.')
        require(set(path.name for path in (run / 'plans').glob('*.pt')) == set(row['path'] for row in plans),
                'Actual new plan file inventory differs from its seal.')
        require(set(path.name for path in (parent_run / 'cache').glob('*.pt')) == set(row['path'] for row in rows),
                'Actual parent cache inventory differs from its manifest.')
        parent_plans = {(row['seed'], row['group']): row for row in parent_seal['plans']}
        require(len(parent_plans) == 192 and set(parent_plans) == expected_plans,
                'Parent sealed-plan keys differ.')
        declarations = declared_scope['cases']
        declared_keys = [(row['seed'], row['phase'], row['group']) for row in declarations]
        require(len(declared_keys) == len(set(declared_keys)) == 576 and set(declared_keys) == expected_cache,
                'Conditioning algebra declaration coverage differs.')
        declared_by_key = {key: row for key, row in zip(declared_keys, declarations)}
        checkpoints = seal['checkpoint_hashes']
        new_records = {row['seed']: row for row in completed['downstream']}
        parent_records = {row['seed']: row for row in previous_completed['downstream']}
        require(set(checkpoints) == {str(seed) for seed in SEEDS}
                and set(new_records) == set(parent_records) == set(SEEDS),
                'Terminal checkpoint or training-receipt inventory differs.')
        for seed in SEEDS:
            checkpoint = bound_file(run / 'downstream', 's%d_final.pt' % seed, checkpoints[str(seed)])
            result['counters']['blind_checkpoint_files_byte_hashed'] += 1
            record = new_records[seed]
            require(record['final_checkpoint_sha256'] == file_hash(checkpoint)
                    and record['epochs'] == 4 and record['parameter_count'] == 151872
                    and record['initial_state_sha256'] == parent_records[seed]['initial_state_sha256']
                    and record['optimizer_updates'] == parent_records[seed]['optimizer_updates'],
                    'Matched training receipt or terminal checkpoint byte binding differs.')
        import torch
        require(not torch.cuda.is_available() and torch.cuda.device_count() == 0
                and torch._C._cuda_getDeviceCount() == 0, 'A GPU is visible in this CPU-only audit.')
        torch.set_num_threads(4)
        result['runtime'] = dict(torch_version=torch.__version__, cpu_threads=torch.get_num_threads())
        result['provenance'] = provenance
        result['parent_provenance'] = parent_provenance
        result['inputs'] = dict(protocol_sha256=file_hash(args.protocol), completed_result_sha256=file_hash(completed_path),
                                parent_cache_manifest_sha256=file_hash(named['parent_cache_manifest']),
                                parent_seal_sha256=file_hash(named['parent_seal']),
                                new_seal_sha256=file_hash(seal_path), conditioning_scope_sha256=file_hash(scope_path),
                                auditor_source_sha256=file_hash(__file__))
        metadata = {}
        validation_targets = {}
        census = {seed: {phase: dict(cases=0, full_target_block=0, exact_same_mean=0,
                                    exact_same_log_count=0, exact_same_summary=0)
                         for phase in ('train', 'validation')} for seed in SEEDS}
        counters = result['counters']
        for row in rows:
            path = bound_file(parent_run / 'cache', row['path'], row['sha256'])
            result['active_file'] = 'parent/cache/' + path.name
            case = torch.load(path, map_location='cpu')
            counters['parent_cache_files_deserialized'] += 1
            meta = check_cache(case, row, p, torch, counters)
            key = (row['seed'], row['phase'], row['group'])
            metadata[key] = meta
            adj, targets = case['target_adjacency'], case['x_targets']
            b, t = adj.shape
            actual_count = adj.sum(1, keepdim=True)
            actual_mean = adj.matmul(targets) / actual_count
            neutral = torch.ones_like(adj)
            neutral_mean = neutral.matmul(targets) / neutral.sum(1, keepdim=True)
            neutral_count = torch.full_like(actual_count, meta['quota']['target_quota'] / b)
            actual_log = torch.log1p(actual_count)
            neutral_log = torch.log1p(neutral_count)
            full = meta['quota']['target_quota'] == b * t
            require(full == bool((adj == 1).all()), 'Full-target metadata differs from actual adjacency.')
            same_mean = torch.equal(actual_mean, neutral_mean)
            same_log = torch.equal(actual_log, neutral_log)
            algebra = dict(seed=row['seed'], phase=row['phase'], group=row['group'],
                           all_targets_attached=full, exact_same_mean=same_mean,
                           exact_same_log_count=same_log, exact_same_summary=same_mean and same_log,
                           maximum_mean_difference=float((actual_mean - neutral_mean).abs().max()),
                           maximum_log_count_difference=float((actual_log - neutral_log).abs().max()))
            require(algebra == declared_by_key[key], 'Saved-input algebra declaration differs from independent recount.')
            require(not full or algebra['exact_same_summary'],
                    'A full target block unexpectedly changes direct summary information.')
            values = census[row['seed']][row['phase']]
            values['cases'] += 1
            values['full_target_block'] += int(full)
            for flag in ('exact_same_mean', 'exact_same_log_count', 'exact_same_summary'):
                values[flag] += int(algebra[flag])
            result['conditioning_cases'].append(algebra)
            counters['conditioning_algebra_cases'] += 1
            if row['phase'] == 'validation':
                validation_targets[(row['seed'], row['group'])] = case['target_adjacency']
            del case
        for seed in SEEDS:
            require(census[seed]['train']['cases'] == 128 and census[seed]['validation']['cases'] == 64
                    and census[seed]['train']['full_target_block'] == 84
                    and census[seed]['validation']['full_target_block'] == 50,
                    'The registered full-target information limit differs from saved inputs.')
        result['conditioning_census'] = census
        for row in plans:
            path = bound_file(run / 'plans', row['path'], row['sha256'])
            result['active_file'] = 'blind/plans/' + path.name
            adj = torch.load(path, map_location='cpu')
            counters['blind_plan_files_deserialized'] += 1
            tensor_tree(adj, torch, counters)
            key = (row['seed'], row['group'])
            meta = metadata[(row['seed'], 'validation', row['group'])]
            require(row['quota'] == meta['quota']
                    and row['upstream_feature_sha256'] == meta['upstream_feature_sha256']
                    and row['target_block_sha256'] == meta['upstream_target_block_sha256'],
                    'Blind plan does not bind the same fixed upstream input.')
            digest = tensor_hash(adj)
            require(digest == row['topology_sha256'], 'Blind topology content hash differs from its seal.')
            t, c, b = meta['targets'], meta['candidates'], meta['injected_nodes']
            roles = role_counts(adj, t, c, b, meta['quota']['edge_budget'], torch, meta['quota'])
            require(torch.equal(adj[:, :t], validation_targets[key]), 'A blind plan changed upstream target actions.')
            old = parent_plans[key]
            require(old['quota'] == meta['quota']
                    and old['frozen_upstream_feature_sha256'] == meta['upstream_feature_sha256']
                    and old['frozen_upstream_target_block_sha256'] == meta['upstream_target_block_sha256'],
                    'Blind/parent comparator upstream bindings differ.')
            case_key = hashlib.sha256(json.dumps([row['seed'], row['group'],
                                                  'downstream_blind_tensor_audit_v1']).encode('ascii')).hexdigest()[:20]
            result['validation_geometry'].append(dict(
                opaque_case_key=case_key, seed=row['seed'], targets=t, candidates=c, injected_nodes=b,
                planned_quota=meta['quota'], actual_roles=roles,
                upstream_feature_sha256=meta['upstream_feature_sha256'],
                upstream_target_block_sha256=meta['upstream_target_block_sha256'],
                exactly_matches_visible_topology=digest == old['topology_sha256']['quota_learned'],
                exactly_matches_random_topology=digest == old['topology_sha256']['quota_random']))
            del adj
        require(counters['parent_cache_files_deserialized'] == 576
                and counters['blind_plan_files_deserialized'] == 192
                and counters['conditioning_algebra_cases'] == 576, 'Incomplete saved-tensor/algebra coverage.')
        result['topology_identity_counts'] = dict(
            validation_cases=192,
            exact_blind_visible_matches=sum(row['exactly_matches_visible_topology'] for row in result['validation_geometry']),
            exact_blind_random_matches=sum(row['exactly_matches_random_topology'] for row in result['validation_geometry']),
            interpretation='Content-hash equality compares stored adjacency plans. Equal direct summaries alone do not imply equal trained policies or outputs.')
        result.update(status='passed', active_file=None,
                      scope='Independent saved-cache and blind-plan geometry, byte/content bindings, quota formula and direct-summary algebra only.',
                      feature_scope='New plans contain adjacency only; feature equality is checked through bindings to parent cached tensors, not in executed DGL graphs.',
                      checkpoint_scope='Terminal checkpoint bytes are hashed but no parameter checkpoint is deserialized or model instantiated; receipt fields do not independently replay fitting.',
                      budget_scope='Total budgets bind saved parent metadata; no original dataset is read to recompute budgets. Per-auxiliary-node degrees may change.',
                      chronology_scope='Manifest byte bindings are checked; timestamps and absence of unrecorded earlier calls are not established.',
                      information_scope='The blind adapter hides direct realized-attachment mean/count only. Frozen auxiliary features can retain correlated information; equal summaries do not force equal separately trained outputs.',
                      independence='Separately authored saved-artifact audit, not human verification or new classifier evaluation.')
    except BaseException as caught:
        error = caught
        result['error'] = repr(caught)
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['process_cpu_seconds'] = time.process_time() - cpu_started
        with out.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps({name: result[name] for name in ('status', 'counters', 'GPU_hours', 'wall_seconds')},
                         allow_nan=False), flush=True)
    if error is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
