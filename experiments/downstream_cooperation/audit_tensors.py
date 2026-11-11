"""Independently recount sealed downstream geometry from saved own tensors.

Only this study's cached upstream captures and final adjacency plans are
deserialized. No original dataset, classifier, generator or trained parameter
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
SCHEMA = 'monti_downstream_saved_tensor_geometry_audit_v1'


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    run = pathlib.Path(args.run_root).resolve()
    destination = pathlib.Path(args.output).resolve()
    require(not destination.exists(), 'An audit output already exists; do not overwrite it.')
    result = dict(schema=SCHEMA, status='failed', active_file=None,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  counters=dict(cache_tensor_files_deserialized=0, final_plan_files_deserialized=0,
                                saved_tensor_objects_checked=0, classifier_or_generator_instances=0,
                                model_forwards=0, optimizer_updates=0, original_dataset_loads=0,
                                model_parameter_checkpoint_loads=0, TEST_queries=0, API_calls=0),
                  GPU_hours=0, validation_role_counts=[], cache_metadata_rows=[])
    error = None
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Audit requires explicitly empty CUDA visibility.')
        p = read_json(args.protocol)
        require(p.get('schema') == 'monti_downstream_cooperation_v1'
                and p.get('status') == 'frozen' and p.get('owner_execution_authorized') is True
                and p.get('final_TEST_allowed') is False and tuple(p['seeds']) == SEEDS
                and tuple(p['arms']) == ARMS, 'Wrong frozen study protocol.')
        require(len(p['train_groups']) == len(set(p['train_groups'])) == 128
                and len(p['valid_groups']) == len(set(p['valid_groups'])) == 64
                and not set(p['train_groups']) & set(p['valid_groups']), 'Registered cohort differs.')
        expected_provenance = dict(source_commit=p['source_manifest']['source_commit'],
                                   runtime_commit=p['runtime_commit'], protocol_sha256=file_hash(args.protocol))
        completed = read_json(run / 'result.json')
        require(completed.get('status') == 'completed' and completed.get('provenance') == expected_provenance,
                'Only a completed same-protocol run may be audited; no partial-output audit.')
        cache_manifest_path = run / 'upstream_cache_manifest.json'
        seal_path = run / 'sealed_before_victim.json'
        cache_manifest = read_json(cache_manifest_path)
        seal = read_json(seal_path)
        require(cache_manifest['provenance'] == seal['provenance'] == expected_provenance,
                'Cache and final-plan manifest provenance differs.')
        require(file_hash(cache_manifest_path) == seal['upstream_cache_sha256'],
                'Upstream cache manifest differs from the final-plan seal binding.')
        require(file_hash(seal_path) == completed['pre_victim_seal_sha256'],
                'Final-plan seal differs from the completed-run binding.')
        rows = cache_manifest['cases']
        plans = seal['plans']
        expected_cache_keys = {(seed, phase, group) for seed in SEEDS
                               for phase, groups in (('train', p['train_groups']),
                                                     ('validation', p['valid_groups'])) for group in groups}
        actual_cache_keys = [(row['seed'], row['phase'], row['group']) for row in rows]
        require(len(rows) == 576 and len(set(actual_cache_keys)) == 576
                and set(actual_cache_keys) == expected_cache_keys, 'Upstream cache coverage differs.')
        expected_plan_keys = {(seed, group) for seed in SEEDS for group in p['valid_groups']}
        actual_plan_keys = [(row['seed'], row['group']) for row in plans]
        require(len(plans) == 192 and len(set(actual_plan_keys)) == 192
                and set(actual_plan_keys) == expected_plan_keys, 'Final plan coverage differs.')
        require(set(path.name for path in (run / 'cache').glob('*.pt')) == set(row['path'] for row in rows)
                and set(path.name for path in (run / 'plans').glob('*.pt')) == set(row['path'] for row in plans),
                'Actual tensor file inventory differs from sealed manifests.')
        import torch
        require(not torch.cuda.is_available() and torch.cuda.device_count() == 0
                and torch._C._cuda_getDeviceCount() == 0, 'A GPU is visible in the CPU audit.')
        torch.set_num_threads(4)
        result['runtime'] = dict(torch_version=torch.__version__, cpu_threads=torch.get_num_threads())
        result['provenance'] = expected_provenance
        result['inputs'] = dict(protocol_sha256=file_hash(args.protocol),
                                upstream_cache_manifest_sha256=file_hash(cache_manifest_path),
                                sealed_plan_manifest_sha256=file_hash(seal_path),
                                completed_result_sha256=file_hash(run / 'result.json'),
                                auditor_source_sha256=file_hash(__file__))
        validation_captures = {}
        metadata_by_key = {}
        counters = result['counters']
        for row in rows:
            path = bound_file(run / 'cache', row['path'], row['sha256'])
            result['active_file'] = 'cache/' + path.name
            case = torch.load(path, map_location='cpu')
            counters['cache_tensor_files_deserialized'] += 1
            checked = check_cache(case, row, p, torch, counters)
            key = (row['seed'], row['phase'], row['group'])
            metadata_by_key[key] = checked
            result['cache_metadata_rows'].append(checked)
            if row['phase'] == 'validation':
                # Retain only tensors needed to compare final plans. Original
                # and candidate raw feature matrices are not loaded again.
                validation_captures[(row['seed'], row['group'])] = {
                    name: case[name] for name in ('native_adjacency', 'target_adjacency',
                                                  'original_candidates', 'random_candidates')}
            del case
        for row in plans:
            path = bound_file(run / 'plans', row['path'], row['sha256'])
            result['active_file'] = 'plans/' + path.name
            matrices = torch.load(path, map_location='cpu')
            counters['final_plan_files_deserialized'] += 1
            require(isinstance(matrices, dict) and set(matrices) == set(ARMS), 'Saved final plan arms differ.')
            tensor_tree(matrices, torch, counters)
            key = (row['seed'], row['group'])
            meta = metadata_by_key[(row['seed'], 'validation', row['group'])]
            cached = validation_captures[key]
            require(row['frozen_upstream_feature_sha256'] == meta['upstream_feature_sha256']
                    and row['frozen_upstream_target_block_sha256'] == meta['upstream_target_block_sha256']
                    and row['quota'] == meta['quota'], 'Final plans do not bind the same frozen upstream input.')
            require(torch.equal(matrices['native_monti'], cached['native_adjacency']),
                    'Native final plan differs from its original capture.')
            t, c, b = meta['targets'], meta['candidates'], meta['injected_nodes']
            role_rows = {}
            for arm in ARMS:
                matrix = matrices[arm]
                require(tensor_hash(matrix) == row['topology_sha256'][arm], 'Sealed topology content hash differs.')
                quota = None if arm == 'native_monti' else meta['quota']
                role_rows[arm] = role_counts(matrix, t, c, b, meta['quota']['edge_budget'], torch, quota)
                if arm != 'native_monti':
                    require(torch.equal(matrix[:, :t], cached['target_adjacency']),
                            'A quota arm changed the cached upstream target block.')
                if arm in ('quota_original', 'quota_random'):
                    name = 'original_candidates' if arm == 'quota_original' else 'random_candidates'
                    require(torch.equal(matrix[:, t:t + c], cached[name]),
                            'A frozen or random final control differs from its cached selection.')
            case_key = hashlib.sha256(json.dumps([row['seed'], row['group'],
                                                  'downstream_tensor_audit_v1']).encode('ascii')).hexdigest()[:20]
            result['validation_role_counts'].append(dict(
                opaque_case_key=case_key, seed=row['seed'], targets=t, candidates=c,
                injected_nodes=b, planned_role_allocation=meta['quota'], arms=role_rows,
                frozen_upstream_feature_sha256=meta['upstream_feature_sha256'],
                frozen_upstream_target_block_sha256=meta['upstream_target_block_sha256']))
            del matrices
        require(counters['cache_tensor_files_deserialized'] == 576
                and counters['final_plan_files_deserialized'] == 192, 'Incomplete saved tensor audit.')
        result['quota_summary'] = {
            'all_cache_cases': len(rows),
            'projected_cache_cases': sum(row['quota']['projection_applied'] for row in result['cache_metadata_rows']),
            'validation_cases': len(plans),
            'projected_validation_cases': sum(row['planned_role_allocation']['projection_applied']
                                              for row in result['validation_role_counts']),
            'target_fraction_min_validation': min(row['planned_role_allocation']['target_fraction']
                                                  for row in result['validation_role_counts']),
            'target_fraction_max_validation': max(row['planned_role_allocation']['target_fraction']
                                                  for row in result['validation_role_counts']),
        }
        result.update(status='passed', active_file=None,
                      scope='Independent saved-tensor hashes, IDs, quota projection and owned-adjacency geometry only.',
                      shared_features_scope='Final plans contain adjacency only; shared features are checked through sealed cache content bindings, not inspected in an executed graph.',
                      budget_scope='Total budgets are checked against hash-bound own cache/manifest metadata, not recomputed from the original dataset. Per-auxiliary-node degree equality across arms is not required or claimed.',
                      chronology_scope='Seal byte bindings checked; wall-clock creation order and absence of earlier unrecorded calls are not established.',
                      independence='Separately authored audit; not human verification or a new prediction recount.')
    except BaseException as caught:
        error = caught
        result['error'] = repr(caught)
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['process_cpu_seconds'] = time.process_time() - cpu_started
        with destination.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write('\n')
        print(json.dumps({name: result[name] for name in ('status', 'counters', 'GPU_hours', 'wall_seconds')},
                         allow_nan=False), flush=True)
    if error is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
