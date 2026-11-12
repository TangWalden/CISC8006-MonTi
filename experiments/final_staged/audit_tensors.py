"""Independently check full formal TRAIN/TEST caches and sealed three-arm geometry.

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
import re
import signal
import statistics
import time
import traceback


ARMS = ('native_monti', 'quota_random', 'quota_learned')
SEEDS = (0, 1, 2)
ORIGINAL_NODES = 16488
SCHEMA = 'monti_final_staged_saved_tensor_geometry_audit_v1'
DEADLINE = None


def check_time():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise TimeoutError('Registered offline saved-tensor audit wall limit reached.')


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            check_time()
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
            and relative.name not in ('', '.', '..') and '\\' not in name,
            'Unsafe manifest file name.')
    directory = pathlib.Path(directory)
    require(directory.is_dir() and not directory.is_symlink(), 'Tensor directory missing or linked.')
    candidate = directory / relative.name
    require(candidate.is_file() and not candidate.is_symlink()
            and candidate.resolve().parent == directory.resolve(), 'Manifest tensor file missing or escaping.')
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
    require(targets >= 1 and 0 <= candidates <= 128 and injected >= 1 and edges >= 1,
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
    members = protocol['train_members'] if phase == 'train' else protocol['test_members']
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



def main(argv=None):
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'audit-source-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    args = parser.parse_args(argv)
    require(1 <= args.max_seconds <= 900, 'Offline audit cap must be within1..900seconds.')
    started, cpu_started = time.monotonic(), time.process_time()
    DEADLINE = started + args.max_seconds
    run = pathlib.Path(args.run_root).resolve(strict=True)
    destination = pathlib.Path(args.output).resolve()
    require(run.is_dir() and not destination.exists(), 'Existing run and new exclusive output are required.')
    result = dict(schema=SCHEMA, status='failed', active_file=None,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        max_seconds=args.max_seconds, GPU_hours=0,
        counters=dict(cache_tensor_files_deserialized=0, final_plan_files_deserialized=0,
            saved_tensor_objects_checked=0, classifier_or_generator_instances=0,
            model_forwards=0, optimizer_updates=0, original_dataset_loads=0,
            model_parameter_checkpoint_loads=0, TEST_queries=0, API_calls=0),
        test_role_counts=[], cache_metadata_rows=[])
    error = None
    armed = hasattr(signal, 'SIGALRM')
    def stopped(signum, frame):
        raise TimeoutError('Registered offline saved-tensor audit wall limit reached.')
    if armed:
        signal.signal(signal.SIGALRM, stopped)
        signal.alarm(args.max_seconds)
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicitly empty CUDA visibility is required.')
        post_path = pathlib.Path(args.audit_source_manifest).resolve(strict=True)
        post = read_json(post_path)
        own_sha = file_hash(__file__)
        require(isinstance(post.get('source_commit'), str)
                and re.fullmatch('[0-9a-f]{40}', post['source_commit']) is not None
                and post.get('files', {}).get('experiments/final_staged/audit_tensors.py') == own_sha,
                'Separate committed audit-source manifest does not bind these auditor bytes.')
        p = read_json(args.protocol)
        require(p.get('schema') == 'monti_final_staged_v1' and p.get('status') == 'frozen'
                and p.get('owner_execution_authorized') is True and p.get('final_TEST_allowed') is True
                and tuple(p['seeds']) == SEEDS and tuple(p['arms']) == ARMS
                and p['runtime_commit'] == 'ca093d8b12ea153a076b02bbdbc4ede58508560c',
                'Wrong frozen formal study protocol.')
        require(len(p['train_groups']) == len(set(p['train_groups'])) == 937
                and len(p['test_groups']) == len(set(p['test_groups'])) == 909
                and not set(p['train_groups']) & set(p['test_groups']), 'Full registered cohorts differ.')
        memberships = {}
        for phase, size, occurrences in (('train', 937, 11211), ('test', 909, 10515)):
            members = p[phase + '_members']
            require(set(members) == {str(group) for group in p[phase + '_groups']}
                    and len(members) == size
                    and all(nodes and len(nodes) == len(set(nodes))
                            and all(type(node) is int and 0 <= node < ORIGINAL_NODES for node in nodes)
                            for nodes in members.values()), 'Registered ordered membership differs.')
            require(sum(map(len, members.values())) == occurrences == p['cohort'][phase + '_target_occurrences'],
                    'Complete occurrence denominator differs.')
            memberships[phase] = {node for nodes in members.values() for node in nodes}
            require(len(memberships[phase]) == p['cohort']['unique_' + phase + '_targets'],
                    'Registered unique target count differs.')
        require(not memberships['train'] & memberships['test'], 'TRAIN/TEST target node identities overlap.')
        provenance = dict(source_commit=p['source_manifest']['source_commit'],
            runtime_commit=p['runtime_commit'], protocol_sha256=file_hash(args.protocol))
        completed_path = run / 'result.json'
        completed = read_json(completed_path)
        require(completed.get('status') == 'completed' and completed.get('provenance') == provenance
                and completed.get('full_TEST_evaluated') is True,
                'Only a completed same-protocol formal run may be audited; partial outcomes are forbidden.')
        cache_path, seal_path = run / 'upstream_cache_manifest.json', run / 'sealed_before_victim.json'
        cache_manifest, seal = read_json(cache_path), read_json(seal_path)
        require(cache_manifest['provenance'] == seal['provenance'] == provenance
                and file_hash(cache_path) == seal['upstream_cache_sha256']
                and file_hash(seal_path) == completed['pre_victim_seal_sha256'],
                'Source/protocol or sealed manifest byte bindings differ.')
        require(cache_manifest['optimizer_updates_before_seal'] == 0 and cache_manifest['victim_loaded'] is False
                and seal['official_GAGA_loaded'] is False and seal['fitting_finished'] is True,
                'Sealed fitting/evaluation boundary declarations differ.')
        require(cache_manifest['receipts'] == completed['upstream_receipts']
                and len(cache_manifest['receipts']) == 3
                and {row['seed'] for row in cache_manifest['receipts']} == set(SEEDS),
                'Formal upstream receipt inventory differs.')
        for receipt in cache_manifest['receipts']:
            formal = p['formal_baselines'][str(receipt['seed'])]
            loaded = receipt['loader_receipt']
            require(receipt['before'] == receipt['after']
                    and receipt['optimizer_updates'] == receipt['classifier_queries_during_generation'] == 0
                    and receipt['checkpoint_sha256'] == loaded['checkpoint_sha256'] == formal['checkpoint_ref']['sha256']
                    and loaded['capture_epoch'] == loaded['best_epoch'] == formal['best_epoch']
                    and loaded['completed_epochs'] == loaded['scheduler_restored_steps'] == formal['scheduler_restored_steps']
                    and loaded['official_GAGA_loaded'] is False and loaded['upstream_optimizer_steps'] == 0,
                    'Formal checkpoint, scheduler or frozen-state receipt binding differs.')
        rows, plans = cache_manifest['cases'], seal['plans']
        expected_cache = {(seed, phase, group) for seed in SEEDS
            for phase, groups in (('train', p['train_groups']), ('test', p['test_groups'])) for group in groups}
        actual_cache = [(row['seed'], row['phase'], row['group']) for row in rows]
        require(len(rows) == len(set(actual_cache)) == 5538 and set(actual_cache) == expected_cache,
                'The full5538 TRAIN/TEST cache cohort differs.')
        plan_by_key = {(row['seed'], row['group']): row for row in plans}
        require(len(plans) == len(plan_by_key) == 2727
                and set(plan_by_key) == {(seed, group) for seed in SEEDS for group in p['test_groups']},
                'The full2727 TEST plan cohort differs.')
        require(set(path.name for path in (run / 'cache').glob('*.pt')) == {row['path'] for row in rows}
                and set(path.name for path in (run / 'plans').glob('*.pt')) == {row['path'] for row in plans},
                'Actual serialized file inventory differs from sealed manifests.')
        require(seal['downstream_records'] == completed['downstream']
                and len(seal['downstream_records']) == 3, 'Terminal downstream record copies differ.')
        checkpoint_hashes = {}
        for row in seal['downstream_records']:
            seed = row['seed']
            wanted = seal['downstream_checkpoint_hashes'][str(seed)]
            require(seed in SEEDS and wanted == row['final_checkpoint_sha256']
                    and row['epochs'] == 4 and row['parameter_count'] == 151872,
                    'Terminal scorer identity differs.')
            checkpoint_hashes[str(seed)] = file_hash(bound_file(run / 'downstream', 's%d_final.pt' % seed, wanted))
        require(set(checkpoint_hashes) == set(map(str, SEEDS)), 'Terminal scorer seed inventory differs.')
        import torch
        require(not torch.cuda.is_available() and torch.cuda.device_count() == 0
                and torch._C._cuda_getDeviceCount() == 0, 'A GPU is visible in the CPU-only offline audit.')
        torch.set_num_threads(4)
        result['runtime'] = dict(torch_version=torch.__version__, cpu_threads=torch.get_num_threads(),
            memory_scope='One cache and one three-arm plan live at a time; summaries retain metadata only.')
        result['provenance'] = provenance
        result['audit_source_binding'] = dict(source_commit=post['source_commit'],
            manifest_sha256=file_hash(post_path), auditor_source_sha256=own_sha,
            relationship='Separate committed audit source; original execution manifest/protocol unmodified.')
        result['inputs'] = dict(protocol_sha256=file_hash(args.protocol),
            upstream_cache_manifest_sha256=file_hash(cache_path), sealed_plan_manifest_sha256=file_hash(seal_path),
            completed_result_sha256=file_hash(completed_path), terminal_checkpoint_byte_hashes=checkpoint_hashes)
        counters = result['counters']
        for row in rows:
            check_time()
            require(row['path'] == 's%d_%s_g%d.pt' % (row['seed'], row['phase'], row['group']),
                    'Cache filename is not bound to its declared seed/phase/group.')
            path = bound_file(run / 'cache', row['path'], row['sha256'])
            result['active_file'] = 'cache/' + path.name
            case = torch.load(path, map_location='cpu')
            counters['cache_tensor_files_deserialized'] += 1
            meta = check_cache(case, row, p, torch, counters)
            result['cache_metadata_rows'].append(meta)
            if row['phase'] == 'test':
                plan = plan_by_key[row['seed'], row['group']]
                require(plan['path'] == 's%d_g%d.pt' % (row['seed'], row['group']),
                        'Plan filename is not bound to its seed/group.')
                path = bound_file(run / 'plans', plan['path'], plan['sha256'])
                result['active_file'] = 'plans/' + path.name
                matrices = torch.load(path, map_location='cpu')
                counters['final_plan_files_deserialized'] += 1
                require(isinstance(matrices, dict) and set(matrices) == set(ARMS), 'Sealed three-arm plan differs.')
                tensor_tree(matrices, torch, counters)
                require(plan['quota'] == meta['quota']
                        and plan['frozen_upstream_feature_sha256'] == meta['upstream_feature_sha256']
                        and plan['frozen_upstream_target_block_sha256'] == meta['upstream_target_block_sha256']
                        and set(plan['topology_sha256']) == set(ARMS),
                        'Final plan differs from its frozen upstream feature/target bindings.')
                require(torch.equal(matrices['native_monti'], case['native_adjacency']),
                        'Native final plan differs from its original frozen capture.')
                t, c, b = meta['targets'], meta['candidates'], meta['injected_nodes']
                role_rows = {}
                for arm in ARMS:
                    matrix = matrices[arm]
                    require(tensor_hash(matrix) == plan['topology_sha256'][arm], 'Sealed topology tensor hash differs.')
                    role_rows[arm] = role_counts(matrix, t, c, b, meta['quota']['edge_budget'], torch,
                                                 None if arm == 'native_monti' else meta['quota'])
                    if arm != 'native_monti':
                        require(torch.equal(matrix[:, :t], case['target_adjacency']),
                                'Random/trained completion altered the fixed upstream target block.')
                    if arm == 'quota_random':
                        require(torch.equal(matrix[:, t:t+c], case['random_candidates']),
                                'Random completion differs from its cached fixed random selection.')
                opaque = hashlib.sha256(json.dumps([row['seed'], row['group'],
                    'final_staged_tensor_audit_v1']).encode('ascii')).hexdigest()[:20]
                result['test_role_counts'].append(dict(opaque_case_key=opaque, seed=row['seed'],
                    targets=t, candidates=c, injected_nodes=b, planned_role_allocation=meta['quota'], arms=role_rows,
                    frozen_upstream_feature_sha256=meta['upstream_feature_sha256'],
                    frozen_upstream_target_block_sha256=meta['upstream_target_block_sha256']))
                del matrices
            del case
        require(counters['cache_tensor_files_deserialized'] == 5538
                and counters['final_plan_files_deserialized'] == len(result['test_role_counts']) == 2727,
                'Incomplete formal saved-tensor geometry audit.')
        test_roles = result['test_role_counts']
        ratios = [row['planned_role_allocation']['target_fraction'] for row in test_roles]
        result['quota_summary'] = dict(all_cache_cases=5538,
            projected_cache_cases=sum(row['quota']['projection_applied'] for row in result['cache_metadata_rows']),
            test_cases=2727, projected_test_cases=sum(row['planned_role_allocation']['projection_applied'] for row in test_roles),
            exact_half_test_cases=sum(row['planned_role_allocation']['exact_half'] for row in test_roles),
            target_fraction_min_test=min(ratios), target_fraction_max_test=max(ratios),
            target_fraction_case_mean_test=statistics.mean(ratios), target_fraction_case_median_test=statistics.median(ratios),
            target_fraction_edge_weighted_test=sum(row['planned_role_allocation']['target_quota'] for row in test_roles) /
                                               sum(row['planned_role_allocation']['edge_budget'] for row in test_roles))
        result.update(status='passed', active_file=None,
            scope='Full registered own saved tensors: content hashes, ordered IDs, independent quota algebra, budgets and owned-adjacency geometry. No TEST predictions read.',
            physical_mapping_scope='Distinct original/contiguous injected IDs, binary cells and strict lower-triangular ownership establish unique nonself undirected pairs; no executed DGL graph is inspected.',
            shared_features_scope='Final plans store adjacency only; shared auxiliary features are verified through frozen cache content and seal bindings, not independently against an executed graph.',
            budget_scope='Node/edge budgets checked against frozen own cache and manifest metadata, not independently regenerated from the original graph. Individual auxiliary degrees may differ.',
            chronology_scope='File/tensor bindings checked; wall-clock chronology and absence of unrecorded calls are not proved.',
            independence='Adapted separately authored offline audit; not human verification or a new experiment.')
    except BaseException as caught:
        error = caught
        result['error'] = repr(caught)
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if armed:
            signal.alarm(0)
        DEADLINE = None
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
