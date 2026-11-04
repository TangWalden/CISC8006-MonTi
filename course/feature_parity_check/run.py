"""CPU-only canonical own-teacher checks on all384 previously saved graphs.

All negative development outcomes are already known. This is a POST_HOC
implementation diagnostic, not preregistered efficacy confirmation. Only the
fixed independently fitted full-TRAIN GAGA teacher is constructed. Official
victim values are read exclusively from immutable saved prediction JSONL files.
No generator/GCN/victim construction, new affine classifier forwards, gradient,
optimizer update, API call, or TEST evaluation is permitted.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import pathlib
import signal
import subprocess
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = 'monti_feature_parity_diagnostic_v1'
ARMS = ('baseline', 'gaga_pgd')
SEEDS = (3, 4, 5)
SEQUENCE_TOLERANCES = dict(atol=2e-5, rtol=2e-5)
LOGIT_TOLERANCES = dict(atol=2e-4, rtol=2e-4)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def repository_file(name):
    relative = pathlib.Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Only repository-relative input paths are allowed.')
    path = (ROOT/relative).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise RuntimeError('A frozen diagnostic input is missing or outside the repository.')
    return path


def checked(reference, receipts):
    path = repository_file(reference['path'])
    actual = sha256(path)
    if actual != reference['sha256']:
        raise RuntimeError('Frozen diagnostic input bytes changed: '+reference['path'])
    receipts[reference['path']] = actual
    return path


def index_jsonl(path, expected_groups):
    result = {}
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = int(row['group'])
            if identity in result:
                raise RuntimeError('Duplicate group in a frozen diagnostic input.')
            result[identity] = row
    if set(result) != set(expected_groups) or len(result) != 128:
        raise RuntimeError('Diagnostic input must retain all128 fixed VALID groups.')
    return result


def validate_registration(path, receipts):
    protocol = read_json(path)
    if (protocol.get('schema') != SCHEMA or protocol.get('status') != 'frozen'
            or protocol.get('post_hoc') is not True):
        raise RuntimeError('A frozen explicitly post-hoc parity registration is required.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=ROOT, universal_newlines=True).strip():
        raise RuntimeError('Commit tracked files before this diagnostic.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, universal_newlines=True).strip()
    if head != protocol['source_commit']:
        raise RuntimeError('Diagnostic execution HEAD differs from its new registration.')
    required = {file.relative_to(ROOT).as_posix() for file in pathlib.Path(__file__).parent.glob('*.py')}
    if not required <= set(protocol['source_files']):
        raise RuntimeError('New diagnostic package source freeze is incomplete.')
    for name, wanted in protocol['source_files'].items():
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', name],
                                cwd=ROOT, stderr=subprocess.STDOUT)
        checked(dict(path=name, sha256=wanted), receipts)
    for reference in protocol['inputs']:
        checked(reference, receipts)
    if (tuple(protocol['seeds']) != SEEDS or tuple(protocol['arms']) != ARMS
            or len(protocol['valid_groups']) != 128 or len(set(protocol['valid_groups'])) != 128
            or protocol['deadline_seconds'] != 1800 or protocol['cpu_threads'] != 8
            or protocol['sequence_tolerances'] != SEQUENCE_TOLERANCES
            or protocol['logit_tolerances'] != LOGIT_TOLERANCES
            or set(protocol['runs']) != {'3', '4', '5'}):
        raise RuntimeError('Frozen diagnostic scope/tolerances/resources differ.')
    old_path = checked(protocol['base_experiment_ref'], receipts)
    old = read_json(old_path)
    if (tuple(old['seeds']) != SEEDS or old['valid_groups'] != protocol['valid_groups']
            or old['final_test_allowed'] is not False):
        raise RuntimeError('The existing experiment identity or no-TEST boundary differs.')
    # Old feature_polish.validate_protocol is deliberately not called: the new
    # committed HEAD differs. Instead every reused base source byte is bound
    # both to the existing experiment and to the new diagnostic registration.
    for name, wanted in old['source_files'].items():
        if protocol['source_files'].get(name) != wanted:
            raise RuntimeError('A reused scientific source differs from the original experiment: '+name)
    for reference in old['inputs']+old.get('retained_inputs', []):
        checked(reference, receipts)
    if protocol.get('teacher_ref', old['teacher_ref']) != old['teacher_ref']:
        raise RuntimeError('Diagnostic teacher differs from the completed original full-TRAIN fit.')
    checked(old['teacher_ref'], receipts)
    runs = {}
    for seed in SEEDS:
        spec = protocol['runs'][str(seed)]
        result_path = checked(spec['result_ref'], receipts)
        groups_path = checked(spec['groups_ref'], receipts)
        result = read_json(result_path)
        if (result.get('status') != 'completed' or result['seed'] != seed or result['groups'] != 128
                or result['source_commit'] != old['source_commit']
                or result['protocol_sha256'] != protocol['base_experiment_ref']['sha256']
                or result['groups_jsonl_sha256'] != sha256(groups_path)
                or result['teacher_ref'] != old['teacher_ref']):
            raise RuntimeError('A retained completed refinement identity/provenance differs.')
        rows = index_jsonl(groups_path, protocol['valid_groups'])
        exports = {int(row['group']): row for row in spec['captures']}
        if len(spec['captures']) != 128 or set(exports) != set(rows):
            raise RuntimeError('All128 saved captures must be frozen for each seed.')
        old_exports = {int(row['group']): row for row in result['exports']}
        if len(result['exports']) != 128 or set(old_exports) != set(rows):
            raise RuntimeError('Existing refinement capture manifest is incomplete.')
        directory = result_path.parent
        captures = {}
        for group, reference in exports.items():
            export_path = checked(reference, receipts)
            old_export = old_exports[group]
            if (export_path != (directory/old_export['path']).resolve()
                    or old_export != rows[group]['graph_capture']
                    or reference['sha256'] != old_export['sha256']):
                raise RuntimeError('Diagnostic capture differs from the original run/row receipt.')
            captures[group] = export_path
        if set(spec['victim_prediction_refs']) != set(ARMS):
            raise RuntimeError('Freeze both old official-victim endpoint channels only.')
        victim = {arm:index_jsonl(checked(spec['victim_prediction_refs'][arm], receipts), protocol['valid_groups'])
                  for arm in ARMS}
        runs[seed] = dict(result=result, rows=rows, captures=captures, victim=victim)
    return protocol, old, runs, dict(source_commit=head, protocol_sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:
        raise RuntimeError('Private diagnostic output must remain below ignored runs/.')
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    counters = dict(hard_graph_reconstructions=0, affine_builders=0, canonical_sequence_updates=0,
        fresh_teacher_forward_dispatches=0, fresh_teacher_completed_forwards=0,
        stored_affine_comparisons=0, guard_alias_checks=0, capture_tensor_loads=0,
        data_load_dispatches=0, data_loads_completed=0, teacher_load_dispatches=0, teacher_loads_completed=0,
        new_affine_model_forwards=0, generator_model_constructions=0, generator_forwards=0,
        GCN_model_constructions=0, GCN_forwards=0, official_victim_model_constructions=0,
        official_victim_forwards=0, optimizer_steps=0, autograd_calls=0, API_calls=0, TEST_queries=0)
    result = dict(status='failed', scope='POST_HOC_after_completed_negative_development_results',
                  official_victim_loaded=False, counters=counters,
                  immutable_content_hashes=dict(before=None, after=None))
    runtime_metadata = dict(cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch_cuda_available=None, torch_visible_gpu_count=None, native_visible_gpu_count=None)
    receipts, violations, joins, comparisons = {}, [], [], []
    def stop(signum, frame):
        raise TimeoutError('Frozen1800second CPU diagnostic deadline reached.')
    signal.signal(signal.SIGALRM, stop);signal.signal(signal.SIGTERM, stop);signal.alarm(1800)
    try:
        protocol_path = pathlib.Path(args.protocol).resolve()
        protocol, old, runs, provenance = validate_registration(protocol_path, receipts)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise RuntimeError('The CPU-only diagnostic requires explicitly empty CUDA visibility.')
        write_json(output/'registered.json', dict(provenance=provenance,
            own_source_sha256=sha256(__file__), base_experiment_ref=protocol['base_experiment_ref'],
            scope='All3seeds/all128savedVALIDgraphs/baseline and terminalGAGAfeatures; no method adaptation',
            source_and_input_hashes=receipts, expected_captures=384, expected_canonical_updates=768,
            expected_fresh_own_teacher_forwards=768, new_affine_model_forwards=0,
            official_victim_access='Saved logits only; zero constructor/checkpoint/forward calls'))
        import dgl
        import numpy as np
        import scipy.sparse as sp
        import torch
        import torch.nn.functional as F
        from utils.utils import set_seed
        from course.dual_surrogate.runtime import load_data, load_original_view, model_hash, checked_logits
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.source_role_fix.saved_teacher_diagnostic import reconstruct
        from course.feature_polish.affine import build_affine
        from course.feature_polish.refine import graph_with_features, tensor_sha256, preserves_baseline_gcn_zeros
        from course.feature_polish.teacher import load_teacher
        runtime_metadata['torch_cuda_available'] = bool(torch.cuda.is_available())
        runtime_metadata['torch_visible_gpu_count'] = int(torch.cuda.device_count())
        runtime_metadata['native_visible_gpu_count'] = int(torch._C._cuda_getDeviceCount())
        if (runtime_metadata['torch_cuda_available'] or runtime_metadata['torch_visible_gpu_count'] != 0
                or runtime_metadata['native_visible_gpu_count'] != 0):
            raise RuntimeError('Both Torch/native visible GPU counts must be zero.')
        torch.set_num_threads(8);set_seed(20261104);dgl.seed(20261104);dgl.random.seed(20261104)
        counters['data_load_dispatches'] += 1
        data = load_data(old, 20261104)
        counters['data_loads_completed'] += 1
        if data.device.type != 'cpu':
            raise RuntimeError('Diagnostic data must remain on CPU.')
        view = load_original_view(data)
        counters['teacher_load_dispatches'] += 1
        teacher, teacher_manifest = load_teacher(old, data)
        counters['teacher_loads_completed'] += 1
        teacher_before = model_hash(teacher)
        if teacher.training or any(parameter.requires_grad or parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError('The sole diagnostic teacher is not frozen in eval mode.')
        if any(run['result']['teacher_state_sha256'] != teacher_before for run in runs.values()):
            raise RuntimeError('The loaded own teacher differs from its original completed refinement identity.')
        original_features_before = tensor_sha256(data.features)
        cache_before = dict(sums=tensor_sha256(view.feat_sum), degrees=tensor_sha256(view.d))
        result['immutable_content_hashes']['before'] = dict(teacher=teacher_before,
            features=original_features_before, feat_sum=cache_before['sums'], degrees=cache_before['degrees'])
        minimum, maximum = data.features.min(0).values, data.features.max(0).values
        original_count = data.graph.num_nodes()

        def note(kind, seed=None, group=None, arm=None, detail=None):
            record = dict(kind=kind)
            if seed is not None:record['seed'] = seed
            if group is not None:record['group'] = group
            if arm is not None:record['arm'] = arm
            if detail is not None:record['detail'] = detail
            violations.append(record)

        unique_targets = sorted({int(node) for row in runs[3]['rows'].values() for node in row['node_ids']})
        if len(unique_targets) != 374:
            raise RuntimeError('All128 groups must retain their374unique VALID targets.')
        for run in runs.values():
            if sorted({int(node) for row in run['rows'].values() for node in row['node_ids']}) != unique_targets:
                raise RuntimeError('The unique target cohort differs across seeds.')
        # Pure topology/cache arithmetic: do not silently replace original
        # incidence weights with binary values before validating the assumption.
        adjacency = sp.csr_matrix(view.adj, copy=True)
        adjacency.sum_duplicates();adjacency.eliminate_zeros()
        binary = not adjacency.data.size or bool(np.isfinite(adjacency.data).all() and (adjacency.data == 1).all())
        asymmetric_entries = int((adjacency != adjacency.T).nnz)
        symmetric = asymmetric_entries == 0
        zero_diagonal = not bool(np.any(adjacency.diagonal()))
        unknown = view.group_masks['train_masked'].detach().cpu().numpy()
        if (not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0)
                or not np.array_equal(adjacency.data, adjacency.data.astype(np.int64))
                or not np.isin(unknown, [0, 1]).all()):
            raise RuntimeError('Invalid source topology weights or unknown-mask arithmetic.')
        one = adjacency[unique_targets].astype(np.int64)
        paths = (one@adjacency.astype(np.int64)).tocoo()
        keep = paths.col != np.asarray(unique_targets, dtype=np.int64)[paths.row]
        two = sp.coo_matrix((np.ones(int(keep.sum()), dtype=np.int64), (paths.row[keep], paths.col[keep])), shape=one.shape).tocsr()
        two.sum_duplicates();two.eliminate_zeros();two.data.fill(1)
        expected_degrees = [np.asarray(reach@unknown.astype(np.int64)).reshape(-1)+1 for reach in (one, two)]
        cached_degrees = [view.d[torch.tensor(unique_targets), token, 0].detach().cpu().numpy() for token in (3, 6)]
        cached_two = sp.csr_matrix(view.adj2, copy=True)
        cached_two.sum_duplicates();cached_two.eliminate_zeros()
        cached_two_binary = not cached_two.data.size or bool(np.isfinite(cached_two.data).all() and (cached_two.data == 1).all())
        cached_two_zero_diagonal = not bool(np.any(cached_two.diagonal()))
        cached_two_selected_support = cached_two[unique_targets].copy()
        cached_two_selected_support.data = np.ones(cached_two_selected_support.nnz, dtype=np.int64)
        cached_two_selected_matches = (cached_two_selected_support != two).nnz == 0
        checks = dict(original_adjacency_binary=binary, original_adjacency_symmetric=symmetric,
            original_adjacency_asymmetric_matrix_entries=asymmetric_entries,
            original_adjacency_zero_diagonal=zero_diagonal, original_adj2_binary=cached_two_binary,
            original_adj2_zero_diagonal=cached_two_zero_diagonal,
            original_adj2_selected_support_matches=cached_two_selected_matches,
            unique_VALID_targets=374, masked_self_count_each_hop=1, unknown_degree_checks=[])
        for token, expected, cached in zip((3, 6), expected_degrees, cached_degrees):
            error = np.abs(expected-cached)
            checks['unknown_degree_checks'].append(dict(token=token, exact_match=bool(np.array_equal(expected, cached)),
                mismatched_targets=int(np.count_nonzero(error)), maximum_absolute_count_error=float(error.max())))
        if not all((binary, symmetric, zero_diagonal, cached_two_binary, cached_two_zero_diagonal,
                    cached_two_selected_matches, *(row['exact_match'] for row in checks['unknown_degree_checks']))):
            note('original_topology_or_unknown_cache_assumption', detail=checks)
        result['original_topology_and_cache_checks'] = checks

        def compare(actual, expected, tolerance):
            if actual.shape != expected.shape:
                return dict(passed=False, shape_matches=False, max_absolute_error=None, max_tolerance_fraction=None)
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise RuntimeError('Nonfinite sequence/logits in retained parity comparisons.')
            error = (actual-expected).abs()
            bound = tolerance['atol']+tolerance['rtol']*expected.abs()
            return dict(passed=bool((error <= bound).all()), shape_matches=True,
                max_absolute_error=float(error.max()), max_tolerance_fraction=float((error/bound).max()))

        def measures(logits, labels):
            prediction = logits.argmax(1)
            benign_margin = logits[:, 0]-logits[:, 1]
            return dict(occurrences=len(logits), wrong=int((prediction != labels).sum()),
                mean_benign_minus_fraud_margin=float(benign_margin.mean()),
                mean_softplus=float(F.softplus(-benign_margin).mean()))

        def teacher_forward(sequence, count):
            if counters['fresh_teacher_forward_dispatches'] >= 768:
                raise RuntimeError('The frozen768own-teacher forward cap is exhausted.')
            counters['fresh_teacher_forward_dispatches'] += 1
            logits = checked_logits(teacher(sequence), count)
            counters['fresh_teacher_completed_forwards'] += 1
            return logits

        write_json(output/'original_topology_and_cache_checks.json', checks)
        with torch.no_grad(), (output/'comparisons.jsonl').open('x', encoding='utf-8', newline='\n') as stream, (output/'endpoint_joins.jsonl').open('x', encoding='utf-8', newline='\n') as join_stream:
            for seed in SEEDS:
                run = runs[seed]
                for group in protocol['valid_groups']:
                    saved = run['rows'][group]
                    payload_path = run['captures'][group]
                    if sha256(payload_path) != saved['graph_capture']['sha256']:
                        raise RuntimeError('Saved capture changed after the pre-ML hash gate.')
                    counters['capture_tensor_loads'] += 1
                    payload = torch.load(payload_path, map_location='cpu')
                    if set(payload) != {'capture', 'features'} or set(payload['features']) != set(old['arms']):
                        raise RuntimeError('Retained capture branch schema differs.')
                    capture = payload['capture']
                    graph_record = dict(saved, actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                    clean_identity = dict(group=group, node_ids=saved['node_ids'], labels=saved['labels'])
                    graph, targets, candidates, injected = reconstruct(data, capture, graph_record, clean_identity)
                    counters['hard_graph_reconstructions'] += 1
                    x0 = capture['x_atk'].detach()
                    if not torch.equal(payload['features']['baseline'], x0):
                        raise RuntimeError('Saved baseline features differ from the shared graph capture.')
                    if tensor_sha256(capture['adj_attack']) != saved['adjacency_content_sha256']:
                        raise RuntimeError('Shared capture adjacency identity differs.')
                    allowed, _ = preserves_baseline_gcn_zeros(torch.tensor(saved['final_own_logits']['baseline']['gcn']),
                        torch.tensor(saved['final_own_logits']['gaga_pgd']['gcn']))
                    chosen = 'gaga_pgd' if allowed else 'baseline'
                    alias_matches = (allowed == saved['guard']['accepted']
                        and torch.equal(payload['features']['gaga_guarded'], payload['features'][chosen])
                        and saved['final_own_logits']['gaga_guarded'] == saved['final_own_logits'][chosen])
                    counters['guard_alias_checks'] += 1
                    if not alias_matches:note('saved_guard_alias', seed, group)
                    counters['canonical_sequence_updates'] += 1
                    canonical_base = updated_sequences(view, graph, targets, candidates, injected, training=False)
                    affine = build_affine(view, graph, targets, candidates, injected, x0, base_sequence=canonical_base)
                    counters['affine_builders'] += 1
                    endpoints = {}
                    labels = torch.tensor(saved['labels'], dtype=torch.long)
                    for arm in ARMS:
                        features = payload['features'][arm].detach()
                        if (features.shape != x0.shape or features.dtype != torch.float32
                                or not bool(((features >= minimum)&(features <= maximum)).all())
                                or tensor_sha256(features) != saved['feature_statistics'][arm]['content_sha256']):
                            raise RuntimeError('Retained endpoint features violate their frozen shape/box/hash.')
                        if arm == 'baseline':
                            canonical = canonical_base
                        else:
                            local = graph_with_features(graph, original_count, features)
                            counters['canonical_sequence_updates'] += 1
                            canonical = updated_sequences(view, local, targets, candidates, injected, training=False)
                        affine_sequence = affine.sequence(features)
                        sequence_comparison = compare(affine_sequence, canonical, SEQUENCE_TOLERANCES)
                        logits = teacher_forward(canonical, len(targets))
                        stored = checked_logits(torch.tensor(saved['final_own_logits'][arm]['gaga_teacher'], dtype=torch.float32), len(targets))
                        logit_comparison = compare(logits, stored, LOGIT_TOLERANCES)
                        current_metrics, stored_metrics = measures(logits, labels), measures(stored, labels)
                        class_differences = int((logits.argmax(1) != stored.argmax(1)).sum())
                        victim_row = run['victim'][arm][group]
                        if (victim_row['seed'] != seed or victim_row['arm'] != arm
                                or victim_row['node_ids'] != saved['node_ids'] or victim_row['labels'] != saved['labels']
                                or victim_row['common_capture_sha256'] != saved['graph_capture']['sha256']
                                or victim_row['injected_feature_content_sha256'] != saved['feature_statistics'][arm]['content_sha256']
                                or victim_row['hard_adjacency_content_sha256'] != saved['adjacency_content_sha256']):
                            raise RuntimeError('Saved victim endpoint identity/geometry/feature provenance differs.')
                        victim_logits = checked_logits(torch.tensor(victim_row['gaga_logits'], dtype=torch.float32), len(targets))
                        victim_metrics = measures(victim_logits, labels)
                        row = dict(seed=seed, group=group, arm=arm, targets=len(targets),
                            capture_sha256=saved['graph_capture']['sha256'],
                            feature_content_sha256=saved['feature_statistics'][arm]['content_sha256'],
                            sequence=sequence_comparison, teacher_logits=logit_comparison,
                            teacher_class_differences=class_differences,
                            teacher_canonical_metrics=current_metrics, teacher_saved_affine_metrics=stored_metrics,
                            teacher_mean_loss_difference=current_metrics['mean_softplus']-stored_metrics['mean_softplus'],
                            teacher_mean_margin_difference=current_metrics['mean_benign_minus_fraud_margin']-stored_metrics['mean_benign_minus_fraud_margin'],
                            canonical_teacher_logits=logits.tolist(), stored_affine_teacher_logits=stored.tolist(),
                            saved_victim_metrics=victim_metrics, saved_victim_logits=victim_logits.tolist(),
                            node_ids=saved['node_ids'], labels=saved['labels'], guarded_alias_matches=alias_matches)
                        counters['stored_affine_comparisons'] += 1
                        comparisons.append({key:value for key,value in row.items() if key not in
                            ('canonical_teacher_logits', 'stored_affine_teacher_logits', 'saved_victim_logits', 'node_ids', 'labels')})
                        if not sequence_comparison['passed']:note('affine_sequence_parity', seed, group, arm, sequence_comparison)
                        if not logit_comparison['passed']:note('canonical_teacher_vs_saved_affine_logits', seed, group, arm, logit_comparison)
                        if class_differences:
                            changed = logits.argmax(1) != stored.argmax(1)
                            note('within_tolerance_teacher_argmax_flip' if logit_comparison['passed']
                                 else 'canonical_teacher_vs_saved_affine_class', seed, group, arm,
                                 dict(count=class_differences,
                                      canonical_changed_max_absolute_margin=float((logits[changed, 0]-logits[changed, 1]).abs().max()),
                                      stored_changed_max_absolute_margin=float((stored[changed, 0]-stored[changed, 1]).abs().max())))
                        stream.write(json.dumps(row, allow_nan=False)+'\n');stream.flush()
                        endpoints[arm] = dict(teacher=logits, victim=victim_logits)
                    base, final = endpoints['baseline'], endpoints['gaga_pgd']
                    def changes(before, after):
                        margins_before = before[:, 0]-before[:, 1]
                        margins_after = after[:, 0]-after[:, 1]
                        losses_before, losses_after = F.softplus(-margins_before), F.softplus(-margins_after)
                        return dict(mean_benign_margin_change=float((margins_after-margins_before).mean()),
                            mean_softplus_change=float((losses_after-losses_before).mean()),
                            benign_margin_increases=int((margins_after > margins_before).sum()),
                            benign_margin_decreases=int((margins_after < margins_before).sum()),
                            wrong_count_change=int((after.argmax(1) != labels).sum()-(before.argmax(1) != labels).sum()),
                            per_target_benign_margin_change=(margins_after-margins_before).tolist(),
                            per_target_softplus_change=(losses_after-losses_before).tolist())
                    join = dict(seed=seed, group=group, targets=len(targets),
                        canonical_own_teacher=changes(base['teacher'], final['teacher']),
                        saved_victim=changes(base['victim'], final['victim']),
                        scale_caveat='Separate uncalibrated logit scales; two-endpoint changes are not derivatives or paired slopes.')
                    joins.append(join);join_stream.write(json.dumps(join, allow_nan=False)+'\n');join_stream.flush()
                    if len(joins)%32 == 0:print('FEATURE_PARITY_GRAPHS', len(joins), flush=True)
                    del graph, payload, capture, affine, canonical_base, endpoints
        expected = dict(hard_graph_reconstructions=384, affine_builders=384, canonical_sequence_updates=768,
            fresh_teacher_forward_dispatches=768, fresh_teacher_completed_forwards=768,
            stored_affine_comparisons=768, guard_alias_checks=384, capture_tensor_loads=384,
            data_load_dispatches=1, data_loads_completed=1, teacher_load_dispatches=1, teacher_loads_completed=1,
            new_affine_model_forwards=0, generator_model_constructions=0, generator_forwards=0,
            GCN_model_constructions=0, GCN_forwards=0, official_victim_model_constructions=0,
            official_victim_forwards=0, optimizer_steps=0, autograd_calls=0, API_calls=0, TEST_queries=0)
        if counters != expected:
            raise RuntimeError('Completed diagnostic counters differ from the frozen384/768 design.')
        immutable_after = dict(teacher=model_hash(teacher), features=tensor_sha256(data.features),
            feat_sum=tensor_sha256(view.feat_sum), degrees=tensor_sha256(view.d))
        result['immutable_content_hashes']['after'] = immutable_after
        if (immutable_after != result['immutable_content_hashes']['before'] or teacher.training
                or any(parameter.grad is not None or parameter.requires_grad for parameter in teacher.parameters())
                ):
            raise RuntimeError('The frozen own teacher, original features or caches changed during diagnostic.')
        grouped = []
        for seed in SEEDS:
            for arm in ARMS:
                selected = [row for row in comparisons if row['seed'] == seed and row['arm'] == arm]
                grouped.append(dict(seed=seed, arm=arm, graphs=len(selected),
                    occurrences=sum(row['targets'] for row in selected),
                    sequence_parity_failures=sum(not row['sequence']['passed'] for row in selected),
                    logit_parity_failures=sum(not row['teacher_logits']['passed'] for row in selected),
                    class_differences=sum(row['teacher_class_differences'] for row in selected),
                    maximum_sequence_absolute_error=max((row['sequence']['max_absolute_error'] for row in selected
                        if row['sequence']['max_absolute_error'] is not None), default=None),
                    maximum_teacher_logit_absolute_error=max((row['teacher_logits']['max_absolute_error'] for row in selected
                        if row['teacher_logits']['max_absolute_error'] is not None), default=None),
                    canonical_teacher_wrong=sum(row['teacher_canonical_metrics']['wrong'] for row in selected),
                    saved_affine_teacher_wrong=sum(row['teacher_saved_affine_metrics']['wrong'] for row in selected),
                    saved_victim_wrong=sum(row['saved_victim_metrics']['wrong'] for row in selected)))
        result.update(status='completed', provenance=provenance, parity_passed=not violations,
            sequence_numeric_parity_passed=all(row['sequence']['passed'] for row in comparisons),
            teacher_numeric_logit_parity_passed=all(row['teacher_logits']['passed'] for row in comparisons),
            teacher_argmax_identity=all(row['teacher_class_differences'] == 0 for row in comparisons),
            guard_alias_integrity=all(row['guarded_alias_matches'] for row in comparisons),
            composite_parity_interpretation='Numeric tolerance, argmax identity, guard alias and cache assumptions are separate checks; a within-tolerance near-zero class flip alone is not a large adapter-error claim.',
            violation_count=len(violations), summaries=grouped, teacher_state_sha256=teacher_before,
            teacher_manifest_schema=teacher_manifest['schema'], immutable_teacher_original_features_caches=True,
            unique_VALID_targets_checked=374, registered_comparison_rows=768,
            limitations=['All previous negative outcomes known; this is a post-hoc implementation diagnostic.',
                'Canonical/affine agreement cannot prove efficacy, novelty, transfer or human verification.',
                'Repeated memberships/seeds/contexts are correlated; no IID occurrence uncertainty.',
                'Official-victim endpoint values are saved only; logit scales are separate and uncalibrated.',
                'No affine-path model replay; canonical fresh own-teacher logits are compared to immutable stored affine logits.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__, error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        result.update(counters=counters, completed_comparison_rows=len(comparisons), completed_endpoint_joins=len(joins),
            violation_count=len(violations), input_files_hashed=len(receipts),
            process_cpu_seconds=time.process_time()-cpu_started, wall_seconds=time.monotonic()-started,
            GPU_hours=0, **runtime_metadata)
        write_json(output/'violations.json', violations)
        write_json(output/'input_hashes.json', receipts)
        output_hashes = {name:sha256(output/name) for name in ('registered.json', 'original_topology_and_cache_checks.json',
            'comparisons.jsonl', 'endpoint_joins.jsonl', 'violations.json', 'input_hashes.json') if (output/name).is_file()}
        result['output_hashes'] = output_hashes
        write_json(output/'result.json', result)
        print('FEATURE_PARITY_DIAGNOSTIC_END', json.dumps(result, allow_nan=False), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
