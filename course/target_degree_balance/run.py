"""Seal all structural variants before any classifier construction or query.

This post-hoc exploratory study changes target attachments only. It preserves
saved injected features, target/candidate order, all non-target destinations,
row target counts, row totals, physical injected degrees and exact edge budgets.
Balancing a structural potential does not guarantee improved attack efficacy.
"""
import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = 'monti_target_degree_balance_v1'
ARMS = ('baseline', 'balanced_target', 'matched_random')
SEEDS = (3, 4, 5)


def sha256(path):
    result = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def checked(reference, receipts):
    relative = pathlib.Path(reference['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Only repository-relative frozen inputs are allowed.')
    path = (ROOT/relative).resolve()
    if ROOT not in path.parents or not path.is_file() or sha256(path) != reference['sha256']:
        raise RuntimeError('A frozen input is missing, unsafe or changed: '+reference['path'])
    receipts[reference['path']] = reference['sha256']
    return path


def index_jsonl(path, groups):
    indexed = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():continue
            row = json.loads(line);group = int(row['group'])
            if group in indexed:raise RuntimeError('Duplicate frozen group row.')
            indexed[group] = row
    if len(indexed) != 128 or set(indexed) != set(groups):
        raise RuntimeError('All128 original VALID cases must remain in this study.')
    return indexed


def validate_registration(path, receipts):
    p = read_json(path)
    if (p.get('schema') != SCHEMA or p.get('status') != 'frozen'
            or p.get('post_hoc') is not True or p.get('owner_execution_authorized') is not True):
        raise RuntimeError('A frozen owner-authorized post-hoc balancing registration is required.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, universal_newlines=True).strip():
        raise RuntimeError('Commit tracked source before the registered run.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, universal_newlines=True).strip()
    if head != p['source_commit']:raise RuntimeError('Execution HEAD differs from the new registration.')
    required = {file.relative_to(ROOT).as_posix() for file in pathlib.Path(__file__).parent.glob('*.py')}
    if not required <= set(p['source_files']):raise RuntimeError('New balancing package source freeze is incomplete.')
    for name, wanted in p['source_files'].items():
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', name], cwd=ROOT, stderr=subprocess.STDOUT)
        checked(dict(path=name, sha256=wanted), receipts)
    for reference in p['inputs']:checked(reference, receipts)
    if (tuple(p['seeds']) != SEEDS or tuple(p['arms']) != ARMS or p['deadline_seconds'] != 5400
            or p['cpu_threads'] != 8 or len(p['valid_groups']) != 128 or len(set(p['valid_groups'])) != 128
            or set(p['runs']) != {'3', '4', '5'}):
        raise RuntimeError('Frozen arms/cohort/resource bounds differ.')
    old_path = checked(p['base_experiment_ref'], receipts)
    old = read_json(old_path)
    if (tuple(old['seeds']) != SEEDS or old['valid_groups'] != p['valid_groups']
            or old['final_test_allowed'] is not False or old['execution_device'] != 'cpu'):
        raise RuntimeError('Saved CPU baseline experiment identity differs.')
    for name, wanted in old['source_files'].items():
        if p['source_files'].get(name) != wanted:
            raise RuntimeError('Original scientific implementation changed: '+name)
    for reference in old['inputs']+old.get('retained_inputs', []):checked(reference, receipts)
    parity_path = checked(p['parity_result_ref'], receipts)
    parity = read_json(parity_path)
    if (parity['status'] != 'completed' or parity['parity_passed'] is not True
            or parity['violation_count'] != 0 or parity['completed_comparison_rows'] != 768
            or parity['sequence_numeric_parity_passed'] is not True
            or parity['teacher_numeric_logit_parity_passed'] is not True
            or parity['teacher_argmax_identity'] is not True):
        raise RuntimeError('The completed all-case plumbing parity diagnostic did not pass.')
    parity_registered_path = parity_path.parent/'registered.json'
    parity_registered = read_json(checked(dict(path=parity_registered_path.relative_to(ROOT).as_posix(),
        sha256=parity['output_hashes']['registered.json']), receipts))
    if parity_registered['base_experiment_ref'] != p['base_experiment_ref']:
        raise RuntimeError('Passing parity diagnostic belongs to a different baseline experiment.')
    # Parity is an implementation rationale, never a model/arm selection score.
    runs = {}
    for seed in SEEDS:
        spec = p['runs'][str(seed)]
        result_path = checked(spec['result_ref'], receipts)
        result = read_json(result_path)
        rows_path = checked(spec['groups_ref'], receipts)
        if (result['status'] != 'completed' or result['seed'] != seed or result['groups'] != 128
                or result['source_commit'] != old['source_commit']
                or result['protocol_sha256'] != p['base_experiment_ref']['sha256']
                or result['groups_jsonl_sha256'] != sha256(rows_path)):
            raise RuntimeError('A retained source-capture run is incomplete or has different provenance.')
        rows = index_jsonl(rows_path, p['valid_groups'])
        captures = {int(reference['group']):reference for reference in spec['captures']}
        originals = {int(reference['group']):reference for reference in result['exports']}
        if (len(spec['captures']) != 128 or len(result['exports']) != 128
                or set(captures) != set(rows) or set(originals) != set(rows)):
            raise RuntimeError('Freeze all128 saved captures for every seed.')
        paths = {}
        for group, reference in captures.items():
            capture_path = checked(reference, receipts)
            if (originals[group] != rows[group]['graph_capture']
                    or reference['sha256'] != originals[group]['sha256']
                    or capture_path != (result_path.parent/originals[group]['path']).resolve()):
                raise RuntimeError('A frozen capture does not match its original result and group row.')
            paths[group] = capture_path
        runs[seed] = dict(rows=rows, captures=paths)
    return p, old, runs, dict(source_commit=head, protocol_sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True);parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:
        raise RuntimeError('Private outputs must remain below ignored runs/.')
    output.mkdir(parents=True, exist_ok=False);(output/'geometry').mkdir()
    started, cpu_started = time.monotonic(), time.process_time()
    counts = dict(source_capture_tensor_loads=0, source_captures_prepared=0,
        geometry_graph_reconstructions=0, canonical_sequence_updates=0, first_hop_invariant_checks=0,
        prepared_artifact_tensor_loads=0, evaluation_graph_reconstructions=0,
        original_gcn_load_dispatches=0, original_gcn_models_loaded=0,
        official_gaga_load_dispatches=0, official_gaga_models_loaded=0,
        original_gcn_forward_dispatches=0, original_gcn_completed_forwards=0,
        official_gaga_forward_dispatches=0, official_gaga_completed_forwards=0,
        original_gcn_clean_forwards=0, official_gaga_clean_forwards=0,
        original_gcn_attacked_forwards=0, official_gaga_attacked_forwards=0,
        fresh_teacher_model_constructions=0, fresh_teacher_forwards=0,
        generator_model_constructions=0, generator_forwards=0, optimizer_steps=0,
        autograd_calls=0, API_calls=0, TEST_queries=0)
    runtime = dict(cuda_visibility=os.environ.get('CUDA_VISIBLE_DEVICES'), torch_cuda_available=None,
                   torch_visible_gpu_count=None, native_visible_gpu_count=None)
    result = dict(status='failed', counters=counts, classifier_loading_after_geometry_seal=False,
                  immutable_content_hashes=dict(before=None, after=None))
    receipts, prepared = {}, []
    def stop(signum, frame):raise TimeoutError('Frozen5400second CPU balancing deadline reached.')
    signal.signal(signal.SIGALRM, stop);signal.signal(signal.SIGTERM, stop);signal.alarm(5400)
    try:
        protocol_path = pathlib.Path(args.protocol).resolve()
        p, old, runs, provenance = validate_registration(protocol_path, receipts)
        if runtime['cuda_visibility'] != '':raise RuntimeError('CPU study requires explicitly empty CUDA visibility.')
        write_json(output/'registered.json', dict(provenance=provenance, own_source_sha256=sha256(__file__),
            base_experiment_ref=p['base_experiment_ref'], parity_result_ref=p['parity_result_ref'],
            source_and_input_hashes=receipts, source_captures=384, arms=list(ARMS),
            model_load_gate='All384sourcecaptures/1152variants/1152canonical sequences and exact first-hop invariants sealed first',
            matching='Random matches balanced final per-row turnover k from original ones/zeros; no model output enters geometry'))
        import dgl
        import numpy as np
        import torch
        from utils.utils import get_n_ids, set_seed
        from course.dual_surrogate.runtime import load_data, load_original_view, load_classifier, model_hash, gcn_logits, checked_logits, transitions
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.source_role_fix.saved_teacher_diagnostic import reconstruct
        from course.feature_polish.refine import tensor_sha256
        from course.target_degree_balance.balance_target import balance_targets
        runtime['torch_cuda_available'] = bool(torch.cuda.is_available())
        runtime['torch_visible_gpu_count'] = int(torch.cuda.device_count())
        runtime['native_visible_gpu_count'] = int(torch._C._cuda_getDeviceCount())
        if runtime['torch_cuda_available'] or runtime['torch_visible_gpu_count'] or runtime['native_visible_gpu_count']:
            raise RuntimeError('Both Torch/native visible GPU counts must be zero.')
        torch.set_num_threads(8);set_seed(20261104);dgl.seed(20261104);dgl.random.seed(20261104)
        data = load_data(old, 20261104);view = load_original_view(data)
        if data.device.type != 'cpu':raise RuntimeError('Structural study data must remain on CPU.')
        original_count = data.graph.num_nodes()
        immutable_before = dict(features=tensor_sha256(data.features), feat_sum=tensor_sha256(view.feat_sum), degrees=tensor_sha256(view.d))
        result['immutable_content_hashes']['before'] = dict(immutable_before)
        nonidentity = {str(seed):0 for seed in SEEDS}
        with torch.no_grad(), (output/'geometry.jsonl').open('x', encoding='utf-8', newline='\n') as geometry_stream:
            for seed in SEEDS:
                for group in p['valid_groups']:
                    saved = runs[seed]['rows'][group]
                    path = runs[seed]['captures'][group]
                    if sha256(path) != saved['graph_capture']['sha256']:raise RuntimeError('Saved capture changed after source freeze.')
                    counts['source_capture_tensor_loads'] += 1
                    payload = torch.load(path, map_location='cpu')
                    if set(payload) != {'capture', 'features'}:raise RuntimeError('Source capture schema differs.')
                    capture = payload['capture'];adjacency = capture['adj_attack'];x0 = capture['x_atk']
                    if (not bool(torch.isfinite(adjacency).all())
                            or not bool(((adjacency == 0)|(adjacency == 1)).all())):
                        raise RuntimeError('Validate binary source adjacency before integer conversion.')
                    if tensor_sha256(adjacency) != saved['adjacency_content_sha256']:
                        raise RuntimeError('Source adjacency content differs from its original frozen row receipt.')
                    if (not torch.equal(payload['features']['baseline'], x0)
                            or tensor_sha256(x0) != saved['feature_statistics']['baseline']['content_sha256']):
                        raise RuntimeError('Original x0 differs from its source baseline branch/hash.')
                    targets = [int(value) for value in capture['target_ids'].tolist()]
                    candidates = [int(value) for value in capture['candidate_ids'].tolist()]
                    injected_ids = list(range(original_count, original_count+len(x0)))
                    if targets != saved['node_ids']:raise RuntimeError('Source target order changed.')
                    plain = adjacency.to(torch.int64).cpu().tolist()
                    proposal = balance_targets(plain, targets, candidates, injected_ids, seed=seed, group_key=group)
                    if tuple(proposal['arms']) != ARMS:raise RuntimeError('Exact three-arm core output differs.')
                    t, n = len(targets), len(injected_ids)
                    variants = {arm:torch.tensor(proposal['arms'][arm], dtype=adjacency.dtype) for arm in ARMS}
                    if not torch.equal(variants['baseline'], adjacency):raise RuntimeError('The core baseline changed the original adjacency.')
                    sequences, graph_metadata = {}, {}
                    baseline_degrees, baseline_sequence = None, None
                    identity_hashes = dict(features=tensor_sha256(x0), targets=tensor_sha256(capture['target_ids']),
                        candidates=tensor_sha256(capture['candidate_ids']))
                    for arm in ARMS:
                        variant = variants[arm]
                        if (variant.shape != adjacency.shape or not torch.equal(variant[:, t:], adjacency[:, t:])
                                or not torch.equal(variant.sum(1), adjacency.sum(1))
                                or not torch.equal(variant[:, :t].sum(1), adjacency[:, :t].sum(1))
                                or int(variant.sum()) != saved['edge_budget']):
                            raise RuntimeError('Target-only balancing changed a frozen role/count/edge constraint.')
                        for row in range(n):
                            removed = int(((adjacency[row, :t] == 1)&(variant[row, :t] == 0)).sum())
                            added = int(((adjacency[row, :t] == 0)&(variant[row, :t] == 1)).sum())
                            expected_k = 0 if arm == 'baseline' else proposal['row_plan'][row]['final_k']
                            if removed != added or removed != expected_k:
                                raise RuntimeError('Balanced/random final per-row turnover k is not matched.')
                        variant_capture = dict(capture, adj_attack=variant)
                        graph_record = dict(saved, actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                        clean_identity = dict(group=group, node_ids=saved['node_ids'], labels=saved['labels'])
                        graph, ids, candidate_tensor, injected = reconstruct(data, variant_capture, graph_record, clean_identity)
                        counts['geometry_graph_reconstructions'] += 1
                        physical_degrees = (graph.in_degrees(injected)-1).to(torch.int64)
                        if not torch.equal(physical_degrees, torch.tensor(proposal['physical_injected_degrees'], dtype=torch.int64)):
                            raise RuntimeError('Native reconstructed physical injected degrees differ from the core contract.')
                        if arm == 'baseline':baseline_degrees = physical_degrees
                        elif not torch.equal(physical_degrees, baseline_degrees):raise RuntimeError('A variant changes physical injected degrees.')
                        sequence = updated_sequences(view, graph, ids, candidate_tensor, injected, training=False)
                        counts['canonical_sequence_updates'] += 1
                        if arm == 'baseline':baseline_sequence = sequence
                        if not torch.equal(sequence[:, [0, 1, 2]], baseline_sequence[:, [0, 1, 2]]):
                            raise RuntimeError('SELF or first-hop known-class tokens changed before the model gate.')
                        counts['first_hop_invariant_checks'] += 1
                        sequences[arm] = sequence.detach().cpu()
                        graph_metadata[arm] = dict(adjacency_content_sha256=tensor_sha256(variant),
                            canonical_sequence_content_sha256=tensor_sha256(sequence),
                            actual_injected_nodes=len(injected),
                            actual_injected_nonself_undirected_edges=(graph.num_edges()-data.graph.num_edges()-len(injected))//2,
                            physical_injected_degrees=physical_degrees.tolist(),
                            first_hop_tokens0_1_2_exact=True)
                        del graph
                    if not torch.equal(variants['balanced_target'], adjacency):nonidentity[str(seed)] += 1
                    if (identity_hashes != dict(features=tensor_sha256(capture['x_atk']),
                            targets=tensor_sha256(capture['target_ids']), candidates=tensor_sha256(capture['candidate_ids']))
                            or not torch.equal(capture['x_atk'], x0)):
                        raise RuntimeError('Structural preparation changed frozen features or ID order.')
                    artifact_path = output/'geometry'/('seed%d_group%d.pt'%(seed, group))
                    artifact = dict(capture={key:value.detach().cpu() for key,value in capture.items()},
                        adjacencies={arm:value.detach().cpu() for arm,value in variants.items()}, canonical_sequences=sequences)
                    with artifact_path.open('xb') as handle:torch.save(artifact, handle)
                    record = dict(seed=seed, group=group, node_ids=saved['node_ids'], labels=saved['labels'], targets=t,
                        node_budget=saved['node_budget'], edge_budget=saved['edge_budget'],
                        source_capture_sha256=saved['graph_capture']['sha256'], immutable_capture_content_hashes=identity_hashes,
                        artifact=dict(path=artifact_path.relative_to(output).as_posix(), sha256=sha256(artifact_path)),
                        variants=graph_metadata, geometry={key:value for key,value in proposal.items() if key != 'arms'})
                    prepared.append(record);counts['source_captures_prepared'] += 1
                    geometry_stream.write(json.dumps(record, allow_nan=False)+'\n');geometry_stream.flush()
                    if len(prepared)%32 == 0:print('TARGET_BALANCE_GEOMETRY', len(prepared), flush=True)
                    del payload, capture, artifact, sequences, variants
        if (counts['source_captures_prepared'] != 384 or counts['geometry_graph_reconstructions'] != 1152
                or counts['canonical_sequence_updates'] != 1152 or counts['first_hop_invariant_checks'] != 1152
                or any(counts[name] for name in ('original_gcn_load_dispatches','official_gaga_load_dispatches',
                    'original_gcn_forward_dispatches','official_gaga_forward_dispatches'))):
            raise RuntimeError('The complete geometry/first-hop seal must precede every classifier construction.')
        geometry_totals = []
        for seed in SEEDS:
            selected = [row for row in prepared if row['seed'] == seed]
            target_occurrences = sum(row['targets'] for row in selected)
            unique_targets = len({node for row in selected for node in row['node_ids']})
            if len(selected) != 128 or target_occurrences != 1091 or unique_targets != 374:
                raise RuntimeError('Whole-seed geometry changed target/group denominators before models.')
            for arm in ARMS:
                nodes = sum(row['variants'][arm]['actual_injected_nodes'] for row in selected)
                edges = sum(row['variants'][arm]['actual_injected_nonself_undirected_edges'] for row in selected)
                if (nodes, edges) != (7776, 358491):
                    raise RuntimeError('Whole-seed physical resource totals changed before models.')
                geometry_totals.append(dict(seed=seed, arm=arm, groups=128, target_occurrences=1091,
                    unique_targets=374, summed_injected_nodes=nodes, summed_nonself_undirected_edges=edges))
        marker = dict(status='geometry_sealed_before_any_classifier', provenance=provenance,
            source_captures=384, variants=1152, canonical_sequences=1152,
            balanced_nonidentity_cases_by_seed=nonidentity, counters=dict(counts),
            geometry_jsonl_sha256=sha256(output/'geometry.jsonl'),
            artifacts=[dict(seed=row['seed'],group=row['group'],**row['artifact']) for row in prepared],
            original_content_hashes=immutable_before, whole_seed_geometry_totals=geometry_totals,
            structural_interpretation='Fixed-total target-degree convex potential optimum, not efficacy or a GCN class-protection guarantee.')
        marker_path = output/'geometry_sealed.json';write_json(marker_path, marker)
        result.update(provenance=provenance, geometry_marker_sha256=sha256(marker_path),
            balanced_nonidentity_cases_by_seed=nonidentity, all_geometry_prepared_before_classifier=True)
        if any(value == 0 for value in nonidentity.values()):
            result.update(status='completed_no_headroom', stop_reason='At least one complete seed has zero nonidentity balanced cases; no classifier is loaded or queried.')
        else:
            result['classifier_loading_after_geometry_seal'] = True
            counts['original_gcn_load_dispatches'] += 1
            gcn = load_classifier('gcn_base', data);counts['original_gcn_models_loaded'] += 1
            counts['official_gaga_load_dispatches'] += 1
            victim = load_classifier('gaga_victim', data);counts['official_gaga_models_loaded'] += 1
            model_before = dict(original_gcn=model_hash(gcn), official_gaga=model_hash(victim))
            result['immutable_content_hashes']['before'].update(model_before)
            for model in (gcn, victim):
                if model.training or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError('Both original evaluation classifiers must be frozen in eval mode.')
            def classify(name, graph=None, ids=None, sequence=None, clean=False):
                key = 'original_gcn' if name == 'gcn' else 'official_gaga'
                if counts[key+'_forward_dispatches'] >= 1280:raise RuntimeError('Frozen1280classifier forward cap reached.')
                counts[key+'_forward_dispatches'] += 1
                logits = (gcn_logits(gcn, graph, ids, training=False) if name == 'gcn'
                          else checked_logits(victim(sequence), len(ids)))
                counts[key+'_completed_forwards'] += 1
                counts[key+('_clean_forwards' if clean else '_attacked_forwards')] += 1
                return logits
            sequence_path = ROOT/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy'
            clean_sequence = np.load(sequence_path, allow_pickle=False, mmap_mode='r')
            clean_predictions = {}
            with torch.no_grad(), (output/'clean_predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
                for group in p['valid_groups']:
                    ids = torch.tensor(get_n_ids(data.target_assignment, group), device=data.device, dtype=torch.long)
                    ga = classify('gcn', graph=data.graph, ids=ids, clean=True)
                    batch = torch.from_numpy(np.array(clean_sequence[ids.cpu().numpy()], copy=True)).to(data.device)
                    gb = classify('gaga', ids=ids, sequence=batch, clean=True)
                    labels = data.labels[ids].cpu();clean_predictions[group] = (ga.argmax(1).cpu(), gb.argmax(1).cpu())
                    stream.write(json.dumps(dict(group=group,node_ids=ids.tolist(),labels=labels.tolist(),
                        gcn_logits=ga.tolist(),gaga_logits=gb.tolist()),allow_nan=False)+'\n');stream.flush()
            totals = {(arm,seed):{name:dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0)
                       for name in ('gcn','gaga')} for arm in ARMS for seed in SEEDS}
            streams = {(arm,seed):(output/('%s_seed%d_predictions.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n')
                       for arm in ARMS for seed in SEEDS}
            try:
                with torch.no_grad():
                    for record in prepared:
                        seed, group = record['seed'], record['group']
                        artifact_path = output/record['artifact']['path']
                        if sha256(artifact_path) != record['artifact']['sha256']:raise RuntimeError('A sealed structural artifact changed before inference.')
                        counts['prepared_artifact_tensor_loads'] += 1
                        payload = torch.load(artifact_path,map_location='cpu')
                        if set(payload) != {'capture','adjacencies','canonical_sequences'}:raise RuntimeError('Sealed artifact schema changed.')
                        capture = payload['capture']
                        for arm in ARMS:
                            adjacency = payload['adjacencies'][arm];sequence = payload['canonical_sequences'][arm]
                            if (tensor_sha256(adjacency) != record['variants'][arm]['adjacency_content_sha256']
                                    or tensor_sha256(sequence) != record['variants'][arm]['canonical_sequence_content_sha256']):
                                raise RuntimeError('Sealed structural adjacency or canonical sequence changed.')
                            for key, field in (('x_atk','features'),('target_ids','targets'),('candidate_ids','candidates')):
                                if tensor_sha256(capture[key]) != record['immutable_capture_content_hashes'][field]:
                                    raise RuntimeError('Saved x0 or target/candidate order changed after geometry seal.')
                            variant_capture = dict(capture,adj_attack=adjacency)
                            saved = runs[seed]['rows'][group]
                            graph_record = dict(saved,actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                            identity = dict(group=group,node_ids=saved['node_ids'],labels=saved['labels'])
                            graph,ids,candidates,injected = reconstruct(data,variant_capture,graph_record,identity)
                            counts['evaluation_graph_reconstructions'] += 1
                            ga = classify('gcn',graph=graph,ids=ids)
                            gb = classify('gaga',ids=ids,sequence=sequence.to(data.device))
                            labels = data.labels[ids].cpu()
                            row = dict(seed=seed,group=group,arm=arm,node_ids=ids.tolist(),labels=labels.tolist(),
                                gcn_logits=ga.tolist(),gaga_logits=gb.tolist(),node_budget=saved['node_budget'],edge_budget=saved['edge_budget'],
                                actual_injected_nodes=len(injected),actual_injected_edges=(graph.num_edges()-data.graph.num_edges()-len(injected))//2,
                                source_capture_sha256=record['source_capture_sha256'],prepared_artifact_sha256=record['artifact']['sha256'],
                                injected_feature_content_sha256=record['immutable_capture_content_hashes']['features'],
                                candidate_order_content_sha256=record['immutable_capture_content_hashes']['candidates'],
                                target_order_content_sha256=record['immutable_capture_content_hashes']['targets'],
                                variant_adjacency_content_sha256=record['variants'][arm]['adjacency_content_sha256'],
                                canonical_sequence_content_sha256=record['variants'][arm]['canonical_sequence_content_sha256'])
                            for index,name in enumerate(('gcn','gaga')):
                                predicted = (ga if name == 'gcn' else gb).argmax(1).cpu()
                                transition = transitions(labels,clean_predictions[group][index],predicted)
                                row[name+'_transitions'] = transition
                                for key,value in transition.items():totals[arm,seed][name][key] += value
                            streams[arm,seed].write(json.dumps(row,allow_nan=False)+'\n');streams[arm,seed].flush()
                            del graph
                        del payload
            finally:
                for stream in streams.values():stream.close()
            if (counts['prepared_artifact_tensor_loads'] != 384 or counts['evaluation_graph_reconstructions'] != 1152
                    or counts['original_gcn_completed_forwards'] != 1280 or counts['official_gaga_completed_forwards'] != 1280
                    or counts['original_gcn_clean_forwards'] != 128 or counts['official_gaga_clean_forwards'] != 128
                    or counts['original_gcn_attacked_forwards'] != 1152 or counts['official_gaga_attacked_forwards'] != 1152):
                raise RuntimeError('Full retained-cohort evaluation counts differ.')
            summaries = []
            for seed in SEEDS:
                for arm in ARMS:
                    if any(row['occurrences'] != 1091 for row in totals[arm,seed].values()):
                        raise RuntimeError('Complete occurrence denominator differs.')
                    summaries.append(dict(seed=seed,arm=arm,groups=128,unique_targets=374,totals=totals[arm,seed],
                        gcn_mr_percent=100*totals[arm,seed]['gcn']['attacked_wrong']/1091,
                        gaga_mr_percent=100*totals[arm,seed]['gaga']['attacked_wrong']/1091))
            result.update(status='completed',summaries=summaries,
                baseline_channel='Fresh CPU reconstruction and classifier inference, not old stored predictions',
                official_gaga_channel='All1152canonical sequences sealed before classifier loading',
                no_gcn_class_protection_assumed=True)
        immutable_after = dict(features=tensor_sha256(data.features),feat_sum=tensor_sha256(view.feat_sum),degrees=tensor_sha256(view.d))
        if result['status'] == 'completed':
            immutable_after.update(original_gcn=model_hash(gcn),official_gaga=model_hash(victim))
            if any(parameter.grad is not None or parameter.requires_grad for model in (gcn,victim) for parameter in model.parameters()):
                raise RuntimeError('An evaluation classifier gradient buffer or parameter flag changed.')
        result['immutable_content_hashes']['after'] = immutable_after
        if immutable_after != result['immutable_content_hashes']['before']:raise RuntimeError('Original models/features/caches changed.')
        result.update(immutable_content_hashes_retained=True,
            interpretation='Post-hoc structural hypothesis; fixed-total potential optimality is not attack efficacy, novelty or course approval.',
            final_TEST_used=False,experimental_API_calls=0)
    except BaseException as error:
        traceback.print_exc();result.update(status='failed',error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        write_json(output/'input_hashes.json',receipts)
        result.update(counters=counts,prepared_cases=len(prepared),process_cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.monotonic()-started,GPU_hours=0,**runtime)
        names=['registered.json','geometry.jsonl','geometry_sealed.json','clean_predictions.jsonl','input_hashes.json']
        names.extend('%s_seed%d_predictions.jsonl'%(arm,seed) for arm in ARMS for seed in SEEDS)
        result['output_hashes'] = {name:sha256(output/name) for name in names if (output/name).is_file()}
        write_json(output/'result.json',result)
        print('TARGET_DEGREE_BALANCE_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status'] not in ('completed','completed_no_headroom'):raise SystemExit(1)


if __name__ == '__main__':main()
