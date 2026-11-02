"""Bridge retained validation attacks to 32 label-free LLM evidence cards.

No attacker is initialized, no API is called, and no test prediction is made.
All ML imports and execution are inside main after a committed-source guard.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import traceback

from course.fusion_pilot.self_evidence import (
    ROOT, Limits, extract_current_context, load_fixed_data, load_graph_branch,
    load_self_checkpoint, private_output, sha256, write_json,
)
from course.llm_fusion.contract import BranchEvidence, EvidenceCard, GraphSummary, public_payload

ARMS = ('off', 'always', 'gated')
GROUPS = (1185, 1267, 1285, 1415)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def check_committed_source(expected_commit):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    if actual != expected_commit:
        raise RuntimeError('Bridge HEAD does not match the explicitly frozen source commit.')
    checked = {}
    for relative in ('course/fusion_pilot/extract_attacked.py',
                     'course/fusion_pilot/self_evidence.py', 'course/llm_fusion/contract.py',
                     'models/victim_model.py', 'utils/utils.py'):
        committed = subprocess.check_output(['git', 'show', actual + ':' + relative], cwd=str(ROOT))
        # Compare logical UTF-8 text because Windows checkouts can use CRLF.
        current = (ROOT / relative).read_bytes().replace(b'\r\n', b'\n')
        if current != committed.replace(b'\r\n', b'\n'):
            raise RuntimeError('Uncommitted bridge dependency: ' + relative)
        checked[relative] = sha256(ROOT / relative)
    return actual, checked


def initialize_train_masks(data):
    import torch
    graph = data['graph']
    known = torch.zeros(graph.num_nodes(), dtype=torch.long)
    known[torch.as_tensor(data['idx_train'], dtype=torch.long)] = 1
    labels = graph.ndata['y'].long().cpu()
    graph.ndata['train_neg'] = (1 - labels) * known
    graph.ndata['train_pos'] = labels * known
    graph.ndata['train_masked'] = 1 - known


def adjacency_without_self(graph):
    adj = graph.adj_external(scipy_fmt='csr').copy()
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1
    adj.sort_indices()
    return adj


def canonical_sequences(graph, node_ids, adjacency=None):
    """Exact 7-token unweighted set semantics; no clean-delta approximation.

    Tokens: center, hop1 benign/fraud/masked, hop2 benign/fraud/masked.
    Both hop sets exclude center; center is added to each masked group exactly
    once, matching DataHandler.group_aggregation. Two-hop paths are Boolean.
    """
    import torch
    adj = adjacency_without_self(graph) if adjacency is None else adjacency
    features = graph.ndata['x'].detach().cpu()
    masks = [graph.ndata[name].detach().cpu().bool()
             for name in ('train_neg', 'train_pos', 'train_masked')]
    if not torch.all(sum(mask.long() for mask in masks) == 1):
        raise RuntimeError('Training-label masks do not form exactly one valid group per node.')
    result = []
    for node in node_ids:
        node = int(node)
        first = set(int(n) for n in adj.indices[adj.indptr[node]:adj.indptr[node+1]])
        first.discard(node)
        second = set()
        for neighbor in first:
            second.update(int(n) for n in adj.indices[adj.indptr[neighbor]:adj.indptr[neighbor+1]])
        second.discard(node)
        tokens = [features[node]]
        for neighbors in (first, second):
            ids = torch.tensor(sorted(neighbors), dtype=torch.long)
            for index, mask in enumerate(masks):
                selected = ids[mask[ids]]
                if index == 2:
                    selected = torch.cat((selected, torch.tensor([node], dtype=torch.long)))
                tokens.append(features[selected].mean(dim=0) if len(selected)
                              else torch.zeros_like(features[node]))
        result.append(torch.stack(tokens))
    sequences = torch.stack(result)
    if not torch.isfinite(sequences).all():
        raise RuntimeError('Nonfinite canonical GAGA sequences.')
    return sequences


def reconstruct(data, payload, expected_group, expected_arm, expected_targets):
    import dgl
    import torch
    base = data['graph']
    total = base.num_nodes()
    if int(payload.get('original_num_nodes', payload.get('N', -1))) != total:
        raise RuntimeError('Attack export original graph size mismatch.')
    if payload.get('target_group') != expected_group or payload.get('arm') != expected_arm:
        raise RuntimeError('Attack export group/arm mismatch.')
    feature = payload['x_atk'].detach().cpu()
    adjacency = payload['adj_attack'].detach().cpu()
    targets = payload['target_ids'].detach().cpu().long().reshape(-1)
    candidates = payload['candidate_ids'].detach().cpu().long().reshape(-1)
    if targets.tolist() != [int(n) for n in expected_targets]:
        raise RuntimeError('Exported target order/membership differs from frozen manifest.')
    if feature.shape != (1, base.ndata['x'].shape[1]) or not torch.isfinite(feature).all():
        raise RuntimeError('Exported feature is nonfinite or violates the one-node budget.')
    if len(set(candidates.tolist())) != len(candidates) or set(targets.tolist()) & set(candidates.tolist()):
        raise RuntimeError('Duplicate or overlapping candidates.')
    if any(n < 0 or n >= total for n in candidates.tolist() + targets.tolist()):
        raise RuntimeError('Illegal original-node index in export.')
    if adjacency.shape != (1, len(targets) + len(candidates) + 1):
        raise RuntimeError('Exported adjacency shape mismatch.')
    if not torch.all((adjacency == 0) | (adjacency == 1)) or int(adjacency.count_nonzero()) != 1:
        raise RuntimeError('Exported attack must contain exactly one binary edge.')
    src, dst = adjacency.nonzero(as_tuple=True)
    if int(src[0]) != 0 or int(dst[0]) >= len(targets):
        raise RuntimeError('One-edge attack does not satisfy mandatory target attachment.')
    if int(payload.get('node_budget', -1)) != 1 or int(payload.get('edge_budget', -1)) != 1:
        raise RuntimeError('Exported budget metadata mismatch.')
    added = dict(x=feature, train_neg=torch.zeros(1, dtype=base.ndata['train_neg'].dtype),
                 train_pos=torch.zeros(1, dtype=base.ndata['train_pos'].dtype),
                 train_masked=torch.ones(1, dtype=base.ndata['train_masked'].dtype))
    graph = dgl.add_nodes(base, 1, added)
    indices = torch.cat((targets, candidates, torch.tensor([total], dtype=torch.long)))
    endpoint = int(indices[dst[0]])
    graph.add_edges([total, endpoint], [endpoint, total])
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    if graph.num_nodes() != total + 1 or graph.num_edges() != base.num_edges() + 3:
        raise RuntimeError('Reconstructed physical node/edge counts violate frozen budgets.')
    if not torch.equal(graph.ndata['x'][:total], base.ndata['x']) or not torch.equal(graph.ndata['x'][total:], feature):
        raise RuntimeError('Reconstruction changed original/generated features.')
    if set(graph.successors(total).tolist()) != {total, endpoint}:
        raise RuntimeError('Reconstructed injected incidence mismatch.')
    return graph, dict(node_budget=1, nonself_undirected_edges=1,
                       selected_endpoint_private=endpoint)


def local_exports(root):
    rows = []
    for arm in ARMS:
        folder = pathlib.Path(root) / (arm + '_seed3')
        summary_path = folder / 'summary.json'
        if not summary_path.is_file():
            summary_path = folder / 'result.json'
        summary = json.loads(summary_path.read_text())
        if summary.get('status') != 'completed':
            raise RuntimeError('Alignment arm did not complete: ' + arm)
        index_path = folder / 'selected_attack_exports.json'
        selected = json.loads(index_path.read_text())
        if len(selected) != 4 or {int(row['target_group']) for row in selected} != set(GROUPS):
            raise RuntimeError('Missing or extra selected raw exports in ' + arm)
        for row in selected:
            path = (folder / row['path']).resolve()
            if folder.resolve() not in path.parents or sha256(path) != row['sha256']:
                raise RuntimeError('Attack export path/digest mismatch.')
            rows.append(dict(arm=arm, group=int(row['target_group']), path=path,
                             sha256=row['sha256'], epoch=int(row['epoch']),
                             index_sha256=sha256(index_path), summary_sha256=sha256(summary_path)))
    return rows


def neighbor_similarity(graph, node):
    import torch
    neighbors = graph.predecessors(int(node))
    neighbors = neighbors[neighbors != int(node)]
    degree = int(len(neighbors))
    if not degree:
        return degree, None, None
    own = graph.ndata['x'][int(node)].unsqueeze(0)
    values = torch.nn.functional.cosine_similarity(own, graph.ndata['x'][neighbors], dim=1).clamp(-1, 1)
    return degree, float(values.mean()), float(values.std(unbiased=False)) if degree > 1 else 0.0


def export_meta_cards(self_run, data, manifest, registration, output, limits):
    """Same-information neural-gate training cards; no additional model call."""
    raw_path, label_path = self_run / 'records.jsonl', self_run / 'labels_private.jsonl'
    completed = json.loads((self_run / 'summary.json').read_text())
    if sha256(raw_path) != completed['records_sha256'] or sha256(label_path) != completed['private_supervision_sha256']:
        raise RuntimeError('Completed self-run evidence/label hashes differ.')
    truths = {}
    with label_path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['self_seed'] == 3 and row['cohort'] == 'meta_train':
                if row['case_id'] in truths:
                    raise RuntimeError('Duplicate meta supervision case.')
                truths[row['case_id']] = row
    expected_nodes = set(int(n) for n in manifest['meta_nodes'])
    seen_nodes, seen_cases = set(), set()
    context = digest(dict(graph_input=registration['graph_input_sha256'], population='original_train_meta',
        self_checkpoint=registration['self_checkpoint_sha256'], graph_checkpoint=registration['graph_checkpoint_sha256'],
        graph_summary_source=registration['source_sha256']['course/fusion_pilot/extract_attacked.py']))
    with (output / 'meta_cards.jsonl').open('x') as cards_file, (output / 'meta_labels_private.jsonl').open('x') as labels_file:
        with raw_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['self_seed'] != 3 or row['cohort'] != 'meta_train':
                    continue
                limits.check()
                truth = truths[row['case_id']]
                node = int(truth['node_id'])
                if node not in expected_nodes or node in seen_nodes or row['case_id'] in seen_cases:
                    raise RuntimeError('Meta membership/identity mismatch.')
                if int(data['graph'].ndata['y'][node]) != int(truth['label']):
                    raise RuntimeError('Meta private label mismatch.')
                degree, mean, std = neighbor_similarity(data['graph'], node)
                if degree != row['nonself_in_degree']:
                    raise RuntimeError('Meta current degree differs from saved clean evidence.')
                card = EvidenceCard(BranchEvidence(row['p_self'], 'uncalibrated'),
                    BranchEvidence(row['p_graph'], 'uncalibrated'), GraphSummary(degree, mean, std),
                    'registered_numeric_pilot')
                cards_file.write(canonical_json(dict(case_id=row['case_id'], context_sha256=context,
                                                     card=public_payload(card))) + '\n')
                labels_file.write(canonical_json(dict(case_id=row['case_id'], label=int(truth['label']))) + '\n')
                seen_nodes.add(node)
                seen_cases.add(row['case_id'])
    if seen_nodes != expected_nodes or seen_cases != set(truths):
        raise RuntimeError('Incomplete same-information meta cohort.')
    return dict(cases=len(seen_cases), additional_model_calls=0,
                graph_meta_evidence='IN_SAMPLE for original graph checkpoint; NOT out-of-fold',
                descriptions_or_label_demonstrations_for_api=False,
                cards_sha256=sha256(output / 'meta_cards.jsonl'),
                labels_private_sha256=sha256(output / 'meta_labels_private.jsonl'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-run', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--alignment-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-source-commit', required=True)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--max-wall-seconds', type=int, default=600)
    parser.add_argument('--max-graph-forwards', type=int, default=100)
    args = parser.parse_args()
    started = time.monotonic()
    output = private_output(args.output)
    limits = Limits(args.max_wall_seconds, args.max_graph_forwards)
    limits.started = started
    result = dict(status='started', api_calls=0, test_inference_calls=0)
    try:
        commit, source_hashes = check_committed_source(args.expected_source_commit)
        self_run = pathlib.Path(args.self_run)
        self_summary = json.loads((self_run / 'summary.json').read_text())
        if self_summary.get('status') != 'completed':
            raise RuntimeError('The registered self-model run did not complete.')
        manifest = json.loads(pathlib.Path(args.manifest).read_text())
        if sha256(args.manifest) != sha256(self_run / 'cohort_manifest_private.json'):
            raise RuntimeError('Manifest differs from the completed self-model run.')
        if [row['group_id'] for row in manifest['probes']] != list(GROUPS):
            raise RuntimeError('Probe group list differs from the frozen four-group cohort.')
        registered = json.loads((self_run / 'registered.json').read_text())
        expected_inputs = registered['checked_sha256']
        for relative, expected in expected_inputs.items():
            if relative.startswith(('Datasets/', 'models/')) and sha256(ROOT / relative) != expected:
                raise RuntimeError('Frozen data/model input changed: ' + relative)
        self_row = next(row for row in self_summary['self_models'] if row['seed'] == 3)
        checkpoint = self_run / self_row['checkpoint']
        if sha256(checkpoint) != self_row['checkpoint_sha256']:
            raise RuntimeError('Self-model checkpoint digest mismatch.')
        exports = local_exports(args.alignment_root)
        registration = dict(source_commit=commit, source_sha256=source_hashes,
            manifest_sha256=sha256(args.manifest), self_run_summary_sha256=sha256(self_run / 'summary.json'),
            self_checkpoint_sha256=self_row['checkpoint_sha256'], self_seed=3,
            graph_checkpoint_sha256=expected_inputs['models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle'],
            graph_input_sha256=expected_inputs['Datasets/GossipCop/GossipCop_0.05.bin'],
            clean_sequence_cache_sha256=expected_inputs['Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy'],
            exports=[dict(row, path=str(row['path'])) for row in exports],
            clean_sequence_check=dict(atol=1e-5, rtol=1e-5, before_any_model_forward=True),
            canonical_rule='unweighted nonself 1-hop; Boolean 2-hop union minus center; center added to masked at both hops',
            limits=dict(wall_seconds=args.max_wall_seconds, graph_forwards=args.max_graph_forwards),
            expected_cards=32, api_calls=0, test_inference_allowed=False)
        registration['source_sha256'] = source_hashes
        write_json(output / 'registered.json', registration)

        import numpy as np
        import torch
        torch.set_num_threads(8)
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Frozen CUDA device unavailable; refusing automatic substitution.')
        data = load_fixed_data()
        initialize_train_masks(data)
        valid = set(int(n) for n in data['idx_valid'])
        query_nodes = sorted(set(int(probe[k]) for probe in manifest['probes'] for k in ('fraud_node', 'benign_node')))
        if not set(query_nodes).issubset(valid):
            raise RuntimeError('A probe query is not in the original validation partition.')
        base = data['graph']
        clean_sequences = canonical_sequences(base, query_nodes)
        cache = np.load(ROOT / 'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy', mmap_mode='r')
        cached = torch.from_numpy(np.array(cache[query_nodes], copy=True)).float()
        difference = (clean_sequences - cached).abs()
        check = dict(max_absolute_difference=float(difference.max()),
                     allclose=bool(torch.allclose(clean_sequences, cached, atol=1e-5, rtol=1e-5)),
                     query_nodes=len(query_nodes), model_forwards_before_check=0,
                     tolerance_atol=1e-5, tolerance_rtol=1e-5)
        write_json(output / 'canonical_clean_cache_check.json', check)
        if not check['allclose']:
            raise RuntimeError('Canonical clean sequences disagree with the frozen cache; no model is called.')
        limits.check()
        meta = export_meta_cards(self_run, data, manifest, registration, output, limits)
        write_json(output / 'meta_export_summary.json', meta)
        self_model = load_self_checkpoint(checkpoint, device, self_row['checkpoint_sha256'])
        graph_model = load_graph_branch('GAGA', device)
        clean_by_node = {n: row for n, row in zip(query_nodes, clean_sequences)}
        by_group = {int(row['group_id']): row for row in manifest['probes']}
        contexts = [dict(arm='clean', group=group, path=None, sha256=None) for group in GROUPS] + exports
        cards, labels, private_rows, counts = [], [], [], []
        for item in contexts:
            limits.check()
            group, arm = item['group'], item['arm']
            probe = by_group[group]
            nodes = [int(probe['fraud_node']), int(probe['benign_node'])]
            if [int(base.ndata['y'][n]) for n in nodes] != [1, 0]:
                raise RuntimeError('Frozen probe pair is not one fraud and one benign node.')
            if arm == 'clean':
                graph = base
                sequences = torch.stack([clean_by_node[n] for n in nodes])
                size = dict(node_budget=0, nonself_undirected_edges=0)
            else:
                payload = torch.load(item['path'], map_location='cpu')
                graph, size = reconstruct(data, payload, group, arm, probe['target_nodes'])
                if int(payload['epoch']) != item['epoch']:
                    raise RuntimeError('Export epoch differs from the selected index.')
                sequences = canonical_sequences(graph, nodes)
            context = digest(dict(graph_input=registration['graph_input_sha256'],
                export_sha256=item['sha256'], group=group, query_nodes=nodes,
                self_checkpoint=registration['self_checkpoint_sha256'],
                graph_checkpoint=registration['graph_checkpoint_sha256'],
                canonical_source=source_hashes['course/fusion_pilot/extract_attacked.py']))
            rows, truths = extract_current_context(data, self_model, 'GAGA', graph_model,
                graph, nodes, device, limits, context, 3, 'probe',
                aligned_sequences=sequences, sequence_context_id=context)
            for row, truth in zip(rows, truths):
                node = int(truth['node_id'])
                degree, similarity_mean, similarity_std = neighbor_similarity(graph, node)
                if degree != row['nonself_in_degree']:
                    raise RuntimeError('Degree summaries disagree.')
                card = EvidenceCard(BranchEvidence(row['p_self'], 'uncalibrated'),
                    BranchEvidence(row['p_graph'], 'uncalibrated'),
                    GraphSummary(degree, similarity_mean, similarity_std), 'registered_numeric_pilot')
                cards.append(dict(case_id=row['case_id'], context_sha256=context, card=public_payload(card)))
                labels.append(dict(case_id=row['case_id'], label=int(truth['label'])))
                private_rows.append(dict(case_id=row['case_id'], context_sha256=context,
                    group_id=group, arm=arm, node_id=node, self_seed=3,
                    label=int(truth['label']), export_sha256=item['sha256'],
                    p_self=row['p_self'], p_graph=row['p_graph'], **size))
            counts.append(dict(arm=arm, group_id=group, cards=len(rows), **size))
            # Retain incremental private progress, including a later failed attempt.
            write_json(output / ('context_%02d.json' % len(counts)), counts[-1])
        if len(cards) != 32 or len({row['case_id'] for row in cards}) != 32:
            raise RuntimeError('The complete fixed 32-case cohort was not produced.')
        if sum(row['label'] for row in labels) != 16 or len(labels) != 32:
            raise RuntimeError('The full case denominator/class balance changed.')
        for name, rows in (('api_cards.jsonl', cards), ('labels_private.jsonl', labels),
                           ('provenance_private.jsonl', private_rows)):
            with (output / name).open('x') as stream:
                for row in rows:
                    stream.write(canonical_json(row) + '\n')
        result.update(status='completed', cards=32, clean_cards=8, attacked_cards=24,
            label_counts=[16, 16], contexts=counts, clean_sequence_check=check,
            same_information_meta=meta,
            api_cards_sha256=sha256(output / 'api_cards.jsonl'),
            labels_private_sha256=sha256(output / 'labels_private.jsonl'),
            provenance_private_sha256=sha256(output / 'provenance_private.jsonl'),
            graph_branch_meta_status='NOT out-of-fold; API probe is original validation only',
            calibration='uncalibrated', training_runs=0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0)
    except BaseException:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        result['counts_and_elapsed'] = limits.report()
        result['process_elapsed_seconds'] = time.monotonic() - started
        result['gpu_wall_hours'] = result['process_elapsed_seconds'] / 3600 if args.device.startswith('cuda') else 0
        write_json(output / 'summary.json', result)


if __name__ == '__main__':
    main()
