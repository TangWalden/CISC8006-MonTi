"""Bounded self-MLP training and numeric branch evidence; no API or test inference.

Dependencies are imported only inside execution functions. Nothing runs on import.
Raw manifests, labels, node identities and checkpoints must stay under runs/.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import time
import traceback
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPLIT_SEED = 20261103
SEEDS = (3, 4, 5)
SELF_SETTINGS = dict(hidden_sizes=[64, 32], max_epochs=30, patience=5,
                     batch_size=256, lr=0.001, weight_decay=0.0)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def stable_order(values, namespace):
    return sorted((int(x) for x in values), key=lambda x: (
        hashlib.sha256(('%s|%s|%s' % (SPLIT_SEED, namespace, x)).encode()).hexdigest(), x))


def private_output(path):
    path = pathlib.Path(path).resolve()
    if (ROOT / 'runs').resolve() not in path.parents:
        raise ValueError('Raw output must be in the ignored repository runs/ directory.')
    path.mkdir(parents=True, exist_ok=False)
    return path


class Limits:
    def __init__(self, seconds, graph_forwards):
        if not 0 < float(seconds) <= 3600 or not 0 < int(graph_forwards) <= 1000:
            raise ValueError('Explicit bounded wall/graph-forward limits required.')
        self.started = time.monotonic()
        self.seconds = float(seconds)
        self.maximum_graph_forwards = int(graph_forwards)
        self.graph_forwards = 0
        self.self_training_forwards = 0
        self.self_evaluation_forwards = 0
        self.optimizer_steps = 0

    def check(self):
        if time.monotonic() - self.started > self.seconds:
            raise RuntimeError('Frozen wall-clock limit reached; preserve this failed attempt.')

    def graph_call(self):
        self.check()
        if self.graph_forwards >= self.maximum_graph_forwards:
            raise RuntimeError('Frozen graph-forward limit reached.')
        self.graph_forwards += 1

    def report(self):
        return dict(elapsed_seconds=time.monotonic() - self.started,
                    graph_forwards=self.graph_forwards,
                    self_training_forwards=self.self_training_forwards,
                    self_evaluation_forwards=self.self_evaluation_forwards,
                    optimizer_steps=self.optimizer_steps)


def verify_registration(config_path, expected_sha):
    if sha256(config_path) != expected_sha:
        raise RuntimeError('Configuration does not match the externally frozen digest.')
    config = json.loads(pathlib.Path(config_path).read_text(encoding='utf-8'))
    if config.get('phase') != 'validation_pilot' or config.get('graph_model') not in ('GCN', 'GAGA'):
        raise ValueError('Only explicit validation_pilot and GCN/GAGA are supported.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), text=True).strip()
    if head != config['source_commit']:
        raise RuntimeError('HEAD differs from the frozen source commit.')
    expected = config['expected_sha256']
    mandatory = {'course/fusion_pilot/self_evidence.py', 'models/victim_model.py',
                 'modules/data_handler.py', 'utils/utils.py',
                 'Datasets/GossipCop/GossipCop_0.05.bin',
                 'Datasets/GossipCop/data_split/target_assign_matrix_0.npz',
                 'Datasets/GossipCop/data_split/train_idx_surro_0.npy',
                 'Datasets/GossipCop/data_split/valid_idx_surro_0.npy',
                 'Datasets/GossipCop/data_split/test_idx_surro_0.npy',
                 'Datasets/GossipCop/data_split/train_target_mask_0.npy',
                 'Datasets/GossipCop/data_split/valid_target_mask_0.npy',
                 'Datasets/GossipCop/data_split/test_target_mask_0.npy'}
    model = config['graph_model']
    role = 'victim_models' if model == 'GAGA' else 'surrogate_models'
    mandatory.update({'models/pretrained_models/configs/%s-gossipcop_0.05-0.json' % model,
                      'models/pretrained_models/%s/%s-gossipcop_0.05-0.pickle' % (role, model)})
    if model == 'GAGA':
        mandatory.add('Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy')
    if not mandatory.issubset(expected):
        raise ValueError('Missing frozen input/source digests: %s' % sorted(mandatory - set(expected)))
    checked = {}
    for relative, digest in sorted(expected.items()):
        path = (ROOT / relative).resolve()
        if ROOT not in path.parents:
            raise ValueError('Input escaped the repository.')
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError('Frozen input/source mismatch: ' + relative)
        checked[relative] = actual
    return config, checked


def load_fixed_data():
    import dgl
    import torch
    from utils.utils import load_data
    data = load_data('gossipcop_0.05', split=0, multi_target=True)
    graph = dgl.add_self_loop(dgl.remove_self_loop(data['graph']))
    graph.ndata['degree'] = graph.in_degrees().to(torch.float32) - 1
    data['graph'] = graph
    return data


def prepare_manifest(data, config=None):
    """Only frozen labels/splits/budgets determine selection; no predictor is called."""
    import torch
    from modules.data_handler import DataHandler
    labels = data['graph'].ndata['y'].long().cpu().numpy()
    if set(labels.tolist()) != {0, 1}:
        raise RuntimeError('This frozen pilot requires binary labels 0 and 1.')
    train = set(int(x) for x in data['idx_train'])
    valid = set(int(x) for x in data['idx_valid'])
    test = set(int(x) for x in data['idx_test'])
    if train & valid or train & test or valid & test:
        raise RuntimeError('Original node partitions overlap.')
    portions = {'base_fit_nodes': [], 'base_tune_nodes': [], 'meta_nodes': []}
    clean_valid = []
    for label in (0, 1):
        nodes = stable_order([n for n in train if labels[n] == label], 'train-class-%s' % label)
        fit_stop, tune_stop = int(0.8 * len(nodes)), int(0.8 * len(nodes)) + int(0.1 * len(nodes))
        portions['base_fit_nodes'] += nodes[:fit_stop]
        portions['base_tune_nodes'] += nodes[fit_stop:tune_stop]
        portions['meta_nodes'] += nodes[tune_stop:]
        candidates = stable_order([n for n in valid if labels[n] == label], 'clean-valid-class-%s' % label)
        if len(candidates) < 32:
            raise RuntimeError('Fewer than 32 original-validation nodes in a class.')
        clean_valid += candidates[:32]
    for name in portions:
        portions[name] = sorted(portions[name])
    graph = data['graph']
    holder = SimpleNamespace(graph=graph, target_assignment=data['target_assignment'],
                             rho=0.05, xi=0.1, device=torch.device('cpu'))
    holder.node_budgets = DataHandler.calculate_node_budgets(holder)
    edge_budgets = DataHandler.calculate_edge_budgets(holder)
    eligible = [int(g) for g in data['target_idx_valid']
                if int(holder.node_budgets[g]) == 1 and int(edge_budgets[g]) == 1]
    if len(eligible) < 4:
        raise RuntimeError('Fewer than four one-node/one-edge validation groups.')
    proposed_groups = None if config is None else config.get('probe_group_ids')
    if proposed_groups is None:
        selected = stable_order(eligible, 'probe-groups')[:4]
        probe_rule = 'First four eligible groups by frozen SHA256 order'
    else:
        selected = [int(g) for g in proposed_groups]
        if len(selected) != 4 or len(set(selected)) != 4 or not set(selected).issubset(eligible):
            raise RuntimeError('Explicit probe groups must be four distinct eligible validation groups.')
        probe_rule = 'Explicit root-frozen group list; not claimed as a fresh random sample'
    benign_pool = [n for n in valid if labels[n] == 0 and n not in set(clean_valid)]
    used_benign, probes = set(), []
    for group in selected:
        targets = data['target_assignment'][group].nonzero()[1].tolist()
        if not targets or not set(targets).issubset(valid) or any(labels[n] != 1 for n in targets):
            raise RuntimeError('Invalid target membership in selected group.')
        fraud_node = stable_order(targets, 'probe-fraud-%s' % group)[0]
        benign_node = next(n for n in stable_order(benign_pool, 'probe-benign-%s' % group)
                           if n not in used_benign)
        used_benign.add(benign_node)
        probes.append(dict(group_id=group, target_nodes=[int(n) for n in targets],
                           fraud_node=fraud_node, benign_node=benign_node,
                           node_budget=1, edge_budget=1))
    manifest = dict(schema='fusion_pilot_cohort_v1', split_seed=SPLIT_SEED,
                    dataset='gossipcop_0.05', split=0, self_seeds=list(SEEDS),
                    self_settings=SELF_SETTINGS, **portions)
    manifest.update(clean_valid_nodes=sorted(clean_valid), probes=probes,
                    cohort_rule='SHA256(seed|namespace|integer-ID), ascending digest then ID',
                    train_split_rule='Per class: floor(.8*n) fit, floor(.1*n) tune, remainder meta',
                    probe_group_rule=probe_rule,
                    probe_benign_rule='Unique across probe groups and disjoint from clean-valid cohort',
                    graph_meta_evidence='IN_SAMPLE: official graph checkpoint saw original training labels; NOT out-of-fold',
                    calibration='uncalibrated', test_inference_allowed=False,
                    budget_semantics='retained upstream incidence helper, not paper unique-neighbor formula')
    for name in ('base_fit_nodes', 'base_tune_nodes', 'meta_nodes', 'clean_valid_nodes'):
        manifest[name + '_label_counts'] = [int(sum(labels[n] == c for n in manifest[name])) for c in (0, 1)]
        if min(manifest[name + '_label_counts']) < 1:
            raise RuntimeError('A required cohort does not contain both classes.')
    return manifest


def self_logits(model, features):
    """Exact feature-only MLP layers, without fabricating DGL blocks."""
    value = features
    for layer in model.layers:
        value = model.activation(layer(value))
    return model.linear_layer(value)


def train_self(data, manifest, seed, device, output, limits):
    import torch
    from models.victim_model import MLP
    from utils.utils import set_seed
    set_seed(seed)
    settings = manifest['self_settings']
    model = MLP(data['graph'].ndata['x'].shape[1], settings['hidden_sizes'], num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    features, labels = data['graph'].ndata['x'], data['graph'].ndata['y'].long()
    fit = torch.tensor(manifest['base_fit_nodes'], dtype=torch.long)
    tune = torch.tensor(manifest['base_tune_nodes'], dtype=torch.long)
    best_loss, stale, best_state, best_epoch, history = float('inf'), 0, None, None, []
    for epoch in range(1, settings['max_epochs'] + 1):
        limits.check()
        model.train()
        ordering = fit[torch.randperm(len(fit))]
        loss_sum = 0.0
        for ids in ordering.split(settings['batch_size']):
            limits.check()
            optimizer.zero_grad()
            limits.self_training_forwards += 1
            logits = self_logits(model, features[ids.to(features.device)].to(device))
            loss = torch.nn.functional.cross_entropy(logits, labels[ids.to(labels.device)].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite self-model training loss.')
            loss.backward()
            optimizer.step()
            limits.optimizer_steps += 1
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        tune_loss = 0.0
        with torch.no_grad():
            for ids in tune.split(settings['batch_size']):
                limits.check()
                limits.self_evaluation_forwards += 1
                logits = self_logits(model, features[ids.to(features.device)].to(device))
                tune_loss += float(torch.nn.functional.cross_entropy(logits, labels[ids.to(labels.device)].to(device), reduction='sum'))
        tune_loss /= len(tune)
        if not math.isfinite(tune_loss):
            raise RuntimeError('Nonfinite tuning loss.')
        history.append(dict(epoch=epoch, fit_cross_entropy=loss_sum / len(fit), tune_cross_entropy=tune_loss))
        if tune_loss < best_loss:
            best_loss, stale, best_epoch = tune_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        write_json(output / ('seed%s_epoch%02d.json' % (seed, epoch)), history[-1])
        print('SELF_EPOCH', seed, epoch, json.dumps(history[-1]), flush=True)
        if stale >= settings['patience']:
            break
    model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    checkpoint = output / ('self_mlp_seed%s.pickle' % seed)
    torch.save(dict(state_dict=best_state, feature_dim=features.shape[1], settings=settings,
                    seed=seed, best_epoch=best_epoch, calibration='uncalibrated'), checkpoint)
    return model, dict(seed=seed, best_epoch=best_epoch, completed_epochs=len(history),
                       best_tune_cross_entropy=best_loss, checkpoint=checkpoint.name,
                       checkpoint_sha256=sha256(checkpoint), history=history)


def load_graph_branch(model_name, device):
    """Same classes/configurations as the official loader, with explicit map_location."""
    import torch
    from models.victim_model import GCN, GAGA
    setting = '%s-gossipcop_0.05-0' % model_name
    folder = ROOT / 'models/pretrained_models'
    config = json.loads((folder / 'configs' / (setting + '.json')).read_text())
    model = (GCN(768, config['emb_sizes'], num_classes=2) if model_name == 'GCN'
             else GAGA(768, config['emb_sizes'][0], num_classes=2, num_relations=1))
    role = 'surrogate_models' if model_name == 'GCN' else 'victim_models'
    model.load_state_dict(torch.load(folder / role / (setting + '.pickle'), map_location='cpu'))
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_self_checkpoint(path, device, expected_sha256):
    """Load only an explicitly registered self checkpoint from this runner."""
    import torch
    from models.victim_model import MLP
    if sha256(path) != expected_sha256:
        raise RuntimeError('Self checkpoint digest mismatch.')
    saved = torch.load(path, map_location='cpu')
    if saved['settings'] != SELF_SETTINGS or saved['seed'] not in SEEDS:
        raise RuntimeError('Unregistered self-model architecture or seed.')
    model = MLP(saved['feature_dim'], saved['settings']['hidden_sizes'], num_classes=2)
    model.load_state_dict(saved['state_dict'])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def graph_branch_logits(model_name, model, graph, node_ids, device, limits,
                        sequences=None, context_id=None, sequence_context_id=None):
    """One bounded graph call. Caller must supply current-context GAGA sequences.

    This helper never recomputes caches, initializes an attacker, or opens tests.
    For an attacked GAGA graph, the caller must construct updated sequences first.
    """
    import dgl
    import torch
    limits.check()
    ids = torch.as_tensor(node_ids, dtype=torch.long, device=graph.device)
    with torch.no_grad():
        if model_name == 'GAGA':
            if sequences is None or not context_id or sequence_context_id != context_id:
                raise RuntimeError('GAGA requires explicitly context-matched sequences.')
            limits.graph_call()
            logits = model(torch.as_tensor(sequences, dtype=torch.float32, device=device))
            logits = logits.reshape(len(ids), 2)
        elif model_name == 'GCN':
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
            _, returned_ids, blocks = sampler.sample_blocks(graph, ids)
            limits.graph_call()
            logits = model([block.to(device) for block in blocks], edge_weight=False).reshape(len(ids), 2)
            returned = returned_ids.cpu().tolist()
            requested = ids.cpu().tolist()
            if len(returned) != len(set(returned)) or set(returned) != set(requested):
                raise RuntimeError('GCN output membership mismatch.')
            positions = {node: i for i, node in enumerate(returned)}
            logits = logits[[positions[node] for node in requested]]
        else:
            raise ValueError('Unsupported fixed graph branch.')
    if not torch.isfinite(logits).all():
        raise RuntimeError('Nonfinite graph probabilities.')
    return logits.detach().cpu()


def probability_summary(logits):
    import torch
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(-1)
    return probs[:, 1].tolist(), (logits[:, 1] - logits[:, 0]).tolist(), entropy.tolist()


def extract_records(data, manifest, self_model, graph_model_name, graph_model,
                    device, limits, context_id, cohorts, graph=None, sequences=None,
                    sequence_context_id=None, graph_predictions=None, seed=None):
    """Return label-free numeric cards and a separate private supervision table.

    node_ids are only in supervision. cohorts maps names to explicit original
    validation/train-meta IDs. Reuse graph_predictions only for this context.
    """
    import torch
    graph = data['graph'] if graph is None else graph
    records, supervision = [], []
    features = graph.ndata['x']
    all_ids = [int(n) for ids in cohorts.values() for n in ids]
    if any(n in set(int(x) for x in data['idx_test']) for n in all_ids):
        raise RuntimeError('Test-node inference is forbidden.')
    if len(all_ids) != len(set(all_ids)):
        raise ValueError('Extract overlapping cohorts in separate explicit contexts.')
    for cohort, nodes in cohorts.items():
        for start in range(0, len(nodes), 64):
            limits.check()
            batch = [int(n) for n in nodes[start:start+64]]
            ids = torch.tensor(batch, dtype=torch.long, device=graph.device)
            with torch.no_grad():
                limits.self_evaluation_forwards += 1
                sl = self_logits(self_model, features[ids].to(device)).cpu()
            if graph_predictions is not None:
                if graph_predictions.get('context_id') != context_id:
                    raise RuntimeError('Cross-context graph prediction cache refused.')
                gl = torch.tensor([graph_predictions['logits'][n] for n in batch], dtype=torch.float32)
            else:
                seq = None if sequences is None else sequences[batch]
                gl = graph_branch_logits(graph_model_name, graph_model, graph, batch, device, limits,
                                         seq, context_id, sequence_context_id)
            ps, ms, es = probability_summary(sl)
            pg, mg, eg = probability_summary(gl)
            degrees = graph.in_degrees(ids).cpu().tolist()
            norms = features[ids].norm(dim=1).cpu().tolist()
            for i, node in enumerate(batch):
                # Exactly one self-loop per graph node is required by this interface.
                self_edges = int((graph.successors(node) == node).sum())
                if self_edges != 1:
                    raise RuntimeError('Current graph must have exactly one self-loop per node.')
                case = hashlib.sha256(('%s|%s|%s|%s' % (context_id, cohort, seed, node)).encode()).hexdigest()
                record = dict(case_id=case, cohort=cohort, self_seed=seed, context_id=context_id,
                              p_self=ps[i], p_graph=pg[i], self_margin=ms[i], graph_margin=mg[i],
                              self_entropy=es[i], graph_entropy=eg[i],
                              nonself_in_degree=int(degrees[i])-1, feature_l2_norm=float(norms[i]),
                              calibration='uncalibrated', graph_branch=graph_model_name,
                              content_kind='numeric_features_only', raw_text_available=False)
                if not all(math.isfinite(record[k]) for k in ('p_self', 'p_graph', 'self_margin',
                        'graph_margin', 'self_entropy', 'graph_entropy', 'feature_l2_norm')):
                    raise RuntimeError('Nonfinite evidence value.')
                records.append(record)
                supervision.append(dict(case_id=case, cohort=cohort, self_seed=seed,
                                         node_id=node, label=int(graph.ndata['y'][node])))
    return records, supervision


def extract_current_context(data, self_model, graph_model_name, graph_model, graph,
                            node_ids, device, limits, context_id, seed, cohort,
                            aligned_sequences=None, sequence_context_id=None):
    """Convenience API for root's saved attack-graph exports.

    aligned_sequences must contain exactly one row per supplied node, in the
    same order. For GAGA the caller computes them from this attacked graph with
    legitimate train-label masks. A matching string is a caller assertion, not
    independent verification that sequences were computed correctly.
    """
    nodes = [int(n) for n in node_ids]
    if len(nodes) != len(set(nodes)):
        raise ValueError('Duplicate requested nodes in one evidence context.')
    logits = graph_branch_logits(graph_model_name, graph_model, graph, nodes, device,
                                 limits, aligned_sequences, context_id, sequence_context_id)
    predictions = dict(context_id=context_id, logits=dict(zip(nodes, logits.tolist())))
    return extract_records(data, None, self_model, graph_model_name, graph_model,
                           device, limits, context_id, {cohort: nodes}, graph=graph,
                           graph_predictions=predictions, seed=seed)


def run(data, manifest, config, output, limits):
    import numpy as np
    import torch
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Frozen CUDA device is unavailable; no automatic CPU substitution.')
    torch.set_num_threads(int(config.get('cpu_threads', 2)))
    graph = data['graph'].to(device)
    data['graph'] = graph
    branch = load_graph_branch(config['graph_model'], device)
    context = 'clean:' + config['expected_sha256']['Datasets/GossipCop/GossipCop_0.05.bin']
    cohorts = dict(meta_train=manifest['meta_nodes'], clean_valid=manifest['clean_valid_nodes'])
    all_nodes = sorted(set(n for nodes in cohorts.values() for n in nodes))
    sequence = None
    if config['graph_model'] == 'GAGA':
        sequence = np.load(ROOT / 'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy', mmap_mode='r')
    predictions = dict(context_id=context, logits={})
    for start in range(0, len(all_nodes), 64):
        nodes = all_nodes[start:start+64]
        seq = None if sequence is None else np.array(sequence[nodes], copy=True)
        logits = graph_branch_logits(config['graph_model'], branch, graph, nodes, device, limits,
                                     seq, context, context if seq is not None else None)
        predictions['logits'].update({n: row for n, row in zip(nodes, logits.tolist())})
    summaries = []
    with (output / 'records.jsonl').open('x') as records_file, (output / 'labels_private.jsonl').open('x') as labels_file:
        for seed in manifest['self_seeds']:
            model, summary = train_self(data, manifest, seed, device, output, limits)
            records, labels = extract_records(data, manifest, model, config['graph_model'], branch,
                device, limits, context, cohorts, graph=graph, graph_predictions=predictions, seed=seed)
            for row in records:
                records_file.write(json.dumps(row, allow_nan=False) + '\n')
            for row in labels:
                labels_file.write(json.dumps(row, allow_nan=False) + '\n')
            records_file.flush()
            labels_file.flush()
            summaries.append(summary)
    return dict(status='completed', self_models=summaries, clean_records_per_seed=len(all_nodes),
                graph_meta_evidence=manifest['graph_meta_evidence'], calibration='uncalibrated',
                graph_prediction_semantics='Clean explicit node cohorts in fixed batches of 64; not historical target-group replay',
                runtime=dict(python=platform.python_version(), torch=torch.__version__, device=str(device),
                             gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0),
                probes_executed=False, api_calls=0, test_inference_calls=0,
                records_sha256=sha256(output / 'records.jsonl'),
                private_supervision_sha256=sha256(output / 'labels_private.jsonl'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare-manifest', 'run'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--expected-manifest-sha256')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = private_output(args.output)
    result = dict(status='started', mode=args.mode, created_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    limits = None
    try:
        config, checked = verify_registration(args.config, args.expected_config_sha256)
        limits = Limits(config['max_wall_seconds'], config['max_graph_forwards'])
        limits.started = started
        limits.check()
        write_json(output / 'registered.json', dict(config=config, config_sha256=args.expected_config_sha256,
                   checked_sha256=checked, command_mode=args.mode, api_calls=0, test_inference_allowed=False))
        data = load_fixed_data()
        if args.mode == 'prepare-manifest':
            manifest = prepare_manifest(data, config)
            write_json(output / 'cohort_manifest_private.json', manifest)
            result.update(status='completed', manifest_sha256=sha256(output / 'cohort_manifest_private.json'),
                          new_model_calls=0, training_runs=0)
        else:
            if not args.manifest or not args.expected_manifest_sha256:
                raise ValueError('run requires the pre-training manifest and externally fixed digest.')
            if sha256(args.manifest) != args.expected_manifest_sha256:
                raise RuntimeError('Manifest digest mismatch.')
            manifest = json.loads(pathlib.Path(args.manifest).read_text())
            # Recompute deterministic membership from frozen metadata before any model call.
            if manifest != prepare_manifest(data, config):
                raise RuntimeError('Manifest differs from deterministic cohort construction.')
            write_json(output / 'cohort_manifest_private.json', manifest)
            result.update(run(data, manifest, config, output, limits))
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        result['process_elapsed_seconds'] = time.monotonic() - started
        result['gpu_wall_hours'] = (result['process_elapsed_seconds'] / 3600.0
            if args.mode == 'run' and 'config' in locals() and config.get('device', '').startswith('cuda') else 0.0)
        if limits is not None:
            result['counts_and_elapsed'] = limits.report()
        write_json(output / 'summary.json', result)


if __name__ == '__main__':
    main()
