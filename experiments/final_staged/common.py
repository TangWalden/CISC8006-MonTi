"""Source-bound CPU runtime for the registered full staged comparison.

The published pilot scorer/quota implementation is reused unchanged. The
formal checkpoint loader is a separate adapter; legacy pilot loaders are never
called. Only TRAIN is used for fitting; TEST plans are sealed before detectors.
"""
import hashlib
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
from types import MethodType, SimpleNamespace

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
ARMS = ('native_monti', 'quota_random', 'quota_learned')
SEEDS = (0, 1, 2)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False); stream.write('\n')


def verify_protocol(path):
    p = read_json(path)
    if (p.get('schema') != 'monti_final_staged_v1' or p.get('status') != 'frozen'
            or p.get('owner_execution_authorized') is not True or p.get('final_TEST_allowed') is not True
            or tuple(p['arms']) != ARMS or tuple(p['seeds']) != SEEDS
            or len(p['train_groups']) != 937 or len(p['test_groups']) != 909
            or len(set(p['train_groups'])) != 937 or len(set(p['test_groups'])) != 909
            or set(p['train_groups']) & set(p['test_groups'])):
        raise RuntimeError('Frozen owner-authorized full comparison protocol differs.')
    for phase in ('train', 'test'):
        members = p[phase + '_members']
        if (set(members) != {str(group) for group in p[phase + '_groups']}
                or any(not nodes or len(nodes) != len(set(nodes)) for nodes in members.values())):
            raise RuntimeError('Complete ordered target membership differs.')
    runtime = pathlib.Path(p['runtime_root']).resolve()
    if pathlib.Path(p['source_root']).resolve() != SOURCE_ROOT or pathlib.Path(p['asset_root']).resolve() != runtime:
        raise RuntimeError('Source/runtime root binding differs.')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != RUNTIME_COMMIT:
        raise RuntimeError('Historical runtime must remain at its clean registered source.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip():
        raise RuntimeError('Historical runtime tracked source is dirty.')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=runtime, text=True).strip() != p['runtime_tree']:
        raise RuntimeError('Historical runtime tree changed.')
    for name, wanted in p['source_manifest']['files'].items():
        relative = pathlib.PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('methods', 'experiments', 'tests'):
            raise RuntimeError('Unsafe or shadowing study source inventory.')
        if sha(SOURCE_ROOT / name) != wanted: raise RuntimeError('Study source changed: ' + name)
    for ref in p['input_refs']:
        relative = pathlib.PurePosixPath(ref['path'])
        if relative.is_absolute() or '..' in relative.parts: raise RuntimeError('Unsafe input path.')
        if sha(runtime / ref['path']) != ref['sha256']: raise RuntimeError('Frozen scientific input changed: ' + ref['path'])
    if (p['device']['kind'] != 'cpu' or p['device']['cuda_visible_devices'] != ''
            or p['settings']['downstream_epochs'] != 4 or p['settings']['quota_policy'] != 'closest_feasible'
            or p['settings']['target_fraction'] != .5
            or p['settings']['hidden_dimension'] != 64
            or p['settings']['learning_rate'] != .001
            or p['settings']['weight_decay'] != .0001
            or p['resource']['threads'] != 4
            or p['resource']['main_inner_seconds'] > 43200
            or p['resource']['CPU_phase_seconds'] > 57600
            or p['resource']['automatic_retry'] is not False):
        raise RuntimeError('Fixed model, quota or CPU execution settings changed.')
    expected = dict(generator_forwards=5538, train_GCN_forwards=11244,
                    evaluation_GCN_forwards=9090, evaluation_GAGA_forwards=9090,
                    optimizer_updates=11244, autograd_calls=11244,
                    validation_queries=0, API_calls=0)
    if p.get('logical_main_caps') != expected:
        raise RuntimeError('Formal logical query bounds changed.')
    qualification = dict(primary='quota_learned-minus-native_monti_gaga',
                         require_mean_gain_positive=True,
                         require_all_three_seed_deltas_positive=True,
                         require_nominal_paired_t95_lower_above_zero=True,
                         random_comparison='secondary_only_not_a_new_selection_rule',
                         controls_not_candidates=True)
    if p.get('qualification') != qualification:
        raise RuntimeError('Registered primary criterion or secondary-control role changed.')
    # The archive supplies course/models/modules; the new source supplies only
    # the isolated experiment and method. Neither historical source is edited.
    for root in (str(runtime), str(SOURCE_ROOT)):
        if root not in sys.path: sys.path.insert(0, root)
    return p, dict(source_commit=p['source_manifest']['source_commit'],
                   runtime_commit=RUNTIME_COMMIT, protocol_sha256=sha(path))


def check_device():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU visibility must be explicit.')
    expected_lib = '/home/tjy/miniconda3/envs/monti/lib'
    if os.environ.get('LD_LIBRARY_PATH') != expected_lib or os.environ.get('DGLBACKEND') != 'pytorch':
        raise RuntimeError('The locked DGL dynamic-library environment is missing.')
    import torch
    import dgl
    import numpy as np
    import scipy
    versions = dict(python=sys.version.split()[0], torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__)
    if (versions['python'] != '3.8.20' or versions['torch'] != '1.13.1+cu117'
            or not versions['dgl'].startswith('1.1.2') or versions['numpy'] != '1.24.4' or versions['scipy'] != '1.10.1'):
        raise RuntimeError('Locked ML versions differ: ' + str(versions))
    if torch.cuda.is_available() or torch.cuda.device_count() or torch._C._cuda_getDeviceCount():
        raise RuntimeError('No Torch/native GPU may be visible.')
    torch.set_num_threads(4)
    return dict(versions=versions, device='cpu', GPU_hours=0, cpu_threads=4)


def precheck_resources(output, p):
    """Metadata-only admission before any model/dataset construction.

    These conservative available-space thresholds are not a peak-RAM or final
    artifact-size guarantee. No process belonging to another job is changed.
    """
    minimum_disk = 20 * 1024 ** 3
    minimum_memory = 8 * 1024 ** 3
    if (p['resource'].get('minimum_disk_free_bytes') != minimum_disk
            or p['resource'].get('minimum_memory_available_bytes') != minimum_memory):
        raise RuntimeError('Frozen resource admission thresholds changed.')
    output = pathlib.Path(output).resolve()
    if (pathlib.Path(p['runtime_root']).resolve() / 'runs') not in output.parents:
        raise RuntimeError('Exclusive output must remain under runtime ignored runs/.')
    disk = shutil.disk_usage(str(output)).free
    memory = {}
    for line in pathlib.Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
        key, value = line.split(':', 1)
        if key == 'MemAvailable': memory[key] = int(value.split()[0]) * 1024
    available = memory.get('MemAvailable', -1)
    cpus = len(os.sched_getaffinity(0))
    if disk < minimum_disk or available < minimum_memory or cpus < 4:
        raise RuntimeError('Registered CPU disk/memory/core admission failed.')
    return dict(disk_free_bytes=disk, memory_available_bytes=available,
                affinity_logical_cpus=cpus, minimum_disk_free_bytes=minimum_disk,
                minimum_memory_available_bytes=minimum_memory,
                expected_cache_files=5538, expected_final_plan_files=2727,
                scope='Metadata admission only; no claimed peak-resource guarantee.')


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode('ascii')
    return hashlib.sha256(header + b'\0' + tensor.numpy().tobytes()).hexdigest()


def state_hash(model):
    from course.dual_surrogate.runtime import model_hash
    return model_hash(model)


def freeze(model):
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False); parameter.grad = None
    return state_hash(model)


def assert_frozen(model, wanted):
    if state_hash(model) != wanted or any(v.requires_grad or v.grad is not None for v in model.parameters()):
        raise RuntimeError('Frozen model weights, buffers or gradient flags changed.')


def load_engine(p, seed, output, deadline):
    from experiments.final_staged.loader import load_engine as formal_loader
    engine, data, receipt, hashes = formal_loader(p, seed, output, deadline)
    if (data.device.type != 'cpu' or engine.victim_model is not None
            or set(engine.teachers) != {'gcn_base'} or engine.optimizer_steps != 0
            or not hasattr(engine, 'capture_epoch')):
        raise RuntimeError('Formal loader must expose a frozen CPU TRAIN-only engine.')
    assert_frozen(engine.model, hashes['upstream'])
    assert_frozen(engine.teachers['gcn_base'], hashes['teacher'])
    return engine, data, receipt, hashes


def capture_case(engine, data, seed, group, phase):
    import torch
    import torch.nn.functional as F
    from utils.utils import get_n_ids
    from methods.downstream_quota import plan_quota, choose_target_edges, choose_edges
    ids = torch.tensor(get_n_ids(data.target_assignment, group), dtype=torch.long, device=data.device)
    if not bool((data.labels[ids] == 1).all()): raise RuntimeError('Original fraud target list changed.')
    b, e = int(data.node_budgets[group]), int(data.edge_budgets[group])
    observed = {}
    def projection_hook(module, args, value): observed['edge_projection'] = F.normalize(value.detach(), dim=-1).clone()
    hook = engine.model.linear_edge.register_forward_hook(projection_hook)
    engine.current_group = int(group)
    engine.current_epoch, engine.phase = int(engine.capture_epoch), phase
    try:
        with torch.no_grad(): graph = engine.get_perturbed_graph(ids, b, e, training=False)
        del graph
    finally: hook.remove()
    native = {key: value.detach().clone() for key, value in engine.capture.items()}
    engine.capture = engine.live_capture = None
    cids = native['candidate_ids']; c = len(cids); original = data.graph.num_nodes()
    source_ids = list(range(original, original + b)); target_ids = ids.tolist(); candidate_ids = cids.tolist()
    if (len(set(target_ids)) != len(target_ids) or len(set(candidate_ids)) != c
            or set(target_ids) & set(candidate_ids)
            or any(node < 0 or node >= original for node in target_ids + candidate_ids)
            or not torch.equal(native['target_ids'], ids)
            or native['x_atk'].shape != (b, data.feat_dim)
            or native['adj_attack'].shape != (b, len(ids) + c + b)
            or not bool(((native['adj_attack'] == 0) | (native['adj_attack'] == 1)).all())
            or int(native['adj_attack'].sum()) != e
            or not bool((native['adj_attack'][:, :len(ids)].sum(1) >= 1).all())
            or any(value.requires_grad or not bool(torch.isfinite(value).all()) for value in native.values())):
        raise RuntimeError('Captured original IDs, features, native budgets or tensor boundary changed.')
    q = plan_quota(len(ids), c, b, e)
    # Retain the historical scoring role e[src] in the frozen upstream. This
    # is not an unregistered correction to e[T+C+src]. Exact ties use SHA IDs.
    projected = observed['edge_projection']
    scores = (projected[:b, None, :] * projected[None, :len(ids) + c, :]).sum(-1)
    targets = choose_target_edges(scores[:, :len(ids)].tolist(), q['target_quota'], seed, group, source_ids, target_ids)
    original_candidates = choose_edges(scores[:, len(ids):].tolist(), q['candidate_quota'], seed, group, source_ids, candidate_ids, 'original_candidate_scores')
    # Seed-local private random scores are shared across every later use.
    key = int(hashlib.sha256(json.dumps([seed, group, 'random_downstream_v1']).encode()).hexdigest(), 16)
    rng = random.Random(key)
    random_scores = [[rng.random() for _ in range(c)] for _ in range(b)]
    random_candidates = choose_edges(random_scores, q['candidate_quota'], seed, group, source_ids, candidate_ids, 'random_candidate_scores')
    return dict(seed=seed, group=group, ids=ids, candidate_ids=cids, source_ids=source_ids,
                x_aux=native['x_atk'], x_targets=data.features[ids].detach().clone(),
                x_candidates=data.features[cids].detach().clone(), h_candidates=data.graph.ndata['h'][cids].detach().clone(),
                native_adjacency=native['adj_attack'], target_adjacency=torch.tensor(targets, dtype=native['x_atk'].dtype),
                original_candidates=torch.tensor(original_candidates, dtype=native['x_atk'].dtype),
                random_candidates=torch.tensor(random_candidates, dtype=native['x_atk'].dtype), quota=q,
                upstream_feature_sha256=tensor_hash(native['x_atk']),
                upstream_target_block_sha256=tensor_hash(torch.tensor(targets, dtype=native['x_atk'].dtype)))


def candidate_mask(case, scores, purpose='learned_candidates'):
    import torch
    from methods.downstream_quota import choose_edges
    selected = choose_edges(scores.detach().tolist(), case['quota']['candidate_quota'], case['seed'], case['group'],
                            case['source_ids'], case['candidate_ids'].tolist(), purpose)
    return torch.tensor(selected, dtype=scores.dtype, device=scores.device)


def downstream_scores(model, case):
    return model(case['x_aux'], case['x_targets'], case['target_adjacency'], case['x_candidates'], case['h_candidates'])


def adjacency(case, candidates):
    import torch
    b = len(case['source_ids'])
    return torch.cat((case['target_adjacency'], candidates, torch.zeros((b, b), dtype=candidates.dtype, device=candidates.device)), dim=1)


def make_graph(data, case, adj, hard=None):
    """Native physical mapping; selected-edge ST weights only in training."""
    import torch
    import dgl
    base = data.graph; b = len(case['source_ids']); e = case['quota']['edge_budget']
    hard = adj.detach() if hard is None else hard.detach()
    if (not bool(((hard == 0) | (hard == 1)).all()) or int(hard.sum()) != e
            or not bool((hard[:, :len(case['ids'])].sum(1) >= 1).all())):
        raise RuntimeError('Physical graph mask or mandatory target incidence invalid.')
    source, destination = hard.nonzero(as_tuple=True)
    node_ids = torch.cat((case['ids'], case['candidate_ids'], torch.tensor(case['source_ids'], dtype=torch.long)))
    physical_source, physical_destination = source + base.num_nodes(), node_ids[destination]
    if bool((physical_source == physical_destination).any()): raise RuntimeError('Unexpected injected self edge.')
    graph = dgl.add_nodes(base, b, {'x': case['x_aux'].detach()})
    graph.edata['weight'] = torch.ones(graph.num_edges(), dtype=adj.dtype)
    weights = adj[source, destination]
    graph.add_edges(physical_source, physical_destination, {'weight': weights})
    graph.add_edges(physical_destination, physical_source, {'weight': weights})
    graph = dgl.add_self_loop(dgl.remove_self_loop(graph))
    if graph.num_nodes() != base.num_nodes() + b or graph.num_edges() != base.num_edges() + b + 2 * e:
        raise RuntimeError('Actual physical budgets changed.')
    if not torch.equal(graph.ndata['x'][base.num_nodes():], case['x_aux']): raise RuntimeError('Frozen features changed.')
    return graph


def canonical(data, view, graph, case):
    import torch
    from modules.data_handler import DataHandler
    original = data.graph.num_nodes(); injected = torch.arange(original, graph.num_nodes(), dtype=torch.long)
    local = graph.local_var()
    for name, value in view.group_masks.items():
        suffix = torch.ones(len(injected), dtype=value.dtype) if name == 'train_masked' else torch.zeros(len(injected), dtype=value.dtype)
        local.ndata[name] = torch.cat((value, suffix))
    oracle = SimpleNamespace(**vars(view))
    oracle.group_agg_tensor = MethodType(DataHandler.group_agg_tensor, oracle)
    oracle.feat_agg_tensor = MethodType(DataHandler.feat_agg_tensor, oracle)
    sequence = DataHandler.get_updated_sequences(oracle, local, case['ids'], case['candidate_ids'], injected, edge_weight=False)
    if sequence.shape != (len(case['ids']), 7, data.feat_dim) or not bool(torch.isfinite(sequence).all()):
        raise RuntimeError('Canonical victim input invalid.')
    return sequence


def softplus_loss(logits):
    import torch.nn.functional as F
    return F.softplus(logits[:, 1] - logits[:, 0]).mean()
