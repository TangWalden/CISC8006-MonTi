"""Frozen-parent, CPU-only action-summary ablation helpers."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
PARENT_PROTOCOL_SHA = '5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'
PARENT_SOURCE = '9e013cb182b5af71a9d0b719b5a939deda3c88e2'
SEEDS = (3, 4, 5)
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned', 'quota_blind')


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def read_json(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def verify_protocol(path):
    p = read_json(path)
    if (p.get('schema') != 'monti_downstream_blind_v1' or p.get('status') != 'frozen'
            or p.get('owner_execution_authorized') is not True or p.get('final_TEST_allowed') is not False
            or tuple(p['seeds']) != SEEDS or len(p['train_groups']) != 128 or len(p['valid_groups']) != 64
            or p['settings']['downstream_epochs'] != 4 or p['settings']['action_conditioning'] != 'hidden'):
        raise RuntimeError('Registered owner-authorized ablation settings differ.')
    runtime = pathlib.Path(p['runtime_root']).resolve()
    if pathlib.Path(p['source_root']).resolve() != SOURCE_ROOT or pathlib.Path(p['asset_root']).resolve() != runtime:
        raise RuntimeError('Source/runtime binding differs.')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() != RUNTIME_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip()
            or subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=runtime, text=True).strip() != p['runtime_tree']):
        raise RuntimeError('Historical runtime is not at the clean frozen source.')
    for name, wanted in p['source_manifest']['files'].items():
        rel = pathlib.PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ('methods', 'experiments', 'tests'):
            raise RuntimeError('Unsafe study source path.')
        if sha(SOURCE_ROOT / name) != wanted: raise RuntimeError('Study source changed: ' + name)
    for ref in p['input_refs'] + p['parent_refs']:
        rel = pathlib.PurePosixPath(ref['path'])
        if rel.is_absolute() or '..' in rel.parts: raise RuntimeError('Unsafe frozen input path.')
        if sha(runtime / ref['path']) != ref['sha256']: raise RuntimeError('Frozen input changed: ' + ref['path'])
    parent_path = runtime / p['parent_protocol']['path']
    if sha(parent_path) != PARENT_PROTOCOL_SHA: raise RuntimeError('Parent protocol changed.')
    parent = read_json(parent_path)
    result = read_json(runtime / p['parent_result']['path'])
    if (result['status'] != 'completed' or result['provenance']['source_commit'] != PARENT_SOURCE
            or result['provenance']['protocol_sha256'] != PARENT_PROTOCOL_SHA):
        raise RuntimeError('Only the completed fixed parent study may be reused.')
    for name in ('train_groups', 'valid_groups', 'train_members', 'valid_members', 'cohort'):
        if p[name] != parent[name]: raise RuntimeError('The ablation changed its parent cohort: ' + name)
    # New source only supplies isolated modules; historical course/data/model
    # helpers stay in the original clean runtime and are never overwritten.
    for root in (str(runtime), str(SOURCE_ROOT)):
        if root in sys.path: sys.path.remove(root)
        sys.path.insert(0, root)
    return p, dict(source_commit=p['source_manifest']['source_commit'], runtime_commit=RUNTIME_COMMIT,
                   protocol_sha256=sha(path), parent_protocol_sha256=PARENT_PROTOCOL_SHA)


def parent_protocol(p): return read_json(pathlib.Path(p['runtime_root']) / p['parent_protocol']['path'])


def parent_result(p): return read_json(pathlib.Path(p['runtime_root']) / p['parent_result']['path'])


def old_helpers():
    from experiments.downstream_cooperation import common
    return common


def load_cached_case(p, seed, phase, group, torch):
    old = old_helpers()
    root = pathlib.Path(p['parent_run_root'])
    manifest = read_json(pathlib.Path(p['runtime_root']) / p['parent_cache_manifest']['path'])
    found = [row for row in manifest['cases'] if row['seed'] == seed and row['phase'] == phase and row['group'] == group]
    if len(found) != 1: raise RuntimeError('Parent cache coverage differs.')
    row = found[0]
    path = root / 'cache' / row['path']
    if path.parent != root / 'cache' or sha(path) != row['sha256']: raise RuntimeError('Parent cached input changed.')
    case = torch.load(path, map_location='cpu')
    members = p['train_members'] if phase == 'train' else p['valid_members']
    if (case['ids'].tolist() != members[str(group)] or case['seed'] != seed or case['group'] != group
            or old.tensor_hash(case['x_aux']) != case['upstream_feature_sha256']
            or old.tensor_hash(case['target_adjacency']) != case['upstream_target_block_sha256']
            or case['quota'] != row['quota']):
        raise RuntimeError('Frozen cached input content or quota differs.')
    return case


def blind_scores(model, case):
    # Actual target_adjacency is deliberately not a policy argument. It is used
    # only by the fixed physical graph builder and the TRAIN feedback model.
    return model(case['x_aux'], case['x_targets'], case['quota']['target_quota'],
                 case['x_candidates'], case['h_candidates'])


def paired(values):
    import math
    import statistics
    if len(values) != 3: raise RuntimeError('Exactly the registered three seeds are required.')
    mean = statistics.mean(values)
    radius = 4.302652729911275 * statistics.stdev(values) / math.sqrt(3)
    return dict(seed_differences_pp=values, mean_pp=mean, nominal_paired_t95_pp=[mean - radius, mean + radius])


def summarize(rows):
    import statistics
    metrics = []
    for seed in SEEDS:
        for arm in ARMS:
            selected = [row for row in rows if row['seed'] == seed and row['arm'] == arm]
            denominator = sum(row['targets'] for row in selected)
            if len(selected) != 64 or denominator != 571: raise RuntimeError('Full denominator or arm coverage changed.')
            item = dict(seed=seed, arm=arm, groups=64, occurrences=denominator)
            for detector in ('gcn', 'gaga'):
                for name in ('clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries'):
                    item[detector + '_' + name] = sum(row[detector][name] for row in selected)
                item[detector + '_mr_percent'] = 100 * item[detector + '_attacked_wrong'] / denominator
            metrics.append(item)
    means = {d: {a: statistics.mean(row[d + '_mr_percent'] for row in metrics if row['arm'] == a)
                  for a in ARMS} for d in ('gcn', 'gaga')}
    contrasts = {}
    for label, left, right in (('visible_minus_blind', 'quota_learned', 'quota_blind'),
                               ('visible_minus_random', 'quota_learned', 'quota_random'),
                               ('blind_minus_random', 'quota_blind', 'quota_random')):
        differences = [next(row['gaga_mr_percent'] for row in metrics if row['seed'] == seed and row['arm'] == left)
                       - next(row['gaga_mr_percent'] for row in metrics if row['seed'] == seed and row['arm'] == right)
                       for seed in SEEDS]
        contrasts[label] = paired(differences)
    primary = contrasts['visible_minus_blind']
    criterion = all(value > 0 for value in primary['seed_differences_pp']) and primary['nominal_paired_t95_pp'][0] > 0
    return metrics, means, contrasts, criterion
