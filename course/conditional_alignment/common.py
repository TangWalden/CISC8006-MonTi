"""Small provenance helpers; no ML import or side effects."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = pathlib.Path(path)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()


def validate_protocol(path, *, training=False):
    path = pathlib.Path(path).resolve()
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'frozen':
        raise RuntimeError('This run requires a frozen protocol.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Tracked source differs from committed source: ' + dirty)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    code = protocol.get('code', [])
    if not code or not protocol.get('inputs'):
        raise RuntimeError('Frozen protocol must list source and input checksums.')
    required_source = {str(item.relative_to(ROOT)).replace('\\', '/')
                       for item in pathlib.Path(__file__).parent.glob('*.py')}
    listed_source = {entry['path'].replace('\\', '/') for entry in code}
    if not required_source <= listed_source:
        raise RuntimeError('Protocol omits alignment-package source hashes: '+
                           ', '.join(sorted(required_source-listed_source)))
    for entry in code:
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', entry['path']],
                                cwd=ROOT, stderr=subprocess.STDOUT)
    for entry in code + protocol['inputs']:
        relative = pathlib.Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Frozen input paths must be repository-relative.')
        # Registered data paths may be symlinks to the unchanged private input
        # store. Their bytes, not an invented local copy, must match the hash.
        target = ROOT / relative
        if sha256(target) != entry['sha256']:
            raise RuntimeError('Frozen source/input hash mismatch: ' + entry['path'])
    if sha256(ROOT / 'course/configs/baseline.json') != protocol['base_config_sha256']:
        raise RuntimeError('Original baseline configuration changed.')
    if protocol.get('seeds') != [3, 4, 5] or protocol.get('arms') != ['off', 'always', 'gated']:
        raise RuntimeError('Unexpected registered paired seeds or arms.')
    if training and not protocol.get('reference_sha256'):
        raise RuntimeError('Training requires the one frozen reference checksum.')
    if protocol['reference'].get('degree_rule') != 'floor_log2_nonself':
        raise RuntimeError('Unexpected degree matching rule.')
    if protocol['reference'].get('min_count') != 8 or protocol['reference'].get('gate_quantile') != 0.9:
        raise RuntimeError('Registered reference support/tolerance changed.')
    return protocol, dict(source_commit=commit, tracked_changes=dirty,
                          protocol_path=str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
                          protocol_sha256=sha256(path))


def gc_only_config(protocol, seed, gpu):
    config = json.loads((ROOT / 'course/configs/baseline.json').read_text(encoding='utf-8'))
    for key, default in dict(shared_parameters=False, without_candidates=False,
                             pos_encoding=True, ada_budget=True).items():
        config.setdefault(key, default)
    if config['surrogate_model'] != 'GCN' or config['dataset'] != 'gossipcop_0.05':
        raise RuntimeError('Only the registered GossipCop-S/GCN setting is supported.')
    config.update(seed=int(seed), cuda_id=gpu, random_candidates=False,
                  victim_model='GCN', epochs=int(protocol['epochs']),
                  patience=int(protocol['patience']))
    return config
