"""Frozen provenance and small helpers; no ML imports or network access."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARMS = ('gcn_only', 'gaga_only', 'two_gcn', 'gcn_gaga')
EXPECTED_GPU = 'GPU-d7aa6d97-d823-0319-b218-84acc8e26019'


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value, *, exclusive=False):
    with pathlib.Path(path).open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=True)
        stream.write('\n')
        stream.flush()


def validate_protocol(path):
    path = pathlib.Path(path).resolve()
    protocol = json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'frozen' or protocol.get('owner_execution_authorized') is not True:
        raise RuntimeError('A frozen owner-authorized dual-surrogate protocol is required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, universal_newlines=True).strip():
        raise RuntimeError('Commit all tracked changes before execution.')
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, universal_newlines=True).strip()
    if head != protocol['source_commit']:
        raise RuntimeError('Execution HEAD differs from the frozen source.')
    required = {p.relative_to(ROOT).as_posix() for p in pathlib.Path(__file__).parent.glob('*.py')}
    if not required <= set(protocol['source_files']):
        raise RuntimeError('The new package source freeze is incomplete.')
    for name, wanted in protocol['source_files'].items():
        relative = pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Only repository-relative source paths are allowed.')
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name], cwd=ROOT, stderr=subprocess.STDOUT)
        if sha256(ROOT/relative) != wanted:
            raise RuntimeError('Frozen source mismatch: '+name)
    for row in protocol['inputs']:
        relative = pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative) != row['sha256']:
            raise RuntimeError('Frozen input mismatch: '+row['path'])
    for row in protocol.get('reused_teacher_receipts',[]):
        relative=pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative)!=row['sha256']:
            raise RuntimeError('Retained teacher receipt changed: '+row['path'])
    if tuple(protocol['arms']) != ARMS or protocol['seeds'] != [3,4,5]:
        raise RuntimeError('Arms/seeds changed.')
    if protocol['phase_gpu_hours'] != 2 or protocol['epochs'] != 8 or len(protocol['train_groups']) != 256 or len(protocol['valid_groups']) != 128:
        raise RuntimeError('Fixed cohort/epoch/aggregate resource setting changed.')
    protocol['_protocol_sha256'] = sha256(path)
    protocol['_protocol_path'] = str(path)
    return protocol, dict(source_commit=head, protocol_sha256=sha256(path))


def case_seed(seed, phase, epoch, group):
    value = '%d|%s|%d|%d|dual-surrogate-v1' % (seed, phase, epoch, group)
    return int(hashlib.sha256(value.encode('ascii')).hexdigest()[:8],16)


def private_output(path, *, create=False):
    path = pathlib.Path(path).resolve()
    if (ROOT/'runs').resolve() not in path.parents:
        raise RuntimeError('Private outputs must remain below ignored runs/.')
    if create:
        path.mkdir(parents=True,exist_ok=False)
    return path
