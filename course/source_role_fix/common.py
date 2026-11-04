"""Frozen provenance for the narrow edge-score source-role comparison."""
import json
import pathlib
import subprocess

from course.dual_surrogate.common import ROOT, sha256, write_json, private_output, case_seed

ARMS = ('legacy', 'source_fixed')
MASTER = 'course/SOURCE_ROLE_FIX_PROTOCOL_V1.json'


def validate_protocol(path):
    path = pathlib.Path(path).resolve()
    p = json.loads(path.read_text(encoding='utf-8'))
    if p.get('status') != 'frozen' or p.get('owner_execution_authorized') is not True:
        raise RuntimeError('A frozen, owner-authorized source-role protocol is required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit tracked changes before execution.')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip()
    if head != p['source_commit']:
        raise RuntimeError('Execution HEAD differs from frozen source.')
    required = {f.relative_to(ROOT).as_posix() for f in pathlib.Path(__file__).parent.glob('*.py')} | {MASTER}
    if not required <= set(p['source_files']):
        raise RuntimeError('New package source freeze is incomplete.')
    for name,wanted in p['source_files'].items():
        relative = pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Source paths must be repository-relative.')
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        if sha256(ROOT/relative) != wanted:
            raise RuntimeError('Frozen source differs: '+name)
    for row in p['inputs'] + p.get('retained_inputs',[]):
        relative = pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative) != row['sha256']:
            raise RuntimeError('Frozen input differs: '+row['path'])
    master = json.loads((ROOT/MASTER).read_text(encoding='utf-8'))
    for key in ('arms','seeds','epochs','train_groups','valid_groups','export_valid_groups','phase_gpu_hours'):
        if p[key] != master[key]:
            raise RuntimeError('Frozen comparison differs from committed master: '+key)
    if (tuple(p['arms']) != ARMS or p['seeds'] != [3,4,5] or p['epochs'] != 8 or
            p['phase_gpu_hours'] != 2 or len(p['train_groups']) != 256 or len(p['valid_groups']) != 128 or
            len(set(p['train_groups'])) != 256 or len(set(p['valid_groups'])) != 128 or
            set(p['train_groups']) & set(p['valid_groups']) or p['final_test_allowed'] is not False):
        raise RuntimeError('Registered fixed settings or no-TEST boundary differ.')
    if p['preflight']['train_groups'] != [263,238]:
        raise RuntimeError('Preflight groups changed.')
    for key in ('preflight_wall_seconds','train_job_wall_seconds','evaluation_wall_seconds'):
        if not 1 <= int(p[key]) <= 900:
            raise RuntimeError('A registered per-process timer is outside its bound.')
    return p,dict(source_commit=head,protocol_sha256=sha256(path))
