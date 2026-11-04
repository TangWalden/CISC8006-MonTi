"""Committed provenance and resource contracts for fixed-graph refinement."""
import json
import pathlib
import subprocess

from course.dual_surrogate.common import ROOT, sha256, write_json, private_output

ARMS = ('baseline', 'gcn_pgd', 'gaga_pgd', 'gaga_guarded', 'random')
MASTER = 'course/FEATURE_POLISH_PROTOCOL_V1.json'


def validate_protocol(path):
    path = pathlib.Path(path).resolve()
    p = json.loads(path.read_text(encoding='utf-8'))
    if p.get('status') != 'frozen' or p.get('owner_execution_authorized') is not True:
        raise RuntimeError('Frozen owner-authorized refinement protocol required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit tracked changes before execution.')
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip()
    if head != p['source_commit']:
        raise RuntimeError('Execution HEAD differs from the registered source.')
    required = {f.relative_to(ROOT).as_posix() for f in pathlib.Path(__file__).parent.glob('*.py')} | {MASTER}
    if not required <= set(p['source_files']):
        raise RuntimeError('Incomplete new package source freeze.')
    for name,wanted in p['source_files'].items():
        relative = pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Only repository-relative source paths are allowed.')
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        if sha256(ROOT/relative) != wanted:
            raise RuntimeError('Frozen source bytes differ: '+name)
    for row in p['inputs'] + p.get('retained_inputs',[]):
        relative = pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative) != row['sha256']:
            raise RuntimeError('Frozen input bytes differ: '+row['path'])
    master = json.loads((ROOT/MASTER).read_text(encoding='utf-8'))
    for key in ('arms','seeds','train_groups','valid_groups','polish_settings','teacher_settings',
                'phase_gpu_hours','phase_total_wall_hours','preflight','final_test_allowed',
                'teacher_device','teacher_wall_seconds','preflight_wall_seconds','refine_job_wall_seconds',
                'evaluation_wall_seconds','allowed_gpus','minimum_free_gpu_mib','cpu_threads','method','evaluation'):
        if p[key] != master[key]:
            raise RuntimeError('Fixed method contract differs: '+key)
    if (tuple(p['arms']) != ARMS or p['seeds'] != [3,4,5] or
            p['polish_settings'] != dict(steps=20,alpha=.02) or
            len(p['train_groups']) != 256 or len(p['valid_groups']) != 128 or
            len(set(p['valid_groups'])) != 128 or p['teacher_device'] != 'cpu' or
            p['final_test_allowed'] is not False):
        raise RuntimeError('Frozen cohort/method/no-TEST setting differs.')
    if p.get('execution_device') is not None:
        if p['execution_device'] == 'cpu':
            if p.get('cuda_visible_devices') != '' or p.get('physical_gpu_uuid') is not None:
                raise RuntimeError('CPU execution must have empty CUDA visibility and no physical GPU.')
        elif p['execution_device'] == 'cuda:0':
            pair=(p.get('cuda_visible_devices'),p.get('physical_gpu_uuid'))
            if pair not in [(r['index'],r['uuid']) for r in master['allowed_gpus']]:
                raise RuntimeError('Unregistered numeric GPU/UUID pair.')
        else:
            raise RuntimeError('Unregistered execution device.')
    return p,dict(source_commit=head,protocol_sha256=sha256(path))
