"""Frozen-source checks for the separately authorized validation follow-up."""
import hashlib
import json
import pathlib
import subprocess

from course.conditional_alignment.common import ROOT, sha256, write_json, gc_only_config

ARMS = ('legacy_hinge_off','smooth_off','smooth_threshold','smooth_rule','smooth_agent','smooth_random')


def validate_protocol(path):
    path=pathlib.Path(path).resolve()
    protocol=json.loads(path.read_text(encoding='utf-8'))
    if protocol.get('status')!='frozen' or protocol.get('owner_execution_authorized') is not True:
        raise RuntimeError('A frozen owner-authorized follow-up protocol is required.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Tracked sources must remain committed during execution.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if protocol['source_commit']!=head:
        raise RuntimeError('Source HEAD differs from the execution freeze.')
    if tuple(protocol['arms'])!=ARMS or protocol['seeds']!=[3,4,5] or protocol['epochs']!=8:
        raise RuntimeError('Registered arms/seeds/terminal epoch changed.')
    if protocol['alignment_lambda']!=.1 or protocol['softplus_beta']!=1 or protocol['gradient_cap_ratio']!=1:
        raise RuntimeError('Registered loss or gradient-protection settings changed.')
    if protocol['phase_gpu_hours']!=2 or protocol['job_wall_seconds']!=420:
        raise RuntimeError('Registered compute ceiling changed.')
    if len(protocol['train_groups'])!=256 or len(protocol['valid_groups'])!=128:
        raise RuntimeError('Registered group cohort changed.')
    if len(protocol['gradient_probe_groups'])!=8 or not set(protocol['gradient_probe_groups'])<=set(protocol['train_groups']):
        raise RuntimeError('Gradient probes must be the eight frozen TRAIN groups.')
    old=json.loads((ROOT/'course/improvements/pilot_protocol_v1.json').read_text())
    if protocol['train_groups']!=old['train_groups'] or protocol['valid_groups']!=old['valid_groups']:
        raise RuntimeError('Original registered TRAIN/validation membership changed.')
    required={p.relative_to(ROOT).as_posix() for p in pathlib.Path(__file__).parent.glob('*.py')}
    if not required<=set(protocol['source_files']):
        raise RuntimeError('Some new-package source files are omitted from the freeze.')
    for name,wanted in protocol['source_files'].items():
        relative=pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Only repository-relative frozen sources are allowed.')
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        if sha256(ROOT/relative)!=wanted:
            raise RuntimeError('Frozen source mismatch: '+name)
    for row in protocol['inputs']:
        relative=pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(ROOT/relative)!=row['sha256']:
            raise RuntimeError('Frozen input mismatch: '+row['path'])
    for key in ('alignment_reference_file','filter_reference_file'):
        row=protocol[key]
        if sha256(ROOT/row['path'])!=row['sha256']:
            raise RuntimeError('Frozen reference mismatch: '+key)
    if sha256(ROOT/'course/configs/baseline.json')!=protocol['base_config_sha256']:
        raise RuntimeError('The frozen baseline configuration changed.')
    protocol['_protocol_sha256']=sha256(path)
    protocol['_protocol_path']=str(path)
    return protocol,dict(source_commit=head,protocol_sha256=sha256(path),tracked_changes='')


def case_seed(seed,phase,epoch,group):
    raw=('%s|%s|%s|%s|alignment-agent-v2'%(seed,phase,epoch,group)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8],16)
