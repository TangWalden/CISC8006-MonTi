"""Freeze committed source and all scientific inputs before the new phase."""
import argparse
import json
import pathlib
import subprocess
from course.dual_surrogate.common import ROOT,private_output,sha256,write_json
from course.dual_surrogate.verify import SEALED_GAGA_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--continue-protocol');parser.add_argument('--prior-progress');parser.add_argument('--reuse-teachers')
    parser.add_argument('--gpu-index',choices=['0','1']);parser.add_argument('--physical-gpu-uuid')
    parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    continuation=args.continue_protocol is not None
    stage=private_output(args.stage,create=not continuation)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit before freezing the new phase.')
    master=ROOT/'course/DUAL_SURROGATE_PROTOCOL_V1.json'
    if continuation:
        if not args.prior_progress or not args.reuse_teachers or args.attempt<2:raise RuntimeError('Explicit retained inputs required for continuation.')
        previous=pathlib.Path(args.continue_protocol).resolve()
        p=json.loads(previous.read_text(encoding='utf-8'))
        progress=pathlib.Path(args.prior_progress).resolve()
        prior=json.loads(progress.read_text(encoding='utf-8'))
        if any(j['name'].endswith('_seed3') or j['name'].endswith('_seed4') or j['name'].endswith('_seed5') for j in prior['jobs']):
            raise RuntimeError('This continuation supports build/preflight failures only, not replaying attacker trainings.')
        teachers=private_output(args.reuse_teachers)
        result=json.loads((teachers/'result.json').read_text(encoding='utf-8'))
        registered=json.loads((teachers/'registered.json').read_text(encoding='utf-8'))
        original=p.get('construction_amendment',{})
        teacher_source=p.get('teacher_preparation_source_commit',original.get('previous_source',p['source_commit']))
        teacher_protocol=p.get('teacher_preparation_protocol_sha256',original.get('previous_protocol_sha256',sha256(previous)))
        if (result.get('status')!='completed' or
                sha256(teachers/'teacher_artifacts.json')!=result['teacher_artifacts_sha256'] or
                registered['source_commit']!=teacher_source or registered['protocol_sha256']!=teacher_protocol):
            raise RuntimeError('Reused teachers differ from the successful original preparation.')
        p.update(execution_attempt=args.attempt,reuse_teacher_directory=str(teachers.relative_to(ROOT)),
            teacher_preparation_source_commit=teacher_source,teacher_preparation_protocol_sha256=teacher_protocol,
            prior_progress_ref=dict(path=str(progress.relative_to(ROOT)),sha256=sha256(progress)),
            reused_teacher_receipts=[dict(path=str((teachers/name).relative_to(ROOT)),sha256=sha256(teachers/name))
                for name in ('result.json','registered.json','teacher_artifacts.json')],
            construction_amendment=dict(previous_protocol_sha256=sha256(previous),previous_source=p['source_commit'],
                reason='Explicit long indices for original TRAIN masks; preserve completed teachers and failed preflight without changing scientific settings.'))
    else:p=json.loads(master.read_text(encoding='utf-8'))
    if args.gpu_index is not None:
        if not continuation or not args.physical_gpu_uuid:raise RuntimeError('Device change requires explicit continuation and UUID.')
        p['resource_amendment']=dict(previous_visibility=p.get('cuda_visible_devices','0'),previous_uuid=p['physical_gpu_uuid'],
            reason='Shared GPU0 native allocation failure; use available same-model GPU1 without changing other jobs or scientific settings.')
        p.update(cuda_visible_devices=args.gpu_index,physical_gpu_uuid=args.physical_gpu_uuid,minimum_free_gpu_mib=8192)
    old=json.loads((ROOT/'evidence/alignment_agent_v2/execution_protocol_v2.json').read_text(encoding='utf-8'))
    sources=dict(old['source_files'])
    sources.update({f.relative_to(ROOT).as_posix():sha256(f) for f in pathlib.Path(__file__).parent.glob('*.py')})
    sources[master.relative_to(ROOT).as_posix()]=sha256(master)
    for name,wanted in sources.items():
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Registered source changed: '+name)
    inputs=p['inputs'] if continuation else list(old['inputs'])+p.pop('additional_input_manifest')
    for row in inputs:
        if sha256(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Input byte mismatch: '+row['path'])
    # This independent file hash does not deserialize or query the victim.
    victim='models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle'
    if sha256(ROOT/victim)!=SEALED_GAGA_SHA256:raise RuntimeError('Sealed victim bytes changed.')
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=inputs,stage_root=str(stage),sealed_victim_sha256=SEALED_GAGA_SHA256,
        sealed_victim=dict(path=victim,sha256=SEALED_GAGA_SHA256),
        accounting='Two aggregate GPU hours include preparation/preflight/train/evaluation/failures; max outer/guardian.')
    path=stage/('preparation_protocol_v%d.json'%args.attempt);write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
