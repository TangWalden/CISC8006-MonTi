"""Freeze committed source and all scientific inputs before the new phase."""
import argparse
import json
import pathlib
import subprocess
from course.dual_surrogate.common import ARMS,ROOT,private_output,sha256,write_json
from course.dual_surrogate.verify import SEALED_GAGA_SHA256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--continue-protocol');parser.add_argument('--prior-progress');parser.add_argument('--reuse-teachers')
    parser.add_argument('--gpu-index',choices=['0','1']);parser.add_argument('--physical-gpu-uuid')
    parser.add_argument('--reuse-matrix');parser.add_argument('--reuse-passed-preflight',action='store_true')
    parser.add_argument('--train-job-wall-seconds',type=int,choices=[480,900])
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
        training_jobs=[j for j in prior['jobs'] if any(j['name']==a+'_seed'+str(s) for a in ARMS for s in p['seeds'])]
        if training_jobs and not args.reuse_matrix:
            raise RuntimeError('Retain successful attacker jobs explicitly instead of replaying them.')
        if args.reuse_matrix:
            # Only orchestration/timers may change after TRAIN-only construction
            # failure; every scientific source byte remains the original one.
            allowed={'course/dual_surrogate/prepare_protocol.py','course/dual_surrogate/queue.py'}
            for name,wanted in p['source_files'].items():
                if name not in allowed and sha256(ROOT/name)!=wanted:
                    raise RuntimeError('Scientific source changed during timer-only continuation: '+name)
            matrix=private_output(args.reuse_matrix);retained={}
            successful=[j['name'] for j in training_jobs if j['exit_code']==0]
            if len(successful)!=len(set(successful)):raise RuntimeError('Duplicate successful training jobs in prior ledger.')
            for name in successful:
                directory=matrix/name;r=json.loads((directory/'result.json').read_text())
                registered_run=json.loads((directory/'registered.json').read_text())
                if (r.get('status')!='completed' or r['epochs']!=8 or r['optimizer_steps']!=2048 or
                        name!=r['arm']+'_seed'+str(r['seed']) or
                        r['victim_loaded'] or r['validation_performed'] or registered_run['provenance']['source_commit']!=p['source_commit'] or
                        registered_run['provenance']['protocol_sha256']!=sha256(previous) or
                        sha256(directory/'final_model.pt')!=r['final_model_sha256'] or sha256(directory/'final_optimizer.pt')!=r['final_optimizer_sha256']):
                    raise RuntimeError('A retained terminal TRAIN job failed its original receipt: '+name)
                retained[name]=dict(directory=str(directory.relative_to(ROOT)),provenance=registered_run['provenance'],
                    files={f.relative_to(directory).as_posix():sha256(f) for f in directory.rglob('*') if f.is_file()})
            if not retained or (stage/'all12_trainings_complete.json').exists() or (stage/'evaluation_attempt1').exists():
                raise RuntimeError('Timer continuation requires retained TRAIN jobs and no prior victim evaluation.')
            p.update(reused_training_runs=retained,scientific_core_source=p['source_commit'])
        if args.reuse_passed_preflight:
            ref=p['preflight_ref'];r=json.loads((ROOT/ref['path']).read_text())
            if sha256(ROOT/ref['path'])!=ref['sha256'] or r['status']!='passed' or r['optimizer_steps'] or r['sealed_victim_forwards']:
                raise RuntimeError('The retained preflight must be an unchanged passed zero-update TRAIN check.')
            p['reuse_passed_preflight']=True
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
                reason=('Retain completed terminal TRAIN jobs and passed preflight; rerun only incomplete timer failure with unchanged science.' if args.reuse_matrix else
                    'Explicit long indices for original TRAIN masks; preserve completed teachers and failed preflight without changing scientific settings.')))
    else:p=json.loads(master.read_text(encoding='utf-8'))
    if args.train_job_wall_seconds is not None:
        if not continuation or not args.reuse_matrix:raise RuntimeError('A job-timer amendment requires explicit retained TRAIN continuation.')
        p['timing_amendment']=dict(previous_seconds=p['train_job_wall_seconds'],new_seconds=args.train_job_wall_seconds,
            basis='TRAIN-only runtime timeout after2012of2048updates; no VALID/victim outcome observed; aggregate ceiling unchanged.')
        p['train_job_wall_seconds']=args.train_job_wall_seconds
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
