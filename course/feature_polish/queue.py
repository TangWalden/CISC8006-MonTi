"""One serial capped CPU/GPU refinement study; no automatic retry or tuning."""
import argparse
import datetime as dt
import fcntl
import getpass
import json
import os
import pathlib
import signal
import subprocess
import time

from course.feature_polish.common import ROOT,private_output,sha256,validate_protocol,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p,provenance=validate_protocol(args.protocol)
    stage=pathlib.Path(p['stage_root']);out=private_output(args.output,create=True)
    lock=(stage/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pattern=r'(^|/)(python[0-9.]*) .*course\.(dual_surrogate|source_role_fix|feature_polish|alignment_agent|conditional_alignment|fusion_pilot)\.(teacher|refine|run|guard|evaluate|preflight|queue)'
    active=subprocess.run(['pgrep','-u',getpass.getuser(),'-f',pattern],stdout=subprocess.PIPE,universal_newlines=True)
    if active.returncode not in (0,1) or {int(v) for v in active.stdout.split()}-{os.getpid()}:
        raise RuntimeError('Another relevant own method process exists; no duplicate queue.')
    start=time.monotonic();charged=0.;jobs=[];prior_wall=0.
    if 'prior_progress_ref' in p:
        ref=p['prior_progress_ref'];previous=ROOT/ref['path']
        if sha256(previous)!=ref['sha256']:raise RuntimeError('Prior phase accounting changed.')
        prior=json.loads(previous.read_text(encoding='utf-8'));charged=prior['charged_gpu_hours'];jobs=list(prior['jobs'])
        jobs=[dict(row,guardian_path=(previous.parent/(row['name']+'_guardian.json')).relative_to(ROOT).as_posix())
              if 'guardian_path' not in row else row for row in jobs]
        prior_wall=max(float(prior.get('elapsed_wall_seconds',0)),time.time()-dt.datetime.fromisoformat(p['phase_origin_started_utc']).timestamp())
    write_json(out/'queue_progress.json',dict(jobs=jobs,charged_gpu_hours=charged),exclusive=True)
    def run(name,module,arguments,limit,protocol,is_cpu=False):
        nonlocal charged
        if prior_wall+time.monotonic()-start+limit+15>p['phase_total_wall_hours']*3600:
            raise RuntimeError('Whole-phase wall limit prevents another process.')
        if not is_cpu:
            if charged+limit/3600+.02>p['phase_gpu_hours']:
                raise RuntimeError('Aggregate GPU limit prevents another process.')
            memory=subprocess.check_output(['nvidia-smi','--id='+protocol['physical_gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
            if int(memory.strip())<p['minimum_free_gpu_mib']:
                raise RuntimeError('Selected GPU no longer meets the frozen free-memory gate; no model process launched.')
        visible='' if is_cpu else protocol['cuda_visible_devices']
        receipt=out/(name+'_guardian.json')
        command=['env','CUDA_VISIBLE_DEVICES='+visible,'bash','course/run.sh','-m','course.feature_polish.guard',
            '--receipt',str(receipt),'--limit-seconds',str(limit),'--module',module,'--']+arguments
        opened=time.monotonic()
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=limit+15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        elapsed=time.monotonic()-opened
        guardian=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        cost=0. if is_cpu else max(elapsed/3600,float(guardian.get('charged_gpu_wall_hours',0)))
        charged+=cost
        row=dict(name=name,module=module,pid=child.pid,exit_code=code,execution_device='cpu' if is_cpu else 'cuda:0',
            outer_seconds=elapsed,charged_gpu_wall_hours=cost,guardian_sha256=sha256(receipt) if receipt.exists() else None,
            guardian_path=receipt.relative_to(ROOT).as_posix() if receipt.exists() else None)
        jobs.append(row);write_json(out/'queue_progress.json',dict(jobs=jobs,charged_gpu_hours=charged,elapsed_wall_seconds=prior_wall+time.monotonic()-start))
        print('FEATURE_POLISH_JOB_END',json.dumps(row),flush=True)
        if code:raise RuntimeError('A process failed; preserve partial records without automatic retry: '+name)
    try:
        if 'reuse_synthetic_ref' in p:
            ref=p['reuse_synthetic_ref']
            if sha256(ROOT/ref['path'])!=ref['sha256'] or json.loads((ROOT/ref['path']).read_text())['status']!='passed':
                raise RuntimeError('Retained unchanged synthetic check changed.')
            write_json(out/'synthetic_reused.json',dict(reference=ref,new_execution=False),exclusive=True)
        else:
            opened=time.monotonic()
            with (out/'synthetic_checks.log').open('x',encoding='utf-8') as log:
                test=subprocess.run(['env','CUDA_VISIBLE_DEVICES=','bash','course/run.sh','-m','unittest','course.feature_polish.test_affine'],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=120)
            write_json(out/'synthetic_receipt.json',dict(status='passed' if test.returncode==0 else 'failed',exit_code=test.returncode,
                outer_seconds=time.monotonic()-opened,GPU_hours=0,source_commit=p['source_commit'],
                stdout_sha256=sha256(out/'synthetic_checks.log')),exclusive=True)
            if test.returncode:raise RuntimeError('Synthetic affine/feature-gradient checks failed.')
        teacher=ROOT/p['reuse_teacher_directory'] if 'reuse_teacher_directory' in p else stage/'teacher_attempt1'
        base_protocol=pathlib.Path(args.protocol).resolve()
        if 'reuse_teacher_directory' not in p:
            run('full_train_teacher','course.feature_polish.teacher',['--protocol',str(base_protocol),'--output',str(teacher)],p['teacher_wall_seconds'],p,is_cpu=True)
        teacher_result=json.loads((teacher/'result.json').read_text(encoding='utf-8'))
        manifest=teacher/'teacher_artifacts.json'
        if teacher_result['status']!='completed' or sha256(manifest)!=teacher_result['teacher_artifacts_sha256']:
            raise RuntimeError('Fresh full-TRAIN teacher did not finish its frozen fit.')
        # Resource choice uses only free-memory metadata, not predictions. Once
        # selected, one device is retained for all new graphs and conditions.
        q=dict(p,teacher_directory=teacher.relative_to(ROOT).as_posix(),teacher_training_source=p.get('teacher_training_source',p['source_commit']),teacher_preparation_protocol_sha256=p.get('teacher_preparation_protocol_sha256',provenance['protocol_sha256']),
            teacher_ref=dict(path=manifest.relative_to(ROOT).as_posix(),sha256=sha256(manifest)))
        q['retained_inputs']=list(p['retained_inputs'])+[dict(path=(teacher/f).relative_to(ROOT).as_posix(),sha256=sha256(teacher/f))
            for f in ('result.json','registered.json','teacher_artifacts.json','gaga_full_train_teacher.pt')]
        if 'reuse_execution_resource' in p:
            q.update(p['reuse_execution_resource'])
        else:
            resources=[];chosen=None
            for candidate in p['allowed_gpus']:
                raw=subprocess.check_output(['nvidia-smi','--id='+candidate['uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
                free=int(raw.strip());resources.append(dict(candidate,free_mib=free))
                if chosen is None and free>=p['minimum_free_gpu_mib']:chosen=candidate
            q.update(execution_device='cuda:0' if chosen else 'cpu',cuda_visible_devices=chosen['index'] if chosen else '',
                physical_gpu_uuid=chosen['uuid'] if chosen else None,
                resource_selection=dict(available_metadata=resources,selection_before_baseline_generation=True,
                    rule='First allowed GPU with >=8192MiB free, otherwise CPU uniformly'))
        preprotocol=stage/'refinement_protocol_v1.json';write_json(preprotocol,q,exclusive=True)
        cpu=q['execution_device']=='cpu'
        if 'reuse_passed_preflight_ref' in p:
            ref=p['reuse_passed_preflight_ref'];old_protocol_ref=p['preflight_refinement_protocol_ref']
            gate_path=ROOT/ref['path'];old_protocol_path=ROOT/old_protocol_ref['path']
            if sha256(gate_path)!=ref['sha256'] or sha256(old_protocol_path)!=old_protocol_ref['sha256']:
                raise RuntimeError('Retained passed preflight or its original protocol changed.')
            gate=json.loads(gate_path.read_text(encoding='utf-8'))
            original=json.loads(old_protocol_path.read_text(encoding='utf-8'))
            if (gate['status']!='passed' or gate['counts']['optimizer_steps']!=0 or
                    gate['counts']['official_victim_forwards']!=0 or
                    gate['provenance']['source_commit']!=p['preflight_execution_source'] or
                    gate['provenance']['protocol_sha256']!=old_protocol_ref['sha256'] or
                    original['source_commit']!=p['preflight_execution_source'] or
                    sha256(ROOT/'course/feature_polish/preflight.py')!=original['source_files']['course/feature_polish/preflight.py']):
                raise RuntimeError('Retained zero-update/no-victim preflight source/protocol binding differs.')
            q['preflight_ref']=dict(ref)
            write_json(out/'preflight_reused.json',dict(reference=ref,
                original_refinement_protocol=old_protocol_ref,execution_source=p['preflight_execution_source'],
                new_execution=False,optimizer_steps=0,official_victim_forwards=0),exclusive=True)
        else:
            preflight=stage/'preflight_attempt1'
            run('real_train_preflight','course.feature_polish.preflight',['--protocol',str(preprotocol),'--output',str(preflight)],p['preflight_wall_seconds'],q,is_cpu=cpu)
            if json.loads((preflight/'result.json').read_text(encoding='utf-8'))['status']!='passed':
                raise RuntimeError('Exact hard-forward/feature-gradient preflight failed.')
            q['preflight_ref']=dict(path=(preflight/'result.json').relative_to(ROOT).as_posix(),sha256=sha256(preflight/'result.json'))
        protocol=stage/'experiment_protocol_v1.json';write_json(protocol,q,exclusive=True)
        matrix=out/'refinements';matrix.mkdir()
        for seed in p['seeds']:
            directory=matrix/('seed%d'%seed)
            run('refine_seed%d'%seed,'course.feature_polish.refine',['--protocol',str(protocol),'--output',str(directory),'--seed',str(seed)],p['refine_job_wall_seconds'],q,is_cpu=cpu)
            if json.loads((directory/'result.json').read_text(encoding='utf-8'))['status']!='completed':
                raise RuntimeError('A complete refinement result is missing.')
        marker=dict(status='completed',jobs=jobs,charged_gpu_hours=charged,
            refinement_results={str(seed):sha256(matrix/('seed%d'%seed)/'result.json') for seed in p['seeds']})
        write_json(stage/'all3_refinements_complete.json',marker,exclusive=True)
        evaluation=stage/'evaluation_attempt1'
        run('development_evaluation','course.feature_polish.evaluate',['--protocol',str(protocol),'--matrix',str(matrix),'--output',str(evaluation)],p['evaluation_wall_seconds'],q,is_cpu=cpu)
        complete=dict(status='completed',jobs=jobs,charged_gpu_hours=charged,elapsed_wall_seconds=prior_wall+time.monotonic()-start,
            scientific_source=p['source_commit'],preparation_protocol_sha256=provenance['protocol_sha256'],
            experiment_protocol_sha256=sha256(protocol),evaluation_result_sha256=sha256(evaluation/'result.json'),
            execution_device=q['execution_device'],resource_selection=q['resource_selection'],
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),TEST_queries=0,experimental_API_calls=0)
        write_json(out/'queue_complete.json',complete,exclusive=True);write_json(stage/'queue_complete.json',complete,exclusive=True)
    except BaseException as error:
        result=dict(status='failed',error_type=type(error).__name__,error=str(error),jobs=jobs,
            charged_gpu_hours=charged,elapsed_wall_seconds=prior_wall+time.monotonic()-start)
        write_json(out/'queue_failed.json',result,exclusive=True);raise


if __name__=='__main__':main()
