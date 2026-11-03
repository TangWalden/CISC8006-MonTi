"""One serial, capped preparation/TRAIN/development queue; no automatic retry."""
import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import time

from course.dual_surrogate.common import ARMS,ROOT,private_output,sha256,validate_protocol,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p,provenance=validate_protocol(args.protocol)
    stage=pathlib.Path(p['stage_root']);out=private_output(args.output,create=True)
    lock=(stage/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    charged=0.0;jobs=[];attempt=int(p.get('execution_attempt',1))
    if 'prior_progress_ref' in p:
        row=p['prior_progress_ref'];path=ROOT/row['path']
        if sha256(path)!=row['sha256']:raise RuntimeError('Prior accounting bytes changed.')
        previous=json.loads(path.read_text());charged=float(previous['charged_gpu_hours']);jobs=list(previous['jobs'])
    def run(name,module,arguments,limit):
        nonlocal charged
        if charged+limit/3600+.02>p['phase_gpu_hours']:raise RuntimeError('Aggregate ceiling prevents another process.')
        memory=subprocess.check_output(['nvidia-smi','--id='+p['physical_gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
        if int(memory.strip())<int(p.get('minimum_free_gpu_mib',8192)):
            raise RuntimeError('Registered GPU has insufficient free memory; no model process launched.')
        receipt=out/(name+'_guardian.json');command=['env','CUDA_VISIBLE_DEVICES='+p.get('cuda_visible_devices','0'),'bash','course/run.sh','-m',
            'course.dual_surrogate.guard','--receipt',str(receipt),'--limit-seconds',str(limit),'--module',module,'--']+arguments
        start=time.monotonic()
        with (out/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=limit+15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        elapsed=time.monotonic()-start;guardian=json.loads(receipt.read_text()) if receipt.exists() else {}
        cost=max(elapsed/3600,float(guardian.get('charged_gpu_wall_hours',0)));charged+=cost
        row=dict(name=name,module=module,exit_code=code,pid=child.pid,outer_seconds=elapsed,charged_gpu_wall_hours=cost,
            guardian_sha256=sha256(receipt) if receipt.exists() else None)
        jobs.append(row);write_json(out/'queue_progress.json',dict(jobs=jobs,charged_gpu_hours=charged))
        print('DUAL_JOB_END',json.dumps(row),flush=True)
        if code:raise RuntimeError('A process failed; partial records retained, no automatic retry: '+name)
        return row
    try:
        teacher=ROOT/p['reuse_teacher_directory'] if 'reuse_teacher_directory' in p else stage/'teachers_attempt1'
        if 'reuse_teacher_directory' not in p:
            run('teacher_preparation','course.dual_surrogate.prepare_teachers',
                ['--protocol',str(pathlib.Path(args.protocol).resolve()),'--output',str(teacher),'--gpu','0'],p['teacher_wall_seconds'])
        teacher_result=json.loads((teacher/'result.json').read_text())
        if teacher_result['status']!='completed':raise RuntimeError('Teacher quality/readiness failed.')
        q=dict(p,teacher_directory=str(teacher.relative_to(ROOT)),
            teacher_artifacts_ref=dict(path=str((teacher/'teacher_artifacts.json').relative_to(ROOT)),sha256=sha256(teacher/'teacher_artifacts.json')))
        q.pop('_protocol_path',None);q.pop('_protocol_sha256',None)
        preprotocol=stage/('preflight_protocol_v%d.json'%attempt);write_json(preprotocol,q,exclusive=True)
        preflight=stage/('preflight_attempt%d'%attempt)
        run('real_train_preflight','course.dual_surrogate.preflight',
            ['--protocol',str(preprotocol),'--output',str(preflight)],p['preflight_wall_seconds'])
        if json.loads((preflight/'result.json').read_text())['status']!='passed':raise RuntimeError('Preflight did not pass.')
        q['preflight_ref']=dict(path=str((preflight/'result.json').relative_to(ROOT)),sha256=sha256(preflight/'result.json'))
        protocol=stage/('experiment_protocol_v%d.json'%attempt);write_json(protocol,q,exclusive=True)
        for seed in p['seeds']:
            for arm in ARMS:
                name='%s_seed%d'%(arm,seed)
                run(name,'course.dual_surrogate.run',['--protocol',str(protocol),'--output',str(out/name),
                    '--arm',arm,'--seed',str(seed)],p['train_job_wall_seconds'])
                result=json.loads((out/name/'result.json').read_text())
                if result['status']!='completed' or result['optimizer_steps']!=2048:raise RuntimeError('Incomplete terminal training.')
        write_json(stage/'all12_trainings_complete.json',dict(status='completed',jobs=jobs,charged_gpu_hours=charged),exclusive=True)
        evaluation=stage/'evaluation_attempt1'
        run('development_evaluation','course.dual_surrogate.evaluate',
            ['--protocol',str(protocol),'--matrix',str(out),'--output',str(evaluation)],p['evaluation_wall_seconds'])
        complete=dict(status='completed',jobs=jobs,charged_gpu_hours=charged,
            scientific_source=provenance['source_commit'],preparation_protocol_sha256=provenance['protocol_sha256'],
            experiment_protocol_sha256=sha256(protocol),evaluation_result_sha256=sha256(evaluation/'result.json'),
            completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),test_queries=0,LLM_API_calls=0)
        write_json(out/'queue_complete.json',complete,exclusive=True);write_json(stage/'queue_complete.json',complete,exclusive=True)
    except BaseException as error:
        failure=dict(status='failed',jobs=jobs,charged_gpu_hours=charged,error_type=type(error).__name__,error=str(error))
        write_json(out/'queue_failed.json',failure,exclusive=True)
        write_json(stage/(out.name+'_queue_failed.json'),failure,exclusive=True)
        raise


if __name__=='__main__':main()
