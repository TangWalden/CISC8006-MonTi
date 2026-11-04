"""One exclusive serial development queue; no automatic retries or tuning."""
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

from course.source_role_fix.common import ARMS,ROOT,private_output,sha256,validate_protocol,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();p,provenance=validate_protocol(args.protocol)
    stage=pathlib.Path(p['stage_root']);out=private_output(args.output,create=True)
    lock=(stage/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pattern=r'(^|/)(python[0-9.]*) .*course\.(dual_surrogate|source_role_fix|alignment_agent|conditional_alignment|fusion_pilot)\.(run|guard|evaluate|preflight|queue)'
    active=subprocess.run(['pgrep','-u',getpass.getuser(),'-f',pattern],stdout=subprocess.PIPE,universal_newlines=True)
    other={int(value) for value in active.stdout.split()}-{os.getpid()}
    if active.returncode not in (0,1) or other:
        raise RuntimeError('Another relevant own model/queue process exists; no duplicate jobs launched.')
    charged=0.;jobs=[]
    write_json(out/'queue_progress.json',dict(jobs=jobs,charged_gpu_hours=charged),exclusive=True)
    def run(name,module,arguments,limit):
        nonlocal charged
        if charged+limit/3600+.02>p['phase_gpu_hours']:
            raise RuntimeError('Aggregate ceiling prevents another process.')
        memory=subprocess.check_output(['nvidia-smi','--id='+p['physical_gpu_uuid'],'--query-gpu=memory.free','--format=csv,noheader,nounits'],universal_newlines=True)
        if int(memory.strip())<int(p['minimum_free_gpu_mib']):
            raise RuntimeError('Registered GPU has insufficient free memory; no model process launched.')
        receipt=out/(name+'_guardian.json')
        command=['env','CUDA_VISIBLE_DEVICES='+p['cuda_visible_devices'],'bash','course/run.sh','-m',
            'course.source_role_fix.guard','--receipt',str(receipt),'--limit-seconds',str(limit),'--module',module,'--']+arguments
        started=time.monotonic()
        with (out/(name+'.log')).open('x',encoding='utf-8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=limit+15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:code=child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);code=child.wait()
        elapsed=time.monotonic()-started
        guardian=json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        cost=max(elapsed/3600,float(guardian.get('charged_gpu_wall_hours',0)));charged+=cost
        row=dict(name=name,module=module,exit_code=code,pid=child.pid,outer_seconds=elapsed,charged_gpu_wall_hours=cost,
            guardian_sha256=sha256(receipt) if receipt.exists() else None)
        jobs.append(row);write_json(out/'queue_progress.json',dict(jobs=jobs,charged_gpu_hours=charged))
        print('SOURCE_ROLE_JOB_END',json.dumps(row),flush=True)
        if code:raise RuntimeError('A process failed; partial records retained, no automatic retry: '+name)
        return row
    try:
        synthetic_start=time.monotonic()
        with (out/'synthetic_checks.log').open('x',encoding='utf-8') as log:
            test=subprocess.run(['env','CUDA_VISIBLE_DEVICES=','bash','course/run.sh','-m','unittest','course.source_role_fix.test_model'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        synthetic=dict(status='passed' if test.returncode==0 else 'failed',exit_code=test.returncode,
            outer_seconds=time.monotonic()-synthetic_start,GPU_hours=0,source_commit=provenance['source_commit'],
            stdout_log='synthetic_checks.log',stdout_sha256=sha256(out/'synthetic_checks.log'))
        write_json(out/'synthetic_receipt.json',synthetic,exclusive=True)
        if test.returncode:raise RuntimeError('CPU synthetic role/mask/gradient checks failed.')
        protocol=pathlib.Path(args.protocol).resolve()
        preflight=stage/'preflight_attempt1'
        run('real_train_preflight','course.source_role_fix.preflight',
            ['--protocol',str(protocol),'--output',str(preflight)],p['preflight_wall_seconds'])
        if json.loads((preflight/'result.json').read_text(encoding='utf-8'))['status']!='passed':
            raise RuntimeError('Preflight did not pass.')
        q=dict(p,preflight_ref=dict(path=(preflight/'result.json').relative_to(ROOT).as_posix(),sha256=sha256(preflight/'result.json')))
        protocol=stage/'experiment_protocol_v1.json';write_json(protocol,q,exclusive=True)
        if p['companion_saved_graph_diagnostic']['enabled']:
            run('saved_teacher_diagnostic','course.source_role_fix.saved_teacher_diagnostic',
                ['--protocol',str(protocol),'--output',str(stage/'saved_teacher_diagnostic_attempt1')],p['diagnostic_wall_seconds'])
        for seed in p['seeds']:
            for arm in ARMS:
                name='%s_seed%d'%(arm,seed)
                run(name,'course.source_role_fix.run',['--protocol',str(protocol),'--output',str(out/name),
                    '--arm',arm,'--seed',str(seed)],p['train_job_wall_seconds'])
                result=json.loads((out/name/'result.json').read_text(encoding='utf-8'))
                if result['status']!='completed' or result['optimizer_steps']!=2048 or result['arm']!=arm:
                    raise RuntimeError('A terminal TRAIN result is incomplete or mislabeled.')
        write_json(stage/'all6_trainings_complete.json',dict(status='completed',jobs=jobs,charged_gpu_hours=charged),exclusive=True)
        evaluation=stage/'evaluation_attempt1'
        run('development_evaluation','course.source_role_fix.evaluate',
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
