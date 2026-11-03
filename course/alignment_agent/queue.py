"""One serial bounded follow-up matrix; never automatically retry a failure."""
import argparse
import datetime as dt
import fcntl
import json
import os
import pathlib
import random
import re
import signal
import subprocess
import time

from course.alignment_agent.common import ROOT, ARMS, sha256, write_json, validate_protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    protocol,provenance=validate_protocol(args.protocol)
    stage=pathlib.Path(protocol['stage_root'])
    output=pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:
        raise RuntimeError('Use a new output below ignored runs/.')
    lock=(stage/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    output.mkdir(parents=True,exist_ok=False)
    if not (stage/'bridge_claim.json').is_file():
        raise RuntimeError('Start the registered local API bridge before the GPU matrix.')
    claim=json.loads((stage/'bridge_claim.json').read_text())
    if (claim.get('protocol_sha256')!=provenance['protocol_sha256'] or
            type(claim.get('bridge_nonce')) is not str or
            not re.fullmatch(r'[0-9a-f]{32}',claim['bridge_nonce'])):
        raise RuntimeError('The bridge claim does not match this exact execution protocol.')
    charged=float(protocol['diagnostic_gpu_hours'])
    jobs=[]
    try:
        for seed in protocol['seeds']:
            for arm in ARMS:
                name='%s_seed%d'%(arm,seed)
                if charged+420/3600+.02>protocol['phase_gpu_hours']:
                    raise RuntimeError('The aggregate follow-up ceiling prevents another job.')
                extra=[]
                if arm=='smooth_random':
                    original=output/('smooth_agent_seed%d'%seed)/'outer_actions.json'
                    trace=json.loads(original.read_text())
                    actions=trace['actions']
                    if len(actions)!=8 or actions[0] or any(type(v) is not bool for v in actions):
                        raise RuntimeError('The same-seed applied agent schedule is incomplete.')
                    tail=list(actions[1:]);random.Random(20261103+seed).shuffle(tail)
                    schedule=output/('matched_schedule_seed%d.json'%seed)
                    write_json(schedule,dict(seed=seed,protocol_sha256=provenance['protocol_sha256'],
                        actions=[False]+tail,source_enabled_epochs=sum(actions),
                        source_actions_path=str(original.relative_to(ROOT)),source_actions_sha256=sha256(original),
                        derivation_rule='Uniform permutation of actual applied agent ON flags among epochs2..8, without model outcomes',random_seed=20261103+seed),exclusive=True)
                    extra=['--matched-schedule',str(schedule)]
                guardian=output/(name+'_guardian.json')
                command=['env','CUDA_VISIBLE_DEVICES=0','bash','course/run.sh','-m','course.alignment_agent.guard',
                    '--receipt',str(guardian),'--limit-seconds','420','--module','course.alignment_agent.run','--',
                    '--protocol',str(pathlib.Path(args.protocol).resolve()),'--arm',arm,'--seed',str(seed),'--output',str(output/name)]+extra
                started=time.monotonic()
                with (output/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code=child.wait(timeout=430)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:code=child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                elapsed=time.monotonic()-started
                inner=json.loads(guardian.read_text()) if guardian.exists() else {}
                cost=max(elapsed/3600,float(inner.get('charged_gpu_wall_hours',0)))
                charged+=cost
                row=dict(arm=arm,seed=seed,exit_code=code,pid=child.pid,elapsed_process_seconds=elapsed,
                    charged_gpu_wall_hours=cost,guardian_sha256=sha256(guardian) if guardian.exists() else None)
                jobs.append(row)
                write_json(output/'queue_progress.json',dict(jobs=jobs,pending=18-len(jobs),charged_gpu_hours_including_diagnostic=charged))
                print('FOLLOWUP_JOB_END',json.dumps(row),flush=True)
                if code:
                    raise RuntimeError('A job failed; preserve partial output, no automatic retry.')
        complete=dict(status='completed',jobs=jobs,charged_gpu_hours_including_diagnostic=charged,
            protocol_sha256=provenance['protocol_sha256'],completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            test_queries=0,gaga_forwards=0)
        write_json(output/'queue_complete.json',complete,exclusive=True)
        write_json(stage/'queue_complete.json',complete,exclusive=True)
    except BaseException as error:
        failure=dict(status='failed',error_type=type(error).__name__,error=str(error),jobs=jobs,
            charged_gpu_hours_including_diagnostic=charged,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(stage/'queue_failed.json',failure,exclusive=True)
        raise


if __name__=='__main__':
    main()
