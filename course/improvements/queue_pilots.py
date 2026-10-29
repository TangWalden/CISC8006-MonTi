"""Finite nine-run queue. No test evaluation, implicit retries or unrelated process changes."""
import datetime as dt
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from course.runner import ROOT, write_json


def main():
    lock = (ROOT / 'runs/improvement_pilot_queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((ROOT / 'course/improvements/pilot_protocol_v1.json').read_text())
    events = (ROOT / 'evidence/improvement_pilot_events.jsonl').open('x')
    def event(kind, **data):
        row = dict(event=kind, utc=dt.datetime.now(dt.timezone.utc).isoformat(), **data)
        events.write(json.dumps(row)+'\n')
        events.flush()
        print(json.dumps(row), flush=True)
    if list((ROOT / 'evidence/improvements').glob('*/registered.json')):
        raise RuntimeError('Pilot attempts already exist. Review rather than automatically rerunning.')
    event('queue_start', pid=os.getpid(), limit_gpu_hours=protocol['pilot_gpu_hour_limit'])
    pending = [(variant, seed) for seed in protocol['seeds'] for variant in protocol['variants']]
    active, completed_cost = [], 0.
    try:
        while pending or active:
            used = completed_cost + sum(time.monotonic()-job['start'] for job in active)
            if used/3600 >= protocol['pilot_gpu_hour_limit']:
                event('budget_stop', gpu_wall_hours=used/3600)
                raise RuntimeError('Pilot GPU-hour cap reached.')
            occupied = {job['gpu'] for job in active}
            for gpu in [0, 1]:
                if pending and gpu not in occupied:
                    variant, seed = pending.pop(0)
                    log = (ROOT / 'runs' / ('improvement_pilot_%s_seed%d_gpu%d_attempt1.log' % (variant, seed, gpu))).open('x')
                    command = [sys.executable, '-u', '-m', 'course.improvements.run', '--variant', variant, '--seed', str(seed), '--gpu', str(gpu)]
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active.append(dict(process=child, log=log, start=time.monotonic(), variant=variant, seed=seed, gpu=gpu))
                    event('run_start', variant=variant, seed=seed, gpu=gpu, pid=child.pid, command=command)
            for job in active[:]:
                code = job['process'].poll()
                if code is None:
                    continue
                elapsed = time.monotonic()-job['start']
                completed_cost += elapsed
                job['log'].close()
                active.remove(job)
                event('run_end', variant=job['variant'], seed=job['seed'], gpu=job['gpu'], exit_code=code, elapsed_seconds=elapsed)
                if code:
                    raise RuntimeError('A pilot failed; preserve evidence and review before any retry.')
            time.sleep(2)
        subprocess.run([sys.executable, '-m', 'course.improvements.analyze_pilots'], cwd=ROOT, check=True)
        event('all_complete', gpu_wall_hours=completed_cost/3600)
    except BaseException as error:
        event('failed_stop', error=repr(error))
        for job in active:
            if job['process'].poll() is None:
                os.killpg(job['process'].pid, signal.SIGTERM)
                event('own_child_terminated', pid=job['process'].pid, variant=job['variant'], seed=job['seed'])
            job['log'].close()
        raise
    finally:
        events.close()


if __name__ == '__main__':
    main()
