"""Finite six-run comparison with a cumulative GPU wall-time ceiling."""
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time

from course.runner import ROOT, write_json


def main():
    if list((ROOT / 'evidence/runs').glob('*_reproduction_*/registered.json')):
        raise RuntimeError('Formal registrations already exist; audit them before any retry.')
    protocol = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Commit tracked changes before launching frozen baseline.')
    previous_seconds = sum(json.loads(p.read_text()).get('elapsed_seconds', 0) for p in (ROOT / 'evidence/runs').glob('*/result.json'))
    remaining = protocol['gpu_hour_limit'] * 3600 - previous_seconds
    consumed = 0.0
    (ROOT / 'runs').mkdir(exist_ok=True)
    events = ROOT / 'evidence/queue_events.jsonl'

    def event(value):
        value['utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        with events.open('a') as stream:
            stream.write(json.dumps(value) + '\n')
        print(json.dumps(value), flush=True)

    for seed in protocol['seeds']:
        processes = []
        for gpu in (0, 1):
            mode = ('learned' if gpu == 0 else 'random') if seed % 2 == 0 else ('random' if gpu == 0 else 'learned')
            log_path = ROOT / 'runs' / ('formal_%s_seed%d_gpu%d.log' % (mode, seed, gpu))
            stream = log_path.open('x')
            command = [sys.executable, '-u', '-m', 'course.runner', '--config', protocol['configuration'],
                       '--phase', 'reproduction', '--mode', mode, '--seed', str(seed), '--gpu', str(gpu)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((process, stream))
            event(dict(event='start', seed=seed, mode=mode, gpu=gpu, pid=process.pid, command=command, log=str(log_path.relative_to(ROOT))))
        tick = time.monotonic()
        while any(process.poll() is None for process, _ in processes):
            active = sum(process.poll() is None for process, _ in processes)
            time.sleep(5)
            now = time.monotonic()
            consumed += (now-tick) * active
            tick = now
            if consumed >= remaining:
                for process, _ in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                event(dict(event='budget_stop', total_gpu_wall_hours=(consumed+previous_seconds)/3600))
                for process, stream in processes:
                    process.wait()
                    stream.close()
                raise SystemExit(2)
        codes = []
        for process, stream in processes:
            codes.append(process.returncode)
            stream.close()
        event(dict(event='round_end', seed=seed, exit_codes=codes, total_gpu_wall_hours=(consumed+previous_seconds)/3600))
        if any(codes):
            event(dict(event='failed_stop', reason='preserve failure and audit before continuing'))
            raise SystemExit(1)
    subprocess.run([sys.executable, '-m', 'course.analyze'], cwd=ROOT, check=True)
    event(dict(event='all_complete', total_gpu_wall_hours=(consumed+previous_seconds)/3600))


if __name__ == '__main__':
    main()
