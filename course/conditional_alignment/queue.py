"""Serial nine-run queue; charges guardian/process lifetime and stops on failure."""
import argparse
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from course.conditional_alignment.common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--stage-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    stage = pathlib.Path(args.stage_root).resolve()
    output = pathlib.Path(args.output).resolve()
    if (ROOT / 'runs').resolve() not in output.parents:
        raise RuntimeError('Run output must stay under ignored runs/.')
    lock = (stage / 'alignment_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(pathlib.Path(args.protocol).read_text())
    initial = json.loads((stage / 'fusion-self-guardian-attempt1.json').read_text())['charged_gpu_wall_hours']
    charged = float(initial)
    jobs = []
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            # Preserve room for the later bounded 300-second fusion extraction.
            if charged + (480 + 300) / 3600 > 2:
                raise RuntimeError('Aggregate pilot GPU ceiling prevents the next job.')
            name = '%s_seed%d' % (arm, seed)
            receipt = output / (name + '_guardian.json')
            command = ['env', 'CUDA_VISIBLE_DEVICES=0', 'bash', 'course/run.sh', '-m', 'course.new_methods_guard',
                '--receipt', str(receipt), '--limit-seconds', '480', '--module', 'course.conditional_alignment.run', '--',
                '--protocol', str(pathlib.Path(args.protocol).resolve()), '--reference', str(pathlib.Path(args.reference).resolve()),
                '--arm', arm, '--seed', str(seed), '--gpu', '0', '--output', str(output / name)]
            started = time.monotonic()
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=490)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        code = child.wait()
            elapsed = time.monotonic() - started
            guardian = json.loads(receipt.read_text()) if receipt.exists() else None
            cost = max(elapsed / 3600, 0 if guardian is None else guardian['charged_gpu_wall_hours'])
            charged += cost
            row = dict(arm=arm, seed=seed, exit_code=code, pid=child.pid,
                       elapsed_seconds=elapsed, charged_gpu_wall_hours=cost,
                       guardian_sha256=sha256(receipt) if receipt.exists() else None)
            jobs.append(row)
            write_json(output / 'queue_progress.json', dict(jobs=jobs, charged_with_self_gpu_hours=charged, pending=9-len(jobs)))
            print('QUEUE_RUN_END', json.dumps(row), flush=True)
            if code:
                raise RuntimeError('Pilot failed; partial evidence retained, no automatic rerun.')
    write_json(output / 'queue_complete.json', dict(status='completed', jobs=jobs,
        charged_with_self_gpu_hours=charged, api_calls=0, test_queries=0), exclusive=True)


if __name__ == '__main__':
    main()
