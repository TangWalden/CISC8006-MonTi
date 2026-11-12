"""Isolated one-attempt CPU guardian for the action-blind control."""
import argparse
import datetime
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback

from common import verify_protocol, read_json, write_json, sha


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp.%d' % os.getpid())
    write_json(temporary, value)
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--phase', required=True, choices=('preflight', 'run'))
    args = parser.parse_args()
    guardian_start = time.monotonic()
    protocol, provenance = verify_protocol(args.protocol)
    stage = pathlib.Path(args.stage).resolve()
    assets = pathlib.Path(protocol['asset_root']).resolve()
    if (assets / 'runs').resolve() not in stage.parents:
        raise RuntimeError('The isolated guardian stage must remain below ignored runs/.')
    if protocol['device']['kind'] != 'cpu' or protocol['device']['cuda_visible_devices'] != '':
        raise RuntimeError('This guardian permits only the frozen CPU-only device.')
    if protocol['resource']['threads'] != 4 or protocol['resource'].get('automatic_retry') is not False:
        raise RuntimeError('CPU thread or retry policy changed.')
    stage.mkdir(parents=True, exist_ok=True)
    lock_path = assets / 'runs' / 'downstream_blind_exclusive.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = stage / 'phase_state.json'
        if state_path.exists():
            state = read_json(state_path)
            if state['provenance'] != provenance:
                raise RuntimeError('The phase accounting belongs to another frozen source/protocol.')
        else:
            state = dict(provenance=provenance, phase_started_utc=utc(), rows=[], automatic_retry=False)
            write_json(state_path, state)
        if any(row['phase'] == args.phase for row in state['rows']):
            raise RuntimeError('The phase already started; automatic retry or duplicate dispatch is forbidden.')
        preflight = stage / 'preflight_attempt1' / 'result.json'
        if args.phase == 'run':
            passed = [row for row in state['rows'] if row['phase'] == 'preflight'
                      and row.get('status') == 'completed' and row.get('exit_code') == 0]
            if len(passed) != 1:
                raise RuntimeError('The same registered preflight must complete before training.')
            gate = read_json(preflight)
            if gate['status'] != 'passed' or gate['provenance'] != provenance:
                raise RuntimeError('A passed same-protocol preflight is required.')
        origin = datetime.datetime.fromisoformat(state['phase_started_utc'])
        elapsed = (datetime.datetime.now(datetime.timezone.utc) - origin).total_seconds()
        inner = protocol['resource']['preflight_inner_seconds' if args.phase == 'preflight' else 'main_inner_seconds']
        outer = inner + protocol['resource']['outer_extra_seconds']
        if elapsed + outer + 30 > protocol['resource']['CPU_phase_seconds']:
            raise RuntimeError('The remaining whole-phase CPU wall budget cannot fit this job and termination grace.')
        output = stage / ('preflight_attempt1' if args.phase == 'preflight' else 'main_attempt1')
        if output.exists():
            raise RuntimeError('The exclusive phase output already exists.')
        command = [sys.executable, str(pathlib.Path(__file__).with_name('preflight.py' if args.phase == 'preflight' else 'run.py')),
                   '--protocol', str(pathlib.Path(args.protocol).resolve()), '--output', str(output)]
        if args.phase == 'run':
            command += ['--preflight', str(preflight)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        env.update(protocol['environment'])
        row = dict(phase=args.phase, status='running', started_utc=utc(), output=str(output),
                   inner_seconds=inner, outer_seconds=outer, command=command,
                   selected_device=protocol['device'], protocol_sha256=provenance['protocol_sha256'])
        state['rows'].append(row)
        atomic(state_path, state)
        child = None
        child_start = time.monotonic()
        error = None
        returncode = None

        def interrupted(signum, frame):
            raise InterruptedError('Guardian interrupted by signal %d.' % signum)

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            with (stage / (args.phase + '_stdout.log')).open('xb') as stream:
                child = subprocess.Popen(command, cwd=protocol['runtime_root'], env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = child.pid
                atomic(state_path, state)
                try:
                    returncode = child.wait(timeout=outer)
                except subprocess.TimeoutExpired:
                    error = 'Frozen outer deadline reached; preserve partial output without retry.'
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        returncode = child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        returncode = child.wait(timeout=15)
        except BaseException as exception:
            error = repr(exception)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    returncode = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    returncode = child.wait(timeout=15)
            traceback.print_exc()
        finally:
            finish = datetime.datetime.now(datetime.timezone.utc)
            phase_wall = (finish - origin).total_seconds()
            result_path = output / 'result.json'
            saved = None
            result_error = None
            if result_path.exists():
                try:
                    saved = read_json(result_path)
                except BaseException as exception:
                    result_error = repr(exception)
            wanted = 'passed' if args.phase == 'preflight' else 'completed'
            success = (returncode == 0 and saved is not None and saved.get('status') == wanted
                       and saved.get('provenance') == provenance)
            row.update(status='completed' if success else 'failed', exit_code=returncode, error=error,
                       result_read_error=result_error, finished_utc=finish.isoformat(),
                       outer_wall_seconds=time.monotonic() - child_start,
                       guardian_wall_seconds=time.monotonic() - guardian_start,
                       child_result_sha256=sha(result_path) if result_path.exists() else None,
                       whole_phase_wall_seconds_including_review_wait=phase_wall,
                       whole_phase_GPU_hours=0,
                       child_process_cpu_seconds=saved.get('process_cpu_seconds') if saved else None)
            atomic(state_path, state)
            write_json(stage / (args.phase + '_guardian.json'), row)
            print(json.dumps(row, allow_nan=False), flush=True)
        if not success:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
