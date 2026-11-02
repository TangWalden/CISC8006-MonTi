"""Bound one explicitly authorized GPU pilot in this process and retain its cost."""
import argparse
import datetime as dt
import json
import os
import pathlib
import runpy
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--limit-seconds', type=int, required=True)
    parser.add_argument('--module', required=True, choices=[
        'course.fusion_pilot.self_evidence', 'course.conditional_alignment.run',
        'course.fusion_pilot.extract_attacked'])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.limit_seconds <= 900:
        parser.error('Pilot process limit must be between one and nine hundred seconds.')
    receipt = pathlib.Path(args.receipt)
    if receipt.exists():
        raise RuntimeError('Receipt already exists; review the prior attempt.')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  pid=os.getpid(), module=args.module, arguments=args.arguments,
                  limit_seconds=args.limit_seconds, status='started')
    def expired(signum, frame):
        raise TimeoutError('Bounded pilot wall-time limit or termination signal.')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.alarm(args.limit_seconds)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('This pilot requires numeric CUDA visibility zero.')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch._C._cuda_getDeviceCount() != 1:
            raise RuntimeError('PyTorch/native visible-device counts must both equal one.')
        torch.cuda.set_device(0)
        torch.zeros(1, device='cuda:0').sum().item()
        expected = 'GPU-d7aa6d97-d823-0319-b218-84acc8e26019'
        own = []
        for attempt in range(5):
            rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                            '--format=csv,noheader'], text=True)
            own = [row.split(',')[1].strip() for row in rows.splitlines()
                   if row.split(',')[0].strip() == str(os.getpid())]
            if own:
                break
            time.sleep(.1)
        if not own or set(own) != {expected}:
            raise RuntimeError('Own process did not resolve exclusively to the registered physical GPU.')
        result.update(gpu_uuid=expected, own_process_uuids=own,
                      torch=torch.__version__, cuda=torch.version.cuda)
        arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        sys.argv = [args.module] + arguments
        runpy.run_module(args.module, run_name='__main__')
        result['status'] = 'completed'
    except SystemExit as error:
        result.update(status='completed' if error.code in (None, 0) else 'failed',
                      exit_code=error.code)
        raise
    except BaseException as error:
        result.update(status='failed', exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        result['elapsed_seconds'] = time.monotonic() - started
        result['charged_gpu_wall_hours'] = result['elapsed_seconds'] / 3600
        result['cost_policy'] = 'Conservatively charge complete guardian lifetime, including imports and metadata.'
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print('GUARD_RECEIPT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
