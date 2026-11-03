"""Guardian for the separately authorized alignment-agent follow-up only."""
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
    parser.add_argument('--limit-seconds', required=True, type=int)
    parser.add_argument('--module', required=True, choices=(
        'course.alignment_agent.diagnostics', 'course.alignment_agent.run'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.limit_seconds <= 480:
        raise RuntimeError('The follow-up guardian cap cannot exceed480seconds.')
    receipt = pathlib.Path(args.receipt)
    if receipt.exists():
        raise RuntimeError('Preserve the previous attempt; choose a new receipt.')
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        module=args.module, pid=os.getpid(), arguments=args.arguments,
        limit_seconds=args.limit_seconds, status='started')
    def stop(signum, frame):
        raise TimeoutError('Follow-up wall-clock guardian expired or was terminated.')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(args.limit_seconds)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise RuntimeError('Numeric CUDA visibility0 is required.')
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch._C._cuda_getDeviceCount() != 1:
            raise RuntimeError('Torch and native CUDA visible counts must both be1.')
        torch.cuda.set_device(0)
        torch.zeros(1, device='cuda:0').sum().item()
        expected = 'GPU-d7aa6d97-d823-0319-b218-84acc8e26019'
        own = []
        for _ in range(5):
            output = subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], universal_newlines=True)
            own = [s.split(',')[1].strip() for s in output.splitlines()
                   if s.split(',')[0].strip() == str(os.getpid())]
            if own:
                break
            time.sleep(.1)
        if set(own) != {expected}:
            raise RuntimeError('This process is not exclusively on the registered physical GPU.')
        result.update(physical_gpu_uuid=expected, torch=torch.__version__, cuda=torch.version.cuda)
        argv = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
        sys.argv = [args.module] + argv
        runpy.run_module(args.module, run_name='__main__')
        result['status'] = 'completed'
    except SystemExit as error:
        result.update(status='completed' if error.code in (None,0) else 'failed', exit_code=error.code)
        raise
    except BaseException as error:
        result.update(status='failed', exception_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.alarm(0)
        result['elapsed_seconds'] = time.monotonic()-started
        result['charged_gpu_wall_hours'] = result['elapsed_seconds']/3600
        result['accounting'] = 'Complete one-GPU guardian lifetime including setup, failures and API wait; outer queue may charge more.'
        with receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
        print('FOLLOWUP_GUARDIAN',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
