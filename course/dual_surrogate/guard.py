"""One physical-GPU guardian; all failed/idle/process time remains charged."""
import argparse
import datetime as dt
import json
import os
import pathlib
import runpy
import signal
import sys
import time

from course.dual_surrogate.common import EXPECTED_GPU, write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--limit-seconds',type=int,required=True)
    parser.add_argument('--module',required=True,choices=[
        'course.dual_surrogate.prepare_teachers','course.dual_surrogate.preflight',
        'course.dual_surrogate.run','course.dual_surrogate.evaluate'])
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    path=pathlib.Path(args.receipt)
    if path.exists() or not 1<=args.limit_seconds<=900:raise RuntimeError('New receipt and bounded limit required.')
    path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    receipt=dict(module=args.module,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        limit_seconds=args.limit_seconds,pid=os.getpid(),arguments=args.arguments,status='started')
    def stop(signum,frame):raise TimeoutError('Registered GPU guardian deadline reached.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(args.limit_seconds)
    try:
        from course.alignment_agent.diagnostics import own_gpu_uuid
        import torch
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Numeric CUDA visibility0 required.')
        receipt['own_physical_uuids']=own_gpu_uuid(EXPECTED_GPU)
        receipt['torch']=torch.__version__
        argv=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        sys.argv=[args.module]+argv
        runpy.run_module(args.module,run_name='__main__');receipt['status']='completed'
    except SystemExit as error:
        receipt.update(status='completed' if error.code in (None,0) else 'failed',exit_code=error.code)
        raise
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);receipt['elapsed_seconds']=time.monotonic()-start
        receipt['charged_gpu_wall_hours']=receipt['elapsed_seconds']/3600
        receipt['accounting']='Complete one-GPU process guardian; charge max guardian and outer process; includes failures/idle.'
        write_json(path,receipt,exclusive=True)
        print('DUAL_GUARDIAN',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
