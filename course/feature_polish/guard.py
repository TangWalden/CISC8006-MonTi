"""One physical-GPU guardian; all failed/idle/process time remains charged."""
import argparse
import datetime as dt
import json
import os
import pathlib
import runpy
import signal
import sys
import subprocess
import time

from course.feature_polish.common import write_json,validate_protocol


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--limit-seconds',type=int,required=True)
    parser.add_argument('--module',required=True,choices=[
        'course.feature_polish.teacher','course.feature_polish.preflight',
        'course.feature_polish.refine','course.feature_polish.evaluate'])
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    path=pathlib.Path(args.receipt)
    if path.exists() or not 1<=args.limit_seconds<=1800:raise RuntimeError('New receipt and bounded limit required.')
    path.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();cpu_start=time.process_time()
    receipt=dict(module=args.module,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        limit_seconds=args.limit_seconds,pid=os.getpid(),arguments=args.arguments,status='started')
    def stop(signum,frame):raise TimeoutError('Registered GPU guardian deadline reached.')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(args.limit_seconds)
    try:
        argv=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        protocol,_=validate_protocol(argv[argv.index('--protocol')+1])
        import torch
        cpu = args.module == 'course.feature_polish.teacher' or protocol['execution_device'] == 'cpu'
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_available() or torch.cuda.device_count()!=0 or torch._C._cuda_getDeviceCount()!=0:
                raise RuntimeError('Registered CPU process must expose zero CUDA devices.')
            receipt.update(execution_device='cpu',numeric_visibility='',torch_visible_count=0,native_visible_count=0,own_physical_uuids=[])
        else:
            receipt['execution_device']='cuda:0'
            visible=protocol['cuda_visible_devices']
            if visible not in ('0','1') or os.environ.get('CUDA_VISIBLE_DEVICES')!=visible:
                raise RuntimeError('Registered numeric single-GPU visibility required.')
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch._C._cuda_getDeviceCount()!=1:
                raise RuntimeError('Torch/native visible GPU counts must equal one.')
            torch.cuda.set_device(0);torch.zeros(1,device='cuda:0').sum().item()
            own=set()
            for _ in range(5):
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],universal_newlines=True)
                own={row.split(',')[1].strip() for row in rows.splitlines() if row.split(',')[0].strip()==str(os.getpid())}
                if own:break
                time.sleep(.1)
            if own!={protocol['physical_gpu_uuid']}:raise RuntimeError('Own physical GPU UUID differs from registered device.')
            receipt.update(execution_device='cuda:0',own_physical_uuids=sorted(own),numeric_visibility=visible,torch_visible_count=1,native_visible_count=1)
        receipt['torch']=torch.__version__
        sys.argv=[args.module]+argv
        runpy.run_module(args.module,run_name='__main__');receipt['status']='completed'
    except SystemExit as error:
        receipt.update(status='completed' if error.code in (None,0) else 'failed',exit_code=error.code)
        raise
    except BaseException as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        signal.alarm(0);receipt['elapsed_seconds']=time.monotonic()-start
        receipt['process_cpu_seconds']=time.process_time()-cpu_start
        receipt['charged_gpu_wall_hours']=receipt['elapsed_seconds']/3600 if receipt.get('execution_device')=='cuda:0' else 0.0
        receipt['accounting']='CPU device: zero GPU charge and actual CPU/wall clocks; GPU device: maximum outer/guardian lifetime includes failures and idle.'
        write_json(path,receipt,exclusive=True)
        print('FEATURE_POLISH_GUARDIAN',json.dumps(receipt),flush=True)


if __name__=='__main__':main()
