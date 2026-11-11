"""Exclusive, source-bound one-attempt guardian; preflight and run need separate invocations."""
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

from runtime import verify_protocol,load_json,write_json,sha


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path,value):
    temporary=path.with_name(path.name+'.tmp.%d'%os.getpid())
    write_json(temporary,value);os.replace(str(temporary),str(path))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--phase',required=True,choices=('preflight','run'));args=parser.parse_args()
    start=time.monotonic();p,provenance=verify_protocol(args.protocol);stage=pathlib.Path(args.stage).resolve()
    assets=pathlib.Path(p['asset_root']).resolve()
    if (assets/'runs') not in stage.parents:raise RuntimeError('Guardian stage must remain private ignored runs/.')
    stage.mkdir(parents=True,exist_ok=True)
    with (assets/'runs'/'feature_assignment_exclusive.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state_path=stage/'phase_state.json'
        if state_path.exists():
            state=load_json(state_path)
            if state['provenance']!=provenance:raise RuntimeError('Accounting belongs to another frozen source/protocol.')
        else:
            state=dict(provenance=provenance,phase_started_utc=utc(),rows=[],automatic_retry=False)
            write_json(state_path,state)
        if any(row['phase']==args.phase for row in state['rows']):raise RuntimeError('No automatic retry or repeated phase dispatch is permitted.')
        if args.phase=='run':
            passed=[row for row in state['rows'] if row['phase']=='preflight' and row.get('status')=='completed' and row.get('exit_code')==0]
            if len(passed)!=1:raise RuntimeError('The same registered preflight must complete before an independently authorized main invocation.')
            preflight=stage/'preflight_attempt1'/'result.json'
            gate=load_json(preflight)
            if gate['status']!='passed' or gate['provenance']!=provenance:raise RuntimeError('A passed same-protocol preflight is required.')
        origin=datetime.datetime.fromisoformat(state['phase_started_utc'])
        now=datetime.datetime.now(datetime.timezone.utc)
        total_cap=p['resource']['GPU_phase_seconds' if p['device']['kind']=='gpu' else 'CPU_phase_seconds']
        inner=p['resource']['preflight_inner_seconds' if args.phase=='preflight' else 'main_inner_seconds']
        outer=inner+p['resource']['outer_extra_seconds']
        elapsed=(now-origin).total_seconds()
        if elapsed+outer+30>total_cap:raise RuntimeError('Remaining whole-phase wall budget cannot fit this fixed job plus termination grace; no resource expansion.')
        output=stage/('preflight_attempt1' if args.phase=='preflight' else 'main_attempt1')
        if output.exists():raise RuntimeError('Exclusive attempt output already exists.')
        command=[sys.executable,str(pathlib.Path(__file__).with_name('preflight.py' if args.phase=='preflight' else 'run.py')),
            '--protocol',str(pathlib.Path(args.protocol).resolve()),'--output',str(output)]
        if args.phase=='run':command+=['--preflight',str(preflight)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=p['device']['cuda_visible_devices'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',PYTHONUNBUFFERED='1')
        row=dict(phase=args.phase,status='running',started_utc=utc(),output=str(output),inner_seconds=inner,outer_seconds=outer,command=command,
            selected_device=p['device'],protocol_sha256=provenance['protocol_sha256'])
        state['rows'].append(row);atomic(state_path,state)
        child=None;child_start=time.monotonic();failure=None;returncode=None
        try:
            with (stage/(args.phase+'_stdout.log')).open('xb') as stream:
                child=subprocess.Popen(command,cwd=p['runtime_root'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=child.pid;atomic(state_path,state)
                try:returncode=child.wait(timeout=outer)
                except subprocess.TimeoutExpired:
                    failure='Frozen outer deadline reached; preserving partial work without retry.'
                    os.killpg(child.pid,signal.SIGTERM)
                    try:returncode=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);returncode=child.wait(timeout=15)
        except BaseException as error:
            failure=repr(error)
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:returncode=child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);returncode=child.wait(timeout=15)
            traceback.print_exc()
        finally:
            wall=time.monotonic()-child_start
            finish=datetime.datetime.now(datetime.timezone.utc);phase_wall=(finish-origin).total_seconds()
            result_path=output/'result.json';saved=load_json(result_path) if result_path.exists() else None
            success=returncode==0 and saved is not None and saved['status']==('passed' if args.phase=='preflight' else 'completed')
            row.update(status='completed' if success else 'failed',exit_code=returncode,error=failure,finished_utc=finish.isoformat(),
                outer_wall_seconds=wall,child_result_sha256=sha(result_path) if saved is not None else None,
                whole_phase_wall_seconds_including_review_wait=phase_wall,
                whole_phase_GPU_hours=phase_wall/3600 if p['device']['kind']=='gpu' else 0,
                child_process_cpu_seconds=saved.get('process_cpu_seconds') if saved else None)
            atomic(state_path,state);write_json(stage/(args.phase+'_guardian.json'),row)
            print(json.dumps(row,allow_nan=False),flush=True)
        if not success:raise SystemExit(1)


if __name__=='__main__':main()
