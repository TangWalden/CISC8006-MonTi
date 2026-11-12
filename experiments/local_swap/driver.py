"""One exclusive preflight, one main run, one saved-record audit; no retries."""
import argparse,datetime,fcntl,json,os,pathlib,signal,subprocess,sys,time
from helpers import read,write,sha
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();stage=pathlib.Path(a.stage).resolve();p=read(stage/'registration/protocol.json')
    lock=(stage.parent/'local_swap_exclusive.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    receipt=stage/'driver_receipt.json';marker=stage/'launch_once.json';write(marker,dict(pid=os.getpid(),started_utc=utc(),protocol_sha256=sha(stage/'registration/protocol.json')))
    rows=[];start=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib')
    env['PYTHONPATH']=p['source_root']+':'+p['runtime_root']
    status='failed';child=None
    try:
        for phase,limit in [('preflight',p['resources']['preflight_seconds']),('main',p['resources']['main_seconds']),('audit',300)]:
            base=pathlib.Path(__file__).parent;protocol=str(stage/'registration/protocol.json')
            if phase=='audit':cmd=[sys.executable,str(base/'audit.py'),'--protocol',protocol,'--run',str(stage/'main'),'--output',str(stage/'audit.json')]
            else:
                cmd=[sys.executable,str(base/'run.py'),'--protocol',protocol,'--output',str(stage/phase)]
                cmd+=['--preflight'] if phase=='preflight' else ['--preflight-result',str(stage/'preflight/result.json')]
            began=time.monotonic();row=dict(phase=phase,started_utc=utc(),status='running');rows.append(row)
            with (stage/(phase+'.log')).open('xb') as log:
                child=subprocess.Popen(cmd,cwd=p['runtime_root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:rc=child.wait(timeout=limit+30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:rc=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
                    row['deadline_exceeded']=True
            row.update(exit_code=rc,wall_seconds=time.monotonic()-began,finished_utc=utc(),status='completed' if rc==0 else 'failed')
            print(json.dumps(row),flush=True)
            if rc:break
            if phase=='audit':status='completed'
    finally:
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
        write(receipt,dict(status=status,rows=rows,wall_seconds=time.monotonic()-start,finished_utc=utc(),GPU_hours=0,automatic_retry=False))
    if status!='completed':raise SystemExit(1)
if __name__=='__main__':main()
