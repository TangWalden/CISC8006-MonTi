"""One exclusive TRAIN preflight and complete-cohort run; no automatic retry."""
import pathlib,argparse,time,subprocess,os,signal,fcntl,datetime,sys
from common import read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);p=read(s/'protocol.json');lock=(s.parent/'portal_complete_exclusive.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 write(s/'launch_once.json',dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(s/'protocol.json')))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib',PYTHONUNBUFFERED='1');rows=[];start=time.monotonic();status='failed';child=None
 try:
  for name,limit in [('preflight',900),('run',p['resources']['main_seconds'])]:
   began=time.monotonic()
   with (s/(name+'.log')).open('xb') as log:
    child=subprocess.Popen(['nice','-n','10',sys.executable,str(s/'source'/(name+'.py')),'--stage',str(s)],cwd=p['runtime_root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=child.wait(timeout=limit+30)
    except subprocess.TimeoutExpired:
     os.killpg(child.pid,signal.SIGTERM)
     try:rc=child.wait(timeout=15)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
   rows.append(dict(phase=name,pid=child.pid,exit_code=rc,wall_seconds=time.monotonic()-began));print(rows[-1],flush=True)
   if rc:break
   if name=='run':status='completed'
 finally:
  if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15)
  write(s/'guardian_receipt.json',dict(status=status,phases=rows,wall_seconds=time.monotonic()-start,GPU_hours=0,automatic_retry=False))
 if status!='completed':raise SystemExit(1)
if __name__=='__main__':main()
