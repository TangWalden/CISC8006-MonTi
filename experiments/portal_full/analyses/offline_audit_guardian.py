"""Server-local postprocessing of one fixed completed experiment, no retries."""
import pathlib,json,os,time,subprocess,sys,hashlib,fcntl,datetime
STAGE=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v,exclusive=False):
 with p.open('x' if exclusive else 'w') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 s=STAGE;lock=(s/'offline_audit_exclusive.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reg=read(s/'saved_audit_registration.json');assert sha(s/'saved_audit.py')==reg['source_sha256'];assert sha(s/'protocol.json')=='206225b7a23e40e941571fd050630d7eb42f6c2ed7b7d406317f98272885f5a1'
 start=time.monotonic();deadline=start+13*3600;state=s/'offline_audit_state.json'
 write(s/'offline_audit_started.json',dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_source_sha256=reg['source_sha256']),exclusive=True)
 while not (s/'guardian_receipt.json').exists():
  if time.monotonic()>=deadline:raise TimeoutError('No scientific completion receipt within bounded waiting period')
  write(state,dict(status='waiting_for_scientific_completion',pid=os.getpid()));time.sleep(45)
 guardian=read(s/'guardian_receipt.json')
 if guardian['status']!='completed' or read(s/'main/result.json')['status']!='completed':
  result=dict(status='skipped_scientific_failure',graph_classifier_calls=0,parameter_loads=0,GPU_hours=0,waiting_wall_seconds=time.monotonic()-start);write(s/'offline_audit_receipt.json',result,exclusive=True);write(state,result);return
 wait=time.monotonic()-start;write(state,dict(status='auditing_saved_records',pid=os.getpid(),waiting_wall_seconds=wait));began=time.monotonic();error=None;rc=None
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib')
 try:
  with (s/'audit.stdout').open('xb') as out,(s/'audit.stderr').open('xb') as err:
   r=subprocess.run(['nice','-n','10','/home/tjy/miniconda3/envs/monti/bin/python',str(s/'saved_audit.py'),'--stage',str(s)],cwd='/home/tjy/Project/MonTi-course',env=env,stdout=out,stderr=err,timeout=3600);rc=r.returncode
 except BaseException as e:error=dict(type=type(e).__name__,message=str(e))
 result=dict(status='passed' if rc==0 and (s/'saved_audit.json').exists() else 'failed',exit_code=rc,error=error,waiting_wall_seconds=wait,audit_outer_wall_seconds=time.monotonic()-began,audit_source_sha256=reg['source_sha256'],GPU_hours=0,automatic_retry=False)
 write(s/'offline_audit_receipt.json',result,exclusive=True);write(state,result)
 if result['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
