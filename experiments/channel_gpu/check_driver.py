"""Bounded readiness/timing check while the registered CPU run continues."""
import argparse,pathlib,json,subprocess,sys,os,time,signal,concurrent.futures
from multiprocessing.connection import Client
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);p=json.loads((s/'protocol.json').read_text());rows=[];start=time.monotonic();gpu=None;gpu_start=None;status='failed'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib',PYTHONUNBUFFERED='1')
 def run(ds,mode):
  began=time.monotonic()
  with (s/(mode+'_'+ds+'.log')).open('xb') as log:
   child=subprocess.Popen(['nice','-n','10',sys.executable,str(s/'source/preflight.py'),'--stage',str(s),'--dataset',ds,'--mode',mode],env=env,cwd=p['runtime_root'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:rc=child.wait(timeout=180)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGTERM);rc=child.wait(timeout=10)
  return dict(dataset=ds,mode=mode,exit_code=rc,wall_seconds=time.monotonic()-began)
 try:
  for ds in p['datasets']:rows.append(run(ds,'export'));assert rows[-1]['exit_code']==0
  gpu_env=dict(env,CUDA_VISIBLE_DEVICES=str(p['gpu_ordinal']));gpu_start=time.monotonic()
  with (s/'gpu.log').open('xb') as log:gpu=subprocess.Popen([sys.executable,str(s/'source/gpu_service.py'),'--stage',str(s)],env=gpu_env,cwd=p['runtime_root'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while not (s/'gpu_ready.json').exists():
   assert gpu.poll() is None and time.monotonic()-gpu_start<60,'GPU startup failed or timed out';time.sleep(.5)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows.extend(list(pool.map(lambda ds:run(ds,'check'),p['datasets'])))
  assert all(x['exit_code']==0 for x in rows)
  checks={ds:json.loads((s/(ds+'_preflight.json')).read_text()) for ds in p['datasets']};status='completed'
 finally:
  if gpu is not None and gpu.poll() is None:
   if (s/'gpu_ready.json').exists():
    try:
     c=Client(str(s/'teacher.sock'),family='AF_UNIX',authkey=(s/'rpc.key').read_bytes());c.send(dict(action='stop'));c.close()
    except (OSError,EOFError):pass
   try:gpu.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(gpu.pid,signal.SIGTERM);gpu.wait(timeout=10)
  rec=dict(status=status,phases=rows,wall_seconds=time.monotonic()-start,GPU_guardian_seconds=0 if gpu_start is None else time.monotonic()-gpu_start,GPU_hours=0 if gpu_start is None else (time.monotonic()-gpu_start)/3600,TEST_calls=0,CPU_full_run_not_interrupted=True)
  if status=='completed':rec.update(parity_passed=all(v['passed'] for v in checks.values()),checks=checks)
  (s/'check_receipt.json').write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec),flush=True)
if __name__=='__main__':main()
