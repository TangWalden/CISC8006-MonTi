"""Exclusive bounded CPU pipeline, with a two-dataset pre-victim barrier."""
import argparse,pathlib,os,time,subprocess,signal,fcntl,datetime,sys,shutil,concurrent.futures
from full_common import protocol,read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p=protocol(ap.parse_args().stage)
 lock=(s.parent/'gradient_retrieval_exclusive.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 mem=int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 assert mem>=p['resources']['minimum_available_GiB']*1024**3 and shutil.disk_usage(s).free>=p['resources']['minimum_disk_GiB']*1024**3
 write(s/'launch_once.json',dict(pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=sha(s/'protocol.json')))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib',PYTHONUNBUFFERED='1');rows=[];start=time.monotonic();status='failed'
 def run(mode,ds=None,shard=None):
  name=mode+('_'+ds if ds else '')+('_'+str(shard) if shard is not None else '');began=time.monotonic();limit=p['resources'][mode+'_seconds'];cmd=['nice','-n','10',sys.executable,str(s/'source'/('phase.py' if ds else 'audit.py')),'--stage',str(s)]
  if ds:cmd+=['--dataset',ds,'--mode',mode]
  if shard is not None:cmd+=['--shard',str(shard)]
  with (s/(name+'.log')).open('xb') as log:
   child=subprocess.Popen(cmd,cwd=p['runtime_root'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   write(s/(name+'_process.json'),dict(pid=child.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd))
   try:rc=child.wait(timeout=limit+30)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:rc=child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
  rec=dict(phase=name,pid=child.pid,exit_code=rc,wall_seconds=time.monotonic()-began);write(s/(name+'_outer.json'),rec);print(rec,flush=True);return rec
 def batch(tasks):
  with concurrent.futures.ThreadPoolExecutor(max_workers=p['resources']['max_selection_workers']) as pool:
   futures=[pool.submit(run,*t) for t in tasks]
   for f in concurrent.futures.as_completed(futures):rows.append(f.result())
  assert all(r['exit_code']==0 for r in rows),'Preserved failed phase; no automatic retry.'
 try:
  batch([('preflight',d,None) for d in p['datasets']])
  batch([('selection',d,j) for d in p['datasets'] for j in range(p['selection_shards'])])
  seals={}
  for ds,v in p['datasets'].items():
   root=s/ds;entries=[]
   for j in range(p['selection_shards']):
    r=read(root/('selection_'+str(j)+'_result.json'));f=root/('seal_'+str(j)+'.json');assert r['status']=='completed' and sha(f)==r['seal_sha256'];entries+=read(f)['entries']
   expected={(seed,g) for seed in v['seeds'] for g in v['test_groups']};assert len(entries)==len(expected) and {(e['seed'],e['group']) for e in entries}==expected
   entries.sort(key=lambda e:(e['seed'],e['group']));write(root/'sealed_before_victim.json',dict(entries=entries,official_victim_loaded=False));seals[ds]=sha(root/'sealed_before_victim.json')
  write(s/'all_actions_sealed.json',dict(datasets=seals,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),official_TEST_calls_so_far=0))
  batch([('evaluation',d,None) for d in p['datasets']]);batch([('audit',None,None)]);status='completed'
 finally:write(s/'guardian_receipt.json',dict(status=status,phases=rows,wall_seconds=time.monotonic()-start,GPU_hours=0,automatic_retry=False,phase_wall_times_overlap=True))
if __name__=='__main__':main()
