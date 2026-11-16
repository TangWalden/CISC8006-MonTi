"""Closed-parent, frozen-source utilities; no victim use during selection."""
import pathlib,json,hashlib,sys,subprocess,os

def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def setup(stage,dataset):
 s=pathlib.Path(stage).resolve();p=read(s/'protocol.json');assert p['schema']=='gradient_retrieval_v1' and not p['TEST_allowed'];v=p['datasets'][dataset]
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
 for f,h in v['inputs'].items():assert sha(f)==h,f
 rt=pathlib.Path(p['runtime_root']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
 parent=pathlib.Path(v['parent']);sys.path.insert(0,str(parent/'source'));os.chdir(rt)
 import common
 _,old,q=common.setup(str(parent));data,view,teacher,th,benign,mean,std=common.load(q)
 return s,p,v,parent,old,q,data,view,teacher,th,benign,common
def counts():return dict(physical_builds=0,teacher_forwards=0,teacher_contexts=0,backward_calls=0,algebraic_pool_scores=0,official_GAGA_VALID=0,upstream_generations=0,updates=0,TEST_queries=0,API_calls=0)
