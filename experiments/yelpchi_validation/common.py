"""Dataset adapter around hash-pinned shared downstream study code."""
import pathlib,json,hashlib,sys,subprocess,os
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def setup(stage):
 s=pathlib.Path(stage).resolve();p=read(s/'protocol.json');assert p['schema']=='yelpchi_validation_v1' and not p['TEST_allowed']
 rt=pathlib.Path(p['runtime_root']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
 for f,h in p['inputs'].items():assert sha(f)==h,f
 for n,r in p['shared_sources'].items():assert sha(r['path'])==r['sha256'],n
 shared=pathlib.Path(p['shared_sources']['kernel.py']['path']).parent;portal=pathlib.Path(p['shared_sources']['geometry_hard.py']['path']).parent
 support=p['support_source'];sys.path[:0]=[str(s/'source'),str(shared),str(portal),support+'/experiments/block_swap',support,str(rt)]
 os.chdir(p['asset_root'])
 q=dict(runtime_root=str(rt),asset_root=p['asset_root'],teacher_checkpoint=str(s/'teacher/model.pt'),support_source=support,stage=str(s),dataset='yelpchi_rtr')
 return s,p,q
def load(q):
 from adapter import load_data,load_teacher
 import torch
 data,view=load_data(q);teacher,th=load_teacher(q,data);ids=torch.as_tensor(data.idx_train).long();benign=ids[data.labels[ids]==0].numpy();mean=data.features[ids].mean(0);std=data.features[ids].std(0,unbiased=False).clamp_min(1e-6)
 return data,view,teacher,th,benign,mean,std
def index_cards(parent):return {tuple(x['key']):x for x in read(pathlib.Path(parent)/'main/card_seal.json')['cards']}
def oldcard(parent,index,seed,phase,group):
 import torch
 r=index[seed,phase,group];f=pathlib.Path(parent)/'main'/r['path'];assert sha(f)==r['sha256'];return torch.load(f,map_location='cpu')
def counters():return dict(physical_builds=0,teacher_batches=0,teacher_contexts=0,policy_forwards=0,updates=0,official_GAGA_VALID=0,upstream_generations=0,TEST_queries=0,API_calls=0)
def shared(name,replacements=()):
 stage=sys.argv[sys.argv.index('--stage')+1];s,p,q=setup(stage);ref=p['shared_sources'][name];source=pathlib.Path(ref['path']).read_text()
 for before,after in replacements:
  assert source.count(before)==1,(name,before);source=source.replace(before,after)
 return compile(source,ref['path'],'exec')
