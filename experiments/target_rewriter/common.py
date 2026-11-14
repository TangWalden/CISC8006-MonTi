"""Immutable, CPU-only development study utilities."""
import pathlib,json,hashlib,subprocess,sys

def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def setup(stage):
 s=pathlib.Path(stage);p=read(s/'protocol.json');assert p['schema']=='target_rewriter_v1' and not p['TEST_allowed']
 rt=pathlib.Path(p['runtime_root']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
 for n,h in p['inputs'].items():assert sha(rt/n)==h,n
 parent=pathlib.Path(p['parent']);q=read(parent/'main_protocol.json');support=q['support_source']
 sys.path[:0]=[str(s/'source'),str(parent/'source'),support+'/experiments/block_swap',support,str(rt)]
 from helpers import verify
 verify(pathlib.Path(support).parent/'registration/protocol.json')
 return s,p,q
def load(q):
 import torch,numpy as np
 from modules.data_handler import DataHandler
 from experiments.local_swap.loader import CONFIG,DEFAULTS
 from experiments.downstream_cooperation.common import check_device,freeze
 from course.dual_surrogate.runtime import load_original_view
 from models.victim_model import GAGA
 check_device();config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config)
 data.target_idx_test=np.empty(0,dtype=np.int64);view=load_original_view(data)
 teacher=GAGA(data.feat_dim,64,2);teacher.load_state_dict(torch.load(q['teacher_checkpoint'],map_location='cpu'));th=freeze(teacher)
 ids=torch.as_tensor(data.idx_train).long();benign=ids[data.labels[ids]==0].numpy()
 mean=data.features[ids].mean(0);std=data.features[ids].std(0,unbiased=False).clamp_min(1e-6)
 return data,view,teacher,th,benign,mean,std
def index_cards(parent):return {tuple(x['key']):x for x in read(pathlib.Path(parent)/'main/card_seal.json')['cards']}
def oldcard(parent,index,seed,phase,group):
 import torch
 r=index[seed,phase,group];f=pathlib.Path(parent)/'main'/r['path'];assert sha(f)==r['sha256'];return torch.load(f,map_location='cpu')
def counters():return dict(physical_builds=0,teacher_batches=0,teacher_contexts=0,policy_forwards=0,updates=0,official_GAGA_VALID=0,upstream_generations=0,TEST_queries=0,API_calls=0)
