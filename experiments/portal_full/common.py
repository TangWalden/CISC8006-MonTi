"""Frozen author-checkpoint complete-cohort evaluation, isolated from runtime."""
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
 s=pathlib.Path(stage);p=read(s/'protocol.json');assert p['schema']=='portal_complete_cohort_v1' and p['TEST_allowed']
 assert sha(p['cohort_protocol'])==p['cohort_protocol_sha256']
 rt=pathlib.Path(p['runtime_root']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
 parent=pathlib.Path(p['parent']);q=read(parent/'main_protocol.json');assert sha(parent/'main_protocol.json')==p['parent_protocol_sha256']
 for n,h in q['sources'].items():assert sha(parent/'source'/n)==h,n
 assert read(parent/'saved_audit.json')['status']=='passed' and read(parent/'main/result.json')['owner_mean_criterion']=={'neural':True,'direct':True}
 assert sha(q['teacher_checkpoint'])==p['teacher_sha256']==q['teacher_sha256']
 for v in p['models']:assert sha(parent/'main'/v['path'])==v['sha256']
 support=q['support_source'];sys.path[:0]=[str(s/'source'),str(parent/'source'),support+'/experiments/block_swap',support,p['runtime_root']]
 from helpers import verify
 verify(pathlib.Path(support).parent/'registration/protocol.json')
 return s,p,q
def load_teacher(p,q,data):
 import torch
 from models.victim_model import GAGA
 teacher=GAGA(data.feat_dim,64,2);teacher.load_state_dict(torch.load(q['teacher_checkpoint'],map_location='cpu'));teacher.eval()
 for v in teacher.parameters():v.requires_grad_(False)
 return teacher
def card(data,view,case,teacher,counts):
 import torch,numpy as np
 from geometry_hard import proposals,replace
 from experiments.downstream_cooperation.common import make_graph,canonical
 trainids=torch.as_tensor(data.idx_train).long();benign=trainids[data.labels[trainids]==0].numpy();mean=data.features[trainids].mean(0);std=data.features[trainids].std(0,unbiased=False).clamp_min(1e-6)
 donor,bank=proposals(case,view,benign);ids=case['ids'].numpy()
 if bank:
  cols=np.asarray(bank);assert view.adj[ids][:,cols].nnz==0 and view.adj2[ids][:,cols].nnz==0
  assert (view.adj[ids]@view.adj[:,cols]).nnz==0
 values=[];degree=None
 for node in [None]+bank:
  changed=case if node is None else replace(case,donor,node,data.features);g=make_graph(data,changed,changed['native_adjacency'])
  d=g.in_degrees()[case['source_ids']]
  if degree is None:degree=d
  else:assert torch.equal(degree,d)
  with torch.no_grad():values.append(canonical(data,view,g,changed))
  counts['physical_canonical_builds']+=1
 seq=torch.stack(values);t=len(case['ids'])
 with torch.no_grad():z=teacher(seq.reshape(-1,7,data.feat_dim)).reshape(len(values),t,2)
 counts['own_teacher_batched_forwards']+=1;counts['own_teacher_hard_contexts']+=len(values)
 losses=torch.nn.functional.softplus(z[:,:,1]-z[:,:,0]).mean(1)
 aux=case['x_aux'][case['native_adjacency'][:,t+donor]>0].mean(0) if donor is not None else case['x_aux'].mean(0)
 query=torch.cat(((case['x_targets'].mean(0)-mean)/std,(aux-mean)/std,(seq[0,:,4].mean(0)-mean)/std,torch.tensor([np.log1p(t),np.log1p(len(case['source_ids'])),np.log1p(float(view.d[case['ids']].reshape(t,7)[:,4].mean()))],dtype=torch.float32)))
 keys=torch.zeros(len(values),data.feat_dim)
 if bank:keys[1:]=(data.features[torch.tensor(bank)]-mean)/std
 return dict(case=case,donor=donor,bank=bank,sequences=seq,teacher_logits=z,losses=losses,query=query,keys=keys,changes=(seq[:,:,4].mean(1)-seq[0,:,4].mean(0))/std,physical_injected_degrees=degree)
def choose(v,net):
 import torch,random
 from geometry_hard import key
 with torch.no_grad():scores=net(v['query'],v['keys'],v['changes'])
 neural=int(scores.argmax());best=int(v['losses'].argmin());direct=best if float(v['losses'][0]-v['losses'][best])>1e-4 else 0
 case=v['case'];ri=random.Random(key(case['seed'],case['group'],'portal_matched_random')).randrange(1,len(v['bank'])+1) if v['bank'] else 0
 return dict(native=0,neural=neural,random_neural=ri if neural else 0,direct=direct,random_direct=ri if direct else 0),scores
