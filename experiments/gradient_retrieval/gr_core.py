"""HotFlip-inspired discrete retrieval with checked physical relation templates."""
import itertools,random,hashlib
import numpy as np
import torch

def key(*v):return int(hashlib.sha256(repr(v).encode()).hexdigest()[:16],16)
def objective(z,mode):
 m=z[...,1]-z[...,0]
 return (torch.nn.functional.softplus(m) if mode=='softplus' else torch.sigmoid(m)).mean(-1)
def predict(model,seqs,c,mode,grad=False):
 x=torch.stack(seqs).detach().requires_grad_(grad);k,t,_,d=x.shape
 with torch.set_grad_enabled(grad):
  z=model(x.reshape(-1,7,d)).reshape(k,t,2);loss=objective(z,mode)
  g=torch.autograd.grad(loss.sum(),x)[0] if grad else None
 c['teacher_forwards']+=1;c['teacher_contexts']+=k;c['backward_calls']+=int(grad)
 assert torch.isfinite(z).all() and (g is None or torch.isfinite(g).all())
 return z.detach(),loss.detach(),g

class Geometry:
 def __init__(self,data,view,base,seq0,benign,c):
  from geometry_hard import proposals
  self.data,self.view,self.base,self.c=data,view,base,c;self.seq0=seq0
  ids=base['ids'].numpy();cids=base['candidate_ids'].numpy();self.t=len(ids);a=base['native_adjacency']
  banned=np.unique(np.concatenate((ids,cids,view.adj[ids].tocsr().indices,view.adj2[ids].tocsr().indices)))
  self.pool=np.setdiff1d(benign,banned,assume_unique=False).astype(int).tolist()
  d=view.d[base['ids']].reshape(self.t,7)[:,4];used=torch.nonzero(a[:,self.t:self.t+len(cids)].sum(0)>0).flatten().tolist()
  def coverage(j):return (a[a[:,self.t+j]>0,:self.t].sum(0)>0).float()
  self.donors=sorted(used,key=lambda j:(-float((coverage(j)/(1+d)).sum()),key(base['seed'],base['group'],int(cids[j]),'donor')))[:min(3,len(self.pool))]
  self.coverage={j:coverage(j) for j in self.donors};self.anchors={j:self.pool[i] for i,j in enumerate(self.donors)};self.cache={}
 def changed(self,plan):
  from geometry_hard import replace
  v=self.base;assert len(set(plan.values()))==len(plan)
  for j,node in sorted(plan.items()):v=replace(v,j,node,self.data.features)
  return v
 def physical(self,plan,need_den=False):
  from experiments.downstream_cooperation.common import make_graph,canonical
  import dgl
  case=self.changed(plan);g=make_graph(self.data,case,case['native_adjacency']);a=case['native_adjacency'];n=self.t+len(case['candidate_ids'])
  assert torch.equal(g.in_degrees()[case['source_ids']].to(a.dtype),a.sum(1)+a[:,n:].sum(0)+1)
  with torch.no_grad():seq=canonical(self.data,self.view,g,case)
  self.c['physical_builds']+=1
  if not need_den:return seq
  ids=torch.cat((case['ids'],case['candidate_ids'],torch.tensor(case['source_ids'],dtype=torch.long)))
  sub=dgl.remove_self_loop(dgl.node_subgraph(g,ids));assert torch.equal(sub.ndata[dgl.NID],ids)
  adj=torch.zeros((len(ids),len(ids)));src,dst=sub.edges();adj[src,dst]=1
  a2=(adj@adj).bool().float()-torch.eye(len(ids));a2=a2[:self.t]
  org=torch.zeros_like(a2);org[self.view.adj2[case['ids']][:,ids[:n]].nonzero()]=1
  diff=torch.where(a2-org>0,a2,torch.zeros_like(a2));neg=torch.cat((self.view.group_masks['train_neg'][ids[:n]],torch.zeros(len(ids)-n)))
  den=self.view.d[case['ids']].reshape(self.t,7)[:,4]+(diff*neg).sum(1)
  return seq,den.clamp_min(1)
 def template(self,donors):
  k=tuple(sorted(donors))
  if k not in self.cache:self.cache[k]=self.physical({j:self.anchors[j] for j in k},True)
  return self.cache[k]
 def sequence(self,plan):
  if not plan:return self.seq0.clone()
  anchor,den=self.template(plan);seq=anchor.clone()
  for j,node in sorted(plan.items()):seq[:,4]+=(self.coverage[j]/den).unsqueeze(1)*(self.data.features[node]-self.data.features[self.anchors[j]])
  return seq
 def verify(self,plan,tolerance=1e-5):
  seq=self.sequence(plan);hard=self.physical(plan);err=float((seq-hard).abs().max());assert err<=tolerance,('affine_physical_mismatch',err,plan)
  assert torch.isfinite(seq).all();return hard,err
 def random(self,plan,mode):
  rng=random.Random(key(self.base['seed'],self.base['group'],'full_pool_random',mode));nodes=rng.sample(self.pool,len(plan));return dict(zip(sorted(plan),nodes))
 def dose(self,plan):return sum(int(self.base['native_adjacency'][:,self.t+j].sum()) for j in plan)

def search(geo,teacher,mode,c,width=8,horizon=2):
 chosen={};traces=[];lossmode='softplus' if mode=='softplus' else 'boundary'
 _,ll,_=predict(teacher,[geo.seq0],c,lossmode);current=float(ll[0])
 for turn in range(horizon):
  regions=[j for j in geo.donors if j not in chosen];eligible=[n for n in geo.pool if n not in chosen.values()]
  if not regions or not eligible:break
  # Different substituted regions share exact canonical templates. Candidates
  # from the entire same legal pool are ranked without victim information.
  provisional=[]
  for j in regions:
   node=next(n for n in eligible if n not in chosen.values());v=dict(chosen);v[j]=node;provisional.append(v)
  base_seqs=[geo.sequence(v) for v in provisional];plans=[]
  if mode!='uniform':
   _,_,grad=predict(teacher,base_seqs,c,lossmode,True)
   features=geo.data.features[torch.tensor(eligible,dtype=torch.long)]
   for i,j in enumerate(regions):
    _,den=geo.template(provisional[i]);query=((geo.coverage[j]/den).unsqueeze(1)*grad[i,:,4]).sum(0)
    score=features@query;c['algebraic_pool_scores']+=len(eligible)
    # Stable CPU sort with original node ID as the exact-tie convention.
    idx=sorted(range(len(eligible)),key=lambda k:(float(score[k]),eligible[k]))[:width]
    for k in idx:v=dict(chosen);v[j]=eligible[k];plans.append(v)
  else:
   for j in regions:
    rng=random.Random(key(geo.base['seed'],geo.base['group'],turn,j,'uniform_shortlist'))
    for node in rng.sample(eligible,min(width,len(eligible))):v=dict(chosen);v[j]=node;plans.append(v)
  _,values,_=predict(teacher,[geo.sequence(v) for v in plans],c,lossmode)
  best=min(range(len(plans)),key=lambda i:(float(values[i]),tuple(sorted(plans[i].items()))));gain=current-float(values[best])
  traces.append(dict(turn=turn,shortlist_size=len(plans),before=current,best=float(values[best]),gain=gain,accepted=gain>1e-5))
  if gain<=1e-5:break
  chosen=plans[best];current=float(values[best])
 return chosen,traces
