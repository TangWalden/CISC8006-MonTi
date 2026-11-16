"""Type-aware discrete endpoint retrieval; no validation labels or victim queries."""
import random,itertools
import numpy as np
import torch
from gr_core import Geometry as BenignGeometry,key,predict

class Geometry(BenignGeometry):
 def __init__(self,data,view,base,seq0,benign,c):
  super().__init__(data,view,base,seq0,benign,c)
  ids=base['ids'].numpy();cids=base['candidate_ids'].numpy()
  banned=np.unique(np.concatenate((ids,cids,view.adj[ids].tocsr().indices,view.adj2[ids].tocsr().indices)))
  train=torch.as_tensor(data.idx_train).long();fraud=train[data.labels[train]==1].numpy()
  self.pools={0:list(self.pool),1:np.setdiff1d(fraud,banned).astype(int).tolist()}
  self.node_class={n:label for label,pool in self.pools.items() for n in pool}
  self.anchors={(j,label):pool[i] for label,pool in self.pools.items() for i,j in enumerate(self.donors) if i<len(pool)}
  self.cache={}
  for label,pool in self.pools.items():
   if pool:assert bool((view.group_masks['train_neg' if label==0 else 'train_pos'][pool]==1).all())
 def physical(self,plan,need_den=False):
  from experiments.downstream_cooperation.common import make_graph,canonical
  import dgl
  case=self.changed(plan);a=case['native_adjacency'];g=make_graph(self.data,case,a);n=self.t+len(case['candidate_ids'])
  assert torch.equal(g.in_degrees()[case['source_ids']].to(a.dtype),a.sum(1)+a[:,n:].sum(0)+1)
  with torch.no_grad():seq=canonical(self.data,self.view,g,case)
  self.c['physical_builds']+=1
  if not need_den:return seq
  ids=torch.cat((case['ids'],case['candidate_ids'],torch.tensor(case['source_ids'],dtype=torch.long)))
  sub=dgl.remove_self_loop(dgl.node_subgraph(g,ids));assert torch.equal(sub.ndata[dgl.NID],ids)
  adj=torch.zeros((len(ids),len(ids)));src,dst=sub.edges();adj[src,dst]=1
  a2=(adj@adj).bool().float()-torch.eye(len(ids));a2=a2[:self.t]
  org=torch.zeros_like(a2);org[self.view.adj2[case['ids']][:,ids[:n]].nonzero()]=1
  diff=torch.where(a2-org>0,a2,torch.zeros_like(a2));den={}
  for label,name in [(0,'train_neg'),(1,'train_pos')]:
   mask=torch.cat((self.view.group_masks[name][ids[:n]],torch.zeros(len(ids)-n)))
   den[label]=(self.view.d[case['ids']].reshape(self.t,7)[:,4+label]+(diff*mask).sum(1)).clamp_min(1)
  return seq,den
 def template(self,plan):
  k=tuple(sorted((j,self.node_class[node]) for j,node in plan.items()))
  if k not in self.cache:self.cache[k]=self.physical({j:self.anchors[j,label] for j,label in k},True)
  return self.cache[k]
 def sequence(self,plan):
  if not plan:return self.seq0.clone()
  anchor,den=self.template(plan);seq=anchor.clone()
  for j,node in sorted(plan.items()):
   label=self.node_class[node];seq[:,4+label]+=(self.coverage[j]/den[label]).unsqueeze(1)*(self.data.features[node]-self.data.features[self.anchors[j,label]])
  return seq
 def random(self,plan,mode,same_class=True):
  rng=random.Random(key(self.base['seed'],self.base['group'],'channel_random',mode));result={}
  for j,node in sorted(plan.items()):
   pool=self.pools[self.node_class[node]] if same_class else self.pools[0]+self.pools[1]
   result[j]=rng.choice([n for n in pool if n not in result.values()])
  return result

def search(geo,teacher,mode,c,width=8,horizon=2):
 chosen={};traces=[];_,loss,_=predict(teacher,[geo.seq0],c,'boundary');current=float(loss[0])
 for turn in range(horizon):
  proposals=[]
  for j in geo.donors:
   if j in chosen:continue
   for label in [0,1]:
    pool=[n for n in geo.pools[label] if n not in chosen.values()]
    if not pool or (j,label) not in geo.anchors:continue
    proposal=dict(chosen);proposal[j]=pool[0];proposals.append((j,label,pool,proposal))
  if not proposals:break
  plans=[]
  if mode=='channel':
   _,_,grad=predict(teacher,[geo.sequence(v[3]) for v in proposals],c,'boundary',True)
   for i,(j,label,pool,proposal) in enumerate(proposals):
    _,den=geo.template(proposal);query=((geo.coverage[j]/den[label]).unsqueeze(1)*grad[i,:,4+label]).sum(0)
    score=geo.data.features[torch.tensor(pool,dtype=torch.long)]@query;c['algebraic_pool_scores']+=len(pool)
    chosen_nodes=[pool[k] for k in sorted(range(len(pool)),key=lambda k:(float(score[k]),pool[k]))[:width]]
    for node in chosen_nodes:v=dict(chosen);v[j]=node;plans.append(v)
  else:
   assert mode=='uniform'
   for j,label,pool,_ in proposals:
    rng=random.Random(key(geo.base['seed'],geo.base['group'],turn,j,label,'channel_uniform'))
    for node in rng.sample(pool,min(width,len(pool))):v=dict(chosen);v[j]=node;plans.append(v)
  _,values,_=predict(teacher,[geo.sequence(v) for v in plans],c,'boundary')
  best=min(range(len(plans)),key=lambda i:(float(values[i]),tuple(sorted(plans[i].items()))));gain=current-float(values[best])
  traces.append(dict(turn=turn,shortlist=len(plans),before=current,best=float(values[best]),gain=gain,accepted=gain>1e-5))
  if gain<=1e-5:break
  chosen=plans[best];current=float(values[best])
 return chosen,traces
