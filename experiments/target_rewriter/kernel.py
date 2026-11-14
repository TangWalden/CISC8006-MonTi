"""Physical, target-aware region/rule state and exact two-edit TRAIN labels."""
import random,hashlib,itertools
import numpy as np
import torch

def key(*v):return int(hashlib.sha256(repr(v).encode()).hexdigest()[:16],16)
def schedule(case,view,benign):
 from geometry_hard import proposals
 donor,first=proposals(case,view,benign)
 if donor is None:return []
 ids=case['ids'].numpy();cids=case['candidate_ids'].numpy();t=len(ids);a=case['native_adjacency']
 banned=np.unique(np.concatenate((ids,cids,view.adj[ids].tocsr().indices,view.adj2[ids].tocsr().indices)))
 eligible=np.setdiff1d(benign,banned,assume_unique=False)
 d=view.d[case['ids']].reshape(t,7)[:,4]
 used=torch.nonzero(a[:,t:t+len(cids)].sum(0)>0).flatten().tolist()
 def score(j):return float(((a[a[:,t+j]>0,:t].sum(0)>0).float()/(1+d)).sum())
 ordered=sorted(used,key=lambda j:(-score(j),key(case['seed'],case['group'],int(cids[j]),'donor')))
 assert ordered[0]==donor
 result=[dict(donor=donor,bank=first[:4])];taken=set(first)
 for i,j in enumerate(ordered[1:3],1):
  pool=[int(n) for n in eligible if int(n) not in taken]
  if not pool:break
  bank=random.Random(key(case['seed'],case['group'],'rewriter_bank',i)).sample(pool,min(4,len(pool)))
  result.append(dict(donor=j,bank=bank));taken.update(bank)
 return result
def options(steps,used=()):return [(r,b) for r,s in enumerate(steps) if r not in used for b in range(len(s['bank']))]
def apply(base,steps,chosen,features):
 from geometry_hard import replace
 assert len({r for r,b in chosen})==len(chosen)
 v=base
 for r,b in chosen:v=replace(v,steps[r]['donor'],steps[r]['bank'][b],features)
 return v
def physical(data,view,base,changed,c):
 from experiments.downstream_cooperation.common import make_graph,canonical
 assert torch.equal(base['native_adjacency'],changed['native_adjacency']) and torch.equal(base['x_aux'],changed['x_aux'])
 assert torch.equal(base['ids'],changed['ids']);assert len(set(changed['candidate_ids'].tolist()))==len(base['candidate_ids'])
 g=make_graph(data,changed,changed['native_adjacency']);a=base['native_adjacency'];n=len(base['ids'])+len(base['candidate_ids'])
 expected=a.sum(1)+a[:,n:].sum(0)+1;degree=g.in_degrees()[base['source_ids']]
 assert torch.equal(degree.to(expected.dtype),expected)
 with torch.no_grad():seq=canonical(data,view,g,changed)
 c['physical_builds']+=1
 return seq
def predict(teacher,seqs,c):
 values=torch.stack(seqs);k,t,_,d=values.shape
 with torch.no_grad():z=teacher(values.reshape(-1,7,d)).reshape(k,t,2)
 assert torch.isfinite(z).all();c['teacher_batches']+=1;c['teacher_contexts']+=k
 return z,torch.nn.functional.softplus(z[:,:,1]-z[:,:,0]).mean(1)
def state(data,view,base,steps,chosen,seq,z,mean,std):
 # All target-level signals are surrogate outputs or graph/features, not victim truth.
 t=len(base['ids']);targets=torch.cat(((seq[:,0]-mean)/std,(seq[:,4]-mean)/std,(seq[:,6]-mean)/std,(z[:,1]-z[:,0]).unsqueeze(1)),1)
 opts=options(steps,[r for r,b in chosen]);af=[];coverage=[];regions=[];a=base['native_adjacency']
 for r,b in opts:
  j=steps[r]['donor'];rows=a[:,t+j]>0;cov=(a[rows,:t].sum(0)>0).float();aux=base['x_aux'][rows].mean(0)
  af.append(torch.cat(((data.features[steps[r]['bank'][b]]-mean)/std,(aux-mean)/std,torch.tensor([np.log1p(float(rows.sum())),len(chosen)/2,np.log1p(t)],dtype=torch.float32))))
  coverage.append(cov);regions.append(r)
 return dict(targets=targets,actions=torch.stack(af) if af else torch.empty(0,2*data.feat_dim+3),coverage=torch.stack(coverage) if coverage else torch.empty(0,t),regions=regions,options=opts)
def policy(net,v,c):
 with torch.no_grad():lp=net(v['targets'],v['actions'],v['coverage'],v['regions'])
 c['policy_forwards']+=1;assert torch.isfinite(lp).all();return int(lp.argmax()),lp
def dose(base,steps,chosen):return sum(int(base['native_adjacency'][:,len(base['ids'])+steps[r]['donor']].sum()) for r,b in chosen)
def randomize(base,steps,chosen,kind):
 rng=random.Random(key(base['seed'],base['group'],'rewriter_random',kind));regions=[r for r,b in chosen]
 if kind=='joint':
  feasible=[list(rs) for rs in itertools.combinations(range(len(steps)),len(chosen)) if dose(base,steps,[(r,0) for r in rs])==dose(base,steps,chosen)]
  assert feasible;regions=rng.choice(feasible)
 return [(r,rng.randrange(len(steps[r]['bank']))) for r in regions]
def train_card(data,view,base,old,teacher,benign,mean,std,c):
 steps=schedule(base,view,benign);opts=options(steps);empty=();seqs={empty:old['sequences'][0]};zs={empty:old['teacher_logits'][0]};loss={empty:float(old['losses'][0])}
 # Exact physical final graphs for every one/two-region combination. No lookahead pruning by outcome.
 plans=[(x,) for x in opts]+[tuple(sorted((x,y))) for x,y in itertools.combinations(opts,2) if x[0]!=y[0]]
 for start in range(0,len(plans),12):
  chunk=plans[start:start+12];batch=[]
  for chosen in chunk:
   if len(chosen)==1 and chosen[0][0]==0:seq=old['sequences'][chosen[0][1]+1]
   else:seq=physical(data,view,base,apply(base,steps,chosen,data.features),c)
   seqs[chosen]=seq;batch.append(seq)
  z,ll=predict(teacher,batch,c)
  for i,chosen in enumerate(chunk):zs[chosen]=z[i];loss[chosen]=float(ll[i])
 records=[]
 for chosen in [empty]+[(x,) for x in opts]:
  v=state(data,view,base,steps,chosen,seqs[chosen],zs[chosen],mean,std);values=[loss[chosen]]
  for x in v['options']:
   nextplan=tuple(sorted(chosen+(x,)));value=loss[nextplan]
   if not chosen:
    value=min([value]+[loss[tuple(sorted((x,y)))] for y in opts if y[0]!=x[0]])
   values.append(value)
  # Soft labels use final attainable loss; STOP keeps current graph. Not policy gradient/RL.
  val=torch.tensor(values);scale=val.std(unbiased=False).clamp_min(.001)
  v.update(chosen=chosen,value_to_go=val,labels=torch.softmax((val[0]-val)/scale,0));records.append(v)
 return dict(seed=base['seed'],group=base['group'],steps=steps,records=records,terminal_losses=[dict(actions=k,loss=v) for k,v in loss.items()])
