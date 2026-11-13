"""Replace one used candidate slot by one original TRAIN-benign remote node."""
import numpy as np,torch,random,hashlib
def key(*values):return int(hashlib.sha256(repr(values).encode()).hexdigest()[:16],16)
def proposals(case,view,benign,limit=8):
    ids=case['ids'].numpy();cids=case['candidate_ids'].numpy();t=len(ids);c=len(cids);a=case['native_adjacency']
    banned=np.unique(np.concatenate((ids,cids,view.adj[ids].indices,view.adj2[ids].indices)))
    eligible=np.setdiff1d(benign,banned,assume_unique=False)
    used=torch.nonzero(a[:,t:t+c].sum(0)>0).flatten().tolist()
    if not len(eligible) or not used:return None,[]
    d=view.d[case['ids']].reshape(t,7)[:,4]
    def score(j):
        rows=a[:,t+j]>0;affected=(a[rows,:t].sum(0)>0)
        return float((affected.float()/(1+d)).sum())
    donor=min(used,key=lambda j:(-score(j),key(case['seed'],case['group'],int(cids[j]),'donor')))
    rng=random.Random(key(case['seed'],case['group'],'remote_benign_bank'))
    return donor,rng.sample(eligible.tolist(),min(limit,len(eligible)))
def replace(case,donor,node,features):
    v=dict(case);ids=case['candidate_ids'].clone();assert int(node) not in set(ids.tolist())|set(case['ids'].tolist());ids[donor]=int(node)
    v['candidate_ids']=ids;v['x_candidates']=features[ids].clone()
    v.pop('h_candidates',None) # Not used; avoid carrying stale cached representations.
    return v
