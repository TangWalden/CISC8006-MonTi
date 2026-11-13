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
def portal_coefficient(view,v,donor):
    ids=v['ids'];cids=v['candidate_ids'];t=len(ids);c=len(cids);b=len(v['source_ids']);n=t+c+b;orig=torch.cat((ids,cids)).numpy()
    assert (view.adj!=view.adj.T).nnz==0
    a=torch.zeros(n,n);a[:t+c,:t+c]=torch.from_numpy((view.adj[orig][:,orig].toarray()!=0).astype('float32'))
    a[t+c:,:]+=v['native_adjacency'];a[:,t+c:]+=v['native_adjacency'].T
    two=(a@a)>0;two.fill_diagonal_(False);old=torch.zeros(t,n,dtype=torch.bool);old[:,:t+c]=torch.from_numpy(view.adj2[ids.numpy()][:,orig].toarray()!=0)
    added=two[:t]&~old;neg=torch.cat((view.group_masks['train_neg'][torch.cat((ids,cids))],torch.zeros(b)))
    denominator=view.d[ids].reshape(t,7)[:,4]+added.float()@neg
    return added[:,t+donor].float()/denominator.clamp_min(1)
def portal_sequences(first,coeff,features):
    # Remote eligibility removes original one/two-hop paths to every target.
    # All portals have the same known-TRAIN benign group and membership count.
    result=first.unsqueeze(0).expand(len(features),-1,-1,-1).clone()
    result[:,:,4]+=coeff[None,:,None]*(features-features[0])[:,None,:]
    return result
