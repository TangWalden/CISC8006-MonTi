"""Exact injected-feature coefficients for the existing fixed-topology oracle."""
import torch
def coefficients(view,case):
    ids=case['ids'];cids=case['candidate_ids'];t=len(ids);c=len(cids);b=len(case['source_ids']);n=t+c+b
    original=torch.cat((ids,cids)).numpy();adj=view.adj
    if (adj!=adj.T).nnz:raise RuntimeError('Undirected original graph required.')
    a=torch.zeros(n,n);a[:t+c,:t+c]=torch.from_numpy((adj[original][:,original].toarray()!=0).astype('float32'))
    owned=case['native_adjacency'];a[t+c:,:]+=owned;a[:,t+c:]+=owned.T
    assert bool(((a==0)|(a==1)).all())
    two=(a@a)>0;two.fill_diagonal_(False)
    old1=torch.zeros(t,n,dtype=torch.bool);old2=old1.clone()
    old1[:,:t+c]=torch.from_numpy(adj[ids.numpy()][:,original].toarray()!=0)
    old2[:,:t+c]=torch.from_numpy(view.adj2[ids.numpy()][:,original].toarray()!=0)
    added1=(a[:t]>0)&~old1;added2=two[:t]&~old2
    unknown=torch.cat((view.group_masks['train_masked'][torch.cat((ids,cids))],torch.ones(b)))
    d=view.d[ids].reshape(t,7)
    denom1=(d[:,3]+added1.float()@unknown).clamp_min(1)
    denom2=(d[:,6]+added2.float()@unknown).clamp_min(1)
    return added1[:,t+c:].float()/denom1[:,None],added2[:,t+c:].float()/denom2[:,None]
