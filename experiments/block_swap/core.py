"""One outcome-blind donor, learned recipient; exactly matched random block."""
import random
import numpy as np
from experiments.local_swap.core import keyed,descriptors,fit,predict,FEATURE_NAMES

def pool(case,limit=8):
    t=len(case['ids']);c=len(case['candidate_ids']);a=case['native_adjacency'][:,t:t+c].numpy()
    donors=[]
    for old in range(c):
        rows=np.flatnonzero(a[:,old])
        if len(rows):
            recipients=np.flatnonzero(a[rows].sum(0)==0).tolist()
            if recipients:donors.append((old,rows.tolist(),recipients))
    multi=[d for d in donors if len(d[1])>=2]
    eligible=multi or donors
    rng=random.Random(keyed(case['seed'],case['group'],'block_swap_donor_v1'))
    if not eligible:return []
    old,rows,recipients=eligible[rng.randrange(len(eligible))]
    return [dict(old=old,new=new,rows=rows) for new in rng.sample(recipients,min(limit,len(recipients)))]

def exchange(base,t,c,action):
    out=base.clone()
    if action is None:return out
    old,new,rows=action['old'],action['new'],action['rows']
    assert old!=new and 0<=old<c and 0<=new<c
    assert rows==base[:,t+old].nonzero(as_tuple=True)[0].tolist()
    assert rows and bool((base[rows,t+new]==0).all())
    out[rows,t+old]=0;out[rows,t+new]=1
    return out

def check(base,changed,t,c,action):
    import torch
    assert torch.equal(changed,exchange(base,t,c,action))
    assert torch.equal(base[:,:t],changed[:,:t]) and torch.equal(base[:,t+c:],changed[:,t+c:])
    assert torch.equal(base.sum(1),changed.sum(1))
    assert int((base!=changed).sum())==(0 if action is None else 2*len(action['rows']))
    assert bool(((changed==0)|(changed==1)).all())

def features(case,adj):
    actions=case['block_pool'];expanded=dict(case)
    expanded['pool']=[[r,a['old'],a['new']] for a in actions for r in a['rows']]
    d=descriptors(expanded,adj);output=[]
    for action in actions:
        # Same 32 features as v1, averaged across the exact affected rows.
        # No extra feature family or victim information is introduced.
        rows=[np.outer(d[r][1],d[r][0][action['new']]-d[r][0][action['old']]).reshape(-1) for r in action['rows']]
        output.append(np.mean(rows,axis=0))
    return np.asarray(output,dtype=np.float64).reshape((-1,len(FEATURE_NAMES)))

def choose(model,X,actions,seed,group):
    gains=predict(model,X);rank=int(np.argmax(gains)) if len(actions) else None
    active=rank is not None and float(gains[rank])>model['threshold']
    r=random.Random(keyed(seed,group,'block_swap_random_v1')).randrange(len(actions)) if actions else None
    return dict(learned=actions[rank] if active else None,matched_random=actions[r] if active else None,
        always_random=actions[r] if actions else None,predicted_gains=gains.tolist(),active=active,threshold=model['threshold'])
