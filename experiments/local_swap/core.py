"""Outcome-blind exchange pools and an antisymmetric, context-conditioned ranker.

The model fits observed hard-graph loss differences, not topology gradients.
No node identities, labels, victim predictions or upstream checkpoint objects
enter its inputs. Antisymmetry is with respect to the same frozen base context;
it does not assert that multi-step graph interventions commute.
"""
import hashlib
import math
import random

DESCRIPTORS=('aux_cosine','target_mean_cosine','target_max_cosine','target_min_cosine',
             'target_context_cosine','log_original_degree','target_neighbor_fraction',
             'attached_candidate_neighbor_fraction')
CONTEXTS=('constant','log_target_count','log_aux_count','log_row_candidate_count')
FEATURE_NAMES=[c+'*delta_'+d for c in CONTEXTS for d in DESCRIPTORS]

def keyed(*parts):
    return int(hashlib.sha256(repr(parts).encode('ascii')).hexdigest()[:16],16)

def proposal_pool(block, seed, group, limit=8):
    """Uniform without-replacement sample of row/remove/add legal triples."""
    rows=[];total=0
    for r,values in enumerate(block):
        if any(v not in (0,1) for v in values):raise ValueError('Nonbinary candidate block.')
        present=[i for i,v in enumerate(values) if v==1]
        absent=[i for i,v in enumerate(values) if v==0]
        n=len(present)*len(absent)
        if n:rows.append((r,present,absent,total,n));total+=n
    indices=random.Random(keyed(seed,group,'local_swap_pool_v1')).sample(range(total),min(limit,total))
    result=[]
    for k in indices:
        for r,old,new,start,n in rows:
            if start<=k<start+n:
                offset=k-start;result.append([r,old[offset//len(new)],new[offset%len(new)]]);break
    return result

def exchange(adjacency, targets, candidates, triple):
    out=adjacency.clone()
    if triple is not None:
        r,old,new=triple
        if not(0<=r<len(out) and 0<=old<candidates and 0<=new<candidates and old!=new):raise ValueError('Illegal exchange indices.')
        if out[r,targets+old]!=1 or out[r,targets+new]!=0:raise ValueError('Exchange must remove an existing edge and add an absent edge.')
        out[r,targets+old]=0;out[r,targets+new]=1
    return out

def assert_exchange(base,changed,t,c,chosen):
    import torch
    if (base.shape!=changed.shape or not bool(((changed==0)|(changed==1)).all())
            or not torch.equal(base[:,:t],changed[:,:t]) or not torch.equal(base[:,t+c:],changed[:,t+c:])
            or not torch.equal(base.sum(1),changed.sum(1))):
        raise RuntimeError('Exchange changed a protected block, row budget or binary geometry.')
    expected=exchange(base,t,c,chosen)
    if not torch.equal(expected,changed):raise RuntimeError('Exchange does not match registered triple.')
    if int((base!=changed).sum())!=(0 if chosen is None else 2):raise RuntimeError('Wrong turnover.')

def descriptors(case,original_adjacency):
    import numpy as np
    x=case['x_candidates'].double().numpy();aux=case['x_aux'].double().numpy()
    targets=case['x_targets'].double().numpy();hc=case['h_candidates'].double().numpy()
    ht=case['h_targets'].double().numpy();a=case['native_adjacency'].numpy()
    t=len(targets);c=len(x);b=len(aux);ids=case['ids'].numpy();cids=case['candidate_ids'].numpy()
    def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
    xn=unit(x);tn=unit(targets);an=unit(aux);hn=unit(hc)
    target_cos=tn@xn.T
    degree=np.asarray(original_adjacency[cids].sum(1)).reshape(-1)
    result={}
    for r in sorted({triple[0] for triple in case['pool']}):
        attached=np.flatnonzero(a[r,:t]);cand=np.flatnonzero(a[r,t:t+c])
        if not len(attached):raise RuntimeError('Native row has no target attachment.')
        target_mean=unit(targets[attached].mean(0,keepdims=True))[0]
        hmean=unit(ht[attached].mean(0,keepdims=True))[0]
        connected=np.asarray(original_adjacency[cids][:,ids[attached]].sum(1)).reshape(-1)/len(attached)
        covered=(np.asarray(original_adjacency[cids][:,cids[cand]].sum(1)).reshape(-1)/len(cand)
                 if len(cand) else np.zeros(c))
        d=np.stack([xn@an[r],xn@target_mean,target_cos[attached].max(0),target_cos[attached].min(0),
                    hn@hmean,np.log1p(degree),connected,covered],axis=1)
        context=np.array([1.,math.log1p(t),math.log1p(b),math.log1p(len(cand))])
        result[r]=(d,context)
    return result

def features(case,original_adjacency):
    import numpy as np
    desc=descriptors(case,original_adjacency);rows=[]
    for r,old,new in case['pool']:
        d,context=desc[r];rows.append(np.outer(context,d[new]-d[old]).reshape(-1))
    out=np.asarray(rows,dtype=np.float64).reshape((-1,len(FEATURE_NAMES)))
    if not np.isfinite(out).all():raise RuntimeError('Nonfinite model input.')
    return out

def fit(X,y,regularization=.1):
    """32-weight, zero-intercept ridge; exact float64 TRAIN-only fit."""
    import numpy as np
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if len(X)<2 or X.shape!=(len(y),32) or not np.isfinite(X).all() or not np.isfinite(y).all():raise ValueError('Invalid fit rows.')
    scale=np.sqrt(np.mean(X*X,axis=0));scale=np.maximum(scale,1e-8)
    ys=max(float(np.sqrt(np.mean(y*y))),1e-12);z=X/scale
    w=np.linalg.solve(z.T@z/len(z)+regularization*np.eye(32),z.T@(y/ys)/len(z))
    return dict(weights=w.tolist(),feature_rms=scale.tolist(),gain_rms=ys,
                threshold=max(1e-6,.1*ys),regularization=regularization,training_rows=len(y),
                train_rmse=float(np.sqrt(np.mean((z@w*ys-y)**2))),intercept=0.,feature_names=FEATURE_NAMES)

def predict(model,X):
    import numpy as np
    return np.asarray(X)/np.asarray(model['feature_rms'])@np.asarray(model['weights'])*model['gain_rms']

def choose(model,X,pool,seed,group):
    import numpy as np
    predictions=predict(model,X)
    if len(pool)!=len(predictions):raise ValueError('Pool/features mismatch.')
    if not np.isfinite(predictions).all():raise RuntimeError('Nonfinite prediction.')
    ranked=int(np.argmax(predictions)) if len(pool) else None
    active=ranked is not None and float(predictions[ranked])>model['threshold']
    ri=random.Random(keyed(seed,group,'local_swap_random_v1')).randrange(len(pool)) if pool else None
    return dict(learned=pool[ranked] if active else None,
                matched_random=pool[ri] if active else None,
                always_random=pool[ri] if pool else None,
                predicted_gains=predictions.tolist(),active=active,threshold=model['threshold'])
