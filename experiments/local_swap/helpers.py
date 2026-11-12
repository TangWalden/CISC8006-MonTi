"""Shared provenance and frozen graph reconstruction helpers."""
import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    with pathlib.Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(path):
    p=read(path);assert p['schema']=='monti_local_swap_v1' and p['status']=='frozen' and not p['final_TEST_allowed']
    assert pathlib.Path(p['source_root']).resolve()==ROOT and p['seeds']==[0,1,2]
    rt=pathlib.Path(p['runtime_root']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
    for n,h in p['source_manifest']['files'].items():assert sha(ROOT/n)==h,n
    for r in p['input_refs']:assert sha(rt/r['path'])==r['sha256'],r['path']
    for root in (str(rt),str(ROOT)):
        if root not in sys.path:sys.path.insert(0,root)
    return p,dict(source_commit=p['source_manifest']['source_commit'],protocol_sha256=sha(path),runtime_commit=p['runtime_commit'])
def capture(engine,data,seed,group,phase,p):
    import torch
    from utils.utils import get_n_ids
    from experiments.local_swap.core import proposal_pool
    from experiments.downstream_cooperation.common import tensor_hash
    ids=torch.tensor(get_n_ids(data.target_assignment,group),dtype=torch.long)
    assert ids.tolist()==p[phase+'_members'][str(group)] and bool((data.labels[ids]==1).all())
    b,e=int(data.node_budgets[group]),int(data.edge_budgets[group])
    engine.current_group=group;engine.current_epoch=0;engine.phase='local_swap_'+phase
    with torch.no_grad():native_graph=engine.get_perturbed_graph(ids,b,e,False,False)
    cap={k:v.detach().clone() for k,v in engine.capture.items()};engine.capture=engine.live_capture=None
    cids=cap['candidate_ids'];t=len(ids);c=len(cids);adj=cap['adj_attack']
    assert len(set(cids.tolist()))==c and not set(cids.tolist())&set(ids.tolist())
    assert adj.shape==(b,t+c+b)
    case=dict(seed=seed,group=group,ids=ids,candidate_ids=cids,source_ids=list(range(data.graph.num_nodes(),data.graph.num_nodes()+b)),
        x_aux=cap['x_atk'],x_targets=data.features[ids].detach().clone(),x_candidates=data.features[cids].detach().clone(),
        h_targets=data.graph.ndata['h'][ids].detach().clone(),h_candidates=data.graph.ndata['h'][cids].detach().clone(),
        native_adjacency=adj,quota=dict(edge_budget=e),feature_sha256=tensor_hash(cap['x_atk']))
    case['pool']=proposal_pool(adj[:,t:t+c].tolist(),seed,group,p['settings']['pool_size'])
    return case,native_graph
