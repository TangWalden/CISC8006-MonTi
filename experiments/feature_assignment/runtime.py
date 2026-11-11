"""Shared source-bound runtime; ML libraries import only after verification."""
import copy
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from types import MethodType,SimpleNamespace

SOURCE_ROOT=pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT='ca093d8b12ea153a076b02bbdbc4ede58508560c'
ARMS=('baseline','relation_assignment','gcn_assignment','matched_random_permutation')
PROPOSALS=ARMS[1:]
SEEDS=(3,4,5)
TRAIN_PREFLIGHT=(263,238,84,125)
BASE_PATH='runs/feature_polish_20261104_attempt4/experiment_protocol_v1.json'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
PREVIOUS_PATH='runs/candidate_target_reallocation_20261105_attempt1/protocol_v1.json'
PREVIOUS_SHA='24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14'
TEACHER_MANIFEST='runs/feature_polish_20261104_attempt1/teacher_attempt1/teacher_artifacts.json'
TEACHER_SHA='de1bb484c201a2e60c7e2b63047587e8e1123ee465ba5eb72ddfe226ab60f111'
TEACHER_CHECKPOINT_SHA='1952387acec4473bef7246aa0db5ef44f9971b727d89a44086af701989e4c3fd'
UUIDS={'0':'GPU-d7aa6d97-d823-0319-b218-84acc8e26019','1':'GPU-1bb984a3-e0f6-138a-cfb7-dc3365d8f9cf'}


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(8*1024*1024),b''):h.update(value)
    return h.hexdigest()


def load_json(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def relative_file(root,name):
    name=pathlib.Path(name)
    if name.is_absolute() or '..' in name.parts:raise RuntimeError('Only explicit root-relative input paths are accepted.')
    path=(pathlib.Path(root)/name).resolve()
    if pathlib.Path(root).resolve() not in path.parents or not path.is_file():raise RuntimeError('Input missing or escapes its explicit root: '+str(name))
    return path


def verify_source_manifest(root,manifest):
    mandatory={'methods/feature_assignment.py','methods/target_reallocation.py'}|{
        'experiments/feature_assignment/'+name for name in ('runtime.py','prepare.py','run.py','preflight.py','driver.py')}
    if not mandatory<=set(manifest['files']):raise RuntimeError('Study source archive inventory is incomplete.')
    if any(pathlib.PurePosixPath(name).parts[0] not in ('methods','experiments','tests') for name in manifest['files']):
        raise RuntimeError('Minimal study archive must not shadow historical course/modules/models packages.')
    for name,wanted in manifest['files'].items():
        if sha(relative_file(root,name))!=wanted:raise RuntimeError('New study source changed: '+name)


def verify_protocol(path):
    path=pathlib.Path(path).resolve();p=load_json(path)
    if (p['schema']!='monti_feature_assignment_validation_v1' or p['status']!='frozen' or
            p['owner_execution_authorized'] is not True or p['final_TEST_allowed'] is not False or tuple(p['arms'])!=ARMS or
            tuple(p['seeds'])!=SEEDS or len(p['valid_groups'])!=128 or len(set(p['valid_groups']))!=128):
        raise RuntimeError('Wrong frozen development protocol/arms/cohort.')
    source=pathlib.Path(p['source_root']).resolve();runtime=pathlib.Path(p['runtime_root']).resolve();assets=pathlib.Path(p['asset_root']).resolve()
    if source!=SOURCE_ROOT or assets!=runtime:raise RuntimeError('Source/runtime/asset root binding differs.')
    verify_source_manifest(source,p['source_manifest'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()!=RUNTIME_COMMIT:
        raise RuntimeError('Historical runtime must stay at publicca093; do not sync curated main over it.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip():
        raise RuntimeError('Historical runtime tracked files changed.')
    tree=subprocess.check_output(['git','rev-parse',RUNTIME_COMMIT+'^{tree}'],cwd=runtime,text=True).strip()
    if tree!=p['runtime_tree']:raise RuntimeError('Historical public tree binding differs.')
    roots={'runtime':runtime,'asset':assets}
    for ref in p['input_refs']:
        if sha(relative_file(roots[ref['root']],ref['path']))!=ref['sha256']:
            raise RuntimeError('A frozen runtime/data/old receipt/checkpoint input changed: '+ref['path'])
    if p['settings']!={'acceptance_improvement':1e-7,'iterations':1,'loss':'uniform_mean_softplus_fraud_minus_benign','gradient_point':'baseline_x0'}:
        raise RuntimeError('Assignment, gradient or acceptance settings changed.')
    for name in (str(runtime),str(source)):
        if name not in sys.path:sys.path.insert(0,name)
    return p,dict(study_source_commit=p['source_manifest']['source_commit'],runtime_commit=RUNTIME_COMMIT,protocol_sha256=sha(path))


def check_device(p):
    info=p['device'];expected=info['cuda_visible_devices']
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise RuntimeError('Visible device differs from frozen selection.')
    if info['kind']=='gpu':
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        selected=[line.split(',') for line in raw.splitlines() if line.split(',')[0].strip()==expected]
        if len(selected)!=1 or selected[0][1].strip()!=info['uuid'] or int(selected[0][2].strip())<8192:
            raise RuntimeError('Frozen selected GPU no longer meets physical UUID/free-memory gate; no automatic device fallback.')
    import torch
    import numpy as np
    import scipy
    import dgl
    versions=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,dgl=dgl.__version__)
    if (versions['python']!='3.8.20' or versions['torch']!='1.13.1+cu117' or versions['numpy']!='1.24.4' or
            versions['scipy']!='1.10.1' or not versions['dgl'].startswith('1.1.2')):
        raise RuntimeError('Locked ML runtime versions changed: '+str(versions))
    torch.set_num_threads(8)
    counts=(torch.cuda.device_count(),torch._C._cuda_getDeviceCount())
    if info['kind']=='cpu':
        if torch.cuda.is_available() or counts!=(0,0):raise RuntimeError('CPU fallback must have zero Torch/native visible GPUs.')
        return dict(kind='cpu',torch_count=0,native_count=0,GPU_hours=0,versions=versions)
    if counts!=(1,1):raise RuntimeError('Both Torch/native GPU counts must be1.')
    torch.zeros(1,device='cuda:0')
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    own={line.split(',')[1].strip() for line in raw.splitlines() if line.split(',')[0].strip()==str(os.getpid())}
    if own!={info['uuid']}:raise RuntimeError('This process does not own the frozen physical GPU UUID.')
    return dict(kind='gpu',torch_count=1,native_count=1,physical_uuid=info['uuid'],numeric_visibility=expected,versions=versions)


def tensor_hash(value):
    value=value.detach().cpu().contiguous();header=json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode()
    return hashlib.sha256(len(header).to_bytes(8,'big')+header+value.numpy().tobytes()).hexdigest()


def source_state(model):
    from course.dual_surrogate.runtime import model_hash
    return model_hash(model)


def assert_frozen(model,wanted):
    if source_state(model)!=wanted or any(v.grad is not None or v.requires_grad for v in model.parameters()):
        raise RuntimeError('A frozen classifier parameter/buffer/gradient flag changed.')


def data_hashes(data,view):
    values=[('features',data.features),('graph_features',data.graph.ndata['x']),('feat_sum',view.feat_sum),('degrees',view.d)]
    if 'h' in data.graph.ndata:values.append(('cached_h_GCN',data.graph.ndata['h']))
    return {name:tensor_hash(value) for name,value in values}


def load_engine(p,seed,output,deadline):
    from course.feature_polish.refine import make_baseline_engine
    base=copy.deepcopy(load_json(pathlib.Path(p['asset_root'])/BASE_PATH));base['valid_groups']=list(p['valid_groups'])
    return make_baseline_engine(base,seed,output,deadline)


def load_own_teacher(p,data):
    from course.feature_polish.teacher import load_teacher
    return load_teacher(dict(teacher_ref=dict(path=TEACHER_MANIFEST,sha256=TEACHER_SHA),
        teacher_training_source='f3892f06335e4d91813cc537089694df3ec54ce5',
        teacher_preparation_protocol_sha256='c3794d4f8ee11458f6c405b574911059b4113feb6898509613c1727cd2fd4967',
        source_commit=p['source_manifest']['source_commit']),data)


def graph_with_features(data,graph,x):
    import torch
    local=graph.local_var();local.ndata['x']=torch.cat((data.features.detach(),x),dim=0)
    return local


def canonical(data,view,graph,ids,candidates,injected,x):
    import torch
    from modules.data_handler import DataHandler
    copied=graph_with_features(data,graph,x)
    for name,value in view.group_masks.items():
        suffix=torch.ones(len(injected),dtype=value.dtype,device=graph.device) if name=='train_masked' else torch.zeros(len(injected),dtype=value.dtype,device=graph.device)
        copied.ndata[name]=torch.cat((value.to(graph.device),suffix))
    oracle=SimpleNamespace(**vars(view));oracle.group_agg_tensor=MethodType(DataHandler.group_agg_tensor,oracle)
    oracle.feat_agg_tensor=MethodType(DataHandler.feat_agg_tensor,oracle)
    sequence=DataHandler.get_updated_sequences(oracle,copied,ids,candidates,injected,edge_weight=False)
    if sequence.shape!=(len(ids),7,data.feat_dim) or not bool(torch.isfinite(sequence).all()):raise RuntimeError('Native canonical input invalid.')
    return sequence


def fixed_blocks(data,graph,ids):
    import dgl
    sampler=dgl.dataloading.MultiLayerFullNeighborSampler(2)
    _,output,blocks=sampler.sample_blocks(graph,ids)
    if not output.equal(ids):raise RuntimeError('Frozen GCN blocks changed target order.')
    source_ids=blocks[0].srcdata[dgl.NID].long()
    signature=[]
    for block in blocks:
        src,dst=block.edges(order='eid')
        signature.append(dict(src_nodes=tensor_hash(block.srcdata[dgl.NID]),dst_nodes=tensor_hash(block.dstdata[dgl.NID]),
            src_arcs=tensor_hash(src),dst_arcs=tensor_hash(dst)))
    def forward(model,x):
        import torch
        local=[block.local_var() for block in blocks]
        full=torch.cat((data.features.detach().to(dtype=x.dtype),x),dim=0)
        local[0].srcdata['x']=full[source_ids]
        return model(local,edge_weight=False)
    return forward,signature


def make_case(p,engine,data,view,seed,group,phase='validation'):
    import torch
    from utils.utils import get_n_ids
    from course.source_role_fix.saved_teacher_diagnostic import reconstruct
    from course.feature_polish.affine import build_affine
    from methods.target_reallocation import reallocate_candidates
    engine.current_group=group;engine.phase=phase;engine.current_epoch=8
    ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
    nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
    with torch.no_grad():native=engine.get_perturbed_graph(ids,nb,eb,False)
    del native
    capture={k:v.detach().clone() for k,v in engine.capture.items()};engine.capture=engine.live_capture=None
    original=data.graph.num_nodes();candidates=capture['candidate_ids'];injected=torch.arange(original,original+nb,device=data.device)
    with torch.no_grad():
        if not bool(torch.isfinite(capture['adj_attack']).all() and ((capture['adj_attack']==0)|(capture['adj_attack']==1)).all()):
            raise RuntimeError('Generated adjacency must be finite exact binary before conversion.')
        allocation=reallocate_candidates(capture['adj_attack'].long().cpu().tolist(),ids.cpu().tolist(),candidates.cpu().tolist(),
            injected.cpu().tolist(),seed=seed,group_key=group)
        random_adj=torch.tensor(allocation['arms']['random_target_fill'],dtype=capture['adj_attack'].dtype,device=data.device)
        labels=data.labels[ids].detach().cpu().tolist()
        info=dict(group=group,node_ids=ids.cpu().tolist(),labels=labels,node_budget=nb,edge_budget=eb,actual_injected_nodes=nb,actual_injected_edges=eb)
        graph,ids,candidates,injected=reconstruct(data,dict(capture,adj_attack=random_adj),info,info)
        x0=capture['x_atk'];s0=canonical(data,view,graph,ids,candidates,injected,x0)
        affine=build_affine(view,graph,ids,candidates,injected,x0,base_sequence=s0)
    gcn_forward,block_signature=fixed_blocks(data,graph,ids)
    return dict(capture=capture,random_fill_adjacency=random_adj,graph=graph,ids=ids,candidates=candidates,injected=injected,
        x0=x0,baseline_sequence=s0,affine=affine,gcn_forward=gcn_forward,block_signature=block_signature,
        info=info,physical_degrees=(graph.in_degrees(injected)-1).long(),allocation=allocation)


def softplus(logits):
    import torch.nn.functional as F
    return F.softplus(logits[:,1]-logits[:,0]).mean()


def multiset_hash(x):
    rows=x.detach().cpu().contiguous().numpy();hashes=sorted(hashlib.sha256(row.tobytes()).digest() for row in rows)
    return hashlib.sha256(b''.join(hashes)).hexdigest()


def check_permuted(case,x,permutation):
    import torch
    x0=case['x0'];perm=permutation.to(device=x0.device,dtype=torch.long)
    if sorted(perm.cpu().tolist())!=list(range(len(x0))):raise RuntimeError('Feature mapping is not a bijection.')
    expected=x0.index_select(0,perm)
    if tensor_hash(x)!=tensor_hash(expected) or multiset_hash(x)!=multiset_hash(x0):raise RuntimeError('Feature multiset/row bytes changed.')
    graph=case['graph'];injected=case['injected']
    if not torch.equal(graph.in_degrees(injected)-1,case['physical_degrees']):raise RuntimeError('Fixed injection degrees changed.')
    return dict(multiset_sha256=multiset_hash(x),feature_sha256=tensor_hash(x),
        moved_positions=int((perm!=torch.arange(len(perm),device=perm.device)).sum()),
        changed_feature_rows=int((x!=x0).any(1).sum()),raw_displacement_linf=float((x-x0).abs().max()),
        raw_displacement_l2=float((x-x0).double().norm()))
