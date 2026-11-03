"""Shared frozen classifiers/data views. Victim loading is explicit and delayed."""
import contextlib
import hashlib
import json
import pathlib
import random
from types import SimpleNamespace

import dgl
import numpy as np
import scipy.sparse as sp
import torch

from models.victim_model import GAGA, GCN
from modules.data_handler import DataHandler
from course.dual_surrogate.common import ROOT, sha256
from course.dual_surrogate.verify import ORIGINAL_GCN_SHA256,SEALED_GAGA_SHA256

_VERIFIED_TEACHER_DIRECTORIES = {}


def verify_teacher_directory(directory):
    directory = pathlib.Path(directory).resolve()
    manifest_path=directory/'teacher_artifacts.json'
    identity=(str(directory),sha256(manifest_path))
    if identity in _VERIFIED_TEACHER_DIRECTORIES:
        return _VERIFIED_TEACHER_DIRECTORIES[identity]
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('both_quality_gates_passed') is not True:
        raise RuntimeError('Both independent teacher quality gates must pass.')
    cache=manifest['cache_manifest']
    if sha256(directory/cache['path'])!=cache['sha256']:raise RuntimeError('FIT cache manifest changed.')
    entries=list(manifest['cache']['files'].values())
    for row in manifest['teachers'].values():
        entries.append(dict(path=row['checkpoint'],sha256=row['checkpoint_sha256']))
    for row in entries:
        relative=pathlib.Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or sha256(directory/relative)!=row['sha256']:
            raise RuntimeError('Teacher/cache bytes changed: '+row['path'])
    _VERIFIED_TEACHER_DIRECTORIES[identity]=manifest
    return manifest


def checked_logits(value, count):
    if value.ndim==1 and count==1:value=value.unsqueeze(0)
    if value.shape!=(count,2) or not bool(torch.isfinite(value).all()):
        raise RuntimeError('Classifier shape or finite-value contract failed.')
    return value


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        header = json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode()
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8,'big')); digest.update(header)
        digest.update(len(raw).to_bytes(8,'big')); digest.update(raw)
    return digest.hexdigest()


@contextlib.contextmanager
def fork_rng(seed):
    py, npstate = random.getstate(), np.random.get_state()
    cpu = torch.random.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state(0).clone() if torch.cuda.is_available() else None
    try:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if gpu is not None:torch.cuda.manual_seed(seed)
        yield
    finally:
        random.setstate(py); np.random.set_state(npstate)
        torch.random.set_rng_state(cpu)
        if gpu is not None:torch.cuda.set_rng_state(gpu,0)


def config_for(seed, epochs=8):
    config = json.loads((ROOT/'course/configs/baseline.json').read_text(encoding='utf-8'))
    config.update(seed=seed,cuda_id=0,victim_model='GCN',surrogate_model='GCN',
        epochs=epochs,valid_epoch=epochs,patience=epochs,batch_size=1,random_candidates=False,
        shared_parameters=False,without_candidates=False,pos_encoding=True,ada_budget=True)
    return config


def load_data(protocol, seed):
    data = DataHandler(config_for(seed))
    if not set(protocol['train_groups']) <= set(map(int,data.target_idx_train)):
        raise RuntimeError('TRAIN group crosses the original partition.')
    if not set(protocol['valid_groups']) <= set(map(int,data.target_idx_valid)):
        raise RuntimeError('VALID group crosses the original partition.')
    data.target_idx_train = np.array(protocol['train_groups'],dtype=np.int64)
    data.target_idx_valid = np.array(protocol['valid_groups'],dtype=np.int64)
    return data


def load_fit_view(data, directory, protocol):
    directory = pathlib.Path(directory)
    ref = protocol['teacher_artifacts_ref']
    if sha256(ROOT/ref['path']) != ref['sha256']:
        raise RuntimeError('New teacher artifact manifest bytes changed.')
    manifest = verify_teacher_directory(directory)
    adj = sp.load_npz(directory/'original_adjacency.npz')
    adj2 = sp.load_npz(ROOT/protocol['data']['adj2_cache'])
    masks = torch.load(directory/'fit_group_masks.pt',map_location='cpu')
    view = SimpleNamespace(graph=data.graph,features=data.features,device=data.device,feat_dim=data.feat_dim,
        adj=adj,adj2=adj2,feat_sum=torch.from_numpy(np.load(directory/'fit_feat_sum.npy')),
        d=torch.from_numpy(np.load(directory/'fit_degrees.npy')),group_masks=masks)
    return view, manifest


def load_original_view(data, adj=None, adj2=None):
    directory = ROOT/'Datasets/GossipCop/data_split'
    if adj is None:
        adj = sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr()
        adj.setdiag(0); adj.eliminate_zeros()
    if adj2 is None:
        adj2 = sp.load_npz(directory/'gossipcop_0.05_gaga_adj2_0.npz')
    train = torch.zeros(data.graph.num_nodes(),dtype=torch.float32,device=data.graph.device)
    train[torch.as_tensor(data.idx_train,device=data.graph.device)] = 1
    masks = dict(train_neg=(1-data.labels.float())*train,
        train_pos=data.labels.float()*train,train_masked=1-train)
    return SimpleNamespace(graph=data.graph,features=data.features,device=data.device,feat_dim=data.feat_dim,
        adj=adj,adj2=adj2,group_masks=masks,
        feat_sum=torch.from_numpy(np.load(directory/'gossipcop_0.05_gaga_feat_sum_0.npy')),
        d=torch.from_numpy(np.load(directory/'gossipcop_0.05_gaga_degree_0.npy')))


def load_classifier(role, data, teacher_directory=None):
    if role in ('gcn_new','gaga_new'):verify_teacher_directory(teacher_directory)
    if role == 'gcn_base':
        model = GCN(data.feat_dim,[64,64],num_classes=2)
        path = ROOT/'models/pretrained_models/surrogate_models/GCN-gossipcop_0.05-0.pickle'
    elif role == 'gcn_new':
        model = GCN(data.feat_dim,[64,64],num_classes=2)
        path = pathlib.Path(teacher_directory)/'gcn2_teacher.pt'
    elif role in ('gaga_new','gaga_victim'):
        model = GAGA(data.feat_dim,64,2,num_relations=1,n_hops=2,n_heads=4,ff_dim=128,n_layers=3,dropout=.1)
        path = (pathlib.Path(teacher_directory)/'gaga_teacher.pt' if role == 'gaga_new'
                else ROOT/'models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle')
    else:
        raise ValueError('Unregistered classifier role.')
    if role=='gcn_base' and sha256(path)!=ORIGINAL_GCN_SHA256:raise RuntimeError('Original GCN checkpoint changed.')
    if role=='gaga_victim' and sha256(path)!=SEALED_GAGA_SHA256:raise RuntimeError('Sealed GAGA checkpoint changed.')
    model.load_state_dict(torch.load(path,map_location='cpu'))
    model.to(data.device).eval()
    for value in model.parameters(): value.requires_grad_(False)
    return model


def gcn_logits(model, graph, targets, *, training=False, full_graph=False):
    if full_graph:
        return checked_logits(model([graph]*model.num_layers,edge_weight=training)[targets.to(graph.device)],len(targets))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.num_layers)
    _, output, blocks = sampler.sample_blocks(graph,targets.to(graph.device))
    if not torch.equal(output,targets.to(output.device)):
        raise RuntimeError('GCN target ordering changed.')
    return checked_logits(model([block.to(graph.device) for block in blocks],edge_weight=training),len(targets))


def transitions(labels, clean, attacked):
    if len(labels) != len(clean) or len(clean) != len(attacked):
        raise RuntimeError('Transition denominator differs.')
    cw, aw = clean != labels, attacked != labels
    return dict(occurrences=len(labels),clean_wrong=int(cw.sum()),attacked_wrong=int(aw.sum()),
        new_wrong=int((~cw & aw).sum()),recoveries=int((cw & ~aw).sum()))
