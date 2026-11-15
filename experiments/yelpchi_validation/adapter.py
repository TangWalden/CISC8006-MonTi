"""Explicit YelpChi data/classifier mapping, never the GossipCop victim loader."""
import pathlib,json,os
from types import SimpleNamespace
from common import read,sha
def load_data(q):
 import torch,numpy as np,scipy.sparse as sp
 from modules.data_handler import DataHandler
 from experiments.downstream_cooperation.common import check_device
 from course.dual_surrogate.runtime import original_train_masks
 check_device();root=pathlib.Path(q['asset_root']);os.chdir(root)
 cfg=read(root/'best_models/configs/GCN-GAGA-yelpchi_rtr-best_config.json');cfg.update(random_candidates=False,shared_parameters=False,without_candidates=False,pos_encoding=True,ada_budget=True,victim_model='GCN')
 data=DataHandler(cfg);data.target_idx_test=np.empty(0,dtype=np.int64);data._asset_root=root;data._runtime_root=pathlib.Path(q['runtime_root']);assert data.feat_dim==32 and data.num_nodes==45900
 d=root/'Datasets/YelpChi_RTR/data_split';adj=sp.csr_matrix(data.graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros()
 view=SimpleNamespace(graph=data.graph,features=data.features,device=data.device,feat_dim=data.feat_dim,adj=adj,adj2=sp.load_npz(d/'yelpchi_rtr_gaga_adj2_0.npz'),group_masks=original_train_masks(data.labels,data.idx_train),feat_sum=torch.from_numpy(np.load(d/'yelpchi_rtr_gaga_feat_sum_0.npy',allow_pickle=False)),d=torch.from_numpy(np.load(d/'yelpchi_rtr_gaga_degree_0.npy',allow_pickle=False)))
 return data,view
def load_teacher(q,data):
 import torch
 from models.victim_model import GAGA
 from experiments.downstream_cooperation.common import freeze
 f=pathlib.Path(q['teacher_checkpoint']);receipt=read(f.parent/'result.json');assert receipt['status']=='completed' and receipt['quality_passed'] and sha(f)==receipt['checkpoint_sha256']
 model=GAGA(data.feat_dim,64,2);model.load_state_dict(torch.load(f,map_location='cpu'));return model,freeze(model)
def load_classifier(role,data):
 import torch
 from models.victim_model import GAGA,GCN
 from experiments.downstream_cooperation.common import freeze
 if role=='gaga_victim':
  path=data._asset_root/'models/pretrained_models/victim_models/GAGA-yelpchi_rtr-0.pickle';expected='5422b9ed067f268b81d810e9eb9fe1bb0b8662d78715c98e9f7d8aed49cd751f';model=GAGA(data.feat_dim,64,2)
 elif role=='gcn_base':
  path=data._asset_root/'models/pretrained_models/surrogate_models/GCN-yelpchi_rtr-0.pickle';expected='be3c7be91f1041af07a8dcd3c036a32b8f4cbc39bb8107a8499049a41f0781b4';cfg=read(data._runtime_root/'models/pretrained_models/configs/GCN-yelpchi_rtr-0.json');model=GCN(data.feat_dim,cfg['emb_sizes'],2)
 else:raise ValueError(role)
 assert sha(path)==expected;model.load_state_dict(torch.load(path,map_location='cpu'),strict=True);freeze(model);return model
def load_upstream(q,data):
 import torch,numpy as np,dgl
 from modules.experiment_handler import ExperimentHandler
 from experiments.downstream_cooperation.common import freeze
 root=pathlib.Path(q['asset_root']);cfg=read(root/'best_models/configs/GCN-GAGA-yelpchi_rtr-best_config.json');assert cfg['seed']==2 and cfg['node_budget']==.05 and cfg['edge_budget']==.5
 cfg.update(random_candidates=False,shared_parameters=False,without_candidates=False,pos_encoding=True,ada_budget=True)
 class Frozen(ExperimentHandler):
  def load_pretrained_models(self):
   self.surrogate_model=load_classifier('gcn_base',self.data);self.sampler_surrogate=dgl.dataloading.MultiLayerFullNeighborSampler(self.surrogate_model.num_layers);self.victim_model=None
  def get_node_representations(self):self.data.graph.ndata['h']=torch.from_numpy(np.load(root/'Datasets/YelpChi_RTR/h_GCN.npy',allow_pickle=False))
  def run(self):raise RuntimeError('Upstream training forbidden')
  def test(self):raise RuntimeError('TEST forbidden')
 engine=Frozen(cfg,data);path=root/'best_models/checkpoints/GCN-GAGA-yelpchi_rtr-best_model.pickle';assert sha(path)=='18b4b85e44d360f350e14587786c2cf31133497da54cac2d70854e64aeeeaca4';engine.model.load_state_dict(torch.load(path,map_location='cpu'),strict=True);h=freeze(engine.model)
 def no_update(*a,**k):raise RuntimeError('Frozen author optimizer')
 engine.optimizer.step=no_update
 return engine,h
def capture(engine,data,view,seed,phase,group,p):
 import torch,dgl
 from course.dual_surrogate.runtime import fork_rng
 from course.dual_surrogate.common import case_seed
 from experiments.downstream_cooperation.common import make_graph,canonical
 ids=torch.tensor(p[phase+'_members'][str(group)],dtype=torch.long);assert ids.tolist()==data.target_assignment[group].nonzero()[1].tolist();b,e=int(data.node_budgets[group]),int(data.edge_budgets[group]);observed={}
 def hook(module,args,out):
  subgraph=args[0];x,adj,c=out;assert x is not None and adj is not None and c is not None
  observed.update(x_aux=x.detach().clone(),adj=adj.detach().clone(),candidate_ids=subgraph.ndata[dgl.NID][c].detach().clone())
 handle=engine.model.register_forward_hook(hook)
 try:
  with torch.no_grad(),fork_rng(case_seed(seed,'local_swap_'+phase,0,group)):g=engine.get_perturbed_graph(ids,b,e,False,False)
 finally:handle.remove()
 assert g is not None and g.num_nodes()==data.num_nodes+b;assert g.num_edges()-data.graph.num_edges()==b+2*e
 cids=observed['candidate_ids'];assert not set(ids.tolist())&set(cids.tolist()) and len(set(cids.tolist()))==len(cids)
 case=dict(seed=seed,group=group,ids=ids,candidate_ids=cids,source_ids=list(range(data.num_nodes,data.num_nodes+b)),x_aux=observed['x_aux'],x_targets=data.features[ids].clone(),x_candidates=data.features[cids].clone(),native_adjacency=observed['adj'],quota=dict(edge_budget=e))
 assert case['native_adjacency'].shape==(b,len(ids)+len(cids)+b) and int(case['native_adjacency'].sum())==e
 with torch.no_grad():seq=canonical(data,view,g,case)
 return case,seq
