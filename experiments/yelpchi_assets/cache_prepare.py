"""YelpChi asset/cache preparation only: no training or efficacy evaluation."""
import pathlib,json,hashlib,sys,os,time,signal,subprocess
S=pathlib.Path(__file__).resolve().parent;R=S.parent.parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
 os.chdir(S);start=time.monotonic();cpu=time.process_time();counts={'GCN_representation_forwards':0,'MonTi_generations':0,'GAGA_forwards':0,'optimizer_updates':0,'TEST_queries':0,'GPU_hours':0}
 reg=json.loads((S/'cache_registration.json').read_text());result={'status':'failed','scope':'Asset compatibility and deterministic input caches only, no attack or efficacy evaluation','counters':counts}
 signal.signal(signal.SIGALRM,lambda *x:(_ for _ in ()).throw(TimeoutError('cache preparation limit')));signal.alarm(1800)
 try:
  assert subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()==reg['runtime_commit']
  assert not subprocess.check_output(['git','-C',str(R),'status','--porcelain','--untracked-files=no'],text=True).strip()
  for rel,h in reg['files'].items():assert sha(S/rel)==h,rel
  for rel,h in reg['runtime_sources'].items():assert sha(R/rel)==h,rel
  sys.path.insert(0,str(R));import torch,numpy as np,dgl
  assert os.environ['CUDA_VISIBLE_DEVICES']=='' and torch.cuda.device_count()==0
  torch.set_num_threads(4)
  from modules.data_handler import DataHandler
  from modules.experiment_handler import ExperimentHandler
  from models.victim_model import GCN,GAGA
  from utils.utils import set_seed
  cfg=json.loads((S/'best_models/configs/GCN-GAGA-yelpchi_rtr-best_config.json').read_text());cfg.update(random_candidates=False,shared_parameters=False,without_candidates=False,pos_encoding=True,ada_budget=True)
  data=DataHandler(cfg)
  assert data.num_nodes==45900 and data.feat_dim==32
  parts=[set(map(int,getattr(data,'idx_'+x))) for x in ['train','valid','test']]
  assert all(not parts[i]&parts[j] for i in range(3) for j in range(i))
  assert set.union(*parts)==set(range(data.num_nodes))
  target_info={}
  for phase,part in zip(['train','valid','test'],parts):
   groups=list(map(int,getattr(data,'target_idx_'+phase)));target_info[phase]={'groups':len(groups),'occurrences':0,'unique_targets':set()}
   for group in groups:
    ids=data.target_assignment[group].nonzero()[1];assert set(map(int,ids))<=part
    target_info[phase]['occurrences']+=len(ids);target_info[phase]['unique_targets'].update(map(int,ids))
   target_info[phase]['unique_targets']=len(target_info[phase]['unique_targets'])
  gcncfg=json.loads((R/'models/pretrained_models/configs/GCN-yelpchi_rtr-0.json').read_text());gcn=GCN(data.feat_dim,gcncfg['emb_sizes'],num_classes=2)
  gcn.load_state_dict(torch.load(S/'models/pretrained_models/surrogate_models/GCN-yelpchi_rtr-0.pickle',map_location='cpu'),strict=True);gcn.eval()
  for v in gcn.parameters():v.requires_grad_(False)
  sampler=dgl.dataloading.MultiLayerFullNeighborSampler(gcn.num_layers)
  loader=dgl.dataloading.DataLoader(data.graph,torch.arange(data.num_nodes),sampler,batch_size=cfg['batch_size_pre'],num_workers=0,shuffle=False)
  h=torch.zeros(data.num_nodes,cfg['emb_dim'])
  with torch.no_grad():
   for _,ids,blocks in loader:h[ids]=gcn(blocks,prediction=False);counts['GCN_representation_forwards']+=1
  assert bool(torch.isfinite(h).all());np.save(S/'Datasets/YelpChi_RTR/h_GCN.npy',h.numpy())
  # Strict compatibility checks without invoking either attack or victim.
  dummy=object.__new__(ExperimentHandler);dummy.config=cfg;dummy.data=data;dummy.device=torch.device('cpu');set_seed(cfg['seed'])
  upstream=ExperimentHandler.initialize_attack_model(dummy);upstream.load_state_dict(torch.load(S/'best_models/checkpoints/GCN-GAGA-yelpchi_rtr-best_model.pickle',map_location='cpu'),strict=True)
  victim=GAGA(data.feat_dim,64,2);victim.load_state_dict(torch.load(S/'models/pretrained_models/victim_models/GAGA-yelpchi_rtr-0.pickle',map_location='cpu'),strict=True)
  outputs={}
  for f in (S/'Datasets/YelpChi_RTR').rglob('*'):
   if f.is_file() and str(f.relative_to(S)) not in reg['files']:outputs[str(f.relative_to(S))]={'sha256':sha(f),'bytes':f.stat().st_size}
  result.update(status='completed',nodes=data.num_nodes,features=data.feat_dim,split_nodes={x:len(v) for x,v in zip(['train','valid','test'],parts)},target_metadata=target_info,outputs=outputs,strict_author_and_victim_compatible=True,cache_label_scope='Only original TRAIN labels enter GAGA groups. Full transductive topology/features as upstream code. No TEST prediction or label-outcome evaluation.')
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(S/'cache_result.json',result)
if __name__=='__main__':main()
