"""Canonical portal-token equivalence and physical-budget TRAIN preflight."""
import argparse,pathlib,json,sys,time,hashlib
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=json.loads((s/'registration.json').read_text())
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h
 sys.path[:0]=[str(s/'source'),p['support_source']+'/experiments/block_swap',p['support_source'],p['runtime_root']]
 from helpers import verify
 verify(pathlib.Path(p['support_source']).parent/'registration/protocol.json')
 import torch,numpy as np
 from modules.data_handler import DataHandler
 from models.victim_model import GAGA
 from experiments.local_swap.loader import CONFIG,DEFAULTS
 from experiments.downstream_cooperation.common import check_device,make_graph,canonical
 from course.dual_surrogate.runtime import load_original_view
 from geometry_hard import proposals,replace
 from model import PortalRanker
 check_device();start=time.monotonic();cpu=time.process_time();config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config);view=load_original_view(data)
 trainids=torch.as_tensor(data.idx_train).long();benign=trainids[data.labels[trainids]==0].numpy();assert sha(p['teacher_checkpoint'])==p['teacher_sha256']
 model=GAGA(data.feat_dim,64,2);model.load_state_dict(torch.load(p['teacher_checkpoint'],map_location='cpu'));model.eval()
 up=json.loads((pathlib.Path(p['capture_parent'])/'upstream_seal.json').read_text());rows=[];forwards=0
 for ref in up['cases']:
  if ref['phase']!='train' or ref['seed']!=0:continue
  f=pathlib.Path(p['capture_parent'])/ref['path'];assert sha(f)==ref['sha256'];case=torch.load(f,map_location='cpu');donor,bank=proposals(case,view,benign)
  if donor is None:continue
  first=replace(case,donor,bank[0],data.features);graph=make_graph(data,first,first['native_adjacency']);old=make_graph(data,case,case['native_adjacency'])
  assert torch.equal(graph.in_degrees()[case['source_ids']],old.in_degrees()[case['source_ids']]);assert torch.equal(first['x_aux'],case['x_aux'])
  with torch.no_grad():seq=canonical(data,view,graph,first)
  errors=[]
  for j,node in enumerate(bank[:3]):
   changed=replace(case,donor,node,data.features);g=make_graph(data,changed,changed['native_adjacency'])
   assert torch.equal(g.in_degrees()[case['source_ids']],old.in_degrees()[case['source_ids']])
   assert torch.equal(g.ndata['x'][case['source_ids']],case['x_aux'])
   with torch.no_grad():actual=canonical(data,view,g,changed);z=model(actual);forwards+=1
   assert bool(torch.isfinite(actual).all()) and bool(torch.isfinite(z).all())
   errors.append(dict(physical_graph_passed=True,teacher_logits_finite=True))
  ranker=PortalRanker(data.feat_dim);query=torch.zeros(3*data.feat_dim+3);keys=torch.randn(9,data.feat_dim);changes=torch.randn(9,data.feat_dim);keys[0]=0;changes[0]=0;opt=torch.optim.Adam(ranker.parameters(),lr=.001)
  opt.zero_grad();scores=ranker(query,keys,changes);assert float(scores[0])==0;loss=torch.nn.functional.cross_entropy(scores.unsqueeze(0),torch.tensor([1]));loss.backward();assert sum(float(x.grad.abs().sum()) for x in ranker.parameters() if x.grad is not None)>0;opt.step()
  rows.append(dict(group=ref['group'],bank_count=len(bank),donor_slot=donor,canonical_checks=errors,physical_degrees_preserved=True,synthetic_ranker_update_passed=True))
  if len(rows)==2:break
 assert len(rows)==2
 out=dict(status='passed',cases=rows,own_teacher_forwards=forwards,original_victim_forwards=0,TEST_queries=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
 with (s/'preflight_result.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
