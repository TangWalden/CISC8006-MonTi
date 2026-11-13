"""Physical counterfactual memory retrieval, listwise learning and matched controls."""
import pathlib,json,sys,time,argparse,hashlib,signal,statistics,random
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=read(s/'main_protocol.json')
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h
 assert read(s/'preflight_hard_v2_result.json')['status']=='passed' and sha(p['teacher_checkpoint'])==p['teacher_sha256']
 sys.path[:0]=[str(s/'source'),p['support_source']+'/experiments/block_swap',p['support_source'],p['runtime_root']]
 from helpers import verify
 verify(pathlib.Path(p['support_source']).parent/'registration/protocol.json')
 import torch,numpy as np
 from modules.data_handler import DataHandler
 from models.victim_model import GAGA
 from experiments.local_swap.loader import CONFIG,DEFAULTS
 from experiments.downstream_cooperation.common import check_device,make_graph,canonical,state_hash,assert_frozen,tensor_hash
 from course.dual_surrogate.runtime import load_original_view,load_classifier
 from geometry_hard import proposals,replace,key
 from model import PortalRanker
 check_device();out=s/'main';out.mkdir(exist_ok=False)
 for n in ('cards','models'):(out/n).mkdir()
 start=time.monotonic();cpu=time.process_time();counts=dict(cards=0,physical_canonical_builds=0,teacher_TRAIN=0,teacher_VALID_selection=0,neural_updates=0,training_visits=0,no_choice_skips=0,official_GAGA_VALID=0,generations=0,TEST_queries=0)
 result=dict(status='failed',counters=counts,GPU_hours=0,protocol_sha256=sha(s/'main_protocol.json'))
 signal.signal(signal.SIGALRM,lambda *x:(_ for _ in ()).throw(TimeoutError('Frozen portal deadline')));signal.alarm(7200)
 try:
  config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config);data.target_idx_test=np.empty(0,dtype=np.int64);view=load_original_view(data)
  trainids=torch.as_tensor(data.idx_train).long();benign=trainids[data.labels[trainids]==0].numpy();mean=data.features[trainids].mean(0);std=data.features[trainids].std(0,unbiased=False).clamp_min(1e-6)
  teacher=GAGA(data.feat_dim,64,2);teacher.load_state_dict(torch.load(p['teacher_checkpoint'],map_location='cpu'));teacher.eval()
  for v in teacher.parameters():v.requires_grad_(False)
  th=state_hash(teacher);baseparent=pathlib.Path(p['base_cards_parent']);index0={tuple(v['key']):v for v in read(baseparent/'card_seal.json')['cards']};index={}
  print('STAGE physical_counterfactual_cards',flush=True)
  for i,((seed,phase,group),ref) in enumerate(index0.items()):
   f=baseparent/ref['path'];assert sha(f)==ref['sha256'];old=torch.load(f,map_location='cpu');case=old['case'];native=old['sequence'];donor,bank=proposals(case,view,benign)
   sequences=[native];features=[]
   for node in bank:
    changed=replace(case,donor,node,data.features);g=make_graph(data,changed,changed['native_adjacency'])
    with torch.no_grad():sequences.append(canonical(data,view,g,changed))
    counts['physical_canonical_builds']+=1;features.append(data.features[node])
   values=torch.stack(sequences);t=len(case['ids'])
   with torch.no_grad():logits=teacher(values.reshape(-1,7,data.feat_dim)).reshape(len(values),t,2)
   counts['teacher_TRAIN' if phase=='train' else 'teacher_VALID_selection']+=1
   losses=torch.nn.functional.softplus(logits[:,:,1]-logits[:,:,0]).mean(1)
   if donor is not None:
    rows=case['native_adjacency'][:,t+donor]>0;aux=case['x_aux'][rows].mean(0)
   else:aux=case['x_aux'].mean(0)
   query=torch.cat(((case['x_targets'].mean(0)-mean)/std,(aux-mean)/std,(native[:,4].mean(0)-mean)/std,torch.tensor([np.log1p(t),np.log1p(len(case['source_ids'])),np.log1p(float(view.d[case['ids']].reshape(t,7)[:,4].mean()))],dtype=torch.float32)))
   keys=torch.zeros(len(values),data.feat_dim)
   if bank:keys[1:]=(torch.stack(features)-mean)/std
   changes=(values[:,:,4].mean(1)-native[:,4].mean(0))/std
   card=dict(case=case,donor=donor,bank=bank,query=query,keys=keys,changes=changes,sequences=values,teacher_logits=logits,losses=losses)
   path=out/'cards'/('s%d_%s_g%d.pt'%(seed,phase,group));torch.save(card,path);index[seed,phase,group]=dict(path=str(path.relative_to(out)),sha256=sha(path));counts['cards']+=1
   if (i+1)%32==0:print('CARDS',i+1,576,flush=True)
  assert counts['cards']==576 and counts['teacher_TRAIN']==384 and counts['teacher_VALID_selection']==192;assert_frozen(teacher,th)
  write(out/'card_seal.json',dict(cards=[dict(key=list(k),**v) for k,v in index.items()],official_victim_loaded=False))
  def card(seed,phase,group):
   ref=index[seed,phase,group];f=out/ref['path'];assert sha(f)==ref['sha256'];return torch.load(f,map_location='cpu')
  models=[];plans=[]
  print('STAGE TRAIN_listwise_rankers',flush=True)
  with (out/'training.jsonl').open('x') as stream:
   for seed in p['seeds']:
    torch.manual_seed(2026111400+seed);net=PortalRanker(data.feat_dim);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    for epoch in range(8):
     order=list(p['train_groups']);random.Random(key(seed,epoch,'portal_training_order')).shuffle(order)
     for group in order:
      v=card(seed,'train',group);opt.zero_grad();scores=net(v['query'],v['keys'],v['changes']);scale=v['losses'].std(unbiased=False).clamp_min(.001);target=torch.softmax((v['losses'][0]-v['losses'])/scale,0)
      loss=-(target*torch.log_softmax(scores,0)).sum();assert bool(torch.isfinite(loss));counts['training_visits']+=1
      if len(scores)>1:
       loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();counts['neural_updates']+=1
      else:counts['no_choice_skips']+=1
      stream.write(json.dumps(dict(seed=seed,epoch=epoch+1,group=group,loss=float(loss),options=len(scores),updated=len(scores)>1))+'\n')
     stream.flush();print('TRAIN',seed,epoch+1,flush=True)
    net.eval();path=out/'models'/('s%d.pt'%seed);torch.save(net.state_dict(),path);models.append(dict(seed=seed,path=str(path.relative_to(out)),sha256=sha(path)))
    for group in p['valid_groups']:
     v=card(seed,'valid',group)
     with torch.no_grad():scores=net(v['query'],v['keys'],v['changes'])
     neural=int(scores.argmax());best=int(v['losses'].argmin());direct=best if float(v['losses'][0]-v['losses'][best])>1e-4 else 0
     ri=random.Random(key(seed,group,'portal_matched_random')).randrange(1,len(v['bank'])+1) if v['bank'] else 0
     selected=dict(native=0,neural=neural,random_neural=ri if neural else 0,direct=direct,random_direct=ri if direct else 0)
     plans.append(dict(seed=seed,group=group,selected=selected,neural_scores=scores.tolist(),teacher_losses=v['losses'].tolist(),bank=v['bank'],donor=v['donor']))
  assert counts['training_visits']==3072 and counts['neural_updates']+counts['no_choice_skips']==3072;write(out/'sealed_before_victim.json',dict(plans=plans,models=models,official_victim_loaded=False));seal_hash=sha(out/'sealed_before_victim.json')
  victim=load_classifier('gaga_victim',data);vh=state_hash(victim);rows=[]
  print('STAGE sealed_VALID_physical_scoring',flush=True)
  with (out/'evaluation.jsonl').open('x') as stream:
   for i,entry in enumerate(plans):
    seed,group=entry['seed'],entry['group'];v=card(seed,'valid',group);base=v['case'];original=make_graph(data,base,base['native_adjacency']);degree=original.in_degrees()[base['source_ids']]
    for arm in p['arms']:
     choice=entry['selected'][arm];case=base if choice==0 else replace(base,v['donor'],v['bank'][choice-1],data.features);g=make_graph(data,case,case['native_adjacency']);assert torch.equal(g.in_degrees()[case['source_ids']],degree);assert torch.equal(case['x_aux'],base['x_aux'])
     with torch.no_grad():seq=canonical(data,view,g,case);error=float((seq-v['sequences'][choice]).abs().max());assert error<2e-5,error;z=victim(seq)
     counts['official_GAGA_VALID']+=1;assert counts['official_GAGA_VALID']<=960
     row=dict(seed=seed,group=group,arm=arm,node_ids=case['ids'].tolist(),choice=choice,portal_id=None if choice==0 else v['bank'][choice-1],candidate_ids=case['candidate_ids'].tolist(),gaga_logits=z.tolist(),teacher_logits=v['teacher_logits'][choice].tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_wrong=int((v['teacher_logits'][choice].argmax(1)==0).sum()),actual_nodes=len(case['source_ids']),actual_edges=int(case['native_adjacency'].sum()),owned_mask_sha256=tensor_hash(case['native_adjacency']),feature_sha256=tensor_hash(case['x_aux']),canonical_repeat_error=error)
     rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
  assert_frozen(teacher,th);assert_frozen(victim,vh);assert sha(out/'sealed_before_victim.json')==seal_hash
  metrics=[]
  for seed in p['seeds']:
   for arm in p['arms']:
    rr=[r for r in rows if r['seed']==seed and r['arm']==arm];n=sum(len(r['node_ids']) for r in rr);assert len(rr)==64 and n==571
    metrics.append(dict(seed=seed,arm=arm,MR=100*sum(r['wrong'] for r in rr)/n,teacher_MR=100*sum(r['teacher_wrong'] for r in rr)/n,active_cases=sum(r['choice']>0 for r in rr)))
  means={arm:statistics.mean(r['MR'] for r in metrics if r['arm']==arm) for arm in p['arms']};qualified={};contrasts={}
  total=lambda arm:sum(r['wrong'] for r in rows if r['arm']==arm)
  for mode in ('neural','direct'):
   qualified[mode]=total(mode)>max(total('native'),total('random_'+mode))
   for control in ('native','random_'+mode):
    ds=[next(r['MR'] for r in metrics if r['seed']==seed and r['arm']==mode)-next(r['MR'] for r in metrics if r['seed']==seed and r['arm']==control) for seed in p['seeds']];avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrasts[mode+'__'+control]=dict(deltas_pp=ds,mean_pp=avg,nominal_t95=[avg-half,avg+half])
  result.update(status='completed',metrics=metrics,means=means,owner_mean_criterion=qualified,contrasts=contrasts,limitations=p['limitations'])
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
