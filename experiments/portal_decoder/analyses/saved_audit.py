"""Saved portal audit. Replays new policy only; never replays GAGA classifiers."""
import pathlib,json,time,sys,argparse,hashlib,random,statistics
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=read(s/'main_protocol.json');r=s/'main';result=read(r/'result.json');assert result['status']=='completed';assert sha(s/'main_protocol.json')==result['protocol_sha256']
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h
 start=time.monotonic();cpu=time.process_time();sys.path[:0]=[str(s/'source'),p['support_source']+'/experiments/block_swap',p['support_source'],p['runtime_root']]
 from helpers import verify
 verify(pathlib.Path(p['support_source']).parent/'registration/protocol.json');assert sha(p['teacher_checkpoint'])==p['teacher_sha256']
 import torch,numpy as np,dgl,scipy.sparse as sp
 from types import SimpleNamespace
 from geometry_hard import proposals,replace,key
 from model import PortalRanker
 from experiments.downstream_cooperation.common import tensor_hash,state_hash
 torch.set_num_threads(4);checks=0
 def ok(v):
  nonlocal checks
  assert v;checks+=1
 rt=pathlib.Path(p['runtime_root']);graph=dgl.load_graphs(str(rt/'Datasets/GossipCop/GossipCop_0.05.bin'))[0][0];trainids=np.load(rt/'Datasets/GossipCop/data_split/train_idx_surro_0.npy',allow_pickle=False);alllabels=graph.ndata.pop('y');benign=trainids[alllabels[torch.as_tensor(trainids).long()].numpy()==0];del alllabels
 adj=sp.csr_matrix(graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros();adj2=sp.load_npz(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_adj2_0.npz');degree=torch.from_numpy(np.load(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_degree_0.npy',allow_pickle=False));view=SimpleNamespace(adj=adj,adj2=adj2,d=degree)
 stats=graph.ndata['x'][torch.as_tensor(trainids).long()];mean=stats.mean(0);std=stats.std(0,unbiased=False).clamp_min(1e-6)
 seal=read(r/'sealed_before_victim.json');plans={(v['seed'],v['group']):v for v in seal['plans']};refs=read(r/'card_seal.json')['cards'];index={tuple(v['key']):v for v in refs};ok(len(index)==576 and len(plans)==192)
 baseparent=pathlib.Path(p['base_cards_parent']);ok(sha(baseparent/'card_seal.json')==p['base_card_seal_sha256']);baseindex={tuple(v['key']):v for v in read(baseparent/'card_seal.json')['cards']}
 option_graphs=0;cardloads=0
 for k,ref in index.items():
  f=r/ref['path'];ok(sha(f)==ref['sha256']);v=torch.load(f,map_location='cpu');cardloads+=1;case=v['case'];bref=baseindex[k];bf=baseparent/bref['path'];ok(sha(bf)==bref['sha256']);original=torch.load(bf,map_location='cpu')['case']
  for name in ['ids','candidate_ids','x_aux','native_adjacency','x_targets']:ok(torch.equal(case[name],original[name]))
  ok(case['source_ids']==original['source_ids']);donor,bank=proposals(case,view,benign);ok(donor==v['donor'] and bank==v['bank'])
  losses=torch.nn.functional.softplus(v['teacher_logits'][:,:,1]-v['teacher_logits'][:,:,0]).mean(1);ok(torch.allclose(losses,v['losses'],rtol=1e-6,atol=1e-6));option_graphs+=len(bank)
  if bank:
   ids=case['ids'].numpy();cols=np.array(bank);ok(set(bank)<=set(benign.tolist()));ok(not set(bank)&set(case['ids'].tolist()+case['candidate_ids'].tolist()))
   ok(adj[ids][:,cols].nnz==0 and adj2[ids][:,cols].nnz==0);ok((adj[ids]@adj[:,cols]).nnz==0)
  t=len(case['ids']);aux=case['x_aux'][case['native_adjacency'][:,t+donor]>0].mean(0) if donor is not None else case['x_aux'].mean(0)
  query=torch.cat(((case['x_targets'].mean(0)-mean)/std,(aux-mean)/std,(v['sequences'][0,:,4].mean(0)-mean)/std,torch.tensor([np.log1p(t),np.log1p(len(case['source_ids'])),np.log1p(float(degree[case['ids']].reshape(t,7)[:,4].mean()))],dtype=torch.float32)))
  ok(torch.allclose(query,v['query'],rtol=1e-5,atol=1e-5));keys=torch.zeros(len(bank)+1,mean.numel())
  if bank:keys[1:]=(graph.ndata['x'][torch.tensor(bank)]-mean)/std
  changes=(v['sequences'][:,:,4].mean(1)-v['sequences'][0,:,4].mean(0))/std
  ok(torch.allclose(v['keys'],keys,rtol=1e-5,atol=1e-5));ok(torch.allclose(v['changes'],changes,rtol=1e-5,atol=1e-5))
 ok(option_graphs==result['counters']['physical_canonical_builds'])
 train=[json.loads(x) for x in (r/'training.jsonl').read_text().splitlines()];ok(len(train)==3072);ok(sum(x['updated'] for x in train)==result['counters']['neural_updates']);ok(sum(not x['updated'] for x in train)==result['counters']['no_choice_skips'])
 nets={};policy_forwards=0
 for ref in seal['models']:
  seed=ref['seed'];f=r/ref['path'];ok(sha(f)==ref['sha256']);torch.manual_seed(2026111400+seed);net=PortalRanker(mean.numel());initial=state_hash(net);net.load_state_dict(torch.load(f,map_location='cpu'));net.eval();ok(state_hash(net)!=initial);nets[seed]=net
  for epoch in range(8):
   rr=[x for x in train if (x['seed'],x['epoch'])==(seed,epoch+1)];order=list(p['train_groups']);random.Random(key(seed,epoch,'portal_training_order')).shuffle(order);ok([x['group'] for x in rr]==order);ok(all(np.isfinite(x['loss']) and x['updated']==(x['options']>1) for x in rr))
 evals=[json.loads(x) for x in (r/'evaluation.jsonl').read_text().splitlines()];ok(len(evals)==960)
 for (seed,group),entry in plans.items():
  ref=index[seed,'valid',group];v=torch.load(r/ref['path'],map_location='cpu');case=v['case']
  with torch.no_grad():scores=nets[seed](v['query'],v['keys'],v['changes']);policy_forwards+=1
  ok(np.allclose(scores.numpy(),entry['neural_scores'],rtol=1e-4,atol=1e-5));neural=int(scores.argmax());best=int(v['losses'].argmin());direct=best if float(v['losses'][0]-v['losses'][best])>1e-4 else 0
  ri=random.Random(key(seed,group,'portal_matched_random')).randrange(1,len(v['bank'])+1) if v['bank'] else 0
  expected=dict(native=0,neural=neural,random_neural=ri if neural else 0,direct=direct,random_direct=ri if direct else 0);ok(entry['selected']==expected)
  for arm,choice in expected.items():
   rr=[e for e in evals if (e['seed'],e['group'],e['arm'])==(seed,group,arm)];ok(len(rr)==1);row=rr[0];candidate=case['candidate_ids'].tolist()
   if choice:candidate[v['donor']]=v['bank'][choice-1]
   ok(row['candidate_ids']==candidate and row['choice']==choice);ok(row['portal_id']==(None if choice==0 else v['bank'][choice-1]));ok(row['node_ids']==case['ids'].tolist());ok(row['actual_nodes']==len(case['source_ids']) and row['actual_edges']==int(case['native_adjacency'].sum()))
   ok(row['owned_mask_sha256']==tensor_hash(case['native_adjacency']) and row['feature_sha256']==tensor_hash(case['x_aux']));ok(row['canonical_repeat_error']<2e-5)
   for field,name in [('gaga_logits','wrong'),('teacher_logits','teacher_wrong')]:ok(int((np.asarray(row[field]).argmax(1)==0).sum())==row[name])
   ok(np.allclose(np.asarray(row['teacher_logits']),v['teacher_logits'][choice].numpy(),rtol=0,atol=0))
 metrics=[];totals={}
 for seed in p['seeds']:
  for arm in p['arms']:
   rr=[e for e in evals if e['seed']==seed and e['arm']==arm];n=sum(len(e['node_ids']) for e in rr);ok(len(rr)==64 and n==571);wrong=sum(e['wrong'] for e in rr);totals[seed,arm]=wrong;metrics.append(dict(seed=seed,arm=arm,MR=100*wrong/n))
 for x in result['metrics']:ok(abs(x['MR']-next(y['MR'] for y in metrics if (x['seed'],x['arm'])==(y['seed'],y['arm'])))<1e-10)
 means={arm:statistics.mean(x['MR'] for x in metrics if x['arm']==arm) for arm in p['arms']};owner={};paired=[]
 for mode in ('neural','direct'):
  owner[mode]=sum(totals[seed,mode] for seed in p['seeds'])>max(sum(totals[seed,c] for seed in p['seeds']) for c in ('native','random_'+mode));ok(owner[mode]==result['owner_mean_criterion'][mode])
  for control in ('native','random_'+mode):
   ds=[100*(totals[seed,mode]-totals[seed,control])/571 for seed in p['seeds']];avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);ok(np.allclose(ds,result['contrasts'][mode+'__'+control]['deltas_pp'],atol=1e-10,rtol=0));ok(abs(avg-result['contrasts'][mode+'__'+control]['mean_pp'])<1e-10);ok(np.allclose([avg-half,avg+half],result['contrasts'][mode+'__'+control]['nominal_t95'],atol=1e-10,rtol=0))
   for seed in p['seeds']:
    plus=minus=wins=ties=losses=0;examples=[]
    for group in p['valid_groups']:
     get=lambda arm:next(e for e in evals if (e['seed'],e['group'],e['arm'])==(seed,group,arm));a=np.asarray(get(mode)['gaga_logits']).argmax(1);b=np.asarray(get(control)['gaga_logits']).argmax(1);gain=int(((a==0)&(b==1)).sum());lost=int(((a==1)&(b==0)).sum());d=gain-lost;plus+=gain;minus+=lost;wins+=d>0;ties+=d==0;losses+=d<0
     if d:examples.append(dict(group=group,targets=len(a),gained=gain,lost=lost,net=d))
    paired.append(dict(seed=seed,mode=mode,control=control,gained=plus,lost=minus,net=plus-minus,group_wins=wins,group_ties=ties,group_losses=losses,largest_outcome_selected_changes=sorted(examples,key=lambda e:-abs(e['net']))[:5]))
 report=dict(status='passed',checks=checks,metrics=metrics,means=means,owner_mean_criterion=owner,paired=paired,raw_graph_loads=1,physical_counterfactual_cards=option_graphs,hard_action_contexts=option_graphs+576,batched_teacher_forwards=576,saved_card_loads=cardloads,downstream_parameter_loads=3,policy_forwards=policy_forwards,pretrained_parameter_loads=0,graph_classifier_forwards=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,scope='Root-authored saved policy/eligibility/graph-budget/logit recount; not independent implementation/human review. Does not rerun classifier or full canonical input construction.')
 with (s/'saved_audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':main()
