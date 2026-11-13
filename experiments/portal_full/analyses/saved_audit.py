"""Full saved input/action/budget/logit recount. No graph-classifier replay."""
import pathlib,argparse,json,sys,time,statistics,random,hashlib
def read(p):return json.loads(pathlib.Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage)
 assert read(s/'guardian_receipt.json')['status']=='completed';sys.path.insert(0,str(s/'source'))
 from common import setup,sha,choose
 s,p,q=setup(s);r=s/'main';result=read(r/'result.json');assert result['status']=='completed';assert result['protocol_sha256']==sha(s/'protocol.json');start=time.monotonic();cpu=time.process_time()
 import torch,numpy as np,dgl,scipy.sparse as sp
 from types import SimpleNamespace
 from geometry_hard import proposals,key
 from model import PortalRanker
 from experiments.downstream_cooperation.common import tensor_hash,state_hash
 torch.set_num_threads(4);ncheck=0
 def ok(value):
  nonlocal ncheck
  assert value;ncheck+=1
 rt=pathlib.Path(p['runtime_root']);graph=dgl.load_graphs(str(rt/'Datasets/GossipCop/GossipCop_0.05.bin'))[0][0];trainids=np.load(rt/'Datasets/GossipCop/data_split/train_idx_surro_0.npy',allow_pickle=False);labels=graph.ndata.pop('y');benign=trainids[labels[torch.tensor(trainids).long()].numpy()==0];del labels
 adj=sp.csr_matrix(graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros();adj2=sp.load_npz(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_adj2_0.npz');degree=torch.from_numpy(np.load(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_degree_0.npy',allow_pickle=False));view=SimpleNamespace(adj=adj,adj2=adj2,d=degree)
 stats=graph.ndata['x'][torch.tensor(trainids).long()];mean=stats.mean(0);std=stats.std(0,unbiased=False).clamp_min(1e-6)
 seal=read(r/'sealed_before_victim.json');entries=seal['entries'];ok(seal['official_victim_loaded'] is False and len(entries)==2727 and len({(x['seed'],x['group']) for x in entries})==2727)
 rows=[json.loads(x) for x in (r/'evaluation.jsonl').read_text().splitlines()];ok(len(rows)==13635);ri={(x['seed'],x['group'],x['arm']):x for x in rows};ok(len(ri)==13635)
 nets={};parent=pathlib.Path(p['parent']);teacher_hard=0;physical=0;policy_calls=0;actual_budgets={seed:dict(nodes=0,nonself_undirected_edges=0) for seed in p['seeds']}
 for ref in p['models']:
  f=parent/'main'/ref['path'];ok(sha(f)==ref['sha256']);net=PortalRanker(mean.numel());net.load_state_dict(torch.load(f,map_location='cpu'));net.eval();nets[ref['seed']]=net;ok(state_hash(net)==seal['downstream_states'][str(ref['seed'])])
 for e in entries:
  seed,group=e['seed'],e['group'];f=r/e['path'];ok(sha(f)==e['sha256']);v=torch.load(f,map_location='cpu');case=v['case'];ids=case['ids'];cids=case['candidate_ids'];b=len(case['source_ids']);t=len(ids);c=len(cids);mask=case['native_adjacency']
  ok(case['seed']==seed and case['group']==group and ids.tolist()==p['test_members'][str(group)]);ok(case['source_ids']==list(range(graph.num_nodes(),graph.num_nodes()+b)));ok(len(set(cids.tolist()))==c and not set(cids.tolist())&set(ids.tolist()));ok(bool(((cids>=0)&(cids<graph.num_nodes())).all()))
  ok(mask.shape==(b,t+c+b) and bool(((mask==0)|(mask==1)).all()));ok(int(mask.sum())==case['quota']['edge_budget']==e['actual_edges']);ok(bool((mask[:,:t].sum(1)>=1).all()));ok(bool((mask[:,t+c:].diagonal()==0).all()))
  expected_degree=(mask.sum(1)+mask[:,t+c:].sum(0)+1).long();ok(torch.equal(v['physical_injected_degrees'].long(),expected_degree));ok(b==e['actual_nodes']);actual_budgets[seed]['nodes']+=b;actual_budgets[seed]['nonself_undirected_edges']+=e['actual_edges']
  ok(torch.equal(case['x_targets'],graph.ndata['x'][ids]) and torch.equal(case['x_candidates'],graph.ndata['x'][cids]));ok(tensor_hash(case['x_aux'])==e['feature_sha256'] and tensor_hash(mask)==e['mask_sha256'])
  donor,bank=proposals(case,view,benign);ok(donor==v['donor'] and bank==v['bank']);physical+=len(bank)+1;teacher_hard+=len(bank)+1
  if bank:
   cols=np.asarray(bank);ok(set(bank)<=set(benign.tolist()));ok(adj[ids.numpy()][:,cols].nnz==0 and adj2[ids.numpy()][:,cols].nnz==0 and (adj[ids.numpy()]@adj[:,cols]).nnz==0)
  aux=case['x_aux'][mask[:,t+donor]>0].mean(0) if donor is not None else case['x_aux'].mean(0);seq=v['sequences'];ok(seq.shape==(len(bank)+1,t,7,mean.numel()) and bool(torch.isfinite(seq).all()))
  query=torch.cat(((case['x_targets'].mean(0)-mean)/std,(aux-mean)/std,(seq[0,:,4].mean(0)-mean)/std,torch.tensor([np.log1p(t),np.log1p(b),np.log1p(float(degree[ids].reshape(t,7)[:,4].mean()))],dtype=torch.float32)));keys=torch.zeros(len(bank)+1,mean.numel())
  if bank:keys[1:]=(graph.ndata['x'][torch.tensor(bank)]-mean)/std
  ok(torch.allclose(query,v['query'],rtol=1e-5,atol=1e-5));ok(torch.allclose(keys,v['keys'],rtol=1e-5,atol=1e-5));ok(torch.allclose((seq[:,:,4].mean(1)-seq[0,:,4].mean(0))/std,v['changes'],rtol=1e-5,atol=1e-5))
  logits=v['teacher_logits'];losses=torch.nn.functional.softplus(logits[:,:,1]-logits[:,:,0]).mean(1);ok(torch.allclose(losses,v['losses'],rtol=1e-6,atol=1e-6))
  actions,scores=choose(v,nets[seed]);policy_calls+=1;ok(actions==e['selected']);ok(np.allclose(scores.detach().numpy(),e['neural_scores'],rtol=1e-4,atol=1e-5))
  for arm,choice in actions.items():
   row=ri[seed,group,arm];ok(row['node_ids']==ids.tolist() and row['choice']==choice);ok(row['portal_id']==(None if not choice else bank[choice-1]));ok(row['feature_sha256']==e['feature_sha256'] and row['mask_sha256']==e['mask_sha256']);ok(row['actual_nodes']==b and row['actual_edges']==e['actual_edges']);z=np.asarray(row['gaga_logits']);tz=np.asarray(row['teacher_logits']);ok(z.shape==(t,2) and np.isfinite(z).all());ok(int((z.argmax(1)==0).sum())==row['wrong']);ok(np.array_equal(tz,logits[choice].numpy()) and int((tz.argmax(1)==0).sum())==row['teacher_wrong'])
 counters=result['counters'];ok(counters['generations']==2727 and counters['policy_forwards']==policy_calls==2727 and counters['updates']==0);ok(counters['own_teacher_batched_forwards']==2727 and counters['own_teacher_hard_contexts']==teacher_hard);ok(counters['physical_canonical_builds']==physical<=24543 and counters['official_GAGA_TEST']==13635)
 totals={};means={};metrics=[]
 for seed in p['seeds']:
  for arm in p['arms']:
   rr=[x for x in rows if x['seed']==seed and x['arm']==arm];den=sum(len(x['node_ids']) for x in rr);ok(len(rr)==909 and den==10515 and len({n for x in rr for n in x['node_ids']})==1221);wrong=sum(x['wrong'] for x in rr);totals[seed,arm]=wrong;value=100*wrong/den;metrics.append(dict(seed=seed,arm=arm,MR=value,wrong=wrong,occurrences=den));mr=next(x for x in result['metrics'] if (x['seed'],x['arm'])==(seed,arm));ok(abs(value-mr['MR'])<1e-10 and wrong==mr['wrong'])
 for arm in p['arms']:means[arm]=statistics.mean(x['MR'] for x in metrics if x['arm']==arm);ok(abs(means[arm]-result['means'][arm])<1e-10)
 owner={};paired=[]
 for mode in ['neural','direct']:
  owner[mode]=sum(totals[seed,mode] for seed in p['seeds'])>max(sum(totals[seed,arm] for seed in p['seeds']) for arm in ['native','random_'+mode]);ok(owner[mode]==result['owner_mean_criterion'][mode])
  for control in ['native','random_'+mode]:
   ds=[100*(totals[seed,mode]-totals[seed,control])/10515 for seed in p['seeds']];avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrast=result['contrasts'][mode+'__'+control];ok(np.allclose(ds,contrast['deltas_pp'],rtol=0,atol=1e-10));ok(abs(avg-contrast['mean_pp'])<1e-10 and np.allclose([avg-half,avg+half],contrast['nominal_t95'],rtol=0,atol=1e-10))
   for seed in p['seeds']:
    gain=lost=wins=ties=losscases=0;examples=[]
    for group in p['test_groups']:
     a=np.asarray(ri[seed,group,mode]['gaga_logits']).argmax(1);b=np.asarray(ri[seed,group,control]['gaga_logits']).argmax(1);g=int(((a==0)&(b==1)).sum());l=int(((a==1)&(b==0)).sum());d=g-l;gain+=g;lost+=l;wins+=d>0;ties+=d==0;losscases+=d<0
     if d:examples.append(dict(group=group,targets=len(a),gained=g,lost=l,net=d))
    paired.append(dict(seed=seed,mode=mode,control=control,gained=gain,lost=lost,net=gain-lost,group_wins=wins,group_ties=ties,group_losses=losscases,outcome_selected_largest_changes=sorted(examples,key=lambda x:-abs(x['net']))[:5]))
 report=dict(status='passed',checks=ncheck,means=means,metrics=metrics,owner_mean_criterion=owner,paired=paired,actual_budgets_sum_separate_scenarios=actual_budgets,raw_graph_loads=1,downstream_parameter_loads=3,policy_forwards=policy_calls,pretrained_parameter_loads=0,graph_classifier_calls=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,scope='Root-authored saved-byte/action/eligibility/degree/logit recount; not independent implementation/human review. No canonical or pretrained-classifier replay. Raw graph and TRAIN label values only; no TEST truth values read.')
 with (s/'saved_audit.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report),flush=True)
if __name__=='__main__':main()
