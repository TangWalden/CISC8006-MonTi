"""Frozen-output residual audit: saved tensors/logits only, no classifier replay."""
import pathlib,json,time,sys,argparse,hashlib,statistics
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=read(s/'downstream_protocol.json');r=s/'downstream';result=read(r/'result.json');assert result['status']=='completed'
 start=time.monotonic();cpu=time.process_time();sys.path[:0]=[p['support_source']+'/experiments/block_swap',p['support_source'],p['runtime_root']]
 from helpers import verify
 verify(pathlib.Path(p['support_source']).parent/'registration/protocol.json')
 assert sha(s/'teacher/model.pt')==p['teacher_sha256']
 import torch,numpy as np
 from experiments.downstream_cooperation.common import tensor_hash
 from experiments.local_swap.core import keyed
 torch.set_num_threads(1);checks=0
 def ok(v):
  nonlocal checks
  assert v;checks+=1
 seal=read(r/'sealed_before_victim.json');cards=read(r/'card_seal.json');index={tuple(v['key']):v for v in cards['cards']};ok(len(index)==576)
 models=seal['models'];ok(len(models)==6)
 for ref in models:ok(sha(r/ref['path'])==ref['sha256'])
 train=[json.loads(x) for x in (r/'training.jsonl').read_text().splitlines()];ok(len(train)==3072)
 for seed in p['seeds']:
  for mode in ('attention','mean'):
   for epoch in range(4):
    rows=[x for x in train if (x['seed'],x['mode'],x['epoch'])==(seed,mode,epoch+1)];generator=torch.Generator().manual_seed(keyed(seed,epoch,'residual_training_order')%(2**63-1));order=torch.randperm(128,generator=generator).tolist()
    ok([x['group'] for x in rows]==[p['train_groups'][i] for i in order]);ok(all(np.isfinite(x['attack_loss']) and 0<=x['penalty']<=.010001 for x in rows))
 evals=[json.loads(x) for x in (r/'evaluation.jsonl').read_text().splitlines()];ok(len(evals)==960)
 import dgl
 graph=dgl.load_graphs(str(pathlib.Path(p['runtime_root'])/'Datasets/GossipCop/GossipCop_0.05.bin'))[0][0]
 graph.ndata.pop('y');features=graph.ndata['x'];trainids=np.load(pathlib.Path(p['runtime_root'])/'Datasets/GossipCop/data_split/train_idx_surro_0.npy',allow_pickle=False)
 std=features[torch.as_tensor(trainids).long()].std(0,unbiased=False);low=features.min(0).values;high=features.max(0).values
 pair=[];largest_native_delta=0.
 for ref in seal['plans']:
  seed,group=ref['seed'],ref['group'];cardref=index[seed,'valid',group];path=r/cardref['path'];ok(sha(path)==cardref['sha256']);v=torch.load(path,map_location='cpu');case=v['case']
  path=r/ref['path'];ok(sha(path)==ref['sha256']);plans=torch.load(path,map_location='cpu');ok(set(plans)==set(p['arms']));ok(bool((plans['native']==0).all()))
  for mode in ('attention','mean'):
   ok(torch.equal(plans[mode].abs(),plans['random_'+mode].abs()))
   generator=torch.Generator().manual_seed(keyed(seed,group,mode,'matched_random_sign')%(2**63-1));sign=torch.randint(0,2,plans[mode].shape,generator=generator).float()*2-1
   ok(torch.equal(plans['random_'+mode],plans[mode].abs()*sign))
  for arm,d in plans.items():
   rr=[x for x in evals if (x['seed'],x['group'],x['arm'])==(seed,group,arm)];ok(len(rr)==1);row=rr[0]
   radius=torch.minimum(std,torch.minimum(case['x_aux']-low,high-case['x_aux']).clamp_min(0));ok(bool((d.abs()<=radius+1e-6).all()));ok(bool(((case['x_aux']+d>=low-1e-6)&(case['x_aux']+d<=high+1e-6)).all()));ok(row['node_ids']==case['ids'].tolist());ok(row['delta_sha256']==tensor_hash(d));ok(row['topology_sha256']==tensor_hash(case['native_adjacency']))
   ok(row['actual_nodes']==len(case['source_ids']) and row['actual_edges']==int(case['native_adjacency'].sum()))
   for field,name in [('gaga_logits','wrong'),('teacher_logits','teacher_wrong')]:ok(int((np.asarray(row[field]).argmax(1)==0).sum())==row[name])
 metrics=[];totals={}
 for seed in p['seeds']:
  for arm in p['arms']:
   rr=[x for x in evals if x['seed']==seed and x['arm']==arm];n=sum(len(x['node_ids']) for x in rr);ok(len(rr)==64 and n==571)
   wrong=sum(x['wrong'] for x in rr);totals[seed,arm]=wrong;metrics.append(dict(seed=seed,arm=arm,MR=100*wrong/n))
 for x in result['metrics']:ok(abs(x['MR']-next(v['MR'] for v in metrics if (v['seed'],v['arm'])==(x['seed'],x['arm'])))<1e-10)
 means={arm:statistics.mean(x['MR'] for x in metrics if x['arm']==arm) for arm in p['arms']};owner={}
 for mode in ('attention','mean'):
  owner[mode]=sum(totals[seed,mode] for seed in p['seeds'])>max(sum(totals[seed,c] for seed in p['seeds']) for c in ('native','random_'+mode));ok(owner[mode]==result['owner_mean_criterion'][mode])
  for control in ('native','random_'+mode):
   for seed in p['seeds']:
    gained=lost=wins=ties=losses=0;examples=[]
    for group in p['valid_groups']:
     get=lambda arm:next(x for x in evals if (x['seed'],x['group'],x['arm'])==(seed,group,arm))
     x,y=get(mode),get(control);a=np.asarray(x['gaga_logits']).argmax(1);b=np.asarray(y['gaga_logits']).argmax(1);plus=int(((a==0)&(b==1)).sum());minus=int(((a==1)&(b==0)).sum());d=plus-minus
     gained+=plus;lost+=minus;wins+=d>0;ties+=d==0;losses+=d<0
     if d:examples.append(dict(group=group,targets=len(a),gained=plus,lost=minus,net=d))
    pair.append(dict(seed=seed,mode=mode,control=control,gained=gained,lost=lost,net=gained-lost,group_wins=wins,group_ties=ties,group_losses=losses,outcome_selected_largest_changes=sorted(examples,key=lambda x:-abs(x['net']))[:5]))
 out=dict(status='passed',checks=checks,metrics=metrics,means=means,owner_mean_criterion=owner,paired=pair,graph_classifier_calls=0,pretrained_parameter_loads=0,downstream_parameter_loads=0,raw_dataset_loads=1,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,
  scope='Root-authored saved record/geometry/amplitude recount, not independent implementation/human review. Featurebox/TRAIN std independently reconstructed from frozen raw numeric features and TRAIN indices; no label values are read.')
 with (s/'saved_audit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
