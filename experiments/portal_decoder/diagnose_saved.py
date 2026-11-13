"""Post-hoc transfer diagnosis; TRAIN-only remote-context feasibility, no models."""
import pathlib,json,time,hashlib,statistics
start=time.monotonic();cpu=time.process_time();rt=pathlib.Path('/home/tjy/Project/MonTi-course');parent=rt/'runs/residual_attention_20261113_main1';out=rt/'runs/portal_decoder_20261113_diagnosis';out.mkdir(exist_ok=False)
rows=[json.loads(x) for x in (parent/'downstream/evaluation.jsonl').read_text().splitlines()];train=[json.loads(x) for x in (parent/'downstream/training.jsonl').read_text().splitlines()]
index={(x['seed'],x['group'],x['arm']):x for x in rows};report=[]
for mode in ('attention','mean'):
 for seed in (0,1,2):
  own=[];official=[];native_margin=[];delta=[]
  for group in sorted({x['group'] for x in rows}):
   a=index[seed,group,mode];b=index[seed,group,'native'];delta.append(a['delta_L2'])
   for la,lb,ta,tb in zip(a['gaga_logits'],b['gaga_logits'],a['teacher_logits'],b['teacher_logits']):
    official.append((la[1]-la[0])-(lb[1]-lb[0]));own.append((ta[1]-ta[0])-(tb[1]-tb[0]));native_margin.append(abs(lb[1]-lb[0]))
  epochs=[]
  for ep in (1,2,3,4):
   rr=[x for x in train if (x['seed'],x['mode'],x['epoch'])==(seed,mode,ep)];epochs.append(dict(epoch=ep,mean_attack_loss=statistics.mean(x['attack_loss'] for x in rr),teacher_TRAIN_wrong=sum(x['teacher_wrong'] for x in rr),occurrences=sum(x['targets'] for x in rr)))
  report.append(dict(seed=seed,mode=mode,epochs=epochs,median_delta_L2=statistics.median(delta),mean_teacher_margin_change=statistics.mean(own),mean_victim_margin_change=statistics.mean(official),median_abs_native_victim_margin=statistics.median(native_margin),teacher_attack_direction_occurrences=sum(x< -1e-6 for x in own),victim_attack_direction_occurrences=sum(x< -1e-6 for x in official),opposing_direction_occurrences=sum(x< -1e-6 and y>1e-6 for x,y in zip(own,official))))
import torch,numpy as np,dgl,scipy.sparse as sp
torch.set_num_threads(1)
graph=dgl.load_graphs(str(rt/'Datasets/GossipCop/GossipCop_0.05.bin'))[0][0];ids=np.load(rt/'Datasets/GossipCop/data_split/train_idx_surro_0.npy',allow_pickle=False);labels=graph.ndata.pop('y');benign=ids[(labels[torch.as_tensor(ids).long()].numpy()==0)];del labels
adj=sp.csr_matrix(graph.adj_external(scipy_fmt='coo')).T.tocsr();adj.setdiag(0);adj.eliminate_zeros();two=sp.load_npz(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_adj2_0.npz').tocsr();degrees=np.load(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_degree_0.npy',allow_pickle=False)
up= json.loads((rt/'runs/local_swap_20261113_attempt1/main/upstream_seal.json').read_text());feasible=[]
for ref in up['cases']:
 if ref['phase']!='train':continue
 f=rt/'runs/local_swap_20261113_attempt1/main'/ref['path'];assert hashlib.sha256(f.read_bytes()).hexdigest()==ref['sha256'];v=torch.load(f,map_location='cpu');target=v['ids'].numpy();c=len(v['candidate_ids']);t=len(target)
 excluded=np.unique(np.concatenate((target,v['candidate_ids'].numpy(),adj[target].indices,two[target].indices)));pool=np.setdiff1d(benign,excluded,assume_unique=False);activecols=int((v['native_adjacency'][:,t:t+c].sum(0)>0).sum())
 d=degrees[target].reshape(t,7);feasible.append(dict(seed=ref['seed'],group=ref['group'],targets=t,eligible_remote_TRAIN_benign=len(pool),used_candidate_columns=activecols,original_known_benign_twohop_empty=int((d[:,4]==0).sum()),original_known_benign_twohop_atmost3=int((d[:,4]<=3).sum())))
summary=dict(status='completed',scope='Transfer arithmetic is post hoc. Retrieval feasibility uses all384TRAIN cases only; no VALID portal selection/outcome measurements.',transfer=report,TRAIN_feasibility=dict(cases=len(feasible),with_remote_benign=sum(x['eligible_remote_TRAIN_benign']>0 for x in feasible),with_replaceable_column=sum(x['eligible_remote_TRAIN_benign']>0 and x['used_candidate_columns']>0 for x in feasible),total_target_occurrences=sum(x['targets'] for x in feasible),known_benign_twohop_empty=sum(x['original_known_benign_twohop_empty'] for x in feasible),known_benign_twohop_atmost3=sum(x['original_known_benign_twohop_atmost3'] for x in feasible)),graph_classifier_calls=0,parameter_loads=0,raw_graph_loads=1,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
for n,v in [('summary.json',summary),('train_feasibility.json',feasible)]:
 with (out/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
print(json.dumps(summary))
