"""Root-authored saved record/tensor audit, no classifier calls."""
import pathlib,argparse,json,statistics,hashlib,time
from gr_common import read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);p=read(s/'protocol.json');began=time.monotonic();cpu=time.process_time()
 import torch
 checks=0;datasets={}
 for ds,v in p['datasets'].items():
  root=s/ds;rows=list(map(json.loads,(root/'evaluation.jsonl').read_text().splitlines()));assert len(rows)==192*len(p['arms']);index={(r['seed'],r['group'],r['arm']):r for r in rows};assert len(index)==len(rows)
  seal=read(root/'sealed_before_victim.json');assert sha(root/'sealed_before_victim.json')==read(root/'selection_result.json')['seal_sha256'];metrics=[];contrasts={}
  for e in seal['entries']:
   f=root/e['path'];assert sha(f)==e['sha256'];record=torch.load(f,map_location='cpu');base=record['base']
   for name,field in [('x_aux','features_sha256'),('native_adjacency','mask_sha256')]:
    t=base[name].detach().cpu().contiguous();header=json.dumps([str(t.dtype),list(t.shape)]).encode('ascii');assert hashlib.sha256(header+b'\0'+t.numpy().tobytes()).hexdigest()==record[field]
   checks+=3
   for arm in p['arms']:
    row=index[e['seed'],e['group'],arm];z=torch.tensor(row['logits']);assert int((z.argmax(1)==0).sum())==row['wrong'];assert row['node_ids']==base['ids'].tolist();assert row['actual_edges']==int(base['native_adjacency'].sum()) and row['actual_nodes']==len(base['source_ids']);assert row['features_sha256']==record['features_sha256'] and row['mask_sha256']==record['mask_sha256'];checks+=5
   for arm in ['softplus','boundary','uniform']:
    a=record['plans'][arm];b=record['plans']['random_'+arm];assert set(a)==set(b) and len(a)<=2 and len(set(a.values()))==len(a) and len(set(b.values()))==len(b)
    assert record['doses'][arm]==record['doses']['random_'+arm];assert all(float(record['errors'][name])<=p['settings']['sequence_tolerance'] for name in [arm,'random_'+arm]);checks+=3
  for seed in v['seeds']:
   for arm in p['arms']:
    rr=[r for r in rows if (r['seed'],r['arm'])==(seed,arm)];n=sum(len(r['node_ids']) for r in rr);assert len(rr)==64 and n==v['occurrences'];assert len(set(i for r in rr for i in r['node_ids']))==v['unique_targets'];checks+=2
    metrics.append(dict(seed=seed,arm=arm,wrong=sum(r['wrong'] for r in rr),occurrences=n,MR=100*sum(r['wrong'] for r in rr)/n,exchanged_edges=sum(r['exchanged_edges'] for r in rr)))
  means={arm:statistics.mean(r['MR'] for r in metrics if r['arm']==arm) for arm in p['arms']}
  for arm in ['softplus','boundary','uniform']:
   for control in ['native','single_neural','rewriter','random_'+arm]+(['uniform'] if arm!='uniform' else []):
    gained=lost=0;deltas=[]
    for seed in v['seeds']:
     delta=0
     for group in v['valid_groups']:
      a=index[seed,group,arm];b=index[seed,group,control];assert a['node_ids']==b['node_ids'];aa=torch.tensor(a['logits']).argmax(1)==0;bb=torch.tensor(b['logits']).argmax(1)==0;gained+=int((aa&~bb).sum());lost+=int((bb&~aa).sum());delta+=a['wrong']-b['wrong'];checks+=1
     deltas.append(100*delta/v['occurrences'])
    half=4.302652729911275*statistics.stdev(deltas)/(3**.5);avg=statistics.mean(deltas);contrasts[arm+'__'+control]=dict(mean_pp=avg,deltas_pp=deltas,gained_occurrences=gained,lost_occurrences=lost,nominal_t95=[avg-half,avg+half])
  primary=all(means['boundary']>means[arm] for arm in ['native','single_neural','rewriter','random_boundary'])
  datasets[ds]=dict(metrics=metrics,means=means,contrasts=contrasts,primary_pass=primary,beats_uniform=means['boundary']>means['uniform'],selections=read(root/'selection_result.json'),evaluation=read(root/'evaluation_result.json'),preflight=read(root/'preflight_result.json'))
 write(s/'summary.json',dict(status='completed',datasets=datasets,both_datasets_pass=all(v['primary_pass'] for v in datasets.values()),root_authored_audit=dict(checks=checks,model_forwards=0,tensor_loads=True,independent_or_human_review=False,wall_seconds=time.monotonic()-began,process_cpu_seconds=time.process_time()-cpu),protocol_sha256=sha(s/'protocol.json'),limitations=p['limitations']));print(dict(status='passed',checks=checks))
if __name__=='__main__':main()
