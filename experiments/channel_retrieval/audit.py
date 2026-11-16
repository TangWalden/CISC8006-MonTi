"""Root-authored saved tensor/JSON checks; no model execution."""
import argparse,pathlib,json,hashlib,statistics,time,collections
from cr_common import read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);p=read(s/'protocol.json');began=time.monotonic();cpu=time.process_time()
 import torch
 checks=0;datasets={}
 for ds,v in p['datasets'].items():
  root=s/ds;rows=list(map(json.loads,(root/'evaluation.jsonl').read_text().splitlines()));index={(r['seed'],r['group'],r['arm']):r for r in rows};assert len(rows)==len(index)==192*len(p['arms']);assert sha(root/'sealed_before_victim.json')==read(root/'selection_result.json')['seal_sha256'];channels={a:collections.Counter() for a in ['channel','random_channel','random_joint','uniform','random_uniform']}
  for e in read(root/'sealed_before_victim.json')['entries']:
   f=root/e['path'];assert sha(f)==e['sha256'];rec=torch.load(f,map_location='cpu');base=rec['base'];checks+=1
   for name,field in [('x_aux','features_sha256'),('native_adjacency','mask_sha256')]:
    t=base[name].detach().cpu().contiguous();header=json.dumps([str(t.dtype),list(t.shape)]).encode('ascii');assert hashlib.sha256(header+b'\0'+t.numpy().tobytes()).hexdigest()==rec[field];checks+=1
   for arm in p['arms']:
    r=index[e['seed'],e['group'],arm];z=torch.tensor(r['logits']);assert int((z.argmax(1)==0).sum())==r['wrong'];assert r['node_ids']==base['ids'].tolist();assert r['actual_edges']==int(base['native_adjacency'].sum()) and r['actual_nodes']==len(base['source_ids']);assert r['features_sha256']==rec['features_sha256'] and r['mask_sha256']==rec['mask_sha256'];checks+=5
   for arm in channels:
    pl=rec['plans'][arm];assert len(pl)<=2 and len(set(pl.values()))==len(pl);assert rec['errors'][arm]<=p['settings']['sequence_tolerance'];channels[arm].update(rec['channels'][arm].values());checks+=2
   for a,b in [('channel','random_channel'),('channel','random_joint'),('uniform','random_uniform')]:
    assert set(rec['plans'][a])==set(rec['plans'][b]) and rec['doses'][a]==rec['doses'][b]
    if b!='random_joint':assert rec['channels'][a]==rec['channels'][b]
    checks+=2
  metrics=[]
  for seed in v['seeds']:
   for arm in p['arms']:
    rr=[r for r in rows if (r['seed'],r['arm'])==(seed,arm)];n=sum(len(r['node_ids']) for r in rr);assert {r['group'] for r in rr}==set(v['valid_groups']);assert n==v['occurrences'] and len(set(i for r in rr for i in r['node_ids']))==v['unique_targets'];checks+=2
    metrics.append(dict(seed=seed,arm=arm,wrong=sum(r['wrong'] for r in rr),occurrences=n,MR=100*sum(r['wrong'] for r in rr)/n,exchanged_edges=sum(r['exchanged_edges'] for r in rr)))
  means={a:statistics.mean(r['MR'] for r in metrics if r['arm']==a) for a in p['arms']};contrasts={}
  for arm in ['channel','uniform']:
   controls=['native','single_neural','rewriter','benign_retrieval','random_'+arm]+(['random_joint','uniform'] if arm=='channel' else [])
   for control in controls:
    gained=lost=0;dd=[]
    for seed in v['seeds']:
     delta=0
     for group in v['valid_groups']:
      a=index[seed,group,arm];b=index[seed,group,control];assert a['node_ids']==b['node_ids'];aa=torch.tensor(a['logits']).argmax(1)==0;bb=torch.tensor(b['logits']).argmax(1)==0;gained+=int((aa&~bb).sum());lost+=int((bb&~aa).sum());delta+=a['wrong']-b['wrong'];checks+=1
     dd.append(100*delta/v['occurrences'])
    avg=statistics.mean(dd);half=4.302652729911275*statistics.stdev(dd)/(3**.5);contrasts[arm+'__'+control]=dict(mean_pp=avg,deltas_pp=dd,gained_occurrences=gained,lost_occurrences=lost,nominal_t95=[avg-half,avg+half])
  primary=all(means['channel']>means[a] for a in ['native','single_neural','rewriter','benign_retrieval','random_channel','random_joint'])
  datasets[ds]=dict(means=means,metrics=metrics,contrasts=contrasts,selected_class_counts={a:dict(vv) for a,vv in channels.items()},primary_pass=primary,beats_uniform=means['channel']>means['uniform'],preflight=read(root/'preflight_result.json'),selection=read(root/'selection_result.json'),evaluation=read(root/'evaluation_result.json'))
 write(s/'summary.json',dict(status='completed',datasets=datasets,both_datasets_pass=all(v['primary_pass'] for v in datasets.values()),root_authored_audit=dict(checks=checks,model_forwards=0,tensor_loads=True,independent_or_human_review=False,wall_seconds=time.monotonic()-began,process_cpu_seconds=time.process_time()-cpu),protocol_sha256=sha(s/'protocol.json'),limitations=p['limitations']));print(dict(status='passed',checks=checks))
if __name__=='__main__':main()
