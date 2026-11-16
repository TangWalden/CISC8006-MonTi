"""Full saved-only tensor, paired prediction and denominator audit."""
import argparse,pathlib,json,statistics,time,collections,hashlib
from full_common import protocol,read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p=protocol(ap.parse_args().stage);start=time.monotonic();cpu=time.process_time()
 import torch
 checks=0;datasets={};sealed=read(s/'all_actions_sealed.json')
 def th(t):
  t=t.detach().cpu().contiguous();return hashlib.sha256(json.dumps([str(t.dtype),list(t.shape)]).encode('ascii')+b'\0'+t.numpy().tobytes()).hexdigest()
 for ds,v in p['datasets'].items():
  root=s/ds;assert read(root/'evaluation_result.json')['status']=='completed';assert sha(root/'sealed_before_victim.json')==sealed['datasets'][ds]
  rows=[json.loads(line) for line in (root/'evaluation.jsonl').read_text().splitlines()];index={(r['seed'],r['group'],r['arm']):r for r in rows};expected={(seed,g,arm) for seed in v['seeds'] for g in v['test_groups'] for arm in p['arms']};assert len(index)==len(rows)==len(expected) and set(index)==expected
  classes={a:collections.Counter() for a in p['arms'][1:]};totals=collections.Counter();entries=read(root/'sealed_before_victim.json')['entries']
  for e in entries:
   f=root/e['path'];assert sha(f)==e['sha256'];rec=torch.load(f,map_location='cpu');base=rec['base'];assert base['ids'].tolist()==v['test_members'][str(e['group'])];assert th(base['x_aux'])==rec['features_sha256'] and th(base['native_adjacency'])==rec['mask_sha256'];checks+=4
   for arm in p['arms']:
    r=index[e['seed'],e['group'],arm];z=torch.tensor(r['logits']);assert z.shape==(len(base['ids']),2) and torch.isfinite(z).all();assert int((z.argmax(1)==0).sum())==r['wrong'];assert r['node_ids']==base['ids'].tolist();assert r['actual_edges']==int(base['native_adjacency'].sum()) and r['actual_nodes']==len(base['source_ids']);assert r['features_sha256']==rec['features_sha256'] and r['mask_sha256']==rec['mask_sha256'];checks+=5
   for arm,pl in rec['plans'].items():
    assert len(pl)<=2 and len(set(pl.values()))==len(pl);assert set(pl)<=set(rec['donors']);assert rec['errors'][arm]<=p['settings']['sequence_tolerance'];assert rec['doses'][arm]==sum(int(base['native_adjacency'][:,len(base['ids'])+j].sum()) for j in pl);classes[arm].update(rec['channels'][arm].values());checks+=4
   for arm in ['random_channel','random_joint']:
    assert set(rec['plans'][arm])==set(rec['plans']['channel']) and rec['doses'][arm]==rec['doses']['channel'];checks+=2
   assert rec['channels']['channel']==rec['channels']['random_channel'];checks+=1;totals.update(rec['selection_counters'])
  metrics=[]
  for seed in v['seeds']:
   for arm in p['arms']:
    rr=[index[seed,g,arm] for g in v['test_groups']];n=sum(len(r['node_ids']) for r in rr);assert n==v['occurrences'] and len(set(n for r in rr for n in r['node_ids']))==v['unique_targets'];checks+=2
    metrics.append(dict(seed=seed,arm=arm,wrong=sum(r['wrong'] for r in rr),occurrences=n,MR=100*sum(r['wrong'] for r in rr)/n,group_macro_MR=statistics.mean(100*r['wrong']/len(r['node_ids']) for r in rr),exchanged_edges=sum(r['exchanged_edges'] for r in rr),actual_nodes=sum(r['actual_nodes'] for r in rr),actual_edges=sum(r['actual_edges'] for r in rr)))
  means={a:statistics.mean(r['MR'] for r in metrics if r['arm']==a) for a in p['arms']};contrasts={}
  for control in ['native','random_channel','random_joint']:
   gained=lost=0;dd=[];cases=collections.Counter()
   for seed in v['seeds']:
    delta=0
    for group in v['test_groups']:
     a=index[seed,group,'channel'];b=index[seed,group,control];aa=torch.tensor(a['logits']).argmax(1)==0;bb=torch.tensor(b['logits']).argmax(1)==0;gained+=int((aa&~bb).sum());lost+=int((bb&~aa).sum());diff=a['wrong']-b['wrong'];delta+=diff;cases['win' if diff>0 else 'loss' if diff<0 else 'tie']+=1;checks+=1
    dd.append(100*delta/v['occurrences'])
   mean=statistics.mean(dd);half=4.302652729911275*statistics.stdev(dd)/(3**.5);contrasts[control]=dict(mean_pp=mean,deltas_pp=dd,all_repeats_positive=all(x>0 for x in dd),gained_occurrences=gained,lost_occurrences=lost,group_seed_cases=dict(cases),nominal_t95_conditional=[mean-half,mean+half])
  for j in range(p['selection_shards']):
   r=read(root/('selection_'+str(j)+'_result.json'));assert r['status']=='completed' and r['counters']['official_GAGA_TEST']==0 and r['counters']['updates']==0
  evaluation=read(root/'evaluation_result.json');assert evaluation['counters']['official_GAGA_TEST']==len(expected)
  datasets[ds]=dict(means=means,metrics=metrics,contrasts=contrasts,selected_class_counts={a:dict(vv) for a,vv in classes.items()},primary_pass=all(means['channel']>means[a] for a in ['native','random_channel','random_joint']),selection_counters=dict(totals),evaluation=evaluation,preflight=read(root/'preflight_result.json'))
 write(s/'summary.json',dict(status='completed',datasets=datasets,both_datasets_pass=all(v['primary_pass'] for v in datasets.values()),protocol_sha256=sha(s/'protocol.json'),root_authored_audit=dict(checks=checks,model_forwards=0,tensor_loads=True,independent_or_human_review=False,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu),limitations=p['limitations']));print(dict(status='passed',checks=checks))
if __name__=='__main__':main()
