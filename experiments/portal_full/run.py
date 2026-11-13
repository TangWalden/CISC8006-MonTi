"""Complete original TEST groups, frozen portal models and sealed choices."""
import argparse,pathlib,signal,time,statistics,os,shutil
from common import setup,read,sha,write,load_teacher,card,choose
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);pre=read(s/'preflight_result.json');assert pre['status']=='passed' and pre['protocol_sha256']==sha(s/'protocol.json')
 import torch,numpy as np,json
 from experiments.downstream_cooperation.common import check_device,state_hash,assert_frozen,tensor_hash
 from experiments.local_swap.loader import load_engine
 from experiments.local_swap.helpers import capture
 from course.dual_surrogate.runtime import load_original_view,load_classifier
 from model import PortalRanker
 check_device();out=s/'main';out.mkdir(exist_ok=False);(out/'cards').mkdir();start=time.monotonic();cpu=time.process_time()
 signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Frozen complete-cohort deadline')));signal.alarm(p['resources']['main_seconds'])
 counts=dict(generations=0,physical_canonical_builds=0,own_teacher_batched_forwards=0,own_teacher_hard_contexts=0,policy_forwards=0,official_GAGA_TEST=0,updates=0,API_calls=0)
 result=dict(status='failed',protocol_sha256=sha(s/'protocol.json'),counters=counts,GPU_hours=0);parent=pathlib.Path(p['parent']);entries=[];receipts=[];teacher=None
 try:
  base=dict(p['author_base'],test_groups=p['test_groups'],test_members=p['test_members']);nets={}
  print('STAGE complete_author_capture_and_physical_choices',flush=True)
  for seed in p['seeds']:
   engine,data,receipt,frozen=load_engine(base,seed,out,start+p['resources']['main_seconds']);receipts.append(receipt);view=load_original_view(data)
   if teacher is None:teacher=load_teacher(p,q,data);th=state_hash(teacher)
   net=PortalRanker(data.feat_dim);ref=next(x for x in p['models'] if x['seed']==seed);net.load_state_dict(torch.load(parent/'main'/ref['path'],map_location='cpu'));net.eval();nh=state_hash(net);nets[seed]=nh
   for i,group in enumerate(p['test_groups']):
    if i%32==0:assert shutil.disk_usage(s).free>=8*1024**3
    case,g=capture(engine,data,seed,group,'test',base);counts['generations']+=1;del g
    v=card(data,view,case,teacher,counts);actions,scores=choose(v,net);counts['policy_forwards']+=1
    path=out/'cards'/('s%d_g%d.pt'%(seed,group));torch.save(v,path)
    entries.append(dict(seed=seed,group=group,path=str(path.relative_to(out)),sha256=sha(path),selected=actions,neural_scores=scores.tolist(),targets=len(case['ids']),actual_nodes=len(case['source_ids']),actual_edges=int(case['native_adjacency'].sum()),feature_sha256=tensor_hash(case['x_aux']),mask_sha256=tensor_hash(case['native_adjacency'])))
    if (i+1)%32==0:print('CARDS',seed,i+1,909,flush=True)
   assert_frozen(engine.model,frozen['upstream']);assert_frozen(engine.teachers['gcn_base'],frozen['teacher']);assert state_hash(net)==nh
  assert counts['generations']==2727 and counts['own_teacher_batched_forwards']==2727 and counts['physical_canonical_builds']<=24543
  write(out/'sealed_before_victim.json',dict(entries=entries,author_receipts=receipts,models=p['models'],downstream_states=nets,official_victim_loaded=False));sealed=sha(out/'sealed_before_victim.json')
  victim=load_classifier('gaga_victim',data);vh=state_hash(victim);rows=[];print('STAGE sealed_complete_TEST_scoring',flush=True)
  with (out/'evaluation.jsonl').open('x') as stream:
   for i,e in enumerate(entries):
    path=out/e['path'];assert sha(path)==e['sha256'];v=torch.load(path,map_location='cpu');case=v['case'];t=len(case['ids'])
    for arm in p['arms']:
     choice=e['selected'][arm]
     with torch.no_grad():z=victim(v['sequences'][choice]).reshape(t,2)
     counts['official_GAGA_TEST']+=1;assert counts['official_GAGA_TEST']<=13635
     row=dict(seed=e['seed'],group=e['group'],arm=arm,node_ids=case['ids'].tolist(),choice=choice,portal_id=None if not choice else v['bank'][choice-1],gaga_logits=z.tolist(),teacher_logits=v['teacher_logits'][choice].tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_wrong=int((v['teacher_logits'][choice].argmax(1)==0).sum()),actual_nodes=e['actual_nodes'],actual_edges=e['actual_edges'],feature_sha256=e['feature_sha256'],mask_sha256=e['mask_sha256']);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    if (i+1)%64==0:print('EVALUATED',i+1,2727,flush=True)
  assert counts['official_GAGA_TEST']==13635 and sha(out/'sealed_before_victim.json')==sealed;assert_frozen(teacher,th);assert_frozen(victim,vh)
  metrics=[]
  for seed in p['seeds']:
   for arm in p['arms']:
    rr=[x for x in rows if x['seed']==seed and x['arm']==arm];n=sum(len(x['node_ids']) for x in rr);assert len(rr)==909 and n==10515
    metrics.append(dict(seed=seed,arm=arm,MR=100*sum(x['wrong'] for x in rr)/n,teacher_MR=100*sum(x['teacher_wrong'] for x in rr)/n,wrong=sum(x['wrong'] for x in rr),occurrences=n,active_cases=sum(x['choice']>0 for x in rr)))
  means={a:statistics.mean(x['MR'] for x in metrics if x['arm']==a) for a in p['arms']};contrasts={};qualified={}
  for mode in ['neural','direct']:
   total=lambda arm:sum(x['wrong'] for x in metrics if x['arm']==arm);qualified[mode]=total(mode)>max(total('native'),total('random_'+mode))
   for control in ['native','random_'+mode]:
    ds=[next(x['MR'] for x in metrics if (x['seed'],x['arm'])==(seed,mode))-next(x['MR'] for x in metrics if (x['seed'],x['arm'])==(seed,control)) for seed in p['seeds']];avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5);contrasts[mode+'__'+control]=dict(deltas_pp=ds,mean_pp=avg,nominal_t95=[avg-half,avg+half])
  result.update(status='completed',metrics=metrics,means=means,contrasts=contrasts,owner_mean_criterion=qualified,scope=p['scope'])
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
