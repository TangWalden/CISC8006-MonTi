"""TRAIN parity, sealed two-channel search, then official development scoring."""
import argparse,time,signal,json,pathlib,itertools
from cr_common import read,write,sha,setup,counts
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--mode',required=True,choices=['preflight','selection','evaluation']);args=ap.parse_args()
 start=time.monotonic();cpu=time.process_time();s,p,parent,old,v,ancestor,ancestor_protocol,q,data,view,teacher,th,benign,common=setup(args.stage,args.dataset)
 import torch
 from gr_core import predict
 from cr_core import Geometry,search
 from experiments.downstream_cooperation.common import freeze,assert_frozen,tensor_hash
 out=s/args.dataset;out.mkdir(exist_ok=True);c=counts();result=dict(status='failed',mode=args.mode,dataset=args.dataset,counters=c,GPU_hours=0)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Frozen phase deadline')));signal.alarm(p['resources'][args.mode+'_seconds'])
 def enforce():
  for k in ['physical_builds','teacher_contexts','backward_calls','official_GAGA_VALID']:assert c[k]<=p['caps'][k],(k,c[k])
 try:
  if args.mode=='preflight':
   index=common.index_cards(ancestor_protocol['parent']);checks=[]
   for seed in v['seeds'][:2]:
    for group in v['preflight_groups']:
     card=common.oldcard(ancestor_protocol['parent'],index,seed,'train',group);geo=Geometry(data,view,card['case'],card['sequences'][0],benign,c);ds=geo.donors[:2]
     assert len(ds)==2 and all(len(pool)>=4 for pool in geo.pools.values())
     plans=[{}]+[{ds[0]:geo.pools[label][-1]} for label in [0,1]]
     plans += [{ds[0]:geo.pools[a][-1],ds[1]:geo.pools[b][-2]} for a,b in itertools.product([0,1],repeat=2)]
     hard=[];fast=[];errors=[]
     for plan in plans:
      h,err=geo.verify(plan,p['settings']['sequence_tolerance']);hard.append(h);fast.append(geo.sequence(plan));errors.append(err)
     zh,_,_=predict(teacher,hard,c,'boundary');zf,_,_=predict(teacher,fast,c,'boundary');le=float((zh-zf).abs().max());assert le<=p['settings']['logit_tolerance'];assert torch.equal(zh.argmax(-1),zf.argmax(-1))
     gradient=[]
     for label in [0,1]:
      sample=fast[1+label];_,_,g=predict(teacher,[sample],c,'boundary',True);direction=torch.zeros_like(sample);direction[:,4+label]=g[0,:,4+label];direction/=direction.norm().clamp_min(1e-12);eps=.001
      _,lv,_=predict(teacher,[sample+eps*direction,sample-eps*direction],c,'boundary');num=float((lv[0]-lv[1])/(2*eps));ana=float((g[0]*direction).sum());assert abs(num-ana)<=max(1e-4,.05*abs(ana)),(num,ana)
      gradient.append(dict(channel=label,norm=float(g[0,:,4+label].norm()),directional_numeric=num,directional_autograd=ana))
     checks.append(dict(seed=seed,group=group,pool_sizes={label:len(pool) for label,pool in geo.pools.items()},sequence_error=max(errors),logit_error=le,channel_gradients=gradient));enforce();print('PREFLIGHT',len(checks),4,flush=True)
   result.update(status='completed',checks=checks)
  elif args.mode=='selection':
   assert all(read(s/d/'preflight_result.json')['status']=='completed' for d in p['datasets']);(out/'plans').mkdir(exist_ok=False);entries=[]
   for e in read(parent/args.dataset/'sealed_before_victim.json')['entries']:
    f=parent/args.dataset/e['path'];assert sha(f)==e['sha256'];prev=torch.load(f,map_location='cpu');base=prev['base'];geo=Geometry(data,view,base,prev['sequences']['native'],benign,c)
    plans={};seqs={};traces={};errors={};cost={}
    for mode in ['channel','uniform']:
     before=dict(c);plan,trace=search(geo,teacher,mode,c,p['settings']['shortlist_per_channel_region'],p['settings']['horizon']);plans[mode]=plan;traces[mode]=trace;cost[mode]={k:c[k]-before[k] for k in c}
     plans['random_'+mode]=geo.random(plan,mode,True)
     names=[mode,'random_'+mode]
     if mode=='channel':plans['random_joint']=geo.random(plan,'joint',False);names.append('random_joint')
     for arm in names:
      seqs[arm],errors[arm]=geo.verify(plans[arm],p['settings']['sequence_tolerance']);assert set(plan)==set(plans[arm]) and geo.dose(plan)==geo.dose(plans[arm])
    for arm,prior in p['reuse'].items():seqs[arm]=prev['sequences'][prior]
    z,_,_=predict(teacher,[seqs[a] for a in p['arms']],c,'boundary')
    rec=dict(seed=e['seed'],group=e['group'],base=base,sequences=seqs,plans=plans,traces=traces,errors=errors,teacher_logits=z,perarm_selection_counters=cost,pool_sizes={label:len(pool) for label,pool in geo.pools.items()},donors=geo.donors,channels={arm:{j:geo.node_class[node] for j,node in plan.items()} for arm,plan in plans.items()},doses={arm:geo.dose(plan) for arm,plan in plans.items()},old_doses={arm:(prev['doses'][prior] if prior in prev['doses'] else prev['old_doses'][prior]) for arm,prior in p['reuse'].items()},features_sha256=tensor_hash(base['x_aux']),mask_sha256=tensor_hash(base['native_adjacency']))
    dest=out/'plans'/('s%d_g%d.pt'%(e['seed'],e['group']));torch.save(rec,dest);entries.append(dict(seed=e['seed'],group=e['group'],path=str(dest.relative_to(out)),sha256=sha(dest),plans=plans,channels=rec['channels'],doses=rec['doses'],max_sequence_error=max(errors.values())));enforce();print('SEALED_PLAN',len(entries),192,flush=True)
   assert len(entries)==192;write(out/'sealed_before_victim.json',dict(entries=entries,official_victim_loaded=False));result.update(status='completed',seal_sha256=sha(out/'sealed_before_victim.json'))
  else:
   for d in p['datasets']:assert read(s/d/'selection_result.json')['status']=='completed';assert sha(s/d/'sealed_before_victim.json')==read(s/d/'selection_result.json')['seal_sha256']
   if args.dataset=='yelpchi':from adapter import load_classifier
   else:from course.dual_surrogate.runtime import load_classifier
   victim=load_classifier('gaga_victim',data);vh=freeze(victim);refs={(r['seed'],r['group'],r['arm']):r for r in map(json.loads,(parent/args.dataset/'evaluation.jsonl').read_text().splitlines())};rows=0
   with (out/'evaluation.jsonl').open('x') as stream:
    for i,e in enumerate(read(out/'sealed_before_victim.json')['entries']):
     assert sha(out/e['path'])==e['sha256'];rec=torch.load(out/e['path'],map_location='cpu');base=rec['base']
     for j,arm in enumerate(p['arms']):
      with torch.no_grad():z=victim(rec['sequences'][arm]).reshape(len(base['ids']),2)
      c['official_GAGA_VALID']+=1;assert torch.isfinite(z).all()
      if arm in p['reuse']:
       ref=refs[e['seed'],e['group'],p['reuse'][arm]];assert float((z-torch.tensor(ref['logits'])).abs().max())<2e-5;assert int((z.argmax(1)==0).sum())==ref['wrong']
      row=dict(seed=e['seed'],group=e['group'],arm=arm,node_ids=base['ids'].tolist(),logits=z.tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_logits=rec['teacher_logits'][j].tolist(),exchanged_edges=rec['doses'][arm] if arm in rec['doses'] else rec['old_doses'][arm],actual_nodes=len(base['source_ids']),actual_edges=int(base['native_adjacency'].sum()),features_sha256=rec['features_sha256'],mask_sha256=rec['mask_sha256']);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows+=1
     if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
     enforce()
   assert_frozen(victim,vh);result.update(status='completed',rows=rows)
  assert_frozen(teacher,th)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/(args.mode+'_result.json'),result)
if __name__=='__main__':main()
