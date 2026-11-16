"""Separate preflight, selection and victim scoring processes."""
import pathlib,argparse,time,signal,json
from gr_common import read,write,sha,setup,counts

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--mode',choices=['preflight','selection','evaluation'],required=True);args=ap.parse_args()
 start=time.monotonic();cpu=time.process_time();s,p,v,parent,old,q,data,view,teacher,th,benign,common=setup(args.stage,args.dataset)
 import torch
 from gr_core import Geometry,predict,search
 from experiments.downstream_cooperation.common import assert_frozen,freeze,tensor_hash
 out=s/args.dataset;out.mkdir(exist_ok=True);c=counts();result=dict(status='failed',mode=args.mode,dataset=args.dataset,counters=c,GPU_hours=0)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Frozen phase deadline')));signal.alarm(p['resources'][args.mode+'_seconds'])
 def enforce():
  for name in ['physical_builds','teacher_contexts','backward_calls']:assert c[name]<=p['caps']['per_dataset_'+name]
  assert c['official_GAGA_VALID']<=p['caps']['per_dataset_official_VALID']
 try:
  if args.mode=='preflight':
   index=common.index_cards(old['parent']);checks=[]
   for seed in v['seeds'][:2]:
    for group in v['preflight_groups']:
     card=common.oldcard(old['parent'],index,seed,'train',group);geo=Geometry(data,view,card['case'],card['sequences'][0],benign,c)
     assert len(geo.donors)>=2 and len(geo.pool)>=4,('preflight_geometry',seed,group)
     ds=geo.donors[:2];pool=geo.pool;plans=[{}, {ds[0]:pool[-1]}, {ds[1]:pool[-2]}, {ds[0]:pool[-1],ds[1]:pool[-2]}, {ds[0]:pool[-2],ds[1]:pool[-1]}];errors=[];hard=[];fast=[]
     for plan in plans:
      seq,err=geo.verify(plan,p['settings']['sequence_tolerance']);errors.append(err);hard.append(seq);fast.append(geo.sequence(plan))
     zh,_,_=predict(teacher,hard,c,'boundary');zf,_,_=predict(teacher,fast,c,'boundary');le=float((zh-zf).abs().max());assert le<=p['settings']['logit_tolerance'];assert torch.equal(zh.argmax(-1),zf.argmax(-1))
     # Actual input-gradient directional finite difference, not an optimizer step.
     _,_,grad=predict(teacher,[fast[1]],c,'boundary',True);direction=grad[0]/grad[0].norm().clamp_min(1e-12);eps=.001
     _,lv,_=predict(teacher,[fast[1]+eps*direction,fast[1]-eps*direction],c,'boundary');numeric=float((lv[0]-lv[1])/(2*eps));analytic=float((grad[0]*direction).sum());assert abs(numeric-analytic)<=max(1e-4,.05*abs(analytic)),(numeric,analytic)
     checks.append(dict(seed=seed,group=group,pool_size=len(pool),sequence_error=max(errors),logit_error=le,gradient_norm=float(grad.norm()),directional_numeric=numeric,directional_autograd=analytic));enforce();print('PREFLIGHT',len(checks),4,flush=True)
   result.update(checks=checks,status='completed')
  elif args.mode=='selection':
   assert all(read(s/d/'preflight_result.json')['status']=='completed' for d in p['datasets']);(out/'plans').mkdir(exist_ok=False);entries=[]
   for e in read(parent/'main/sealed_before_victim.json')['entries']:
    f=parent/'main'/e['path'];assert sha(f)==e['sha256'];prev=torch.load(f,map_location='cpu');base=prev['base'];geo=Geometry(data,view,base,prev['sequences']['native'],benign,c);plans={};seqs={};traces={};errors={};perarm={}
    for mode in ['softplus','boundary','uniform']:
     before=dict(c);plan,trace=search(geo,teacher,mode,c,p['settings']['shortlist_per_region'],p['settings']['horizon']);plans[mode]=plan;traces[mode]=trace;plans['random_'+mode]=geo.random(plan,mode);perarm[mode]={k:c[k]-before[k] for k in c}
     for arm in [mode,'random_'+mode]:seqs[arm],errors[arm]=geo.verify(plans[arm],p['settings']['sequence_tolerance'])
     assert geo.dose(plans[mode])==geo.dose(plans['random_'+mode]);assert set(plans[mode])==set(plans['random_'+mode])
    for arm in ['native','single_neural','rewriter']:seqs[arm]=prev['sequences'][arm]
    z,_,_=predict(teacher,[seqs[a] for a in p['arms']],c,'boundary');record=dict(seed=e['seed'],group=e['group'],base=base,sequences=seqs,plans=plans,traces=traces,errors=errors,teacher_logits=z,perarm_selection_counters=perarm,pool_size=len(geo.pool),donors=geo.donors,doses={a:geo.dose(pl) for a,pl in plans.items()},old_doses={a:prev['exchanged_edges'][a] for a in ['native','single_neural','rewriter']},features_sha256=tensor_hash(base['x_aux']),mask_sha256=tensor_hash(base['native_adjacency']))
    dest=out/'plans'/('s%d_g%d.pt'%(e['seed'],e['group']));torch.save(record,dest);entries.append(dict(seed=e['seed'],group=e['group'],path=str(dest.relative_to(out)),sha256=sha(dest),plans=plans,doses=record['doses'],max_sequence_error=max(errors.values())));enforce();print('SEALED_PLAN',len(entries),192,flush=True)
   assert len(entries)==192;write(out/'sealed_before_victim.json',dict(entries=entries,official_victim_loaded=False));result.update(status='completed',seal_sha256=sha(out/'sealed_before_victim.json'))
  else:
   for d in p['datasets']:assert read(s/d/'selection_result.json')['status']=='completed';assert sha(s/d/'sealed_before_victim.json')==read(s/d/'selection_result.json')['seal_sha256']
   if args.dataset=='yelpchi':from adapter import load_classifier
   else:from course.dual_surrogate.runtime import load_classifier
   victim=load_classifier('gaga_victim',data);vh=freeze(victim);rows=[];refs={(r['seed'],r['group'],r['arm']):r for r in map(json.loads,(parent/'main/evaluation.jsonl').read_text().splitlines())}
   with (out/'evaluation.jsonl').open('x') as stream:
    for i,e in enumerate(read(out/'sealed_before_victim.json')['entries']):
     assert sha(out/e['path'])==e['sha256'];record=torch.load(out/e['path'],map_location='cpu');base=record['base']
     for j,arm in enumerate(p['arms']):
      with torch.no_grad():z=victim(record['sequences'][arm]).reshape(len(base['ids']),2)
      c['official_GAGA_VALID']+=1;assert torch.isfinite(z).all()
      if arm in ['native','single_neural','rewriter']:
       ref=refs[e['seed'],e['group'],arm];assert float((z-torch.tensor(ref['gaga_logits'])).abs().max())<2e-5;assert int((z.argmax(1)==0).sum())==ref['wrong']
      row=dict(seed=e['seed'],group=e['group'],arm=arm,node_ids=base['ids'].tolist(),logits=z.tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_logits=record['teacher_logits'][j].tolist(),exchanged_edges=record['doses'][arm] if arm in record['doses'] else record['old_doses'][arm],actual_nodes=len(base['source_ids']),actual_edges=int(base['native_adjacency'].sum()),features_sha256=record['features_sha256'],mask_sha256=record['mask_sha256']);rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
     if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
     enforce()
   assert_frozen(victim,vh);result.update(status='completed',rows=len(rows))
  assert_frozen(teacher,th)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/(args.mode+'_result.json'),result)
if __name__=='__main__':main()
