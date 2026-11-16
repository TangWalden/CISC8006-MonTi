"""Unchanged action search; complete cohort sealing precedes official TEST."""
import argparse,pathlib,time,signal,json
from full_common import read,write,sha,setup,counts
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--mode',choices=['preflight','selection','evaluation'],required=True);ap.add_argument('--shard',type=int,default=0);a=ap.parse_args()
 start=time.monotonic();cpu=time.process_time();s,p,loaded=setup(a.stage,a.dataset)
 _,old,parent,older,v,ancestor,ancestor_protocol,q,data,view,teacher,th,benign,common=loaded
 import torch
 from gr_core import predict
 from cr_core import Geometry,search
 from experiments.downstream_cooperation.common import freeze,assert_frozen,tensor_hash
 out=s/a.dataset;out.mkdir(exist_ok=True);c=counts();result=dict(status='failed',dataset=a.dataset,mode=a.mode,shard=a.shard,counters=c)
 suffix=('_'+str(a.shard)) if a.mode=='selection' else '';signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Frozen phase deadline')));signal.alarm(p['resources'][a.mode+'_seconds']);ds=p['datasets'][a.dataset]
 def build(base,seq0):
  before=dict(c);geo=Geometry(data,view,base,seq0,benign,c);plan,trace=search(geo,teacher,'channel',c,p['settings']['shortlist_per_channel_region'],p['settings']['horizon'])
  plans=dict(channel=plan,random_channel=geo.random(plan,'channel',True),random_joint=geo.random(plan,'joint',False));seqs=dict(native=seq0);errors={}
  for arm,pl in plans.items():
   seqs[arm],errors[arm]=geo.verify(pl,p['settings']['sequence_tolerance']);assert set(pl)==set(plan) and geo.dose(pl)==geo.dose(plan)
  z,_,_=predict(teacher,[seqs[arm] for arm in p['arms']],c,'boundary')
  for k,cap in p['caps_per_case'].items():assert c[k]-before[k]<=cap,(k,c[k]-before[k])
  return dict(base=base,sequences=seqs,plans=plans,traces=trace,errors=errors,teacher_logits=z,doses={arm:geo.dose(pl) for arm,pl in plans.items()},channels={arm:{j:geo.node_class[n] for j,n in pl.items()} for arm,pl in plans.items()},pool_sizes={label:len(pool) for label,pool in geo.pools.items()},donors=geo.donors,features_sha256=tensor_hash(base['x_aux']),mask_sha256=tensor_hash(base['native_adjacency']),selection_counters={k:c[k]-before[k] for k in c})
 try:
  if a.mode=='preflight':
   checks=[]
   for e in ds['preflight_entries']:
    f=pathlib.Path(p['method_parent'])/a.dataset/e['path'];assert sha(f)==e['sha256'];ref=torch.load(f,map_location='cpu');r=build(ref['base'],ref['sequences']['native'])
    for arm in p['arms']:
     err=float((r['sequences'][arm]-ref['sequences'][arm]).abs().max());assert err<=1e-5,(arm,err)
     if arm!='native':assert r['plans'][arm]==ref['plans'][arm] and r['channels'][arm]==ref['channels'][arm] and r['doses'][arm]==ref['doses'][arm]
     j=old['arms'].index(arm);assert float((r['teacher_logits'][p['arms'].index(arm)]-ref['teacher_logits'][j]).abs().max())<=2e-5
    checks.append(dict(seed=e['seed'],group=e['group'],plans_reproduced=True,sequence_and_teacher_parity=True));print('PREFLIGHT',len(checks),2,flush=True)
   result.update(status='completed',checks=checks,TEST_calls=0)
  elif a.mode=='selection':
   assert all(read(s/d/'preflight_result.json')['status']=='completed' for d in p['datasets']);assert 0<=a.shard<p['selection_shards'];folder=out/('plans_'+str(a.shard));folder.mkdir(exist_ok=False);entries=[]
   items=[(seed,group) for seed in ds['seeds'] for group in ds['test_groups']][a.shard::p['selection_shards']]
   if a.dataset=='gossipcop':native={(e['seed'],e['group']):e for e in ds['native_cards']}
   else:
    from adapter import load_upstream,capture
    engine,eh=load_upstream(q,data);gh=freeze(engine.surrogate_model)
   for i,(seed,group) in enumerate(items):
    if a.dataset=='gossipcop':
     e=native[seed,group];f=pathlib.Path(ds['native_parent'])/e['path'];assert sha(f)==e['sha256'];saved=torch.load(f,map_location='cpu');base,seq0=saved['case'],saved['sequences'][0];del saved
    else:base,seq0=capture(engine,data,view,seed,'test',group,ds);c['upstream_generations']+=1;c['native_canonical_builds']+=1
    assert base['ids'].tolist()==ds['test_members'][str(group)] and base['seed']==seed and base['group']==group
    r=build(base,seq0);dest=folder/('s%d_g%d.pt'%(seed,group));torch.save(r,dest);entries.append(dict(seed=seed,group=group,path=str(dest.relative_to(out)),sha256=sha(dest),plans=r['plans'],channels=r['channels'],doses=r['doses'],max_sequence_error=max(r['errors'].values())))
    if (i+1)%8==0 or i+1==len(items):print('SEALED_PLAN',i+1,len(items),flush=True)
   if a.dataset=='yelpchi':assert_frozen(engine.model,eh);assert_frozen(engine.surrogate_model,gh)
   write(out/('seal_'+str(a.shard)+'.json'),dict(entries=entries,official_victim_loaded=False));result.update(status='completed',cases=len(items),seal_sha256=sha(out/('seal_'+str(a.shard)+'.json')))
  else:
   global_seal=read(s/'all_actions_sealed.json')
   for d in p['datasets']:assert sha(s/d/'sealed_before_victim.json')==global_seal['datasets'][d]
   if a.dataset=='yelpchi':from adapter import load_classifier
   else:from course.dual_surrogate.runtime import load_classifier
   victim=load_classifier('gaga_victim',data);vh=freeze(victim);refs={}
   if a.dataset=='gossipcop':
    assert sha(ds['native_prediction_reference'])==ds['native_prediction_reference_sha256']
    for line in pathlib.Path(ds['native_prediction_reference']).read_text().splitlines():
     r=json.loads(line)
     if r['arm']=='native':refs[r['seed'],r['group']]=r
    assert len(refs)==len(ds['native_cards'])
   rows=0;max_native_error=0
   with (out/'evaluation.jsonl').open('x') as stream:
    entries=read(out/'sealed_before_victim.json')['entries']
    for i,e in enumerate(entries):
     f=out/e['path'];assert sha(f)==e['sha256'];rec=torch.load(f,map_location='cpu');base=rec['base'];assert bool((data.labels[base['ids']]==1).all())
     for j,arm in enumerate(p['arms']):
      with torch.no_grad():z=victim(rec['sequences'][arm]).reshape(len(base['ids']),2)
      c['official_GAGA_TEST']+=1;assert torch.isfinite(z).all()
      if arm=='native' and refs:
       ref=refs[e['seed'],e['group']];err=float((z-torch.tensor(ref['gaga_logits'])).abs().max());max_native_error=max(err,max_native_error);assert err<=2e-5;assert int((z.argmax(1)==0).sum())==ref['wrong'];assert ref['node_ids']==base['ids'].tolist()
      row=dict(seed=e['seed'],group=e['group'],arm=arm,node_ids=base['ids'].tolist(),logits=z.tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_logits=rec['teacher_logits'][j].tolist(),exchanged_edges=rec['doses'].get(arm,0),actual_nodes=len(base['source_ids']),actual_edges=int(base['native_adjacency'].sum()),features_sha256=rec['features_sha256'],mask_sha256=rec['mask_sha256']);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();rows+=1
     if (i+1)%32==0 or i+1==len(entries):print('EVALUATED',i+1,len(entries),flush=True)
   assert rows==len(ds['seeds'])*len(ds['test_groups'])*len(p['arms']);assert_frozen(victim,vh);result.update(status='completed',rows=rows,max_native_logit_error=max_native_error)
  assert_frozen(teacher,th)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/(a.mode+suffix+'_result.json'),result)
if __name__=='__main__':main()
