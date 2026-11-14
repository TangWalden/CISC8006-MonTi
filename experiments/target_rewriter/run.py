"""TRAIN lookahead imitation, learned two-step policies, then sealed VALID scoring."""
import pathlib,argparse,time,signal,random,json,statistics
from common import read,write,sha,setup,load,index_cards,oldcard,counters

def summarize(rows,p):
 metrics=[]
 for seed in p['seeds']:
  for arm in p['arms']:
   rr=[r for r in rows if (r['seed'],r['arm'])==(seed,arm)];n=sum(len(r['node_ids']) for r in rr)
   assert len(rr)==64 and n==571 and len({i for r in rr for i in r['node_ids']})==240
   metrics.append(dict(seed=seed,arm=arm,MR=100*sum(r['wrong'] for r in rr)/n,wrong=sum(r['wrong'] for r in rr),occurrences=n,exchanged_edges=sum(r['exchanged_edges'] for r in rr),changed_slots=sum(len(r['actions']) for r in rr)))
 means={a:statistics.mean(r['MR'] for r in metrics if r['arm']==a) for a in p['arms']};contrasts={}
 for mode,controls in [('rewriter',['native','single_neural','single_direct','random_endpoint','random_joint','greedy','independent']),('greedy',['native','single_neural','random_greedy']),('independent',['native','single_neural','random_independent'])]:
  for control in controls:
   ds=[next(r['MR'] for r in metrics if (r['seed'],r['arm'])==(seed,mode))-next(r['MR'] for r in metrics if (r['seed'],r['arm'])==(seed,control)) for seed in p['seeds']]
   avg=statistics.mean(ds);half=4.302652729911275*statistics.stdev(ds)/(3**.5)
   contrasts[mode+'__'+control]=dict(deltas_pp=ds,mean_pp=avg,nominal_t95=[avg-half,avg+half])
 return dict(metrics=metrics,means=means,contrasts=contrasts,primary_pass=all(means['rewriter']>means[x] for x in ['native','single_neural','single_direct','random_endpoint','random_joint']),one_pp_above_existing=means['rewriter']>=means['single_neural']+1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);args=ap.parse_args();s,p,q=setup(args.stage)
 import torch
 from network import Rewriter
 from kernel import train_card,key,schedule,options,apply,physical,predict,state,policy,randomize,dose
 from experiments.downstream_cooperation.common import assert_frozen,freeze,tensor_hash
 from course.dual_surrogate.runtime import load_classifier
 assert read(s/'preflight/result.json')['status']=='passed'
 out=s/'main';out.mkdir(exist_ok=False)
 for n in ['train_cards','models','plans']:(out/n).mkdir()
 began=time.monotonic();cpu=time.process_time();c=counters();result=dict(status='failed',counters=c,GPU_hours=0,protocol_sha256=sha(s/'protocol.json'));signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Frozen study deadline')));signal.alarm(p['resources']['main_seconds'])
 try:
  data,view,teacher,th,benign,mean,std=load(q);index=index_cards(p['parent']);trainrefs=[];models=[]
  def enforce():
   for k,limit in p['caps'].items():assert c[k]<=limit,(k,c[k],limit)
  print('STAGE TRAIN_exact_two_edit_labels',flush=True)
  for seed in p['seeds']:
   for group in p['train_groups']:
    old=oldcard(p['parent'],index,seed,'train',group);v=train_card(data,view,old['case'],old,teacher,benign,mean,std,c)
    f=out/'train_cards'/('s%d_g%d.pt'%(seed,group));torch.save(v,f);trainrefs.append(dict(seed=seed,group=group,path=str(f.relative_to(out)),sha256=sha(f),states=len(v['records'])));enforce();print('TRAIN_CARDS',len(trainrefs),96,flush=True)
  write(out/'train_seal.json',dict(cards=trainrefs,official_victim_loaded=False))
  print('STAGE TRAIN_hierarchical_imitation',flush=True)
  for seed in p['seeds']:
   torch.manual_seed(2026111400+seed);net=Rewriter(data.feat_dim);opt=torch.optim.Adam(net.parameters(),lr=.0005,weight_decay=.0001)
   items=[]
   for r in trainrefs:
    if r['seed']==seed:
     assert sha(out/r['path'])==r['sha256'];items.extend(torch.load(out/r['path'],map_location='cpu')['records'])
   with (out/('training_s%d.jsonl'%seed)).open('x') as stream:
    for epoch in range(12):
     order=list(range(len(items)));random.Random(key(seed,epoch,'rewrite_train_order')).shuffle(order);losses=[];updates=0
     for i in order:
      v=items[i]
      if not v['options']:continue
      opt.zero_grad();lp=net(v['targets'],v['actions'],v['coverage'],v['regions']);loss=-(v['labels']*lp).sum();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();c['updates']+=1;updates+=1;losses.append(float(loss))
     stream.write(json.dumps(dict(epoch=epoch+1,states=len(items),updates=updates,mean_cross_entropy=statistics.mean(losses) if losses else None))+'\n');stream.flush();enforce();print('TRAIN',seed,epoch+1,flush=True)
   f=out/'models'/('s%d.pt'%seed);torch.save(net.state_dict(),f);models.append(dict(seed=seed,path=str(f.relative_to(out)),sha256=sha(f)))
  write(out/'model_seal.json',dict(models=models,official_victim_loaded=False))
  oldplans={(x['seed'],x['group']):x for x in read(pathlib.Path(p['parent'])/'main/sealed_before_victim.json')['plans']};entries=[]
  print('STAGE VALID_action_selection_no_victim',flush=True)
  for seed in p['seeds']:
   net=Rewriter(data.feat_dim);mr=next(x for x in models if x['seed']==seed);assert sha(out/mr['path'])==mr['sha256'];net.load_state_dict(torch.load(out/mr['path'],map_location='cpu'));nh=freeze(net)
   for group in p['valid_groups']:
    old=oldcard(p['parent'],index,seed,'valid',group);base=old['case'];steps=schedule(base,view,benign);seq0=old['sequences'][0];z0=old['teacher_logits'][0];cache={():seq0};zcache={():z0};losscache={():float(old['losses'][0])}
    before=dict(c)
    def obtain(chosen):
     k=tuple(sorted(chosen))
     if k not in cache:
      if len(k)==1 and k[0][0]==0:cache[k]=old['sequences'][k[0][1]+1]
      else:cache[k]=physical(data,view,base,apply(base,steps,k,data.features),c)
     return cache[k]
    chosen=[];traces=[]
    for turn in range(2):
     seq=obtain(chosen);k=tuple(sorted(chosen))
     if k not in zcache:
      zz,ll=predict(teacher,[seq],c);zcache[k]=zz[0];losscache[k]=float(ll[0])
     v=state(data,view,base,steps,chosen,seq,zcache[k],mean,std);action,lp=policy(net,v,c);traces.append(dict(chosen=list(chosen),options=v['options'],log_probabilities=lp.tolist(),action=action))
     if action==0:break
     chosen.append(v['options'][action-1])
    neural=list(chosen);neural_query_contexts=c['teacher_contexts']-before['teacher_contexts']+1 # Includes reused initial teacher evidence.
    # Greedy and independent controls use the identical legal three-region/four-node pools.
    allopts=options(steps);singles=[(x,) for x in allopts]
    if singles:
     zz,ll=predict(teacher,[obtain(x) for x in singles],c)
     for i,k in enumerate(singles):zcache[k]=zz[i];losscache[k]=float(ll[i])
    positive=sorted([x for x in allopts if losscache[()]-losscache[(x,)]>1e-4],key=lambda x:(losscache[(x,)],x));independent=[]
    for x in positive:
     if x[0] not in [r for r,b in independent]:independent.append(x)
     if len(independent)==2:break
    greedy=[]
    if positive:
     greedy=[positive[0]];more=options(steps,[greedy[0][0]]);pairs=[tuple(sorted(greedy+[x])) for x in more]
     if pairs:
      zz,ll=predict(teacher,[obtain(k) for k in pairs],c)
      for i,k in enumerate(pairs):zcache[k]=zz[i];losscache[k]=float(ll[i])
      best=min(range(len(pairs)),key=lambda i:(float(ll[i]),i))
      if losscache[tuple(greedy)]-float(ll[best])>1e-4:greedy.append(more[best])
    actions=dict(rewriter=neural,random_endpoint=randomize(base,steps,neural,'endpoint'),random_joint=randomize(base,steps,neural,'joint'),greedy=greedy,random_greedy=randomize(base,steps,greedy,'greedy'),independent=independent,random_independent=randomize(base,steps,independent,'independent'))
    seqs={a:obtain(v) for a,v in actions.items()};cases={a:apply(base,steps,v,data.features) for a,v in actions.items()};doses={a:dose(base,steps,v) for a,v in actions.items()}
    oldsel=oldplans[seed,group]['selected'];oldnames=dict(native='native',single_neural='neural',single_direct='direct',random_single='random_neural')
    from geometry_hard import replace
    for a,n in oldnames.items():
     ch=oldsel[n];seqs[a]=old['sequences'][ch];cases[a]=base if not ch else replace(base,old['donor'],old['bank'][ch-1],data.features);actions[a]=[] if not ch else [(0,ch-1)];doses[a]=0 if not ch else int(base['native_adjacency'][:,len(base['ids'])+old['donor']].sum())
    for a,b in [('rewriter','random_endpoint'),('rewriter','random_joint'),('greedy','random_greedy'),('independent','random_independent')]:assert doses[a]==doses[b] and len(actions[a])==len(actions[b])
    finalz,_=predict(teacher,[seqs[a] for a in p['arms']],c)
    payload=dict(base=base,steps=steps,actions=actions,sequences=seqs,candidate_ids={a:v['candidate_ids'] for a,v in cases.items()},exchanged_edges=doses,teacher_logits=finalz,traces=traces,teacher_queries=dict(neural_selection_contexts=neural_query_contexts,greedy_selection_contexts=1+len(singles)+(len(pairs) if positive and more else 0),old_single_selection_contexts=len(old['sequences']),independent_selection_contexts=1+len(singles)),counter_delta={k:c[k]-before[k] for k in c})
    assert payload['teacher_queries']['neural_selection_contexts']<=2 and payload['teacher_queries']['greedy_selection_contexts']<=21
    f=out/'plans'/('s%d_g%d.pt'%(seed,group));torch.save(payload,f);entries.append(dict(seed=seed,group=group,path=str(f.relative_to(out)),sha256=sha(f),actions=actions,exchanged_edges=doses));enforce();print('PLANNED',len(entries),192,flush=True)
   assert_frozen(net,nh)
  assert_frozen(teacher,th);write(out/'sealed_before_victim.json',dict(entries=entries,official_victim_loaded=False));seal=sha(out/'sealed_before_victim.json')
  victim=load_classifier('gaga_victim',data);vh=freeze(victim);rows=[];oldrows={(r['seed'],r['group'],r['arm']):r for r in map(json.loads,(pathlib.Path(p['parent'])/'main/evaluation.jsonl').read_text().splitlines())}
  print('STAGE sealed_official_VALID',flush=True)
  with (out/'evaluation.jsonl').open('x') as stream:
   for i,e in enumerate(entries):
    assert sha(out/e['path'])==e['sha256'];v=torch.load(out/e['path'],map_location='cpu');base=v['base'];t=len(base['ids'])
    for j,arm in enumerate(p['arms']):
     with torch.no_grad():z=victim(v['sequences'][arm]).reshape(t,2)
     c['official_GAGA_VALID']+=1;assert torch.isfinite(z).all()
     if arm in oldnames:
      ref=oldrows[e['seed'],e['group'],oldnames[arm]];assert float((z-torch.tensor(ref['gaga_logits']).reshape(t,2)).abs().max())<2e-5;assert int((z.argmax(1)==0).sum())==ref['wrong']
     row=dict(seed=e['seed'],group=e['group'],arm=arm,node_ids=base['ids'].tolist(),actions=v['actions'][arm],gaga_logits=z.tolist(),wrong=int((z.argmax(1)==0).sum()),teacher_logits=v['teacher_logits'][j].tolist(),exchanged_edges=v['exchanged_edges'][arm],candidate_ids=v['candidate_ids'][arm].tolist(),actual_nodes=len(base['source_ids']),actual_edges=int(base['native_adjacency'].sum()),features_sha256=tensor_hash(base['x_aux']),mask_sha256=tensor_hash(base['native_adjacency']))
     rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
    enforce()
    if (i+1)%16==0:print('EVALUATED',i+1,192,flush=True)
  assert len(entries)==192 and c['official_GAGA_VALID']==192*len(p['arms']);assert sha(out/'sealed_before_victim.json')==seal;assert_frozen(victim,vh);assert_frozen(teacher,th)
  result.update(status='completed',**summarize(rows,p),limitations=p['limitations'])
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-began,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
