"""Native captures and the unchanged single-retrieval baseline fit; no victim."""
import pathlib,argparse,time,signal,importlib.util,random,json,statistics
from common import setup,load,write,sha,read
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);assert read(s/'native_preflight/result.json')['status']=='passed';parent=s/'single';parent.mkdir(exist_ok=False);out=parent/'main';out.mkdir();(out/'cards').mkdir();(out/'models').mkdir();start=time.monotonic();cpu=time.process_time();counts=dict(upstream_generations=0,physical_canonical_builds=0,own_teacher_batched_forwards=0,own_teacher_hard_contexts=0,neural_updates=0,training_visits=0,policy_forwards=0,official_GAGA=0,TEST_queries=0);result=dict(status='failed',counters=counts,GPU_hours=0)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('single baseline ceiling')));signal.alarm(p['resources']['single_seconds'])
 try:
  import torch
  from adapter import load_upstream,capture
  from geometry_hard import key
  from model import PortalRanker
  from experiments.downstream_cooperation.common import assert_frozen
  data,view,teacher,th,benign,mean,std=load(q);spec=importlib.util.spec_from_file_location('shared_portal_card',p['shared_sources']['portal_common.py']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);index={};author_states=[]
  for seed in p['seeds']:
   engine,eh=load_upstream(q,data);author_states.append(eh)
   for phase,groups in [('train',p['single_train_groups']),('valid',p['valid_groups'])]:
    for group in groups:
     case,native=capture(engine,data,view,seed,phase,group,p);counts['upstream_generations']+=1;v=mod.card(data,view,case,teacher,counts);assert float((native-v['sequences'][0]).abs().max())<2e-5
     f=out/'cards'/('s%d_%s_g%d.pt'%(seed,phase,group));torch.save(v,f);index[seed,phase,group]=dict(path=str(f.relative_to(out)),sha256=sha(f));assert counts['upstream_generations']<=576 and counts['physical_canonical_builds']<=5184 and counts['own_teacher_hard_contexts']<=5184
     if len(index)%16==0:print('SINGLE_CARDS',len(index),576,flush=True)
   assert_frozen(engine.model,eh);del engine
  assert len(set(author_states))==1 and len(index)==576;assert_frozen(teacher,th);write(out/'card_seal.json',dict(cards=[dict(key=list(k),**v) for k,v in index.items()],official_victim_loaded=False,author_state_hash=author_states[0]))
  def card(seed,phase,group):
   ref=index[seed,phase,group];f=out/ref['path'];assert sha(f)==ref['sha256'];return torch.load(f,map_location='cpu')
  models=[];plans=[]
  for seed in p['seeds']:
   torch.manual_seed(2026111400+seed);net=PortalRanker(data.feat_dim);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);history=[]
   for epoch in range(8):
    order=list(p['single_train_groups']);random.Random(key(seed,epoch,'portal_training_order')).shuffle(order);losses=[];updates=0
    for group in order:
     v=card(seed,'train',group);opt.zero_grad();scores=net(v['query'],v['keys'],v['changes']);scale=v['losses'].std(unbiased=False).clamp_min(.001);target=torch.softmax((v['losses'][0]-v['losses'])/scale,0);loss=-(target*torch.log_softmax(scores,0)).sum();assert torch.isfinite(loss);counts['training_visits']+=1
     if len(scores)>1:loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();counts['neural_updates']+=1;updates+=1
     losses.append(float(loss))
    history.append(dict(epoch=epoch+1,mean_loss=statistics.mean(losses),updates=updates));print('SINGLE_TRAIN',seed,epoch+1,flush=True)
   net.eval();f=out/'models'/('s%d.pt'%seed);torch.save(net.state_dict(),f);models.append(dict(seed=seed,path=str(f.relative_to(out)),sha256=sha(f),history=history))
   for group in p['valid_groups']:
    v=card(seed,'valid',group)
    with torch.no_grad():scores=net(v['query'],v['keys'],v['changes'])
    counts['policy_forwards']+=1;neural=int(scores.argmax());best=int(v['losses'].argmin());direct=best if float(v['losses'][0]-v['losses'][best])>1e-4 else 0;ri=random.Random(key(seed,group,'portal_matched_random')).randrange(1,len(v['bank'])+1) if v['bank'] else 0
    plans.append(dict(seed=seed,group=group,selected=dict(native=0,neural=neural,direct=direct,random_neural=ri if neural else 0,random_direct=ri if direct else 0),scores=scores.tolist(),teacher_losses=v['losses'].tolist(),bank=v['bank'],donor=v['donor']))
  assert counts['training_visits']==3072 and len(plans)==192;write(out/'sealed_before_victim.json',dict(plans=plans,models=models,official_victim_loaded=False));assert_frozen(teacher,th);result.update(status='completed',author_configuration_seed=2,author_state_hash=author_states[0],original_victim_loaded=False)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result)
if __name__=='__main__':main()
