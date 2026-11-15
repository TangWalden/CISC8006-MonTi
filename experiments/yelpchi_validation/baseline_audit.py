"""Saved-only YelpChi baseline cards/choices, author state and asset integrity."""
import argparse,pathlib,json,random,time
from common import setup,read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);start=time.monotonic();cpu=time.process_time();out=s/'single/main';checks=0
 import torch
 from geometry_hard import key
 def check(v):
  nonlocal checks
  checks+=1
  if not v:raise AssertionError('baseline audit '+str(checks))
 seal=read(out/'card_seal.json');check(len(seal['cards'])==576);cards={}
 for ref in seal['cards']:
  check(sha(out/ref['path'])==ref['sha256']);v=torch.load(out/ref['path'],map_location='cpu');seed,phase,g=ref['key'];base=v['case'];check(base['ids'].tolist()==p[phase+'_members'][str(g)]);check(len(base['source_ids'])==len(base['x_aux']));check(int(base['native_adjacency'].sum())==base['quota']['edge_budget'])
  ll=torch.nn.functional.softplus(v['teacher_logits'][:,:,1]-v['teacher_logits'][:,:,0]).mean(1);check(torch.equal(ll,v['losses']));check(len(v['bank'])+1==len(ll))
  if phase=='valid':cards[seed,g]=v
 plans=read(out/'sealed_before_victim.json');check(len(plans['plans'])==192)
 for m in plans['models']:check(sha(out/m['path'])==m['sha256']);check(len(m['history'])==8)
 for plan in plans['plans']:
  seed,g=plan['seed'],plan['group'];v=cards[seed,g];best=int(v['losses'].argmin());direct=best if float(v['losses'][0]-v['losses'][best])>1e-4 else 0;neural=max(range(len(plan['scores'])),key=lambda i:plan['scores'][i]);ri=random.Random(key(seed,g,'portal_matched_random')).randrange(1,len(v['bank'])+1) if v['bank'] else 0
  expected=dict(native=0,neural=neural,direct=direct,random_neural=ri if neural else 0,random_direct=ri if direct else 0);check(expected==plan['selected']);check(plan['bank']==v['bank'] and plan['donor']==v['donor'])
 r=read(out/'result.json');check(r['status']=='completed' and r['counters']['upstream_generations']==576 and r['counters']['training_visits']==3072 and r['counters']['official_GAGA']==r['counters']['TEST_queries']==0)
 check(sum(h['updates'] for m in plans['models'] for h in m['history'])==r['counters']['neural_updates'])
 write(s/'baseline_audit.json',dict(status='passed',checks=checks,root_authored=True,model_forwards=0,tensor_loads=True,inputs_rechecked=True,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu));print(dict(status='passed',checks=checks))
if __name__=='__main__':main()
