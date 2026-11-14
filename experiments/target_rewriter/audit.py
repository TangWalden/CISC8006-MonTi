"""Saved-tensor/JSON recount, no classifier or policy forwards; root authored."""
import argparse,pathlib,time,json,math,itertools,statistics
from common import setup,read,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);start=time.monotonic();cpu=time.process_time()
 import torch
 from kernel import randomize,dose
 from run import summarize
 out=s/'main';result=read(out/'result.json');assert result['status']=='completed';checks=0
 def check(v):
  nonlocal checks
  checks+=1
  if not v:raise AssertionError('Audit check '+str(checks))
 for ref in read(out/'train_seal.json')['cards']:
  check(sha(out/ref['path'])==ref['sha256']);v=torch.load(out/ref['path'],map_location='cpu');loss={tuple(tuple(x) for x in r['actions']):r['loss'] for r in v['terminal_losses']}
  for rec in v['records']:
   chosen=tuple(rec['chosen']);values=[loss[chosen]]
   for x in rec['options']:
    nk=tuple(sorted(chosen+(x,)));val=loss[nk]
    if not chosen:val=min([val]+[vv for kk,vv in loss.items() if len(kk)==2 and x in kk])
    values.append(val)
   val=torch.tensor(values);check(torch.equal(rec['value_to_go'],val));labels=torch.softmax((val[0]-val)/val.std(unbiased=False).clamp_min(.001),0);check(torch.equal(rec['labels'],labels));check(rec['targets'].shape[0]==rec['coverage'].shape[1]);check(len(rec['options'])==len(rec['regions']))
 for m in read(out/'model_seal.json')['models']:check(sha(out/m['path'])==m['sha256'])
 rows=list(map(json.loads,(out/'evaluation.jsonl').read_text().splitlines()));rowmap={(r['seed'],r['group'],r['arm']):r for r in rows};check(len(rows)==len(rowmap)==2112);stats=[]
 for e in read(out/'sealed_before_victim.json')['entries']:
  check(sha(out/e['path'])==e['sha256']);v=torch.load(out/e['path'],map_location='cpu');base=v['base'];steps=v['steps'];t=len(base['ids']);baseids=base['candidate_ids'].tolist();a=base['native_adjacency']
  for trace in v['traces']:check(trace['action']==max(range(len(trace['log_probabilities'])),key=lambda j:trace['log_probabilities'][j]))
  for mode,kind in [('random_endpoint','endpoint'),('random_joint','joint'),('random_greedy','greedy'),('random_independent','independent')]:
   candidate={'random_endpoint':'rewriter','random_joint':'rewriter','random_greedy':'greedy','random_independent':'independent'}[mode]
   check(v['actions'][mode]==randomize(base,steps,v['actions'][candidate],kind));check(v['exchanged_edges'][mode]==v['exchanged_edges'][candidate]);check(len(v['actions'][mode])==len(v['actions'][candidate]))
  for j,arm in enumerate(p['arms']):
   r=rowmap[e['seed'],e['group'],arm];z=torch.tensor(r['gaga_logits']).reshape(t,2);check(torch.isfinite(z).all());check(int((z.argmax(1)==0).sum())==r['wrong']);check(r['node_ids']==base['ids'].tolist());check(r['candidate_ids']==v['candidate_ids'][arm].tolist());check(r['actual_nodes']==len(base['source_ids']) and r['actual_edges']==int(a.sum()));check(r['teacher_logits']==v['teacher_logits'][j].tolist())
   newids=v['candidate_ids'][arm].tolist();diff=[i for i,(x,y) in enumerate(zip(baseids,newids)) if x!=y]
   check(len(set(newids))==len(baseids));check(not set(newids)&set(base['ids'].tolist()));check(len(diff)==len(v['actions'][arm]));check(sum(int(a[:,t+i].sum()) for i in diff)==r['exchanged_edges'])
   if arm not in ['native','single_neural','single_direct','random_single']:
    expected=list(baseids)
    for reg,b in v['actions'][arm]:expected[steps[reg]['donor']]=steps[reg]['bank'][b]
    check(newids==expected);check(dose(base,steps,v['actions'][arm])==r['exchanged_edges'])
  nr=rowmap[e['seed'],e['group'],'rewriter'];na=rowmap[e['seed'],e['group'],'native'];old=rowmap[e['seed'],e['group'],'single_neural'];rr=rowmap[e['seed'],e['group'],'random_endpoint'];z=torch.tensor(nr['gaga_logits']);oldz=torch.tensor(old['gaga_logits']);zn=torch.tensor(na['gaga_logits']);teacher=v['teacher_logits'][p['arms'].index('rewriter')];teacher0=v['teacher_logits'][p['arms'].index('native')]
  wrong=z.argmax(1)==0;oldwrong=oldz.argmax(1)==0
  feasible=[rs for rs in itertools.combinations(range(len(steps)),len(v['actions']['rewriter'])) if dose(base,steps,[(r,0) for r in rs])==nr['exchanged_edges']]
  stats.append(dict(seed=e['seed'],group=e['group'],targets=t,rewriter_wrong=nr['wrong'],native_wrong=na['wrong'],existing_wrong=old['wrong'],random_wrong=rr['wrong'],gains_vs_existing=int((wrong&~oldwrong).sum()),losses_vs_existing=int((~wrong&oldwrong).sum()),slots=len(v['actions']['rewriter']),exchanged_edges=nr['exchanged_edges'],region_random_feasible_subsets=len(feasible),mean_victim_margin_delta=float(((z[:,1]-z[:,0])-(zn[:,1]-zn[:,0])).mean()),mean_teacher_margin_delta=float(((teacher[:,1]-teacher[:,0])-(teacher0[:,1]-teacher0[:,0])).mean()),teacher_queries=v['teacher_queries']))
 computed=summarize(rows,p)
 for k,v in computed.items():check(v==result[k])
 total_updates=sum(r['updates'] for seed in p['seeds'] for r in map(json.loads,(out/('training_s%d.jsonl'%seed)).read_text().splitlines()));check(total_updates==result['counters']['updates']);check(result['counters']['TEST_queries']==result['counters']['upstream_generations']==0)
 write(s/'saved_audit.json',dict(status='passed',checks=checks,root_authored=True,independent_or_human_review=False,model_forwards=0,tensor_loads=True,protocol_sha256=sha(s/'protocol.json'),result_sha256=sha(out/'result.json'),wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu))
 write(s/'case_diagnosis.json',dict(status='posthoc_after_outcomes',cases=stats,total_gains_vs_existing=sum(r['gains_vs_existing'] for r in stats),total_losses_vs_existing=sum(r['losses_vs_existing'] for r in stats),wins=sum(r['rewriter_wrong']>r['existing_wrong'] for r in stats),ties=sum(r['rewriter_wrong'] ==r['existing_wrong'] for r in stats),losses=sum(r['rewriter_wrong']<r['existing_wrong'] for r in stats),region_random_degenerate=sum(r['region_random_feasible_subsets']<=1 for r in stats),causal_claim=False))
 print(dict(status='passed',checks=checks),flush=True)
if __name__=='__main__':main()
