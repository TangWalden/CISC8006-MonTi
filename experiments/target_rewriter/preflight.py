"""TRAIN-only physical identity, compositional legality and policy-gradient checks."""
import argparse,pathlib,time,signal,itertools
from common import setup,load,index_cards,oldcard,counters,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);out=s/'preflight';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();c=counters();result=dict(status='failed',counters=c,GPU_hours=0)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('Preflight deadline')));signal.alarm(600)
 try:
  import torch,numpy as np
  from network import Rewriter
  from kernel import schedule,physical,apply,predict,state,randomize,dose,train_card
  from experiments.downstream_cooperation.common import assert_frozen
  from checks import synthetic
  checks=synthetic();data,view,teacher,th,benign,mean,std=load(q);index=index_cards(p['parent']);records=[]
  for group in p['preflight_groups']:
   old=oldcard(p['parent'],index,0,'train',group);base=old['case'];steps=schedule(base,view,benign);native=physical(data,view,base,base,c)
   assert float((native-old['sequences'][0]).abs().max())<2e-5
   cols=[n for x in steps for n in x['bank']];ids=base['ids'].numpy()
   assert len(set(cols))==len(cols) and set(cols)<=set(benign.tolist())
   if cols:assert view.adj[ids][:,cols].nnz==0 and view.adj2[ids][:,cols].nnz==0 and (view.adj[ids]@view.adj[:,cols]).nnz==0
   chosen=[(r,0) for r in range(min(2,len(steps)))];changed=apply(base,steps,chosen,data.features);seq=physical(data,view,base,changed,c)
   reverse=physical(data,view,base,apply(base,steps,list(reversed(chosen)),data.features),c);assert torch.equal(seq,reverse)
   z,ll=predict(teacher,[native,seq],c);assert float((z[0]-old['teacher_logits'][0]).abs().max())<2e-5
   if steps:
    single=physical(data,view,base,apply(base,steps,[(0,0)],data.features),c);assert float((single-old['sequences'][1]).abs().max())<2e-5
   for mode in ['endpoint','joint']:
    random= randomize(base,steps,chosen,mode);assert dose(base,steps,chosen)==dose(base,steps,random) and len(random)==len(chosen)
   v=state(data,view,base,steps,[],native,z[0],mean,std);net=Rewriter(data.feat_dim);opt=torch.optim.Adam(net.parameters(),lr=.001)
   lp=net(v['targets'],v['actions'],v['coverage'],v['regions']);assert abs(float(lp.exp().sum())-1)<1e-5
   if len(lp)>1:
    loss=-lp[-1];loss.backward();assert any(x.grad is not None and float(x.grad.abs().sum())>0 for x in net.region.parameters());assert any(x.grad is not None and float(x.grad.abs().sum())>0 for x in net.rule.parameters());opt.step();c['updates']+=1
   records.append(dict(group=group,regions=len(steps),options=len(cols),native_sequence_error=float((native-old['sequences'][0]).abs().max()),composition_order_error=float((seq-reverse).abs().max()),checks='passed'))
  assert_frozen(teacher,th);result.update(status='passed',synthetic=checks,checks=records,official_victim_loaded=False)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,protocol_sha256=sha(s/'protocol.json'));write(out/'result.json',result);print(result,flush=True)
if __name__=='__main__':main()
