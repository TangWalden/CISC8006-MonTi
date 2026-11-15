"""TRAIN-only author identity/cached view and graph-budget checks."""
import argparse,pathlib,time,signal
from common import setup,load,write,sha
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);out=s/'native_preflight';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();counts=dict(upstream_generations=0,physical_builds=0,teacher_batches=0,teacher_contexts=0,official_GAGA=0,TEST_queries=0);result=dict(status='failed',counters=counts,GPU_hours=0)
 signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('native preflight deadline')));signal.alarm(900)
 try:
  import torch,importlib.util
  from adapter import load_upstream,capture
  from experiments.downstream_cooperation.common import assert_frozen,make_graph,canonical
  data,view,teacher,th,benign,mean,std=load(q);engine,eh=load_upstream(q,data);spec=importlib.util.spec_from_file_location('shared_portal_card',p['shared_sources']['portal_common.py']['path']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);records=[]
  cc=dict(physical_canonical_builds=0,own_teacher_batched_forwards=0,own_teacher_hard_contexts=0)
  for group in p['preflight_groups']:
   a,seq=capture(engine,data,view,0,'train',group,p);counts['upstream_generations']+=1
   b,seqb=capture(engine,data,view,0,'train',group,p);counts['upstream_generations']+=1
   assert torch.equal(a['x_aux'],b['x_aux']) and torch.equal(a['candidate_ids'],b['candidate_ids']) and torch.equal(a['native_adjacency'],b['native_adjacency']) and torch.equal(seq,seqb)
   v=mod.card(data,view,a,teacher,cc);error=float((seq-v['sequences'][0]).abs().max());assert error<2e-5
   assert torch.isfinite(v['teacher_logits']).all();records.append(dict(group=group,targets=len(a['ids']),injected_nodes=len(a['source_ids']),actual_edges=int(a['native_adjacency'].sum()),candidate_pool=len(v['bank']),canonical_error=error,author_repeat_identical=True))
  counts.update(physical_builds=cc['physical_canonical_builds'],teacher_batches=cc['own_teacher_batched_forwards'],teacher_contexts=cc['own_teacher_hard_contexts']);assert_frozen(engine.model,eh);assert_frozen(teacher,th);result.update(status='passed',checks=records,author_configuration_seed=2,author_node_budget=.05,author_edge_budget=.5,official_victim_loaded=False)
 finally:
  signal.alarm(0);result.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);write(out/'result.json',result);print(result,flush=True)
if __name__=='__main__':main()
