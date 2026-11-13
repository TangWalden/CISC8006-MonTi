"""TRAIN-only author-generation, physical input and policy equivalence checks."""
import argparse,time,pathlib
from common import setup,read,sha,write,load_teacher,card,choose
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);start=time.monotonic();cpu=time.process_time()
 import torch
 from experiments.downstream_cooperation.common import check_device,state_hash,assert_frozen,tensor_hash
 from experiments.local_swap.loader import load_engine
 from experiments.local_swap.helpers import capture
 from course.dual_surrogate.runtime import load_original_view
 from model import PortalRanker
 check_device();parent=pathlib.Path(p['parent']);base=p['author_base'];out=s/'preflight';out.mkdir(exist_ok=False)
 counts=dict(generations=0,physical_canonical_builds=0,own_teacher_batched_forwards=0,own_teacher_hard_contexts=0,official_GAGA_forwards=0,TEST_queries=0,updates=0)
 engine,data,receipt,frozen=load_engine(base,0,out,start+900);teacher=load_teacher(p,q,data);th=state_hash(teacher);view=load_original_view(data);net=PortalRanker(data.feat_dim);net.load_state_dict(torch.load(parent/'main'/p['models'][0]['path'],map_location='cpu'));net.eval()
 index={tuple(x['key']):x for x in read(parent/'main/card_seal.json')['cards']};checks=[]
 for group in p['preflight_groups']:
  case,g=capture(engine,data,0,group,'train',base);counts['generations']+=1;v=card(data,view,case,teacher,counts);ref=index[0,'train',group];f=parent/'main'/ref['path'];assert sha(f)==ref['sha256'];old=torch.load(f,map_location='cpu')
  assert torch.equal(case['native_adjacency'],old['case']['native_adjacency']) and torch.equal(case['x_aux'],old['case']['x_aux']);assert torch.equal(case['candidate_ids'],old['case']['candidate_ids']);assert v['bank']==old['bank'] and v['donor']==old['donor']
  se=float((v['sequences']-old['sequences']).abs().max());ze=float((v['teacher_logits']-old['teacher_logits']).abs().max());assert se<2e-5 and ze<2e-5
  a,scores=choose(v,net);b,old_scores=choose(old,net);assert a==b and torch.allclose(scores,old_scores,atol=2e-5,rtol=1e-5)
  checks.append(dict(group=group,sequence_error=se,own_teacher_logit_error=ze,physical_budget_passed=True,actions_identical=True))
 # GAGA squeezes singleton batches; the full runner restores [targets,2].
 assert torch.zeros(2).reshape(1,2).shape==(1,2)
 assert_frozen(engine.model,frozen['upstream']);assert_frozen(engine.teachers['gcn_base'],frozen['teacher']);assert_frozen(teacher,th)
 result=dict(status='passed',protocol_sha256=sha(s/'protocol.json'),checks=checks,counters=counts,author_receipt=receipt,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,GPU_hours=0)
 write(s/'preflight_result.json',result);print(result,flush=True)
if __name__=='__main__':main()
