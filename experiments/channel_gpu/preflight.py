"""Export fixed teachers, then test GPU on predetermined development cases."""
import argparse,pathlib,json,sys,time,hashlib
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--mode',choices=['export','check'],required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=json.loads((s/'protocol.json').read_text());sys.path.insert(0,str(pathlib.Path(p['CPU_full_stage'])/'source'))
 from full_common import setup,counts
 _,fp,loaded=setup(p['CPU_full_stage'],a.dataset);_,old,_,_,_,_,_,q,data,view,teacher,th,benign,common=loaded
 import torch
 if a.mode=='export':
  f=s/(a.dataset+'_teacher.pt');assert not f.exists();torch.save(teacher.state_dict(),f);mf=pathlib.Path(sys.modules[teacher.__class__.__module__].__file__);(s/(a.dataset+'_export.json')).write_text(json.dumps(dict(sha256=sha(f),feature_dim=data.feat_dim,model_source=str(mf),model_source_sha256=sha(mf),original_state_hash=th,TEST_calls=0),indent=2)+'\n');return
 from client import predictor
 import cr_core
 from gr_core import predict as cpu_predict
 from experiments.downstream_cooperation.common import assert_frozen
 gpu_predict=predictor(str(s),a.dataset);c=counts();checks=[];start=time.monotonic();cpu=time.process_time()
 for e in fp['datasets'][a.dataset]['preflight_entries']:
  f=pathlib.Path(fp['method_parent'])/a.dataset/e['path'];assert sha(f)==e['sha256'];ref=torch.load(f,map_location='cpu');times={};records={}
  for mode,predict in [('cpu',cpu_predict),('gpu',gpu_predict)]:
   cr_core.predict=predict;geo=cr_core.Geometry(data,view,ref['base'],ref['sequences']['native'],benign,c);began=time.monotonic();plan,traces=cr_core.search(geo,teacher,'channel',c,8,2);seq,err=geo.verify(plan,1e-5);z,_,gradient=predict(teacher,[seq],c,'boundary',True);times[mode]=time.monotonic()-began;records[mode]=(plan,z,gradient)
  ca,cz,cg=records['cpu'];ga,gz,gg=records['gpu'];le=float((cz-gz).abs().max());ge=float((cg-gg).abs().max());same=ca==ga==ref['plans']['channel'];checks.append(dict(seed=e['seed'],group=e['group'],plans_identical=same,CPU_plan=ca,GPU_plan=ga,max_logit_error=le,max_gradient_error=ge,seconds=times,passed=same and le<=2e-5 and ge<=2e-6));print('GPU_PREFLIGHT',a.dataset,len(checks),2,flush=True)
 assert_frozen(teacher,th);(s/(a.dataset+'_preflight.json')).write_text(json.dumps(dict(status='completed',passed=all(x['passed'] for x in checks),checks=checks,counters=c,wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,TEST_calls=0),indent=2)+'\n')
if __name__=='__main__':main()
