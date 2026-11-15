import pathlib,json,hashlib,time
s=pathlib.Path('/home/tjy/Project/MonTi-course/runs/target_rewriter_20261114_attempt1');assert json.loads((s/'guardian_receipt.json').read_text())['status']=='completed';start=time.monotonic();cpu=time.process_time();f=s/'main/evaluation.jsonl';raw=f.read_bytes();rows=[json.loads(x) for x in raw.splitlines()];ix={(r['seed'],r['group'],r['arm']):r for r in rows};plans=json.loads((s/'case_diagnosis.json').read_text())['cases'];result={}
def wrong(r):return [z[0]>=z[1] for z in r['gaga_logits']]
for mode,control in [('rewriter','single_neural'),('rewriter','native'),('rewriter','random_endpoint'),('rewriter','random_joint'),('greedy','independent'),('greedy','rewriter')]:
 gains=[];losses=[];classdiff=0
 for case in plans:
  a=ix[case['seed'],case['group'],mode];b=ix[case['seed'],case['group'],control];assert a['node_ids']==b['node_ids']
  for node,wa,wb in zip(a['node_ids'],wrong(a),wrong(b)):
   classdiff+=wa!=wb
   if wa and not wb:gains.append([case['seed'],case['group'],node])
   if wb and not wa:losses.append([case['seed'],case['group'],node])
 result[mode+'__'+control]=dict(gained_occurrences=len(gains),lost_occurrences=len(losses),unique_gained_targets=len(set(v[2] for v in gains)),unique_lost_targets=len(set(v[2] for v in losses)),class_disagreements=classdiff,case_gains=gains,case_losses=losses)
result['random_region_feasibility']=dict(cases=len(plans),active=sum(r['slots']>0 for r in plans),degenerate_all=sum(r['region_random_feasible_subsets']==1 for r in plans),degenerate_active=sum(r['slots']>0 and r['region_random_feasible_subsets']==1 for r in plans))
result['decision_context_totals']={k:sum(r['teacher_queries'][k] for r in plans) for k in plans[0]['teacher_queries']}
result['scope']='Posthoc saved JSON analysis, no model/tensor/dataset loads. Decision context totals include reused native evidence and are not measured speedups. Unique IDs overlap across repeats.'
result['input_sha256']=hashlib.sha256(raw).hexdigest();result['wall_seconds']=time.monotonic()-start;result['process_cpu_seconds']=time.process_time()-cpu;print(json.dumps(result))