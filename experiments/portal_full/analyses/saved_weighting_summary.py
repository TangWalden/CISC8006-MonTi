"""POST HOC weighting sensitivity on completed saved full-cohort predictions."""
import pathlib,json,hashlib,time,argparse,statistics,collections
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s=pathlib.Path(ap.parse_args().stage);f=s/'main/evaluation.jsonl';assert hashlib.sha256(f.read_bytes()).hexdigest()=='10eec2d1d24e0a6d435e8da2d5d4808be1a015fbb2122c5b82f372fbd71c4621';assert json.loads((s/'offline_audit_receipt.json').read_text())['status']=='passed'
 start=time.monotonic();cpu=time.process_time();rows=[json.loads(x) for x in f.read_text().splitlines()];arms=['native','neural','random_neural','direct','random_direct'];metrics=[]
 for seed in [0,1,2]:
  for arm in arms:
   rr=[x for x in rows if x['seed']==seed and x['arm']==arm];occ=collections.defaultdict(list);macro=[]
   for row in rr:
    wrong=[int(z[0]>=z[1]) for z in row['gaga_logits']];assert sum(wrong)==row['wrong'];macro.append(sum(wrong)/len(wrong))
    for node,w in zip(row['node_ids'],wrong):occ[node].append(w)
   assert len(rr)==909 and len(occ)==1221 and sum(map(len,occ.values()))==10515
   metrics.append(dict(seed=seed,arm=arm,occurrence_MR=100*sum(sum(x) for x in occ.values())/10515,group_equal_MR=100*statistics.mean(macro),target_equal_context_average_MR=100*statistics.mean(statistics.mean(x) for x in occ.values())))
 means={arm:{key:statistics.mean(x[key] for x in metrics if x['arm']==arm) for key in ['occurrence_MR','group_equal_MR','target_equal_context_average_MR']} for arm in arms};deltas={mode:{control:{key:means[mode][key]-means[control][key] for key in means[mode]} for control in ['native','random_'+mode]} for mode in ['neural','direct']}
 out=dict(status='completed',classification='POST HOC; overall occurrence results already known before source registration. Not preregistration or a replacement primary metric.',input_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),metrics=metrics,means=means,descriptive_deltas_pp=deltas,definitions=dict(occurrence_MR='All10515occurrences equalweight perrepeat; frozen primary.',group_equal_MR='Each909scenario group equalweight, mean within-group MR.',target_equal_context_average_MR='Each1221unique target equalweight after averaging its scenario-specific outcomes; not one deduplicated graph/prediction.'),scope='Correlated memberships and adaptive history remain; weighting summaries are descriptive. Zero graph/checkpoint/model/API/TEST queries; saved TEST predictions read.',wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,GPU_hours=0,model_calls=0)
 with (s/'saved_weighting_summary.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
