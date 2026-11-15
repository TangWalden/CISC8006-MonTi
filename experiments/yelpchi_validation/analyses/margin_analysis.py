"""Post-hoc saved-logit checks for the null YelpChi comparison; no inference."""
import pathlib,json,hashlib,time,statistics
root=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();cpu=time.process_time();s=root/'completed';p=json.loads((s/'protocol.json').read_text());rows=list(map(json.loads,(s/'main/evaluation.jsonl').read_text().splitlines()));m={(r['seed'],r['group'],r['arm']):r for r in rows};out={}
 for arm in p['arms']:
  if arm=='native':continue
  vals=[];teacher_before=teacher_after=class_changes=changed=0;examples=[]
  for seed in p['seeds']:
   for group in p['valid_groups']:
    a,b=m[seed,group,arm],m[seed,group,'native'];assert a['node_ids']==b['node_ids']
    for n,z,z0,t,t0 in zip(a['node_ids'],a['gaga_logits'],b['gaga_logits'],a['teacher_logits'],b['teacher_logits']):
     d=z[1]-z[0];d0=z0[1]-z0[0];change=d-d0;ld=max(abs(x-y) for x,y in zip(z,z0));changed+=ld>0;class_changes+=(d<=0)!=(d0<=0);teacher_before+=t0[0]>=t0[1];teacher_after+=t[0]>=t[1];toward=d0*change<0
     vals.append((ld,abs(change),abs(d0),abs(change)/abs(d0) if toward and d0 else 0.0))
     examples.append(dict(seed=seed,group=group,node_id=n,native_margin=d0,changed_margin=d,margin_delta=change,exchanged_edges=a['exchanged_edges']))
  assert len(vals)==690 and class_changes==0
  out[arm]=dict(logit_changed_occurrences=changed,class_changes=class_changes,max_abs_logit_delta=max(x[0] for x in vals),mean_abs_margin_delta=statistics.mean(x[1] for x in vals),max_abs_margin_delta=max(x[1] for x in vals),minimum_native_abs_margin=min(x[2] for x in vals),max_fraction_of_native_margin_closed=max(x[3] for x in vals),teacher_wrong_native=teacher_before,teacher_wrong_arm=teacher_after,examples_posthoc=[min(examples,key=lambda r:(abs(r['native_margin']),r['seed'],r['group'],r['node_id'])),max(examples,key=lambda r:abs(r['margin_delta']))])
 v=dict(status='posthoc_saved_logits_after_null_outcome',source_sha256=sha(__file__),input_sha256=sha(s/'main/evaluation.jsonl'),arms=out,scope='690 repeated occurrences,230unique targets. Margin is fraud minus benign logit. Examples are post-hoc extrema, not representative selection or causal proof. Native is the author attack, not a clean graph. Changes in logits with unchanged classes establish threshold non-crossing on these records, not a sole cause or a universal robustness bound.',model_calls=0,tensor_loads=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
 f=root/'margin_analysis.json';assert not f.exists()
 with f.open('x',encoding='utf-8',newline='\n') as stream:json.dump(v,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps(out['rewriter']))
if __name__=='__main__':main()
