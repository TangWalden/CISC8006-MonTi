"""Post-outcome saved-only dose/transfer diagnosis; no graph classifier calls."""
import pathlib,json,argparse,time,statistics
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',required=True);s=pathlib.Path(p.parse_args().stage)
 read=lambda p:json.loads(p.read_text());assert read(s/'guardian_receipt.json')['exit_code']==0
 result=read(s/'main/result.json');assert result['status']=='completed';assert read(s/'saved_audit.json')['status']=='passed'
 import torch,numpy as np
 start=time.monotonic();cpu=time.process_time();r=s/'main';refs={tuple(x['key']):x for x in read(r/'card_seal.json')['cards']}
 rows=[json.loads(x) for x in (r/'evaluation.jsonl').read_text().splitlines()];rowmap={(x['seed'],x['group'],x['arm']):x for x in rows}
 plans=read(r/'sealed_before_victim.json')['plans'];records=[];summaries=[]
 for plan in plans:
  seed,group=plan['seed'],plan['group'];v=torch.load(r/refs[seed,'valid',group]['path'],map_location='cpu');base=rowmap[seed,group,'native']
  for mode in ['neural','direct']:
   choice=plan['selected'][mode];row=rowmap[seed,group,mode];control=rowmap[seed,group,'random_'+mode]
   native=np.asarray(base['gaga_logits']);new=np.asarray(row['gaga_logits']);tn=np.asarray(base['teacher_logits']);tc=np.asarray(row['teacher_logits']);delta=new[:,1]-new[:,0]-(native[:,1]-native[:,0]);td=tc[:,1]-tc[:,0]-(tn[:,1]-tn[:,0]);seq=v['sequences'][choice]-v['sequences'][0]
   records.append(dict(seed=seed,group=group,mode=mode,active=bool(choice),targets=len(native),gained=int(((new.argmax(1)==0)&(native.argmax(1)==1)).sum()),lost=int(((new.argmax(1)==1)&(native.argmax(1)==0)).sum()),native_wrong=base['wrong'],new_wrong=row['wrong'],random_wrong=control['wrong'],teacher_wrong_delta=row['teacher_wrong']-base['teacher_wrong'],dose_changed_target_count=int((seq.abs().amax((1,2))>1e-6).sum()),per_token_max_abs_change=seq.abs().amax((0,2)).tolist(),victim_abs_delta_margin=np.abs(delta).tolist(),native_abs_margin=np.abs(native[:,1]-native[:,0]).tolist(),signed_teacher_margin_delta=td.tolist(),signed_victim_margin_delta=delta.tolist(),opposite_direction=int(((td*delta)<0).sum())))
 for mode in ['neural','direct']:
  rr=[x for x in records if x['mode']==mode];active=[x for x in rr if x['active']];flat=lambda name:[a for x in active for a in x[name]]
  summaries.append(dict(mode=mode,active_cases=len(active),active_target_occurrences=sum(x['targets'] for x in active),dose_changed_target_occurrences=sum(x['dose_changed_target_count'] for x in active),gained=sum(x['gained'] for x in rr),lost=sum(x['lost'] for x in rr),teacher_wrong_net=sum(x['teacher_wrong_delta'] for x in rr),native_wrong=sum(x['native_wrong'] for x in rr),new_wrong=sum(x['new_wrong'] for x in rr),random_wrong=sum(x['random_wrong'] for x in rr),median_abs_victim_margin_change_on_active=float(np.median(flat('victim_abs_delta_margin'))) if active else None,median_abs_native_victim_margin_on_active=float(np.median(flat('native_abs_margin'))) if active else None,opposite_teacher_victim_direction_occurrences=sum(x['opposite_direction'] for x in active),max_change_by_token=[max(x['per_token_max_abs_change'][i] for x in rr) for i in range(7)]))
 training=[json.loads(x) for x in (r/'training.jsonl').read_text().splitlines()]
 trainmeans=[dict(seed=seed,epoch=epoch,mean_listwise_loss=statistics.mean(x['loss'] for x in training if x['seed']==seed and x['epoch']==epoch),scope='Teacher-target entropy varies across cases; loss alone is not attack efficacy.') for seed in [0,1,2] for epoch in [1,8]]
 out=dict(status='completed',scope='Saved post-hoc development dose/transfer diagnosis. No classifier calls, parameter loads, TEST queries or GPU. Occurrences correlated; active-only magnitude summaries are not full-cohort MR.',summary=summaries,train_loss_endpoints=trainmeans,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu,graph_classifier_calls=0,parameter_loads=0,GPU_hours=0)
 with (s/'saved_effect_diagnosis.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
