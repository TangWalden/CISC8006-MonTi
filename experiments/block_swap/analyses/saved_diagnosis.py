"""Planned descriptive saved-output analysis; requires completed science.

No imports of ML libraries, no dataset/parameter/tensor loads or model calls.
Examples are explicitly outcome-selected for explanation, never selection.
"""
import argparse,hashlib,json,math,pathlib,time
def read(p):return json.loads(pathlib.Path(p).read_text())
def softplus(z):return max(z,0)+math.log1p(math.exp(-abs(z)))
def loss(rows):return sum(softplus(b-a) if b-a<=20 else b-a for a,b in rows)/len(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();root=pathlib.Path(a.stage)
    start=time.monotonic();cpu=time.process_time();r=read(root/'main/result.json');assert r['status']=='completed'
    p=read(root/'registration/protocol.json');seal=read(root/'main/sealed_before_victim.json')
    clean={};cases={};rows=[]
    for line in (root/'main/raw/evaluation.jsonl').read_text().splitlines():
        v=json.loads(line)
        if v['stage']=='clean':clean[v['group']]=v
        else:cases[v['seed'],v['group'],v['arm']]=v
    for s in p['seeds']:
        for control in ['native','matched_random','always_random']:
            wins=ties=losses=gained=lost=0;examples=[]
            for g in p['valid_groups']:
                l=cases[s,g,'block_rank'];c=cases[s,g,control];assert l['node_ids']==c['node_ids']
                lp=[int(b>a) for a,b in l['gaga_logits']];cp=[int(b>a) for a,b in c['gaga_logits']]
                gain=sum(a==0 and b==1 for a,b in zip(lp,cp));lose=sum(a==1 and b==0 for a,b in zip(lp,cp));delta=gain-lose
                gained+=gain;lost+=lose;wins+=delta>0;losses+=delta<0;ties+=delta==0
                examples.append(dict(group=g,targets=len(lp),learned_wrong=sum(x==0 for x in lp),control_wrong=sum(x==0 for x in cp),gained=gain,lost=lose,net=delta))
            rows.append(dict(seed=s,control=control,group_wins=wins,group_ties=ties,group_losses=losses,gross_gained_occurrences=gained,gross_lost_occurrences=lost,net_occurrences=gained-lost,
                outcome_selected_positive_examples=sorted([v for v in examples if v['net']>0],key=lambda v:(-v['net'],v['group']))[:3],
                outcome_selected_negative_examples=sorted([v for v in examples if v['net']<0],key=lambda v:(v['net'],v['group']))[:3]))
    identity=dict(equal_topology_comparisons=0,gcn_class_differences=0,gaga_class_differences=0,max_gcn_logit_difference=0.,max_gaga_logit_difference=0.)
    for s in p['seeds']:
        for g in p['valid_groups']:
            for i,aa in enumerate(p['arms']):
                for bb in p['arms'][i+1:]:
                    a,b=cases[s,g,aa],cases[s,g,bb]
                    if a['topology_hash']!=b['topology_hash']:continue
                    assert a['feature_hash']==b['feature_hash'];identity['equal_topology_comparisons']+=1
                    for model in ('gcn','gaga'):
                        av,bv=a[model+'_logits'],b[model+'_logits']
                        identity[model+'_class_differences']+=sum((x[1]>x[0])!=(y[1]>y[0]) for x,y in zip(av,bv))
                        identity['max_'+model+'_logit_difference']=max(identity['max_'+model+'_logit_difference'],max(abs(x-y) for xr,yr in zip(av,bv) for x,y in zip(xr,yr)))
    gating=[];teacher=[]
    for s in p['seeds']:
        plans=[v for v in seal['plans'] if v['seed']==s];active=[v for v in plans if v['decision']['active']]
        gating.append(dict(seed=s,all_groups=len(plans),eligible_groups=sum(bool(v['pool']) for v in plans),active_groups=len(active),
            active_target_occurrences=sum(len(p['valid_members'][str(v['group'])]) for v in active),
            all_target_occurrences=571,all_unique_targets=240))
        scores=[];gains=[];net=[]
        for v in active:
            g=v['group'];l=cases[s,g,'block_rank'];b=cases[s,g,'native'];scores.append(max(v['decision']['predicted_gains']))
            gains.append(loss(b['gcn_logits'])-loss(l['gcn_logits']));net.append(l['gaga_wrong']-b['gaga_wrong'])
        teacher.append(dict(seed=s,active=len(active),actual_GCN_loss_improves=sum(v>1e-6 for v in gains),actual_GCN_loss_worsens=sum(v< -1e-6 for v in gains),
            mean_predicted_gain=sum(scores)/len(scores) if scores else None,mean_actual_GCN_gain=sum(gains)/len(gains) if gains else None,
            GAGA_net_wrong_on_active=sum(net)))
    parent=pathlib.Path(p['parent_stage'])/'main/raw/training.jsonl'
    old={(v['seed'],v['group']):v for v in [json.loads(line) for line in parent.read_text().splitlines()]}
    fresh=[json.loads(line) for line in (root/'main/raw/training.jsonl').read_text().splitlines()]
    diffs=[];class_differences=0;max_logit=0.
    for v in fresh:
        before=old[v['seed'],v['group']];assert before['node_ids']==v['node_ids']
        av,bv=before['native_logits'],v['native_logits'];diffs.append(abs(loss(av)-loss(bv)))
        class_differences+=sum((x[1]>x[0])!=(y[1]>y[0]) for x,y in zip(av,bv))
        max_logit=max(max_logit,max(abs(x-y) for xr,yr in zip(av,bv) for x,y in zip(xr,yr)))
    native_repeatability=dict(cases=len(fresh),max_abs_loss_difference=max(diffs),mean_abs_loss_difference=sum(diffs)/len(diffs),max_abs_logit_difference=max_logit,class_differences=class_differences,
        scope='Same parent cached native graph, two recorded evaluations; descriptive numerical variability, not a formal error bound for every intervention.')
    output=dict(status='completed',analysis_scope='Descriptive saved-case diagnosis after the registered experiment; no new selection, tuning or causal attribution.',
        result_sha256=hashlib.sha256((root/'main/result.json').read_bytes()).hexdigest(),paired=rows,gating=gating,teacher_transfer=teacher,identity_controls=identity,native_GCN_repeatability=native_repeatability,
        model_queries=0,dataset_loads=0,checkpoint_loads=0,GPU_hours=0,process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-start,
        caution='Occurrences are correlated. Three repeats share one author checkpoint. Examples selected after outcomes do not define a new rule.')
    with (root/'saved_diagnosis.json').open('x',encoding='utf-8') as f:json.dump(output,f,indent=2);f.write('\n')
    print(json.dumps(output))
if __name__=='__main__':main()
