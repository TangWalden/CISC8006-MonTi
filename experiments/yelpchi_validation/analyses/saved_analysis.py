"""Saved JSON interpretation only; no model/tensor loading or new predictions."""
import pathlib,json,hashlib,time,statistics,collections
root=pathlib.Path(__file__).resolve().parent
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def wrong(r):
    # Torch argmax selects the first class on an exact two-class tie.
    return [z[0]>=z[1] for z in r['gaga_logits']]
def main():
    start=time.monotonic();cpu=time.process_time();s=root/'completed'
    assert read(s/'guardian_receipt.json')['status']=='completed'
    assert read(s/'saved_audit.json')['status']=='passed'
    p=read(s/'protocol.json');rows=list(map(json.loads,(s/'main/evaluation.jsonl').read_text().splitlines()))
    m={(r['seed'],r['group'],r['arm']):r for r in rows}
    assert len(m)==len(rows)==len(p['seeds'])*len(p['valid_groups'])*len(p['arms'])
    output={'scope':'Saved-only interpretation after registered results; target occurrences repeat across noise runs. Native means author attack, not a clean graph. No IID occurrence inference.','input_sha256':{n:sha(s/n) for n in ['protocol.json','main/evaluation.jsonl','main/result.json','case_diagnosis.json']},'source_sha256':sha(__file__),'arms':{},'contrasts':{},'model_calls':0,'tensor_loads':0,'GPU_hours':0}
    for arm in p['arms']:
        rr=[r for r in rows if r['arm']==arm]
        assert all(sum(wrong(r))==r['wrong'] for r in rr)
        output['arms'][arm]={'pooled_wrong':sum(r['wrong'] for r in rr),'occurrences':sum(len(r['node_ids']) for r in rr),'group_equal_MR':100*statistics.mean(r['wrong']/len(r['node_ids']) for r in rr),'modified_cases':sum(bool(r['actions']) for r in rr),'actual_exchanged_edges':sum(r['exchanged_edges'] for r in rr)}
    contrasts=[('rewriter',b) for b in ['native','single_neural','single_direct','random_endpoint','random_joint','greedy','independent']]+[('greedy','independent'),('single_neural','random_single'),('greedy','random_greedy'),('independent','random_independent')]
    for a,b in contrasts:
        gains=losses=0;gain_ids=set();loss_ids=set();cases=[];per_seed={}
        for seed in p['seeds']:
            sg=sl=0
            for group in p['valid_groups']:
                x,y=m[seed,group,a],m[seed,group,b];assert x['node_ids']==y['node_ids']
                assert x['actual_nodes']==y['actual_nodes'] and x['actual_edges']==y['actual_edges']
                xx,yy=wrong(x),wrong(y);gg=[n for n,u,v in zip(x['node_ids'],xx,yy) if u and not v];ll=[n for n,u,v in zip(x['node_ids'],xx,yy) if v and not u]
                sg+=len(gg);sl+=len(ll);gain_ids.update(gg);loss_ids.update(ll)
                cases.append({'seed':seed,'group':group,'targets':len(xx),'method_wrong':sum(xx),'control_wrong':sum(yy),'net_wrong':len(gg)-len(ll),'gain_node_ids':gg,'loss_node_ids':ll,'method_exchanged_edges':x['exchanged_edges'],'control_exchanged_edges':y['exchanged_edges']})
            gains+=sg;losses+=sl;per_seed[str(seed)]={'gains':sg,'losses':sl,'delta_pp':100*(sg-sl)/p['valid_occurrences']}
        ordered=sorted(cases,key=lambda r:(r['net_wrong'],r['seed'],r['group']))
        output['contrasts'][a+'__'+b]={'gains':gains,'losses':losses,'net':gains-losses,'unique_gained_targets':len(gain_ids),'unique_lost_targets':len(loss_ids),'targets_with_both_gain_and_loss':len(gain_ids&loss_ids),'wins':sum(r['net_wrong']>0 for r in cases),'ties':sum(r['net_wrong']==0 for r in cases),'loss_cases':sum(r['net_wrong']<0 for r in cases),'identical_class_predictions':gains+losses==0,'per_seed':per_seed,'illustrative_extrema_after_outcomes':[ordered[0],ordered[-1]],'example_caution':'Outcome-selected extrema are illustrative, not representative samples or causal explanations.'}
    output['random_matching']=dict(root_authored_tensor_audit=read(s/'saved_audit.json'),group_diagnosis_summary={k:v for k,v in read(s/'case_diagnosis.json').items() if k!='cases'})
    output['wall_seconds']=time.monotonic()-start;output['process_cpu_seconds']=time.process_time()-cpu
    dest=root/'saved_analysis.json';assert not dest.exists()
    with dest.open('x',encoding='utf-8',newline='\n') as f:json.dump(output,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'completed_saved_only','output_sha256':sha(dest),'model_calls':0,'tensor_loads':0}))
if __name__=='__main__':main()
