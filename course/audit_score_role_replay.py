"""Recount captured roles, ties and device comparisons without model inference."""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def list_sha(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='evidence/score_role_replay_audit_v2.json')
    args=parser.parse_args()
    started=time.monotonic()
    folder=ROOT/'evidence/score_role_replay_v2'
    registered,summary,guard=read(folder/'registered.json'),read(folder/'summary.json'),read(folder/'guardian.json')
    protocol=read(ROOT/registered['protocol'])
    assert summary['status']=='completed' and guard['exit_code']==0
    assert sha(ROOT/registered['protocol'])==registered['protocol_sha256']
    for path,wanted in read(folder/'output_checksums.json').items():
        assert sha(folder/path)==wanted,path
    snapshot_state_checks=[]
    for path,wanted in read(folder/'verified_inputs.json').items():
        if path=='evidence/research_cost_ledger.json' and sha(ROOT/path)!=wanted:
            blob=subprocess.check_output(['git','show',registered['source_commit']+':'+path],cwd=ROOT)
            assert hashlib.sha256(blob).hexdigest()==wanted,path
            snapshot_state_checks.append(path)
        else:
            assert sha(ROOT/path)==wanted,path
    assert guard['guarded_elapsed_seconds']>=summary['elapsed_seconds']
    assert max(guard['guarded_elapsed_seconds'],summary['elapsed_seconds'])/3600<=protocol['gpu_hour_limit']
    raw_path=folder/'raw/measurements.jsonl'
    assert sha(raw_path)==summary['raw_sha256']
    events=records(raw_path)
    starts=[r for r in events if r['event']=='gcn_started']
    ends=[r for r in events if r['event']=='gcn_completed']
    gs=[r for r in events if r['event']=='generation_started']
    ge=[r for r in events if r['event']=='generation_completed']
    assert [r['call'] for r in starts]==[r['call'] for r in ends]==list(range(1,225))
    assert [r['generation'] for r in gs]==[r['generation'] for r in ge]==list(range(1,97))
    identities={(r['attacker_seed'],r['target_group']):r for r in events if r['event']=='identity'}
    assert len(identities)==48 and len(ge)==96 and len(ends)==224
    source={}
    for run in protocol['runs']:
        for row in records(ROOT/run['source_prediction']['path']):
            if row['target_group'] in protocol['groups']:
                source[run['seed'],row['target_group']]=row
    for key,row in identities.items():
        assert row['node_ids']==source[key]['node_ids'] and row['labels']==source[key]['labels']
        assert row['source_predictions']==source[key]['predictions']
    previous=records(ROOT/'evidence/one_edge_diagnosis_v2/raw/predictions.jsonl')
    previous={(r['attacker_seed'],r['target_group']):r for r in previous}
    captured={(r['public']['attacker_seed'],r['public']['target_group'],r['public']['generator']):r for r in ge}
    public_generators={(int(r['attacker_seed']),int(r['target_group']),r['generator']):r for r in table(folder/'generators.csv')}
    tie_cases=reference_disagreements=role_cases=0
    for key,row in captured.items():
        p,m=row['public'],row['private']
        assert set(p)==set(public_generators[key])
        for name,value in p.items():
            assert str(value)==public_generators[key][name],(key,name)
        t,c=p['targets'],p['candidates']
        assert len(m['target_order'])==t and len(m['candidate_order'])==c
        assert m['actual_src_indices']==[0]*(t+c) and m['actual_dst_indices']==list(range(t+c))
        assert p['start_atk']==t+c and p['executed_source_index']==0 and p['paper_source_index']==t+c
        assert p['executed_source_role']=='first_target' and p['physical_source_role']=='injected'
        assert list_sha(m['target_order'])==p['target_order_sha256']
        assert list_sha(m['candidate_order'])==p['candidate_order_sha256']
        assert list_sha(sorted(m['candidate_order']))==p['candidate_set_sha256']
        normalized=np.asarray(m['normalized'],dtype=np.float32)
        actual=np.asarray(m['actual_raw_scores'],dtype=np.float32)
        soft=np.asarray(m['actual_soft_scores'],dtype=np.float32)
        top=np.asarray(m['actual_topk_input'],dtype=np.float32)
        assert normalized.shape[0]==t+c+1 and top.shape==(1,t)
        assert np.isfinite(normalized).all() and np.isfinite(actual).all() and np.isfinite(top).all()
        assert np.array_equal(soft[:t],top[0])
        assert np.allclose((normalized[0:1]*normalized[:t+c]).sum(1),actual,atol=1e-5,rtol=1e-6)
        for vector,prefix in [(actual[:t],'raw'),(top[0],'soft')]:
            exact=int((vector==vector.max()).sum())
            assert exact==p[prefix+'_exact_maxima']
            assert int((vector.max()-vector<=1e-6).sum())==p[prefix+'_near_max_1e_minus6']
            assert int((vector.max()-vector<=1e-5).sum())==p[prefix+'_near_max_1e_minus5']
        selected=p['selected_endpoint_position']
        assert top[0,selected]==top.max()
        assert top[0,p['same_probability_cpu_topk']]==top.max() and top[0,p['same_probability_gpu_topk']]==top.max()
        assert p['same_probability_device_topk_disagrees']==(p['same_probability_cpu_topk']!=p['same_probability_gpu_topk'])
        expected_native=p['same_probability_cpu_topk'] if key[2]=='cpu' else p['same_probability_gpu_topk']
        assert p['same_device_reference_matches_executed']==(expected_native==selected)
        role_cases+=1
        tie_cases+=p['soft_exact_maxima']>1
        reference_disagreements+=p['same_probability_device_topk_disagrees']
    clean={}
    attacked={}
    for row in ends:
        ids=identities[row['attacker_seed'],row['target_group']]['node_ids']
        assert row['alignment']['output_order_sha256']==row['alignment']['block_dst_order_sha256']==list_sha(ids)
        value=np.asarray(row['logits'],dtype=np.float32)
        assert np.isfinite(value).all() and value.shape==(len(ids),2)
        if row['phase']=='clean':
            clean[row['target_group'],row['detector']]=value
        else:
            key=row['attacker_seed'],row['target_group'],row['generator'],row['detector']
            assert key not in attacked
            attacked[key]=value
    assert len(clean)==32 and len(attacked)==192
    drows=table(folder/'detectors.csv')
    matrix=collections.defaultdict(collections.Counter)
    endpoint_attribution=collections.defaultdict(collections.Counter)
    for row in drows:
        seed,group=int(row['attacker_seed']),int(row['target_group'])
        generator,detector=row['generator'],row['detector']
        value=attacked[seed,group,generator,detector]
        labels=np.asarray(identities[seed,group]['labels'])
        predicted=value.argmax(1)
        clean_pred=clean[group,detector].argmax(1)
        correct=predicted==labels
        cc=clean_pred==labels
        assert int(row['residuals'])==int(correct.sum())
        assert int(row['clean_correct'])==int(cc.sum())
        assert int(row['clean_correct_to_wrong'])==int((cc & ~correct).sum())
        assert int(row['clean_wrong_to_correct'])==int((~cc & correct).sum())
        source_match=predicted.tolist()==source[seed,group]['predictions']
        disagreements=sum(a!=b for a,b in zip(predicted.tolist(),source[seed,group]['predictions']))
        old_match=bool(np.array_equal(predicted,np.asarray(previous[seed,group]['reconstructed_logits']).argmax(1)))
        assert (row['source_prediction_match']=='True')==source_match
        assert int(row['source_prediction_disagreements'])==disagreements
        assert (row['previous_cpu_prediction_match']=='True')==old_match
        aggregate=matrix[generator,detector]
        aggregate.update(dict(cases=1,target_occurrences=len(labels),residuals=int(correct.sum()),
            source_concordant_cases=int(source_match),source_prediction_disagreements=disagreements,previous_cpu_concordant_cases=int(old_match)))
        selected=captured[seed,group,generator]['public']['selected_endpoint_position']
        attribution=endpoint_attribution[generator,detector]
        attribution['cases']+=1
        attribution['selected_endpoint_clean_wrong']+=int(not cc[selected])
        attribution['selected_clean_wrong_with_clean_correct_alternative']+=int(not cc[selected] and cc.any())
        attribution['new_flips']+=int((cc & ~correct).sum())
        attribution['new_flips_away_from_selected_endpoint']+=int(sum(index!=selected for index in np.flatnonzero(cc & ~correct)))
    for row in summary['matrix']:
        assert dict(matrix[row['generator'],row['detector']])=={key:value for key,value in row.items() if key not in ['generator','detector']}
    assert role_cases==summary['source_role_first_target_cases']==96
    assert tie_cases==summary['actual_soft_multiple_exact_maxima']==63
    assert reference_disagreements==summary['same_probability_device_topk_disagreement_cases']==32
    contrasts={}
    for generator in ['cpu','gpu']:
        rows=[(attacked[s,g,generator,'cpu'],attacked[s,g,generator,'gpu']) for s,g in identities]
        contrasts[generator]=dict(cases=48,prediction_disagreements=sum(int((a.argmax(1)!=b.argmax(1)).sum()) for a,b in rows),
                                  maximum_logit_difference=max(float(np.abs(a-b).max()) for a,b in rows))
    same_vectors=different_endpoints_with_same_vectors=0
    for seed,group in identities:
        left,right=captured[seed,group,'cpu'],captured[seed,group,'gpu']
        same=np.array_equal(np.asarray(left['private']['actual_topk_input'],dtype=np.float32),np.asarray(right['private']['actual_topk_input'],dtype=np.float32))
        same_vectors+=int(same)
        different_endpoints_with_same_vectors+=int(same and left['public']['selected_endpoint_position']!=right['public']['selected_endpoint_position'])
    result=dict(status='passed',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),script_sha256=sha(pathlib.Path(__file__)),
        gcn_forwards=224,generator_forwards=96,algebra_reference_topk_calls=192,group_checkpoint_cases=48,
        raw_sha256=summary['raw_sha256'],observed_first_target_score_source_cases=role_cases,actual_probability_multiple_maxima_cases=tie_cases,
        mutable_state_checked_against_registered_git_snapshot=snapshot_state_checks,
        same_vector_reference_device_choice_disagreements=reference_disagreements,
        generator_pair_identical_actual_probability_vectors=same_vectors,
        generator_pair_endpoint_differences_with_identical_actual_probability_vectors=different_endpoints_with_same_vectors,
        detector_device_contrasts=contrasts,
        endpoint_attribution=[dict(generator=key[0],detector=key[1],**value) for key,value in endpoint_attribution.items()],
        charged_gpu_wall_hours=max(guard['guarded_elapsed_seconds'],summary['elapsed_seconds'])/3600,
        additional_model_queries=0,audit_gpu_hours=0,elapsed_seconds=time.monotonic()-started,
        limitations='Recount of retained tensors only. Graph transfer identity, exact loaded-weight equality and hook neutrality are source-reviewed adapter checks, not independently regenerated here. Prediction concordance does not establish historical perturbation identity; no correction performance evaluated.')
    with (ROOT/args.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
