"""Offline join of clean correctness and reconstructed endpoint choice."""
import collections
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    folder = ROOT/'evidence/one_edge_diagnosis_v2'
    summary = json.loads((folder/'summary.json').read_text())
    path = folder/'raw/predictions.jsonl'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==summary['raw_prediction_sha256']
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(raw)==48
    counts = {seed:collections.Counter() for seed in [0,1,2]}
    for row in raw:
        total = counts[row['attacker_seed']]
        labels = row['labels']
        predicted = lambda logits:[max(range(len(x)),key=x.__getitem__) for x in logits]
        clean = predicted(row['clean_logits'])
        actual = predicted(row['reconstructed_logits'])
        position = row['node_ids'].index(row['endpoint_node_id'])
        chosen_clean_wrong = clean[position]!=labels[position]
        actual_wrong = sum(a!=b for a,b in zip(actual,labels))
        best_wrong = max(endpoint['incorrect'] for endpoint in row['counterfactuals'])
        headroom = best_wrong-actual_wrong
        total['cases']+=1
        total['selected_endpoint_clean_wrong_cases']+=chosen_clean_wrong
        total['selected_endpoint_clean_correct_cases']+=not chosen_clean_wrong
        if headroom>0:
            total['positive_headroom_cases']+=1
            total['positive_cases_selected_endpoint_clean_wrong']+=chosen_clean_wrong
            total['positive_cases_selected_endpoint_clean_correct']+=not chosen_clean_wrong
            total['net_oracle_extra_wrong']+=headroom
        for endpoint in row['counterfactuals']:
            predicted_endpoint = predicted(endpoint['logits'])
            fresh = [index for index in range(len(labels)) if clean[index]==labels[index] and predicted_endpoint[index]!=labels[index]]
            repaired = [index for index in range(len(labels)) if clean[index]!=labels[index] and predicted_endpoint[index]==labels[index]]
            assert len(fresh)==endpoint['clean_correct_to_wrong']
            total['endpoint_configurations']+=1
            total['configurations_with_new_flips']+=bool(fresh)
            total['configurations_with_multiple_new_flips']+=len(fresh)>1
            total['configurations_with_new_flip_away_from_endpoint']+=any(index!=endpoint['endpoint_position'] for index in fresh)
            total['configurations_with_clean_wrong_to_correct']+=bool(repaired)
            total['configurations_clean_wrong_endpoint_but_new_flip']+=(clean[endpoint['endpoint_position']]!=labels[endpoint['endpoint_position']] and bool(fresh))
    fields=sorted(set().union(*(set(value) for value in counts.values())))
    totals={key:sum(value[key] for value in counts.values()) for key in fields}
    assert totals['endpoint_configurations']==243 and totals['positive_headroom_cases']==7
    result=dict(status='completed',analysis='Post-hoc offline clean-correctness/selected-endpoint join; no new inference',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                raw_prediction_sha256=digest,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                totals=totals,per_attacker_seed=[dict(attacker_seed=seed,**{key:counts[seed][key] for key in fields}) for seed in [0,1,2]],
                additional_model_queries=0,gpu_hours=0,elapsed_seconds=time.monotonic()-started,
                limitations='Conditional on reconstructed selected-cohort features and graphs; not attribution to original perturbations or a universal capacity bound. Clean GCN correctness is not GAGA correctness. No improvement selected.')
    with (ROOT/'evidence/one_edge_endpoint_attribution_v1.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
