"""Recount diagnostic logits without loading any detector or making queries."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(path.read_text())


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def table(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='evidence/one_edge_diagnosis_v2')
    parser.add_argument('--output', default='evidence/one_edge_diagnosis_audit_v2.json')
    args = parser.parse_args()
    started = time.monotonic()
    folder = ROOT/args.directory
    registered, summary = load(folder/'registered.json'), load(folder/'summary.json')
    assert summary['status']=='completed' and summary['surrogate_forward_calls']==355
    protocol = load(ROOT/registered['protocol'])
    assert sha(ROOT/registered['protocol'])==registered['protocol_sha256']
    for name, wanted in load(folder/'output_checksums.json').items():
        assert sha(folder/name)==wanted, name
    for name, wanted in load(folder/'verified_inputs.json').items():
        assert sha(ROOT/name)==wanted, name
    raw = records(folder/'raw/predictions.jsonl')
    progress = records(folder/'raw/progress.jsonl')
    assert sha(folder/'raw/predictions.jsonl')==summary['raw_prediction_sha256']
    assert sha(folder/'raw/progress.jsonl')==summary['raw_progress_sha256']
    starts = [r for r in progress if r['event']=='forward_started']
    ends = [r for r in progress if r['event']=='forward_completed']
    assert [r['call'] for r in starts]==list(range(1,356))
    assert [r['call'] for r in ends]==list(range(1,356))
    phases = {phase:sum(r['phase']==phase for r in ends) for phase in ['clean','reconstruction','repeat','endpoint']}
    assert phases==dict(clean=16,reconstruction=48,repeat=48,endpoint=243)
    groups = {(int(r['attacker_seed']),int(r['target_group'])):r for r in table(folder/'groups.csv')}
    endpoints = {(int(r['attacker_seed']),int(r['target_group']),int(r['endpoint_position'])):r for r in table(folder/'endpoints.csv')}
    expected = {(seed,group) for seed in [0,1,2] for group in protocol['groups']}
    assert len(raw)==len(groups)==48 and len(endpoints)==243
    assert set(groups)==expected and {(r['attacker_seed'],r['target_group']) for r in raw}==expected
    source = {}
    for run in protocol['runs']:
        assert '_surrogate_valid_attack.jsonl' in run['source_prediction']['path']
        for row in records(ROOT/run['source_prediction']['path']):
            if row['target_group'] in protocol['groups']:
                source[run['seed'],row['target_group']] = row

    def measured(value, labels, clean):
        value, clean = np.asarray(value,dtype=np.float32), np.asarray(clean,dtype=np.float32)
        labels = np.asarray(labels,dtype=int)
        predicted, clean_predicted = value.argmax(1), clean.argmax(1)
        correct, clean_correct = predicted==labels, clean_predicted==labels
        def margin(logits):
            other = logits.copy()
            other[np.arange(len(labels)),labels] = -np.inf
            return logits[np.arange(len(labels)),labels]-other.max(1)
        margins, clean_margins = margin(value), margin(clean)
        result = dict(residuals=int(correct.sum()),incorrect=int((~correct).sum()),
                      clean_correct_denominator=int(clean_correct.sum()),
                      clean_correct_to_wrong=int((clean_correct & ~correct).sum()),
                      clean_wrong_to_correct=int((~clean_correct & correct).sum()),
                      mean_correct_margin=float(margins.mean()),min_correct_margin=float(margins.min()),max_correct_margin=float(margins.max()),
                      targets_margin_changed_above_tolerance=int((np.abs(margins-clean_margins)>protocol['margin_change_tolerance']).sum()))
        return result, predicted, margins

    def check_metrics(written, expected_metrics):
        for key, value in expected_metrics.items():
            if isinstance(value,int):
                assert int(written[key])==value, (key,written[key],value)
            else:
                assert np.isclose(float(written[key]),value,rtol=1e-6,atol=1e-5), (key,written[key],value)

    headroom = concordant = stable = residuals = clean_residuals = 0
    margin_gain_records = []
    for row in raw:
        key = row['attacker_seed'],row['target_group']
        public, saved = groups[key], source[key]
        assert row['node_ids']==saved['node_ids'] and row['labels']==saved['labels']
        assert row['source_saved_predictions']==saved['predictions']
        baseline, predicted, baseline_margins = measured(row['reconstructed_logits'],row['labels'],row['clean_logits'])
        repeated = np.asarray(row['repeat_logits'],dtype=np.float32).argmax(1)
        assert int(public['reconstructed_residuals'])==baseline['residuals']
        assert int(public['clean_correct_to_wrong'])==baseline['clean_correct_to_wrong']
        assert int(public['clean_wrong_to_correct'])==baseline['clean_wrong_to_correct']
        assert int(public['clean_residuals'])==baseline['clean_correct_denominator']
        feature_hash = hashlib.sha256(np.asarray(row['feature'],dtype=np.float32).tobytes()).hexdigest()
        assert feature_hash==public['reconstructed_feature_sha256']
        assert (public['source_prediction_match']=='True')==(predicted.tolist()==saved['predictions'])
        assert (public['repeat_prediction_match']=='True')==bool(np.array_equal(repeated,predicted))
        assert len(row['counterfactuals'])==len(row['labels'])==int(public['legal_endpoints'])
        values = []
        for endpoint in row['counterfactuals']:
            position = endpoint['endpoint_position']
            metrics, prediction, margin = measured(endpoint['logits'],row['labels'],row['clean_logits'])
            check_metrics(endpoints[key+(position,)],metrics)
            check_metrics(endpoint,metrics)
            values.append((position,metrics['residuals'],prediction,margin))
        assert sorted(value[0] for value in values)==list(range(len(row['labels'])))
        chosen_position = int(public['reconstructed_endpoint_position'])
        anchor = next(value for value in values if value[0]==chosen_position)
        assert np.array_equal(anchor[2],predicted)
        best = min(values,key=lambda x:(x[1],x[0]))
        gain = baseline['residuals']-best[1]
        assert gain==int(public['oracle_additional_incorrect'])>=0
        assert best[1]==int(public['best_residuals'])
        assert max(value[1] for value in values)==int(public['worst_residuals'])
        assert sum(value[1]<baseline['residuals'] for value in values)==int(public['strictly_better_endpoints'])
        labels = np.asarray(row['labels'])
        gained, lost = (predicted==labels)&(best[2]!=labels), (predicted!=labels)&(best[2]==labels)
        assert int(gained.sum())==int(public['oracle_gained_occurrences'])
        assert int(lost.sum())==int(public['oracle_lost_occurrences'])
        if gained.any():
            margin_gain_records.append(dict(seed=key[0],target_group=key[1],
                 minimum_signed_margin_before=float(baseline_margins[gained].min()),
                 maximum_signed_margin_after=float(best[3][gained].max())))
        headroom+=gain
        residuals+=baseline['residuals']
        clean_residuals+=baseline['clean_correct_denominator']
        concordant+=predicted.tolist()==saved['predictions']
        stable+=bool(np.array_equal(repeated,predicted))
    assert concordant==summary['source_prediction_concordant_cases']
    assert stable==summary['repeat_prediction_concordant_cases']
    assert sum(row['oracle_additional_incorrect'] for row in summary['per_seed'])==headroom
    assert sum(row['reconstructed_residuals'] for row in summary['per_seed'])==residuals
    assert sum(row['clean_residuals'] for row in summary['per_seed'])==clean_residuals
    assert abs(summary['mean_restricted_oracle_headroom_pp']-100*headroom/243)<1e-12
    old = ROOT/'evidence/one_edge_diagnosis_v1'
    failure = load(old/'failure.json')
    old_progress = records(old/'raw/progress.jsonl')
    old_starts = [r for r in old_progress if r['event']=='forward_started']
    old_ends = [r for r in old_progress if r['event']=='forward_completed']
    assert len(old_starts)==len(old_ends)==failure['surrogate_forward_calls']==79
    assert not subprocess.check_output(['git','ls-files','evidence/one_edge_diagnosis*/raw/'],cwd=ROOT).strip()
    result = dict(status='passed',checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  audit_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  audit_script_sha256=sha(pathlib.Path(__file__)),groups=16,group_checkpoint_cases=48,
                  endpoint_configurations=243,completed_attempt_forward_calls=355,
                  preserved_failed_attempt_forward_calls=79,total_recorded_forward_calls=434,
                  phases=phases,reconstructed_residuals=residuals,clean_residuals=clean_residuals,
                  restricted_oracle_additional_incorrect=headroom,source_prediction_concordant_cases=concordant,
                  repeat_prediction_concordant_cases=stable,oracle_switched_prediction_margin_ranges=margin_gain_records,
                  failed_attempt_progress_sha256=sha(old/'raw/progress.jsonl'),
                  gpu_hours=0,elapsed_seconds=time.monotonic()-started,
                  scope='Independent recount from retained logits and source validation predictions; no model loading/inference',
                  limitations='Adapter-only graph/feature equality and repeat endpoint/feature comparisons are source-reviewed and hashed, not independently regenerated. Numerical discrepancies are empirical references, not error bounds. Restricted oracle is not a learned improvement.')
    with (ROOT/args.output).open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
