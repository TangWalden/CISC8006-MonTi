"""Independently reconcile the saved-outcome diagnosis without model inference."""
import argparse
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-inputs', action='store_true')
    parser.add_argument('--directory', default='evidence/results_first_diagnosis_v1')
    parser.add_argument('--output', default='evidence/results_first_diagnosis_audit_v1.json')
    args = parser.parse_args()
    started = time.monotonic()
    folder = ROOT / args.directory
    hashes = {}

    def read_json(name):
        path = folder / name
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())

    def read_csv(name):
        path = folder / name
        hashes[str(path.relative_to(ROOT))] = digest(path)
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def near(left, right):
        assert abs(float(left) - float(right)) < 1e-10, (left, right)

    summary = read_json('summary.json')
    registered = read_json('registered.json')
    boundaries = read_json('bin_boundaries.json')
    inputs = read_json('input_checksums.json')
    runs = read_csv('runs.csv')
    curves = read_csv('validation_curves.csv')
    groups = read_csv('group_outcomes.csv')
    bins = read_csv('training_derived_bins.csv')
    persistence = read_csv('persistence.csv')
    paired = read_csv('paired_changes.csv')
    nodes = read_csv('node_concentration.csv')
    assert summary['status'] == 'completed' and summary['gpu_hours'] == 0
    assert len(runs) == 15 and len(summary['runs']) == 15
    assert not any('_test_' in path for path in inputs)
    assert len({(r['cohort'], r['condition'], r['seed']) for r in runs}) == 15
    for run in runs:
        key = tuple(run[k] for k in ('cohort', 'condition', 'seed'))
        selected = [r for r in curves if tuple(r[k] for k in ('cohort', 'condition', 'seed')) == key]
        selected.sort(key=lambda r: int(r['epoch']))
        assert [int(r['epoch']) for r in selected] == list(range(1, int(run['completed_epochs']) + 1))
        count = int(run['target_occurrences'])
        residuals = [int(r['gcn_correct_after_attack']) for r in selected]
        assert residuals.index(min(residuals)) + 1 == int(run['best_epoch'])
        assert int(run['best_residual_count']) == min(residuals)
        assert int(run['final_residual_count']) == residuals[-1]
        assert int(run['tied_best_epoch_count']) == residuals.count(min(residuals))
        for curve in selected:
            assert int(curve['target_occurrences']) == count
            near(curve['mr_percent'], 100 * (1 - int(curve['gcn_correct_after_attack']) / count))
        near(run['best_mr_percent'], 100 * (1 - min(residuals) / count))
        near(run['final_minus_best_pp'], float(run['final_mr_percent']) - float(run['best_mr_percent']))
    for condition in summary['conditions']:
        key = condition['cohort'], condition['condition']
        chosen = [r for r in runs if (r['cohort'], r['condition']) == key]
        rows = [r for r in groups if (r['cohort'], r['condition']) == key]
        assert len(chosen) == 3 and len(rows) == int(chosen[0]['groups'])
        exposure = sum(int(r['target_occurrences']) for r in chosen)
        residual = sum(int(r['best_residual_count']) for r in chosen)
        assert sum(int(r['target_occurrences']) for r in rows) == exposure
        assert sum(int(r['gcn_correct_after_attack']) for r in rows) == residual
        assert residual == condition['residual_occurrences_across_seeds']
        near(condition['mean_best_gcn_validation_mr_percent'], statistics.mean(float(r['best_mr_percent']) for r in chosen))
        for row in rows:
            assert int(row['target_occurrences']) == 3 * int(row['group_size'])
            near(row['residual_rate'], int(row['gcn_correct_after_attack']) / int(row['target_occurrences']))
            near(row['mean_mr_percent'], 100 * (1 - float(row['residual_rate'])))
        frequencies = [r for r in persistence if (r['cohort'], r['condition']) == key]
        assert sorted(int(r['gcn_correct_in_n_seeds']) for r in frequencies) == [0, 1, 2, 3]
        assert sum(int(r['occurrence_keys']) for r in frequencies) == exposure // 3
        assert sum(int(r['gcn_correct_in_n_seeds']) * int(r['occurrence_keys']) for r in frequencies) == residual
        assert sum(int(r['persistent_residual_occurrence_keys']) for r in rows) == condition['persistent_residual_occurrence_keys']
        node_rows = [r for r in nodes if (r['cohort'], r['condition']) == key]
        near(sum(int(r['gcn_correct_after_attack']) for r in node_rows) / residual, condition['top10_nodes_share_of_residuals'])
        near(sum(int(r['evaluated_occurrences']) for r in node_rows) / exposure, condition['top10_nodes_share_of_all_evaluated_occurrences'])
        for feature, cuts in boundaries[key[0]].items():
            selected_bins = [r for r in bins if (r['cohort'], r['condition'], r['feature']) == key + (feature,)]
            assert len(selected_bins) == len(cuts) + 1
            assert sum(int(r['target_occurrences']) for r in selected_bins) == exposure
            assert sum(int(r['gcn_correct_after_attack']) for r in selected_bins) == residual
            for cell in selected_bins:
                index = int(cell['bin'])
                members = [r for r in rows if bisect.bisect_right(cuts, float(r[feature])) == index]
                assert int(cell['groups']) == len(members)
                denom = sum(int(r['target_occurrences']) for r in members)
                remain = sum(int(r['gcn_correct_after_attack']) for r in members)
                assert int(cell['target_occurrences']) == denom
                assert int(cell['gcn_correct_after_attack']) == remain
                if denom:
                    near(cell['residual_rate'], remain / denom)
                else:
                    assert cell['residual_rate'] == ''
    for row in paired:
        reference = next(r for r in runs if (r['cohort'], r['condition'], r['seed']) == (row['cohort'], row['reference'], row['seed']))
        variant = next(r for r in runs if (r['cohort'], r['condition'], r['seed']) == (row['cohort'], row['variant'], row['seed']))
        net = int(reference['best_residual_count']) - int(variant['best_residual_count'])
        assert net == int(row['net_misclassified_occurrences'])
        assert net == int(row['additional_misclassified_occurrences']) - int(row['fewer_misclassified_occurrences'])
        near(row['difference_pp'], 100 * net / int(row['target_occurrences']))
    fixed_count = raw_count = 0
    if args.raw_inputs:
        # This is explicitly a post-extraction integrity check, not an earlier registration.
        protocol = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
        for item in protocol['inputs'] + protocol['scientific_code']:
            assert digest(ROOT / item['path']) == item['sha256'], item['path']
            fixed_count += 1
        for relative, expected in inputs.items():
            assert digest(ROOT / relative) == expected, relative
            raw_count += 1
        for field, hash_field in [('plan', 'plan_sha256'), ('script', 'script_sha256')]:
            relative = registered[field] if field == 'plan' else 'course/results_first_diagnosis.py'
            assert digest(ROOT / relative) == registered[hash_field]
        assert subprocess.check_output(['git', 'show', registered['source_commit'] + ':course/results_first_diagnosis.py'], cwd=ROOT).strip() == (ROOT / 'course/results_first_diagnosis.py').read_bytes().strip()
    result = dict(status='passed', checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  audit_script_sha256=digest(pathlib.Path(__file__)),
                  scope='Independent derived-table reconciliation; optional post-extraction raw and frozen input hash check',
                  raw_input_hashes_checked=raw_count, frozen_input_and_scientific_hashes_checked=fixed_count,
                  run_count=len(runs), epoch_count=len(curves), condition_count=len(summary['conditions']),
                  group_condition_rows=len(groups), complete_feature_bin_rows=len(bins),
                  audited_derived_file_sha256=hashes, gpu_hours=0, elapsed_seconds=time.monotonic()-started,
                  limitations='No causal confirmation; selected validation snapshots; no new inference or test prediction reads')
    path = ROOT / args.output
    with path.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
