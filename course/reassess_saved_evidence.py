"""Post-hoc arithmetic on published validation aggregates; no ML imports."""

import argparse
import csv
import datetime as dt
from fractions import Fraction
import hashlib
import io
import json
import pathlib
import subprocess
import time


ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT = '9ae6df5fe64cedded5a53f144629a1072d4f1ea1'
INPUTS = [
    'evidence/results_first_diagnosis_v1/group_outcomes.csv',
    'evidence/results_first_diagnosis_v1/training_derived_bins.csv',
    'evidence/results_first_diagnosis_v1/summary.json',
    'evidence/score_role_replay_v2/pairs.csv',
    'evidence/score_role_replay_v2/generators.csv',
    'evidence/score_role_replay_v2/detectors.csv',
    'evidence/improvement_pilot_summary.json',
    'course/configs/baseline.json',
    'modules/experiment_handler.py',
    'modules/data_handler.py',
    'utils/utils.py',
    'course/improvements/policies.py',
    'course/improvements/sampling.py',
]


def git(*args):
    return subprocess.check_output(['git'] + list(args), cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evidence/reassessment_20261102_v1.json')
    args = parser.parse_args()
    destination = (ROOT / args.output).resolve()
    destination.relative_to(ROOT.resolve())
    if destination.exists():
        raise RuntimeError('Use a new output path; previous evidence is immutable.')
    if git('status', '--porcelain', '--untracked-files=no').strip():
        raise RuntimeError('Commit the analysis source before recording its execution.')
    started = time.perf_counter()
    inputs = {path: git('show', SNAPSHOT + ':' + path) for path in INPUTS}
    for path, data in inputs.items():
        if git('show', 'HEAD:' + path) != data:
            raise RuntimeError('A frozen input changed: ' + path)

    def rows(path):
        return list(csv.DictReader(io.StringIO(inputs[path].decode('utf-8'))))

    groups = rows(INPUTS[0])
    metrics = []
    lookups = {}
    for cohort, condition in sorted({(r['cohort'], r['condition']) for r in groups}):
        selected = [r for r in groups if (r['cohort'], r['condition']) == (cohort, condition)]
        index = {int(r['target_group']): r for r in selected}
        if len(index) != len(selected):
            raise RuntimeError('Duplicate group-condition rows')
        for row in selected:
            total, correct = int(row['target_occurrences']), int(row['gcn_correct_after_attack'])
            assert total == int(row['seeds']) * int(row['group_size'])
            assert 0 <= correct <= total
            assert abs(float(row['mean_mr_percent']) - 100 * (total-correct)/total) < 1e-10
        occurrences = sum(int(r['target_occurrences']) for r in selected)
        incorrect = sum(int(r['target_occurrences'])-int(r['gcn_correct_after_attack']) for r in selected)
        occurrence_mr = Fraction(100 * incorrect, occurrences)
        group_mr = 100 * sum((Fraction(int(r['target_occurrences'])-int(r['gcn_correct_after_attack']),
                                      int(r['target_occurrences'])) for r in selected), Fraction()) / len(selected)
        value = dict(cohort=cohort, condition=condition, groups=len(selected), seeds=3,
                     target_occurrences=occurrences, incorrect_occurrences=incorrect,
                     occurrence_mr_percent=float(occurrence_mr), group_equal_mr_percent=float(group_mr),
                     exact_occurrence_mr_fraction=str(occurrence_mr), exact_group_equal_mr_fraction=str(group_mr))
        metrics.append(value)
        lookups[cohort, condition] = (index, value, occurrence_mr, group_mr)
    comparisons = []
    for cohort, arm, reference in [('full_baseline','learned','random'),
                                   ('pilot','coverage','baseline'), ('pilot','size_sampling','baseline')]:
        left, right = lookups[cohort, arm], lookups[cohort, reference]
        assert left[0].keys() == right[0].keys()
        for key in left[0]:
            for field in ['group_size','node_budget','edge_budget','seeds','target_occurrences']:
                assert left[0][key][field] == right[0][key][field], (key, field)
        micro, macro = left[2]-right[2], left[3]-right[3]
        comparisons.append(dict(cohort=cohort, variant=arm, reference=reference,
                                occurrence_difference_pp=float(micro), group_equal_difference_pp=float(macro),
                                direction_reversal=(micro*macro < 0)))
    original = json.loads(inputs[INPUTS[2]])
    for row in original['conditions']:
        actual = lookups[row['cohort'], row['condition']][1]['occurrence_mr_percent']
        assert abs(actual-row['mean_best_gcn_validation_mr_percent']) < 1e-10

    full_union = [r for r in rows(INPUTS[1]) if r['cohort']=='full_baseline' and
                  r['condition']=='learned' and r['feature']=='min_target_degree' and int(r['groups'])==592]
    assert len(full_union) == 1 and int(full_union[0]['target_occurrences']) == 14889
    unique_targets = int(full_union[0]['unique_nodes'])

    pairs = rows(INPUTS[3])
    assert len(pairs)==48 and len({(r['target_group'], r['attacker_seed']) for r in pairs})==48
    pair_strata = []
    for same in [True, False]:
        selected = [r for r in pairs if (r['cpu_gpu_same_candidate_set']=='True')==same]
        pair_strata.append(dict(same_candidate_set=same, cases=len(selected),
                                same_candidate_order=sum(r['cpu_gpu_same_candidate_order']=='True' for r in selected),
                                selected_endpoint_changes=sum(r['cpu_gpu_same_selected_endpoint']=='False' for r in selected)))
    generators = rows(INPUTS[4])
    assert len(generators)==96
    attacked = [r for r in rows(INPUTS[5]) if r['generator']=='gpu' and r['detector']=='gpu']
    assert len(attacked)==48
    n = sum(int(r['target_occurrences']) for r in attacked)
    clean_wrong = n-sum(int(r['clean_correct']) for r in attacked)
    attacked_wrong = n-sum(int(r['residuals']) for r in attacked)
    new_flips = sum(int(r['clean_correct_to_wrong']) for r in attacked)
    recoveries = sum(int(r['clean_wrong_to_correct']) for r in attacked)
    assert attacked_wrong-clean_wrong == new_flips-recoveries
    baseline = json.loads(inputs['course/configs/baseline.json'])
    result = dict(
        schema='monti_saved_evidence_reassessment_v1',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        analysis_kind='Post-hoc descriptive reassessment; preliminary arithmetic was inspected before this script. Not preregistration.',
        analysis_source_commit=git('rev-parse','HEAD').decode().strip(),
        scientific_input_snapshot=SNAPSHOT,
        command=['bundled-python','-m','course.reassess_saved_evidence','--output',args.output],
        input_sha256={p:hashlib.sha256(b).hexdigest() for p,b in inputs.items()},
        analysis_script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        selected_validation_snapshot_aggregates=metrics, paired_summary_comparisons=comparisons,
        overlap_metadata=dict(validation_groups=592, occurrences_per_checkpoint=4963,
                              unique_validation_target_nodes=unique_targets,
                              occurrences_per_unique_target=4963/unique_targets,
                              source='One saved bin spans all groups; no summation of overlapping bin-unique counts.'),
        saved_generator_device_pairs=pair_strata,
        target_orders_equal_across_device_pairs=sum(r['cpu_gpu_same_target_order']=='True' for r in pairs),
        observed_first_target_score_source_calls=sum(r['executed_source_role']=='first_target' for r in generators),
        selected_gpu_transition_recount=dict(cases=len(attacked), target_occurrences=n, clean_wrong=clean_wrong,
                                            attacked_wrong=attacked_wrong, newly_wrong=new_flips, recovered=recoveries),
        configured_training_batch_size=baseline['batch_size'],
        scope_limits=[
            'Only saved GCN validation aggregates; no test predictions, data tensors, checkpoints or new model calls.',
            'Group means average the same three preselected checkpoints, not newly macro-selected checkpoints.',
            'The small ranking reversal is descriptive, not statistical significance, author bias or a replacement primary metric.',
            'Repeated targets/groups and evaluation draws are not independent seed replications.',
            'Same-candidate device pairs still differ numerically; the split does not establish a sole cause.',
            'No changed-group membership or target-order experiment has been performed.',
            'No defense, correction efficacy, method direction, pilot or full test is approved by this analysis.'
        ],
        additional_model_calls=0, additional_training_runs=0, additional_gpu_hours=0,
        elapsed_cpu_wall_seconds=time.perf_counter()-started)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,indent=2,ensure_ascii=True)
        handle.write('\n')
    print(json.dumps(dict(status='completed',output=str(destination.relative_to(ROOT)),
                          comparisons=comparisons,model_calls=0,gpu_hours=0),indent=2))


if __name__ == '__main__':
    main()
