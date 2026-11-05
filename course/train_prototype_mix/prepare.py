"""Freeze one TRAIN-reference feature contrast before reference extraction."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT, private_output, sha256, write_json

BASE = 'runs/feature_polish_20261104_attempt4'
BASE_SHA = 'eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
BASE_SOURCE = '8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
ARMS = ['baseline', 'benign_mix', 'all_train_mix', 'fraud_mix']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit intended source before registering this study.')
    inputs = {}

    def retain(path, wanted=None):
        path = path.resolve()
        if ROOT not in path.parents: raise RuntimeError('Input outside project.')
        value = sha256(path)
        if wanted is not None and value != wanted: raise RuntimeError('Frozen input changed: '+str(path))
        name = path.relative_to(ROOT).as_posix(); inputs[name] = value
        return dict(path=name, sha256=value)

    old_path = ROOT/BASE/'experiment_protocol_v1.json'
    base_ref = retain(old_path, BASE_SHA)
    old = json.loads(old_path.read_text(encoding='utf-8'))
    if old['source_commit'] != BASE_SOURCE: raise RuntimeError('Original capture experiment differs.')
    source_files = dict(old['source_files'])
    for name, wanted in source_files.items():
        if sha256(ROOT/name) != wanted: raise RuntimeError('Reused scientific implementation changed: '+name)
    source_files.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    for row in old['inputs']+old.get('retained_inputs', []): retain(ROOT/row['path'], row['sha256'])
    reference_data = {
        'graph': retain(ROOT/old['data']['graph']),
        'train_indices': retain(ROOT/old['data']['train_indices']),
        'valid_indices': retain(ROOT/'Datasets/GossipCop/data_split/valid_idx_surro_0.npy'),
        'test_indices': retain(ROOT/'Datasets/GossipCop/data_split/test_idx_surro_0.npy')}
    if json.loads((ROOT/BASE/'queue_complete.json').read_text(encoding='utf-8'))['status'] != 'completed':
        raise RuntimeError('Original baseline capture phase is incomplete.')
    retain(ROOT/BASE/'queue_complete.json')
    runs = {}
    for seed in (3, 4, 5):
        directory = ROOT/BASE/'queue_attempt1/refinements'/('seed%d'%seed)
        result = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if (result['status'] != 'completed' or result['groups'] != 128 or result['seed'] != seed
                or result['source_commit'] != BASE_SOURCE or result['protocol_sha256'] != BASE_SHA):
            raise RuntimeError('Original retained seed does not match the complete capture experiment.')
        captures = [dict(group=row['group'], **retain(directory/row['path'], row['sha256'])) for row in result['exports']]
        if [row['group'] for row in captures] != old['valid_groups']:
            raise RuntimeError('Preserve every case and its order; do not filter on past outcomes.')
        retain(directory/'registered.json')
        runs[str(seed)] = dict(result_ref=retain(directory/'result.json'),
            groups_ref=retain(directory/'groups.jsonl', result['groups_jsonl_sha256']), captures=captures)
    synthetic_path = ROOT/'runs/train_prototype_mix_synthetic_20261105_attempt1/receipt.json'
    synthetic_ref = retain(synthetic_path)
    synthetic = json.loads(synthetic_path.read_text(encoding='utf-8'))
    if (synthetic['status'] != 'passed' or synthetic['tests_run'] != 12
            or synthetic['failures'] or synthetic['errors'] or synthetic['real_data_reads']
            or synthetic['model_calls'] or synthetic['GPU_hours']):
        raise RuntimeError('The frozen meaningful synthetic checks must pass before reference extraction.')
    for name in ('core.py', 'synthetic_tests.py'):
        key = 'course/train_prototype_mix/'+name
        if synthetic['source_files'][key] != source_files[key]:
            raise RuntimeError('The synthetic checks used a different new implementation.')
    ledger_path = ROOT/'evidence/research_cost_ledger.json'
    ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger['measured_or_conservatively_charged']+ledger['unmetered_setup_reserve'] > ledger['limit']:
        raise RuntimeError('Research allowance is exhausted.')
    out = private_output(args.output, create=True)
    p = dict(schema='monti_train_prototype_mix_v1', status='frozen', post_hoc=True,
        owner_execution_authorized=True,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_files=source_files, base_experiment_ref=base_ref, synthetic_receipt_ref=synthetic_ref,
        reference_data=reference_data,
        inputs=[dict(path=n, sha256=h) for n, h in inputs.items()], runs=runs,
        seeds=[3, 4, 5], valid_groups=old['valid_groups'], arms=ARMS, alpha=0.5,
        deadline_seconds=5400, cpu_threads=8, execution_device='cpu', cuda_visible_devices='',
        reference='Once only: float64 raw-feature means over UNIQUE original TRAIN rows, cast to float32. Benign class0, all TRAIN, and fraud class1. Pool selection uses TRAIN labels only; held-out IDs are used for partition checks, not their labels. The standard original loader still contains whole-graph label/class-count metadata.',
        box='Original per-coordinate min/max of all unlabelled transductive raw features; same legal box as baseline, no new unit-norm/L2/L0 constraint. Clamp only numerical mixing roundoff; force zero-width dimensions exactly.',
        intervention='Each proposal is exactly0.5*x0+0.5*its fixed reference. No lambda search, gradient, teacher, LLM, relation-distance penalty or outcome-derived gate.',
        preserved='Entire captured adjacency, target membership/order, candidate membership/order, original-original graph/features, every physical node/edge budget and edge role. Only injected raw features change.',
        pre_classifier_gate='Freeze reference tensors/hashes and all384captures times4feature/canonical-sequence variants before any classifier construction. Canonical tokens0/1/2/4/5 must be bit-identical across arms. All cases, including identity changes, stay in the denominator.',
        qualification=dict(primary='benign_mix', min_mean_gain_pp=1.0,
            require_all_seed_deltas_positive=True, require_mean_above_all_train_mix=True,
            require_mean_above_fraud_mix=True, require_paired_t95_lower_above_zero=True,
            controls_not_candidates=True,
            uncertainty='Nominal conditional paired t95 over3generator seeds,df2, with fixed classifiers and repeatedly inspected correlated development data. Not a confirmatory population or teacher-uncertainty interval.'),
        expected_if_evaluated=dict(reference_selections=1, reference_tensor_saves=1,
            source_capture_tensor_loads=384, source_captures_prepared=384,
            geometry_base_graph_reconstructions=384, geometry_feature_graph_variants=1536,
            canonical_sequence_updates=1536, fixed_token_invariant_checks=1536,
            prepared_artifact_tensor_loads=384,
            evaluation_base_graph_reconstructions=384, evaluation_feature_graph_variants=1536,
            original_gcn_load_dispatches=1, original_gcn_models_loaded=1,
            official_gaga_load_dispatches=1, official_gaga_models_loaded=1,
            original_gcn_forward_dispatches=1664, original_gcn_completed_forwards=1664,
            official_gaga_forward_dispatches=1664, official_gaga_completed_forwards=1664,
            original_gcn_clean_forwards=128, official_gaga_clean_forwards=128,
            original_gcn_attacked_forwards=1536, official_gaga_attacked_forwards=1536,
            fresh_teacher_model_constructions=0, fresh_teacher_forwards=0,
            generator_model_constructions=0, generator_forwards=0,
            optimizer_steps=0, autograd_calls=0, API_calls=0, TEST_queries=0),
        budget_snapshot=dict(unit=ledger['unit'], limit=ledger['limit'],
            charged=ledger['measured_or_conservatively_charged'], reserve=ledger['unmetered_setup_reserve'],
            remaining=ledger['remaining_after_reserve'], ledger_sha256_at_registered_source=sha256(ledger_path),
            new_phase_GPU_hours=0),
        context='Manual human continuation after the5November achieved-artifact delivery and deletion of automation monti. Earlier failures remain intact. Statistical references were not extracted before this registration. This is a practical numerical-feature experiment, not a claim of valid generated article text, camouflage, novelty or guaranteed efficacy.',
        automation_recreated=False, final_TEST_allowed=False,
        failure_rule='Retain failed/interrupted outputs, counters and elapsed time. No automatic scientific retry or outcome-dependent arms, ratios, cohorts or seed expansion. A control result never becomes the selected primary after outcomes.')
    path = out/'protocol_v1.json'; write_json(path, p, exclusive=True)
    print(json.dumps(dict(status='frozen', source_commit=p['source_commit'], protocol_sha256=sha256(path), inputs=len(inputs), output=str(path))))


if __name__ == '__main__': main()
