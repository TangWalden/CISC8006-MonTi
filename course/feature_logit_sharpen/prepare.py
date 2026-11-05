"""Freeze label-blind new VALID contexts before any new model calls."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys

from course.feature_polish.common import ROOT, private_output, sha256, write_json
from course.feature_logit_sharpen.run import EXPECTED_COUNTS

BASE = 'runs/feature_polish_20261104_attempt4/experiment_protocol_v1.json'
BASE_SHA = 'eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
BASE_SOURCE = '8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
SELECTION_SEED = 20261105


def compact_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=True).encode('ascii')).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit all intended source before cohort metadata extraction.')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    inputs = {}

    def retain(path, wanted=None):
        path = path.resolve()
        if ROOT not in path.parents: raise RuntimeError('Input outside project.')
        actual = sha256(path)
        if wanted is not None and actual != wanted: raise RuntimeError('Frozen input changed: '+str(path))
        name = path.relative_to(ROOT).as_posix()
        if inputs.setdefault(name, actual) != actual: raise RuntimeError('Conflicting input hashes.')
        return dict(path=name, sha256=actual)

    base_ref = retain(ROOT/BASE, BASE_SHA); old = json.loads((ROOT/BASE).read_text(encoding='utf-8'))
    if old['source_commit'] != BASE_SOURCE: raise RuntimeError('Exact baseline capture-family protocol differs.')
    sources = dict(old['source_files'])
    for name, wanted in sources.items():
        if sha256(ROOT/name) != wanted: raise RuntimeError('Reused scientific code changed: '+name)
    sources.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    for row in old['inputs']+old.get('retained_inputs', []): retain(ROOT/row['path'], row['sha256'])
    for seed in (3, 4, 5):
        entry = old['baseline_runs'][str(seed)]; directory = ROOT/entry['directory']
        retain(directory/'result.json', entry['result_sha256'])
        retain(directory/'final_model.pt', entry['final_model_sha256'])
    names = dict(graph=old['data']['graph'],
        valid_group_ids='Datasets/GossipCop/data_split/valid_target_mask_0.npy',
        target_assignment='Datasets/GossipCop/data_split/target_assign_matrix_0.npz')
    reference_data = {}
    for role, name in names.items():
        if name not in inputs: raise RuntimeError('Metadata was not in the original frozen input inventory: '+name)
        reference_data[role] = retain(ROOT/name, inputs[name])
    prior_result_ref = retain(ROOT/'evidence/train_prototype_mix_v1/result.json')
    prior_audit_ref = retain(ROOT/'evidence/train_prototype_mix_v1/independent_saved_tensor_audit_v1.json')
    prior_audit = json.loads((ROOT/prior_audit_ref['path']).read_text(encoding='utf-8'))
    if prior_audit['status'] != 'passed' or prior_audit['primary_development_criterion']['criterion_met'] is not False:
        raise RuntimeError('First retain the completed negative prototype study and passed integrity audit.')
    synthetic_path = ROOT/'runs/feature_logit_sharpen_synthetic_20261105_attempt1/receipt.json'
    synthetic_ref = retain(synthetic_path); synthetic = json.loads(synthetic_path.read_text(encoding='utf-8'))
    if (synthetic['status'] != 'passed' or synthetic['tests_run'] != 12 or synthetic['failures'] or synthetic['errors']
            or synthetic['real_data_reads'] or synthetic['model_calls'] or synthetic['GPU_hours']):
        raise RuntimeError('Frozen meaningful synthetic checks must pass before the fresh cohort is opened.')
    for name in ('core.py', 'synthetic_tests.py'):
        key = 'course/feature_logit_sharpen/'+name
        if synthetic['source_files'][key] != sources[key]: raise RuntimeError('Synthetic implementation differs.')
    ledger_path = ROOT/'evidence/research_cost_ledger.json'; ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger['measured_or_conservatively_charged']+ledger['unmetered_setup_reserve'] > ledger['limit']:
        raise RuntimeError('Research allowance is exhausted.')
    out = private_output(args.output, create=True)
    write_json(out/'metadata_scope.json', dict(source_commit=source, source_files=sources,
        registered_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), reference_data=reference_data,
        selection_seed=SELECTION_SEED, draw_count=128, new_group_ID_arrays_opened=False,
        scope='Only original VALID group IDs and original target-membership CSR metadata. No graph labels, checkpoints, models or predictions are deserialized in this preparation step. Source is committed before metadata; actual list is frozen before any new model call.'), exclusive=True)
    import numpy as np
    import scipy.sparse as sp
    raw = np.load(ROOT/reference_data['valid_group_ids']['path'], allow_pickle=False)
    if raw.ndim != 1 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise RuntimeError('Original VALID group IDs must be one finite integral vector.')
    group_ids = raw.astype(np.int64)
    if (not np.array_equal(raw.astype(np.float64), group_ids.astype(np.float64))
            or len(set(map(int, group_ids))) != len(group_ids) or (group_ids < 0).any()):
        raise RuntimeError('Original VALID IDs are fractional, negative or duplicated.')
    excluded = list(map(int, old['valid_groups'])); population = set(map(int, group_ids))
    if len(excluded) != 128 or not set(excluded) <= population: raise RuntimeError('Original excluded context list differs.')
    eligible = sorted(population-set(excluded))
    if len(eligible) < 128: raise RuntimeError('Insufficient unused original VALID contexts; do not resample another rule.')
    selected = random.Random(SELECTION_SEED).sample(eligible, 128)
    assignment = sp.load_npz(ROOT/reference_data['target_assignment']['path']).tocsr()
    if max(population) >= assignment.shape[0]: raise RuntimeError('VALID group IDs exceed original assignment rows.')
    def members(group):
        value = np.flatnonzero(assignment.getrow(group).toarray().reshape(-1))
        if not len(value): raise RuntimeError('A selected original target list is empty; retain preparation failure.')
        return set(map(int, value))
    old_members = {group: members(group) for group in excluded}
    old_union = set().union(*old_members.values())
    if len(old_union) != 374 or sum(map(len, old_members.values())) != 1091:
        raise RuntimeError('Original cohort membership does not reproduce its frozen known counts.')
    selected_members = {group: members(group) for group in selected}; union = set().union(*selected_members.values())
    selection = dict(algorithm='python_random_sample_sorted_eligible_v1', rng_seed=SELECTION_SEED, draw_count=128,
        original_valid_group_count=len(population), eligible_count=len(eligible),
        ordered_groups_sha256=compact_hash(selected), runtime=sys.version,
        no_group_size_or_outcome_selection=True, no_target_list_regrouping=True)
    cohort = dict(target_occurrences_per_seed=sum(map(len, selected_members.values())), unique_targets=len(union),
        original_node_count=assignment.shape[1], overlap_unique_targets_with_old_contexts=len(union&old_union),
        new_unique_targets=len(union-old_union), group_target_counts={str(g):len(selected_members[g]) for g in selected},
        target_ids_sha256=compact_hash(sorted(union)), excluded_target_ids_sha256=compact_hash(sorted(old_union)),
        interpretation='New contexts for this postprocessor, not unseen independent news or an untouched validation population. Target identities overlap earlier groups; all original membership lists are unchanged.')
    p = dict(schema='monti_feature_logit_sharpen_v1', status='frozen', post_hoc=True, owner_execution_authorized=True,
        source_commit=source, source_files=sources, inputs=[dict(path=n, sha256=h) for n,h in inputs.items()],
        base_experiment_ref=base_ref, synthetic_receipt_ref=synthetic_ref, reference_data=reference_data,
        prior_result_ref=prior_result_ref, prior_audit_ref=prior_audit_ref,
        seeds=[3,4,5], valid_groups=selected, excluded_valid_groups=excluded, selection=selection, cohort_metadata=cohort,
        arms=['baseline','logit_sharp','linear_contrast','shuffled_delta'], beta=2.0, linear_slope=1.5,
        execution_device='cpu', cuda_visible_devices='', cpu_threads=8, deadline_seconds=5400,
        expected_if_evaluated=dict(EXPECTED_COUNTS),
        baseline_family='The exact source-role legacy terminal8 checkpoint family bound by base8222/eedad, the same family used by prior feature-polish captures. Not alignment-agent legacy_hinge_off and not the formal full-test generator family.',
        generation='Deep-clone base protocol and change only VALID group IDs. Unchanged helper restores8scheduler ticks; DualExperiment validation/epoch8 case RNG. Cache h_GCN is reused. Three preparatory GCN and generator loads occur before seals; proposal GCN hooks prohibit forwards. One common generated graph per seed/context.',
        intervention='Convert raw features to the original unlabeled global coordinate box. Primary u^2/(u^2+(1-u)^2), equivalent to doubling sigmoid-head logits for interior coordinates, not classifier confidence. Linear comparator clips0.5+1.5*(u-0.5). Endpoints and zero-width dimensions remain numerically bounded; float32 can still collapse interior values.',
        shuffle_control='Use the actual stored sharpened-unit delta, permute only movable coordinates within each injected row with frozen SHA role keys. Only PRECLIP unit-L2 displacement is matched; terminal unit/raw movement, clipping and saturation are measured, never claimed matched.',
        headroom_stop_rule='After all384generations/1536proposals seal, if any seed has zero primary nonidentity cases, stop before dedicated GCN/GAGA loads or queries. All384generator forwards and3preparatoryGCNloads were still spent; this is not a zero-model claim.',
        preserved='Every original target list, candidate choice/order, complete captured hard adjacency, physical node/edge budgets and edge roles, original features, fixed GAGA tokens0/1/2/4/5. Only injected numeric features change; no generated article text, stealth or semantic-validity claim.',
        qualification=dict(primary='logit_sharp', min_mean_gain_pp=1.0, require_all_seed_deltas_positive=True,
            require_paired_t95_lower_above_zero=True, require_mean_above_shuffled_delta=True,
            controls_not_candidates=True, linear_comparator_need_not_be_worse=True,
            uncertainty='Nominal conditional paired t95,3fixed generator seeds/df2, repeated target identities and adaptive development history. No population, teacher or confirmatory-generalization claim; no unique-curve advantage is claimed from equivalence to linear contrast.'),
        budget_snapshot=dict(unit=ledger['unit'], limit=ledger['limit'], charged=ledger['measured_or_conservatively_charged'],
            reserve=ledger['unmetered_setup_reserve'], remaining=ledger['remaining_after_reserve'],
            ledger_sha256_at_registered_source=sha256(ledger_path), new_phase_GPU_hours=0),
        context='Separate exploratory hypothesis selected after the known negative fixed0.5prototype study. That phase stays closed and unchanged. This fixed curve is tested on newly ID-sampled original VALID contexts, not appended ratios or seeds to that phase.',
        metadata_scope_sha256=sha256(out/'metadata_scope.json'), automation_recreated=False, final_TEST_allowed=False,
        failure_rule='Retain partial calls/files/time and every failed case; no automatic scientific retry, outcome-based multiplier/cohort/seed expansion, omitted cases or promotion of a favorable control.')
    path = out/'protocol_v1.json'; write_json(path, p, exclusive=True)
    print(json.dumps(dict(status='frozen', source_commit=source, protocol_sha256=sha256(path),
        original_VALID_groups=len(population), eligible_groups=len(eligible), selected_groups=len(selected),
        target_occurrences_per_seed=cohort['target_occurrences_per_seed'], unique_targets=cohort['unique_targets'],
        overlapping_old_targets=cohort['overlap_unique_targets_with_old_contexts'], inputs=len(inputs), output=str(path))))


if __name__ == '__main__': main()
