"""Freeze the owner-selected projected-quota, fixed-upstream CPU pilot."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
BASE_PATH = 'runs/feature_polish_20261104_attempt4/experiment_protocol_v1.json'
BASE_SHA = 'eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
PREVIOUS_PATH = 'runs/candidate_target_reallocation_20261105_attempt1/protocol_v1.json'
PREVIOUS_SHA = '24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14'
CONCURRENT_PATH = 'runs/feature_assignment_20261111_attempt1/registration/protocol.json'
SEEDS = (3, 4, 5)
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned')


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def relative_file(root, name):
    root = pathlib.Path(root).resolve()
    relative = pathlib.Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise RuntimeError('Only root-relative frozen input paths are permitted.')
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise RuntimeError('Input is missing or escapes its bound root: ' + str(name))
    return path


def verify_source_manifest(manifest):
    if not isinstance(manifest.get('source_commit'), str) or len(manifest['source_commit']) != 40:
        raise RuntimeError('A committed scientific source identity is required.')
    files = manifest.get('files', {})
    mandatory = {'methods/downstream_quota.py'} | {
        'experiments/downstream_cooperation/' + name
        for name in ('prepare.py', 'common.py', 'model.py', 'preflight.py', 'run.py', 'driver.py')
    }
    if not mandatory <= set(files):
        raise RuntimeError('The frozen downstream study source inventory is incomplete.')
    for name, wanted in files.items():
        path = pathlib.PurePosixPath(name)
        allowed = name in ('methods/__init__.py', 'tests/__init__.py') or (
            len(path.parts) == 2 and path.parts[0] == 'methods'
            and path.name.startswith('downstream_') and path.suffix == '.py'
        ) or (
            len(path.parts) == 3 and path.parts[:2] == ('experiments', 'downstream_cooperation')
            and path.suffix == '.py'
        ) or (
            len(path.parts) == 2 and path.parts[0] == 'tests'
            and path.name.startswith('test_downstream') and path.suffix == '.py'
        )
        if not allowed or sha(relative_file(SOURCE_ROOT, name)) != wanted:
            raise RuntimeError('Unexpected or changed study source: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runtime = pathlib.Path(args.runtime_root).resolve()
    assets = runtime
    output = pathlib.Path(args.output).resolve()
    if (assets / 'runs').resolve() not in output.parents:
        raise RuntimeError('Registration must remain inside a new ignored runs directory.')
    manifest = read_json(args.source_manifest)
    verify_source_manifest(manifest)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip()
    if head != RUNTIME_COMMIT:
        raise RuntimeError('Keep the historical runtime at its clean public archive commit.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip():
        raise RuntimeError('Historical tracked runtime files have changed.')
    refs = {}

    def retain(name, wanted=None, root_key='asset'):
        actual = sha(relative_file(assets if root_key == 'asset' else runtime, name))
        if wanted is not None and actual != wanted:
            raise RuntimeError('Frozen historical input changed: ' + name)
        key = (root_key, name)
        if refs.setdefault(key, actual) != actual:
            raise RuntimeError('Conflicting input identities.')
        return dict(root=root_key, path=name, sha256=actual)

    retain(BASE_PATH, BASE_SHA)
    base = read_json(assets / BASE_PATH)
    retain(PREVIOUS_PATH, PREVIOUS_SHA)
    previous = read_json(assets / PREVIOUS_PATH)
    # Hash old inputs and runtime source without deserializing a model or tensor.
    for specification in (base, previous):
        for item in specification.get('inputs', []) + specification.get('retained_inputs', []):
            retain(item['path'], item['sha256'])
        for name, wanted in specification.get('source_files', {}).items():
            retain(name, wanted, 'runtime')
    for name, wanted in previous.get('excluded_sharpen_source_files', {}).items():
        retain(name, wanted, 'runtime')
    for name in (
        'course/configs/baseline.json', 'course/feature_polish/refine.py',
        'course/dual_surrogate/runtime.py', 'course/dual_surrogate/experiment.py',
        'course/dual_surrogate/common.py', 'course/dual_surrogate/verify.py',
        'course/source_role_fix/saved_teacher_diagnostic.py',
        'course/dual_surrogate/sequence_adapter.py', 'modules/data_handler.py',
        'modules/experiment_handler.py', 'models/attack_model.py', 'models/victim_model.py',
        'utils/utils.py', 'utils/constants.py',
    ):
        retain(name, root_key='runtime')
    if set(base['baseline_runs']) != {'3', '4', '5'}:
        raise RuntimeError('The three exact recorded legacy checkpoints are required.')
    for seed in SEEDS:
        reference = base['baseline_runs'][str(seed)]
        directory = pathlib.PurePosixPath(reference['directory'])
        retain((directory / 'result.json').as_posix(), reference['result_sha256'])
        retain((directory / 'final_model.pt').as_posix(), reference['final_model_sha256'])
    retain('models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle',
           '729b8624baac7429bd60e7bc2d3741a9d9ee7c4c7df921d618b53ceb0435c383')
    retain('models/pretrained_models/configs/GAGA-gossipcop_0.05-0.json')
    retain('Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',
           'caf9f30772e2977f20ef3dcb4cb127d7cdf73e3986f7cc5599f5753cb3aa1b89')
    ledger_ref = retain('evidence/research_cost_ledger.json')
    ledger = read_json(assets / ledger_ref['path'])
    remaining = ledger['limit'] - ledger['measured_or_conservatively_charged'] - ledger['unmetered_setup_reserve']
    if abs(remaining - ledger['remaining_after_reserve']) > 1e-10 or remaining < 0:
        raise RuntimeError('Historical aggregate GPU accounting is inconsistent.')
    concurrent_ref = retain(CONCURRENT_PATH)
    for name in ('valid_target_mask_0.npy', 'train_target_mask_0.npy', 'target_assign_matrix_0.npz'):
        retain('Datasets/GossipCop/data_split/' + name)
    output.mkdir(parents=True, exist_ok=False)
    scope = dict(
        schema='monti_downstream_cooperation_metadata_scope_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_manifest=manifest, runtime_commit=RUNTIME_COMMIT,
        input_refs=[dict(root=k[0], path=k[1], sha256=v) for k, v in sorted(refs.items())],
        new_ID_array_reads_started=False, outcome_reads_allowed=False,
        scope='Read original TRAIN/VALID group IDs and assignment membership only. Exclude four pre-existing VALID lists. No ML import, model deserialization or predictions.',
        owner_decision='Project each group to the closest feasible half-target quota while preserving its original total budget; report actual ratios.',
    )
    write_json(output / 'metadata_scope.json', scope)

    import numpy as np
    import scipy.sparse as sp
    population = np.load(assets / 'Datasets/GossipCop/data_split/valid_target_mask_0.npy', allow_pickle=False)
    train_population = np.load(assets / 'Datasets/GossipCop/data_split/train_target_mask_0.npy', allow_pickle=False)
    ids = population.astype(np.int64)
    if population.ndim != 1 or not np.array_equal(population, ids) or len(ids) != 592 or len(set(ids.tolist())) != 592:
        raise RuntimeError('Expected the exact original 592 VALID group IDs.')
    concurrent = read_json(assets / CONCURRENT_PATH)
    exclusions = [previous['excluded_original_valid_groups'], previous['excluded_sharpen_valid_groups'],
                  previous['valid_groups'], concurrent['valid_groups']]
    excluded = set(value for row in exclusions for value in row)
    if any(len(row) != 128 or len(set(row)) != 128 for row in exclusions) or len(excluded) != 512:
        raise RuntimeError('Four disjoint, unchanged prior VALID cohorts are required.')
    if not excluded <= set(map(int, ids)):
        raise RuntimeError('A prior cohort includes non-VALID groups.')
    eligible = sorted(set(map(int, ids)) - excluded)
    if len(eligible) != 80:
        raise RuntimeError('Expected 80 unused original contexts; do not redraw or replace.')
    valid = random.Random(20261112).sample(eligible, 64)
    train_base = list(map(int, base['train_groups']))
    if len(train_base) != 256 or len(set(train_base)) != 256 or not set(train_base) <= set(map(int, train_population)):
        raise RuntimeError('The original 256-group TRAIN population changed.')
    train = random.Random(20261111).sample(sorted(train_base), 128)
    if set(train) & set(valid):
        raise RuntimeError('TRAIN and VALID group IDs overlap.')
    assignment = sp.load_npz(assets / 'Datasets/GossipCop/data_split/target_assign_matrix_0.npz').tocsr()

    def members(group):
        return list(map(int, assignment.getrow(group).nonzero()[1]))

    valid_members = {str(group): members(group) for group in valid}
    train_members = {str(group): members(group) for group in train}
    if not all(valid_members.values()) or not all(train_members.values()):
        raise RuntimeError('An empty selected group cannot be silently replaced.')
    valid_nodes = {node for row in valid_members.values() for node in row}
    train_nodes = {node for row in train_members.values() for node in row}
    old_nodes = {node for group in excluded for node in members(group)}
    if train_nodes & valid_nodes:
        raise RuntimeError('Original TRAIN/VALID target identities overlap.')
    protocol = dict(
        schema='monti_downstream_cooperation_v1', status='frozen', post_hoc=True,
        owner_execution_authorized=True, final_TEST_allowed=False,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=str(SOURCE_ROOT), runtime_root=str(runtime), asset_root=str(assets),
        source_manifest=manifest, runtime_commit=RUNTIME_COMMIT,
        runtime_tree=subprocess.check_output(['git', 'rev-parse', RUNTIME_COMMIT + '^{tree}'], cwd=runtime, text=True).strip(),
        input_refs=scope['input_refs'], metadata_scope_sha256=sha(output / 'metadata_scope.json'),
        base_experiment_ref=dict(path=BASE_PATH, sha256=BASE_SHA),
        concurrent_context_exclusion_ref=concurrent_ref,
        seeds=list(SEEDS), arms=list(ARMS), train_groups=train, valid_groups=valid,
        train_members=train_members, valid_members=valid_members,
        selection=dict(train_seed=20261111, valid_seed=20261112, original_train_population=train_base,
                       original_valid_population=list(map(int, ids)), excluded_ordered_valid_cohorts=exclusions,
                       eligible_valid_sorted=eligible, no_size_label_outcome_selection=True,
                       existing_group_membership_unchanged=True),
        cohort=dict(train_groups=128, valid_groups=64, train_target_occurrences=sum(map(len, train_members.values())),
                    valid_target_occurrences=sum(map(len, valid_members.values())),
                    unique_train_targets=len(train_nodes), unique_valid_targets=len(valid_nodes),
                    valid_target_overlap_with_previous_contexts=len(valid_nodes & old_nodes),
                    interpretation='Unused VALID contexts, not independent unseen news; original target identities can overlap earlier development groups.'),
        settings=dict(downstream_epochs=4, target_fraction=0.5, quota_policy='closest_feasible',
                      target_quota_rounding='ceil', mandatory_target_edge_per_injected_node=True,
                      injected_to_injected_edges_in_quota_arms=False,
                      total_external_capacity_failure='stop_before_training_and_preserve_failure',
                      checkpoint='terminal_epoch_only', optimizer='Adam', learning_rate=0.001,
                      weight_decay=0.0001, hidden_dimension=64,
                      loss='uniform_mean_softplus_fraud_minus_benign',
                      upstream_frozen=True, original_GCN_frozen=True, only_downstream_parameters_optimized=True),
        device=dict(kind='cpu', cuda_visible_devices='', uuid=None, selection='CPU-only registered experiment; no GPU fallback or allocation.'),
        environment=dict(LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib',
                         PYTHONHASHSEED='0', DGLBACKEND='pytorch'),
        resource=dict(CPU_phase_seconds=14400, GPU_phase_seconds=0, preflight_inner_seconds=900,
                      main_inner_seconds=10800, outer_extra_seconds=60, threads=4, automatic_retry=False),
        budget=dict(ledger=ledger_ref, charged_GPU_hours=ledger['measured_or_conservatively_charged'],
                    reserve_GPU_hours=ledger['unmetered_setup_reserve'], remaining_GPU_hours=remaining,
                    new_GPU_cap_hours=0, no_ledger_mutation_during_job=True),
        preflight=dict(groups=train[:4], source_seed=3, logit_atol=1e-4, sequence_atol=1e-5,
                       finite_nonzero_downstream_gradient_required=True,
                       upstream_and_classifier_state_hashes_unchanged=True),
        logical_main_caps=dict(generator_forwards=576, train_GCN_forwards=1536,
                               validation_GCN_forwards=832, validation_GAGA_forwards=832,
                               optimizer_updates=1536, autograd_calls=1536, API_calls=0, TEST_queries=0),
        qualification=dict(primary='quota_learned', mean_gain_at_least_pp=1.0,
                           all_three_seed_deltas_positive=True, paired_t95_lower_positive=True,
                           mean_above_both_quota_controls=True, controls_not_candidates=True),
    )
    prior = assets / 'runs/downstream_cooperation_20261111_attempt1/execution/preflight_guardian.json'
    if prior.exists():
        old = read_json(prior)
        if old['status'] != 'failed':
            raise RuntimeError('Only an explicitly reviewed failed startup may precede this amendment.')
        failure_result = read_json(pathlib.Path(old['output']) / 'result.json')
        if any(failure_result['counters'].values()) or 'libcusparse.so.11' not in failure_result.get('error', ''):
            raise RuntimeError('This environment amendment is limited to the zero-query library startup failure.')
        protocol['setup_amendment'] = dict(prior_guardian_sha256=sha(prior), prior_result_sha256=sha(pathlib.Path(old['output']) / 'result.json'),
            prior_guardian_wall_seconds=old['guardian_wall_seconds'], prior_process_cpu_seconds=old['child_process_cpu_seconds'],
            reason='The first launcher omitted the recorded environment lib directory required by CUDA-linked DGL even on CPU.',
            scientific_policy_changed=False, previous_model_queries=0, previous_GPU_hours=0)
        protocol['resource']['CPU_phase_seconds'] -= int(old['guardian_wall_seconds']) + 1
    write_json(output / 'protocol.json', protocol)
    print(json.dumps(dict(status='frozen_no_model_calls', protocol_sha256=sha(output / 'protocol.json'),
                          source_commit=manifest['source_commit'], train_groups=128, valid_groups=64,
                          valid_target_occurrences=protocol['cohort']['valid_target_occurrences'],
                          GPU_hours=0, model_queries=0), sort_keys=True))


if __name__ == '__main__':
    main()
