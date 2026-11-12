"""Register full TRAIN/TEST cohorts and frozen formal upstreams without models."""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
FORMAL_SOURCE = '9f782dfbb98581e8bbe3c10f5ea5de91237e09ab'
REGISTRY_SHA = '5c43fdb892a0657ac6923bb2617484a0db30ed989d48669f5109a5e8c944ca57'
SEEDS = (0, 1, 2)
ARMS = ('native_monti', 'quota_random', 'quota_learned')
FORMAL = {
    0: ('20261028T122617Z_reproduction_learned_s0_772a5d', 4, 14,
        'd1b566626db88b56637fc171c39dee312735a1252608b1c3181bf42cccdcb190',
        'e48d6d03470048114e0ceb6aaf598e15cac17d7267deceff69f59f4ea7ac8db5'),
    1: ('20261028T125935Z_reproduction_learned_s1_672f97', 3, 13,
        '628b1790e09d01a599d5e9907afe5f271df4108588cafde7b78fadf2af331b11',
        '8181cf84ecb2f382bf54a64ece0c2d755cdcc36f4bc8eb493b32d31c7d575b9d'),
    2: ('20261028T132611Z_reproduction_learned_s2_8ba576', 5, 15,
        '93d3515e9a96132bb05201f4815a0bb67976e47a63aeb11633a1627cf6df1fc0',
        '35b66737c49529283cb571e9137f7877d62f172dd910f0d21de1e6fd91f8958e'),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def relative_file(root, name):
    root = pathlib.Path(root).resolve()
    rel = pathlib.PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe input reference.')
    path = (root / str(rel)).resolve(strict=True)
    require(root in path.parents and path.is_file(), 'Input escapes its frozen root.')
    return path


def verify_source(manifest):
    commit = manifest.get('source_commit')
    require(isinstance(commit, str) and len(commit) == 40
            and all(c in '0123456789abcdef' for c in commit), 'A committed source identity is required.')
    files = manifest.get('files', {})
    mandatory = {'methods/downstream_quota.py', 'experiments/downstream_cooperation/common.py',
                 'experiments/downstream_cooperation/model.py',
                 'experiments/downstream_cooperation/budget_audit.py',
                 'tests/test_downstream_budget_audit.py', 'tests/test_downstream_quota.py',
                 'tests/test_downstream_model.py'} | {
        'experiments/final_staged/' + name
        for name in ('prepare.py', 'loader.py', 'common.py', 'preflight.py', 'run.py', 'driver.py')}
    require(mandatory <= set(files), 'The formal study source inventory is incomplete.')
    for name, wanted in files.items():
        rel = pathlib.PurePosixPath(name)
        allowed = name in ('methods/__init__.py', 'tests/__init__.py', 'methods/downstream_quota.py',
                           'experiments/downstream_cooperation/common.py', 'experiments/downstream_cooperation/model.py',
                           'experiments/downstream_cooperation/budget_audit.py') or (
            len(rel.parts) == 3 and rel.parts[:2] == ('experiments', 'final_staged') and rel.suffix == '.py'
        ) or (len(rel.parts) == 2 and rel.parts[0] == 'tests' and rel.suffix == '.py'
              and rel.name.startswith(('test_final_staged', 'test_downstream')))
        require(allowed and sha(relative_file(SOURCE_ROOT, name)) == wanted,
                'Unexpected or changed isolated study source: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root', 'source-manifest', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    runtime = pathlib.Path(args.runtime_root).resolve(strict=True)
    output = pathlib.Path(args.output).resolve()
    require((runtime / 'runs').resolve() in output.parents and not output.exists(),
            'Use a new ignored registration directory inside the historical runs root.')
    manifest = read_json(args.source_manifest)
    verify_source(manifest)
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() == RUNTIME_COMMIT
            and not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip(),
            'Keep the historical runtime at the clean frozen ca093 source.')
    refs = {}

    def retain(name, wanted=None, kind='asset'):
        actual = sha(relative_file(runtime, name))
        require(wanted is None or actual == wanted, 'Frozen bytes changed: ' + name)
        key = kind, name
        require(refs.setdefault(key, actual) == actual, 'Conflicting frozen input identities.')
        return dict(root=kind, path=name, sha256=actual)

    registry_ref = retain('evidence/run_registry.json', REGISTRY_SHA)
    registry = read_json(runtime / registry_ref['path'])
    frozen_ref = retain('course/frozen_protocol.json')
    frozen = read_json(runtime / frozen_ref['path'])
    for item in frozen['inputs'] + frozen.get('caches', []):
        retain(item['path'], item['sha256'])
    for item in frozen['scientific_code']:
        retain(item['path'], item['sha256'], 'runtime')
    for name in ('course/dual_surrogate/common.py', 'course/dual_surrogate/runtime.py',
                 'course/dual_surrogate/experiment.py', 'course/dual_surrogate/verify.py',
                 'course/dual_surrogate/sequence_adapter.py', 'models/victim_model.py',
                 'modules/data_handler.py', 'modules/experiment_handler.py', 'modules/schedulers.py',
                 'models/attack_model.py', 'utils/utils.py', 'utils/constants.py'):
        retain(name, kind='runtime')
    baselines = {}
    for seed in SEEDS:
        run_id, best, terminal, checkpoint_sha, config_sha = FORMAL[seed]
        matched = [row for row in registry if row['registration']['run_id'] == run_id]
        require(len(matched) == 1, 'Exactly one historical formal learned run is required.')
        row = matched[0]
        reg, result = row['registration'], row['status']
        require(reg['commit'] == FORMAL_SOURCE and reg['tracked_changes'] == ''
                and reg['phase'] == 'reproduction' and reg['mode'] == 'learned' and reg['seed'] == seed
                and result['status'] == 'completed' and result['best_epoch'] == best
                and result['completed_epochs'] == terminal and result['checkpoint_sha256'] == checkpoint_sha
                and result['train_groups'] == 937 and result['valid_groups'] == 592,
                'Historical formal checkpoint metadata differs.')
        directory = pathlib.PurePosixPath(row['evidence'])
        config_ref = retain((directory / 'config.json').as_posix(), config_sha)
        config = read_json(runtime / config_ref['path'])
        require(config['seed'] == seed and config['random_candidates'] is False
                and config['without_candidates'] is False and config['ada_budget'] is True
                and config['shared_parameters'] is False and config['pos_encoding'] is True
                and config['masking'] is False and config['max_candidates'] == 128
                and config['dataset'] == 'gossipcop_0.05' and config['split'] == 0
                and config['surrogate_model'] == 'GCN' and config['victim_model'] == 'GAGA',
                'The exact formal learned architecture/flags changed.')
        result_ref = retain((directory / 'result.json').as_posix())
        registered_ref = retain((directory / 'registered.json').as_posix())
        require(read_json(runtime / result_ref['path']) == result
                and read_json(runtime / registered_ref['path']) == reg, 'Registry/receipt copies differ.')
        baselines[str(seed)] = dict(run_id=run_id, execution_source=FORMAL_SOURCE,
            best_epoch=best, completed_epochs=terminal, scheduler_restored_steps=terminal,
            config_ref=config_ref, result_ref=result_ref, registered_ref=registered_ref,
            target_groups_ref=retain((directory / 'target_groups.json').as_posix()),
            checkpoint_ref=retain(result['checkpoint'], checkpoint_sha))
    ledger_ref = retain('evidence/research_cost_ledger.json')
    ledger = read_json(runtime / ledger_ref['path'])
    require(abs(ledger['limit'] - ledger['measured_or_conservatively_charged']
                - ledger['unmetered_setup_reserve'] - ledger['remaining_after_reserve']) < 1e-10,
            'The inherited GPU ledger is inconsistent.')
    split_dir = 'Datasets/GossipCop/data_split/'
    for name in ('train_target_mask_0.npy', 'test_target_mask_0.npy', 'target_assign_matrix_0.npz',
                 'train_idx_surro_0.npy', 'valid_idx_surro_0.npy', 'test_idx_surro_0.npy'):
        retain(split_dir + name)
    output.mkdir(parents=True, exist_ok=False)
    scope = dict(schema='monti_final_staged_metadata_scope_v1',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_manifest=manifest, runtime_commit=RUNTIME_COMMIT, formal_baselines=baselines,
        input_refs=[dict(root=k[0], path=k[1], sha256=v) for k, v in sorted(refs.items())],
        new_ID_array_reads_started=False, model_imports_allowed=False,
        checkpoint_deserialization_allowed=False, outcome_based_selection_allowed=False,
        scope='Read the complete original TRAIN/TEST group-ID arrays and target membership only after this metadata record. No labels, features, predictions, models or training.',
        owner_execution_authorized=True, historical_TEST_baseline_previously_inspected=True)
    write_json(output / 'metadata_scope.json', scope)

    import numpy as np
    import scipy.sparse as sp
    def group_ids(name, count):
        raw = np.load(runtime / (split_dir + name), allow_pickle=False)
        integer = raw.astype(np.int64)
        require(raw.ndim == 1 and np.array_equal(raw, integer)
                and len(integer) == len(set(integer.tolist())) == count, 'Original full group population differs.')
        return list(map(int, integer))
    train = group_ids('train_target_mask_0.npy', 937)
    test = group_ids('test_target_mask_0.npy', 909)
    require(not set(train) & set(test), 'Original TRAIN/TEST groups overlap.')
    for baseline in baselines.values():
        original = read_json(runtime / baseline['target_groups_ref']['path'])
        require(original['train'] == train and len(original['valid']) == 592 and original['test_count'] == len(test),
                'The formal baseline population differs from the full original split.')
    assignment = sp.load_npz(runtime / (split_dir + 'target_assign_matrix_0.npz')).tocsr()
    def memberships(groups):
        return {str(group): list(map(int, assignment.getrow(group).nonzero()[1])) for group in groups}
    train_members, test_members = memberships(train), memberships(test)
    require(all(train_members.values()) and all(test_members.values()), 'Empty groups cannot be filtered or replaced.')
    train_nodes = {n for values in train_members.values() for n in values}
    test_nodes = {n for values in test_members.values() for n in values}
    require(not train_nodes & test_nodes, 'Original TRAIN/TEST target identities overlap.')
    require(sum(map(len, test_members.values())) == 10515, 'The complete formal TEST occurrence denominator differs.')
    split_counts = {}
    for phase in ('train', 'valid', 'test'):
        values = np.load(runtime / (split_dir + phase + '_idx_surro_0.npy'), allow_pickle=False)
        split_counts[phase] = int(len(values))
    require(split_counts == dict(train=6595, valid=3264, test=6629), 'Original all-node split counts differ.')
    preflight = [839, 835, 910, 238]
    require(set(preflight) <= set(train), 'The four preflight cases must all remain original TRAIN groups.')
    protocol = dict(schema='monti_final_staged_v1', status='frozen', owner_execution_authorized=True,
        final_TEST_allowed=True, frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=str(SOURCE_ROOT), runtime_root=str(runtime), asset_root=str(runtime),
        source_manifest=manifest, runtime_commit=RUNTIME_COMMIT,
        runtime_tree=subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=runtime, text=True).strip(),
        input_refs=scope['input_refs'], metadata_scope_sha256=sha(output / 'metadata_scope.json'),
        formal_baselines=baselines, historical_frozen_protocol=frozen_ref, registry_ref=registry_ref,
        seeds=list(SEEDS), arms=list(ARMS), train_groups=train, test_groups=test, valid_groups=[],
        train_members=train_members, test_members=test_members,
        cohort=dict(train_groups=937, test_groups=909, valid_evaluations=0,
            train_target_occurrences=sum(map(len, train_members.values())), test_target_occurrences=10515,
            unique_train_targets=len(train_nodes), unique_test_targets=len(test_nodes), all_split_nodes=split_counts,
            historical_TEST_baseline_previously_inspected=True,
            interpretation='Complete original split; overlapping group-node occurrences are correlated. This is a freshly paired CPU replay, not promised historical prediction identity.'),
        settings=dict(downstream_epochs=4, hidden_dimension=64, optimizer='Adam', learning_rate=.001,
            weight_decay=.0001, gradient_clip_norm=5., initialization_seed_base=20261111,
            training_order='Random(seed*1000+epoch).shuffle(original_train_groups)',
            target_fraction=.5, quota_policy='closest_feasible', target_quota_rounding='ceil',
            mandatory_target_edge_per_injected_node=True, injected_to_injected_edges_in_quota_arms=False,
            total_external_capacity_failure='stop_before_training_and_preserve_failure',
            checkpoint='terminal_epoch_only', loss='uniform_mean_softplus_fraud_minus_benign',
            upstream_frozen=True, original_GCN_frozen=True, only_downstream_parameters_optimized=True,
            native_scoring_role='historical_full_projection_e_src_unchanged',
            scheduler_restore='completed_epochs14_13_15_like_historical_test',
            capture_rng_epoch='selected_best_epoch4_3_5_with_registered_per_case_fork_rng',
            cache_generation='Once per seed/group shared by native, random and learned arms; no regeneration per epoch/arm.',
            official_GAGA_loading='Only after all trained weights and every final TEST topology are sealed.',
            validation_evaluation_allowed=False),
        device=dict(kind='cpu', cuda_visible_devices='', uuid=None),
        environment=dict(LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib', PYTHONHASHSEED='0', DGLBACKEND='pytorch'),
        resource=dict(CPU_phase_seconds=57600, GPU_phase_seconds=0, preflight_inner_seconds=900,
            main_inner_seconds=43200, outer_extra_seconds=120, threads=4, automatic_retry=False,
            minimum_disk_free_bytes=20 * 1024 ** 3, minimum_memory_available_bytes=8 * 1024 ** 3),
        budget=dict(ledger=ledger_ref, charged_GPU_hours=ledger['measured_or_conservatively_charged'],
            reserve_GPU_hours=ledger['unmetered_setup_reserve'], remaining_GPU_hours=ledger['remaining_after_reserve'],
            new_GPU_cap_hours=0, no_ledger_mutation_during_job=True),
        preflight=dict(seed=0, groups=preflight, logit_atol=1e-4,
            finite_nonzero_downstream_gradient_required=True, TEST_allowed=False),
        logical_main_caps=dict(generator_forwards=5538, train_GCN_forwards=11244,
            evaluation_GCN_forwards=9090, evaluation_GAGA_forwards=9090,
            optimizer_updates=11244, autograd_calls=11244, validation_queries=0, API_calls=0),
        logical_preflight_caps=dict(generator_forwards=4, GCN_forwards=12, optimizer_updates=4,
            autograd_calls=4, TEST_queries=0, validation_queries=0, API_calls=0),
        qualification=dict(primary='quota_learned-minus-native_monti_gaga',
            require_mean_gain_positive=True, require_all_three_seed_deltas_positive=True,
            require_nominal_paired_t95_lower_above_zero=True,
            random_comparison='secondary_only_not_a_new_selection_rule', controls_not_candidates=True),
        historical_baseline_limit='43.7280 percent is the old learned/random reproduction result and is not substituted for this matched CPU native arm.')
    write_json(output / 'protocol.json', protocol)
    print(json.dumps(dict(status='frozen_no_model_calls', protocol_sha256=sha(output / 'protocol.json'),
        seeds=list(SEEDS), train_groups=len(train), test_groups=len(test),
        train_target_occurrences=protocol['cohort']['train_target_occurrences'],
        test_target_occurrences=10515, model_calls=0, training_steps=0, GPU_hours=0), sort_keys=True))


if __name__ == '__main__':
    main()
