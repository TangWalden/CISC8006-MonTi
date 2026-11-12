"""Register the action-blind control using completed frozen parent artifacts."""
import argparse
import copy
import datetime
import hashlib
import json
import pathlib
import subprocess


SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
PARENT_SOURCE = '9e013cb182b5af71a9d0b719b5a939deda3c88e2'
PARENT = pathlib.PurePosixPath('runs/downstream_cooperation_20261111_attempt3')
PARENT_RUN = PARENT / 'execution/main_attempt1'
SEEDS = (3, 4, 5)
ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned', 'quota_blind')
TOP_REFERENCES = {
    'parent_protocol': ((PARENT / 'registration/protocol.json').as_posix(),
                        '5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'),
    'parent_result': ((PARENT_RUN / 'result.json').as_posix(),
                      'f19aee765b85199343d49861949995fca0a6e72838954610a8351515977da7be'),
    'parent_cache_manifest': ((PARENT_RUN / 'upstream_cache_manifest.json').as_posix(),
                              '5f7afb09270a7d618d2e37e9cfa54690d5dc4eb38c2ac01481eaa68bfde5ce50'),
    'parent_seal': ((PARENT_RUN / 'sealed_before_victim.json').as_posix(),
                    '9599baaf5d6baf47374e2e49da6d4e15c555dc5359a4116ae0a9b89c95b6af61'),
    'parent_raw_evaluation': ((PARENT_RUN / 'raw/evaluation.jsonl').as_posix(),
                              '5ac8cf19034ffde18008c28e745bbed4da3c98fb7029844a8166f146ac760e7d'),
    'parent_run_guardian': ((PARENT / 'execution/run_guardian.json').as_posix(),
                            '19f543d44377bc7f53160222c849a58c9900e2dbc7d23971943f43e1f59fb890'),
    'parent_logit_audit': ((PARENT / 'post_audits/saved_logit_audit_v1.json').as_posix(),
                           '4df9bc975e5e552aab9d6545e888a56d14ff58228d66a53720c075ad5fd83d3e'),
    'parent_tensor_audit': ((PARENT / 'post_audits/saved_tensor_audit_v1.json').as_posix(),
                            '3427a1fdf6fff74982ba2e6b464f5c9cfa3b2ee29d426e8bb7f0e0be78a78418'),
}


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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative_file(root, name):
    root = pathlib.Path(root).resolve()
    relative = pathlib.Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'Only explicit root-relative input paths are allowed.')
    path = (root / relative).resolve()
    require(root in path.parents and path.is_file() and not path.is_symlink(),
            'Input is missing, linked or outside its bound root: ' + str(name))
    return path


def verify_source_manifest(manifest):
    require(isinstance(manifest.get('source_commit'), str) and len(manifest['source_commit']) == 40,
            'A committed new scientific source identity is required.')
    files = manifest.get('files', {})
    mandatory = {'methods/downstream_quota.py', 'experiments/downstream_cooperation/common.py',
                 'experiments/downstream_cooperation/model.py'} | {
        'experiments/downstream_blind/' + name
        for name in ('prepare.py', 'driver.py', 'common.py', 'model.py', 'preflight.py', 'run.py')}
    require(mandatory <= set(files), 'The new isolated source inventory is incomplete.')
    for name, wanted in files.items():
        path = pathlib.PurePosixPath(name)
        allowed = name in ('methods/__init__.py', 'tests/__init__.py', 'methods/downstream_quota.py',
                           'experiments/downstream_cooperation/common.py', 'experiments/downstream_cooperation/model.py') or (
            len(path.parts) == 3 and path.parts[:2] == ('experiments', 'downstream_blind') and path.suffix == '.py'
        ) or (len(path.parts) == 2 and path.parts[0] == 'tests'
              and path.name.startswith('test_downstream') and path.suffix == '.py')
        require(allowed and sha(relative_file(SOURCE_ROOT, name)) == wanted,
                'Unexpected or changed new source: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', required=True)
    parser.add_argument('--source-manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    runtime = pathlib.Path(args.runtime_root).resolve()
    output = pathlib.Path(args.output).resolve()
    require((runtime / 'runs').resolve() in output.parents,
            'Register this control in a new private ignored runs directory.')
    manifest = read_json(args.source_manifest)
    verify_source_manifest(manifest)
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=runtime, text=True).strip() == RUNTIME_COMMIT,
            'The shared historical runtime must remain at its public archive commit.')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=runtime, text=True).strip(),
            'Historical runtime tracked files are dirty.')
    parent_refs, input_refs = {}, {}

    def retain(name, wanted=None, parent=False, root_key='asset'):
        path = relative_file(runtime, name)
        actual = sha(path)
        require(wanted is None or wanted == actual, 'Frozen input byte hash changed: ' + name)
        mapping = parent_refs if parent else input_refs
        key = (root_key, name)
        require(mapping.setdefault(key, actual) == actual, 'Conflicting byte identities.')
        return dict(root=root_key, path=name, sha256=actual)

    named = {name: retain(path, wanted, parent=True) for name, (path, wanted) in TOP_REFERENCES.items()}
    previous = read_json(runtime / named['parent_protocol']['path'])
    completed = read_json(runtime / named['parent_result']['path'])
    guardian = read_json(runtime / named['parent_run_guardian']['path'])
    parent_provenance = dict(source_commit=PARENT_SOURCE, runtime_commit=RUNTIME_COMMIT,
                             protocol_sha256=named['parent_protocol']['sha256'])
    require(previous.get('schema') == 'monti_downstream_cooperation_v1' and previous.get('status') == 'frozen'
            and previous.get('final_TEST_allowed') is False
            and previous['source_manifest']['source_commit'] == PARENT_SOURCE
            and tuple(previous['seeds']) == SEEDS, 'The parent registration identity differs.')
    require(completed.get('status') == 'completed' and completed.get('provenance') == parent_provenance,
            'Only the exact completed parent run can supply cached inputs and visible comparators.')
    require(guardian['status'] == 'completed' and guardian['exit_code'] == 0
            and guardian['child_result_sha256'] == named['parent_result']['sha256']
            and guardian['protocol_sha256'] == named['parent_protocol']['sha256']
            and guardian['whole_phase_GPU_hours'] == 0, 'Parent guardian completion/binding differs.')
    require(previous['device']['kind'] == 'cpu' and previous['device']['cuda_visible_devices'] == ''
            and previous['settings']['downstream_epochs'] == 4
            and previous['settings']['hidden_dimension'] == 64
            and previous['settings']['learning_rate'] == .001
            and previous['settings']['weight_decay'] == .0001,
            'Parent device, model dimensions or fixed training settings differ.')
    require(len(previous['train_groups']) == len(set(previous['train_groups'])) == 128
            and len(previous['valid_groups']) == len(set(previous['valid_groups'])) == 64
            and not set(previous['train_groups']) & set(previous['valid_groups']), 'Parent cohorts differ.')
    for ref in previous['input_refs']:
        retain(ref['path'], ref['sha256'], root_key=ref['root'])
    parent_source_root = pathlib.Path(previous['source_root']).resolve()
    require(runtime in parent_source_root.parents, 'Parent study source is outside the frozen runtime root.')
    for name, wanted in previous['source_manifest']['files'].items():
        relative = (parent_source_root / name).resolve().relative_to(runtime).as_posix()
        retain(relative, wanted, parent=True)
    for name in ('methods/downstream_quota.py', 'experiments/downstream_cooperation/common.py',
                 'experiments/downstream_cooperation/model.py'):
        require(manifest['files'][name] == previous['source_manifest']['files'][name],
                'The reused parent geometry/model dependency changed: ' + name)
    logit = read_json(runtime / named['parent_logit_audit']['path'])
    tensor = read_json(runtime / named['parent_tensor_audit']['path'])
    require(logit.get('status') == tensor.get('status') == 'passed'
            and logit['recount']['provenance'] == tensor['provenance'] == parent_provenance,
            'Both exact parent audits must have passed.')
    ledger = read_json(runtime / 'evidence/research_cost_ledger.json')
    require(abs(ledger['limit'] - ledger['measured_or_conservatively_charged']
                - ledger['unmetered_setup_reserve'] - ledger['remaining_after_reserve']) < 1e-10,
            'The inherited aggregate GPU ledger is inconsistent.')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'metadata_scope.json', dict(
        schema='monti_downstream_blind_metadata_scope_v1', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_manifest=manifest, runtime_commit=RUNTIME_COMMIT,
        parent_provenance=parent_provenance,
        parent_top_refs=named,
        inherited_input_refs=[dict(root=k[0], path=k[1], sha256=v) for k, v in sorted(input_refs.items())],
        cached_tensor_byte_hashes_started=False, tensor_deserialization_allowed=False,
        scope='Reuse exact parent cohorts and completed cached decisions. Hash own parent caches, plans and visible checkpoints; do not deserialize tensors, import ML runtimes or generate a new upstream result.',
        owner_authorized=True, no_outcome_based_group_or_hyperparameter_selection=True))
    cache = read_json(runtime / named['parent_cache_manifest']['path'])
    seal = read_json(runtime / named['parent_seal']['path'])
    require(cache['provenance'] == seal['provenance'] == parent_provenance
            and seal['upstream_cache_sha256'] == named['parent_cache_manifest']['sha256']
            and completed['pre_victim_seal_sha256'] == named['parent_seal']['sha256'], 'Parent cache/plan seal binding differs.')
    expected_cache = {(seed, phase, group) for seed in SEEDS
                      for phase, groups in (('train', previous['train_groups']), ('validation', previous['valid_groups']))
                      for group in groups}
    keys = [(row['seed'], row['phase'], row['group']) for row in cache['cases']]
    require(len(keys) == len(set(keys)) == 576 and set(keys) == expected_cache,
            'The parent cache inventory is not the exact 576-case Cartesian cohort.')
    full_blocks = {seed: dict(train=0, validation=0) for seed in SEEDS}
    for row in cache['cases']:
        filename = 's%d_%s_g%d.pt' % (row['seed'], row['phase'], row['group'])
        require(row['path'] == filename, 'Unsafe or misbound parent cache name.')
        retain((PARENT_RUN / 'cache' / filename).as_posix(), row['sha256'], parent=True)
        q = row['quota']
        if q['target_quota'] == q['injected_count'] * q['target_count']:
            full_blocks[row['seed']][row['phase']] += 1
    require(all(value == dict(train=84, validation=50) for value in full_blocks.values()),
            'The known full-target-block information limit differs from parent metadata.')
    plan_keys = [(row['seed'], row['group']) for row in seal['plans']]
    require(len(plan_keys) == len(set(plan_keys)) == 192
            and set(plan_keys) == {(seed, group) for seed in SEEDS for group in previous['valid_groups']},
            'The exact 192 parent plan inventory differs.')
    for row in seal['plans']:
        filename = 's%d_g%d.pt' % (row['seed'], row['group'])
        require(row['path'] == filename, 'Unsafe or misbound parent plan name.')
        retain((PARENT_RUN / 'plans' / filename).as_posix(), row['sha256'], parent=True)
    checkpoint_refs = {}
    visible_records = completed['downstream']
    require(len(visible_records) == 3 and {row['seed'] for row in visible_records} == set(SEEDS),
            'Parent terminal visible-model inventory differs.')
    for row in visible_records:
        seed = row['seed']
        wanted = seal['downstream_checkpoint_hashes'][str(seed)]
        require(row['final_checkpoint_sha256'] == wanted and row['epochs'] == 4
                and row['parameter_count'] == 151872, 'Parent visible training identity differs.')
        checkpoint_refs[str(seed)] = retain((PARENT_RUN / 'downstream' / ('s%d_final.pt' % seed)).as_posix(), wanted, parent=True)
    settings = copy.deepcopy(previous['settings'])
    settings.update(action_conditioning='hidden', downstream_epochs=4, hidden_dimension=64,
                    learning_rate=.001, weight_decay=.0001, optimizer='Adam', gradient_clip_norm=5.0,
                    initialization_seed_base=20261111,
                    training_order='Random(seed*1000+epoch).shuffle(original_train_groups)',
                    neutral_target_summary='neutral=ones((B,T)); neutral.matmul(x_targets)/neutral.sum(1,keepdim=True); parent-compatible all-target mean',
                    neutral_log_count='log1p(full((B,1),registered_target_quota/B))',
                    upstream_generation_allowed=False, visible_model_updates_allowed=False)
    protocol = dict(
        schema='monti_downstream_blind_v1', status='frozen', owner_execution_authorized=True,
        post_hoc=True, final_TEST_allowed=False,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=str(SOURCE_ROOT), runtime_root=str(runtime), asset_root=str(runtime),
        runtime_commit=RUNTIME_COMMIT, runtime_tree=previous['runtime_tree'], source_manifest=manifest,
        parent_root=str(runtime / PARENT), parent_protocol_path=str(runtime / named['parent_protocol']['path']),
        parent_run_root=str(runtime / PARENT_RUN), parent_provenance=parent_provenance,
        parent_refs=[dict(root=k[0], path=k[1], sha256=v) for k, v in sorted(parent_refs.items())],
        input_refs=[dict(root=k[0], path=k[1], sha256=v) for k, v in sorted(input_refs.items())],
        parent_checkpoint_refs=checkpoint_refs, parent_visible_records=visible_records,
        parent_visible_result_status='completed', metadata_scope_sha256=sha(output / 'metadata_scope.json'),
        train_groups=previous['train_groups'], valid_groups=previous['valid_groups'],
        train_members=previous['train_members'], valid_members=previous['valid_members'],
        cohort=previous['cohort'], selection=previous['selection'], new_context_selection=False,
        seeds=list(SEEDS), arms=list(ARMS), new_arm='quota_blind', comparators=['quota_learned', 'quota_random'],
        settings=settings, environment=previous['environment'], device=previous['device'],
        resource=dict(CPU_phase_seconds=14400, GPU_phase_seconds=0, main_inner_seconds=10800,
                      preflight_inner_seconds=900, outer_extra_seconds=60, threads=4, automatic_retry=False),
        budget=copy.deepcopy(previous['budget']),
        preflight=dict(groups=list(previous['preflight']['groups']), source_seed=3,
                       logit_atol=1e-4, finite_nonzero_downstream_gradient_required=True,
                       upstream_and_classifier_state_hashes_unchanged=True),
        information_limit=dict(full_target_block_cases_per_seed=full_blocks,
            full_target_block_train_cases_per_seed=84, full_target_block_valid_cases_per_seed=50,
            interpretation='In these cases all targets attach to every auxiliary node, so the visible attachment mean/count equal the blind replacements; no extra summary information is removed there.'),
        logical_main_caps=dict(upstream_generations=0, train_GCN_forwards=1536,
                               evaluation_GCN_forwards=192, official_GAGA_forwards=192,
                               optimizer_updates=1536, autograd_calls=1536,
                               TEST_queries=0, API_calls=0, GPU_hours=0),
        logical_preflight_caps=dict(upstream_generations=0, GCN_forwards=4,
                                    optimizer_updates=4, autograd_calls=4,
                                    GAGA_forwards=0, TEST_queries=0, API_calls=0, GPU_hours=0),
        qualification=dict(primary='visible_minus_blind', visible_arm='quota_learned',
                           require_all_three_seed_deltas_positive=True,
                           require_nominal_paired_t95_lower_above_zero=True,
                           secondary='blind_minus_same_parent_random', controls_not_candidates=True),
    )
    protocol.update(named)
    write_json(output / 'protocol.json', protocol)
    print(json.dumps(dict(status='frozen_no_model_calls', protocol_sha256=sha(output / 'protocol.json'),
                          source_commit=manifest['source_commit'], parent_cache_files_hashed=576,
                          parent_plan_files_hashed=192, parent_visible_checkpoints_hashed=3,
                          model_queries=0, tensor_deserializations=0, GPU_hours=0), sort_keys=True))


if __name__ == '__main__':
    main()
