"""Author-profile saved-logit recount; no tensor/model deserialization or queries.

Validation bodies are adapted from final_staged.audit_saved. Profile/receipt
checks were authored separately from the loader and method. This is an AI-agent
audit, not human verification. One author state is reused across repetitions.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

ARMS = ('native_monti', 'quota_random', 'quota_learned')
SEEDS = (0, 1, 2)


def require(condition, message):
    if not condition:
        raise ValueError(message)


AUTHOR_CHECKPOINT_SHA = '9f5dc7dd594a8e8927449665ef3a07b777199c2436d6fd1cc1f7adf0fdb6c23d'
AUTHOR_CONFIG_SHA = 'e27fd6959f35ab1786f35e4e70c89bec5156da1a6577f4481287310ebf552c2a'


def author_protocol_check(p):
    require(p.get('schema') == 'monti_author_staged_v1'
            and p.get('execution_profile') == 'author_upstream'
            and p.get('guardian_lock') == 'author_staged_exclusive.lock'
            and p.get('formal_baselines') is None,
            'The single-author profile may not impersonate three formal checkpoints.')
    author = p['author_upstream']
    require(author['distinct_upstream_checkpoints'] == 1
            and author['author_configuration_seed'] == 3
            and author['capture_noise_repetition_seeds'] == list(SEEDS)
            and author['checkpoint_ref']['root'] == author['config_ref']['root'] == 'asset'
            and author['checkpoint_ref']['sha256'] == AUTHOR_CHECKPOINT_SHA
            and author['config_ref']['sha256'] == AUTHOR_CONFIG_SHA
            and p['settings']['scheduler_restore'] == 'fresh_official_inference_initial_schedulers_no_steps'
            and p['settings']['capture_rng_epoch'] == 'synthetic_epoch0_seeded_per_case_no_historical_bitwise_claim',
            'Official state/configuration or explicit repetition/scheduler scope changed.')


def author_receipts_check(rows, p):
    author_protocol_check(p)
    require(len(rows) == 3 and {row['seed'] for row in rows} == set(SEEDS),
            'Exactly three downstream/capture-noise repetitions are required.')
    common_states = None
    for row in rows:
        seed, receipt = row['seed'], row['loader_receipt']
        before, after = row['before'], row['after']
        require(set(before) == set(after) == {'upstream', 'teacher'}
                and before == after
                and all(isinstance(value, str) and len(value) == 64
                        and all(ch in '0123456789abcdef' for ch in value) for value in before.values())
                and row['optimizer_updates'] == row['classifier_queries_during_generation'] == 0,
                'Frozen author/GCN state declarations differ or updates were recorded.')
        if common_states is None:
            common_states = before
        require(before == common_states,
                'The three repetitions do not share the identical author and original-GCN states.')
        require(row['checkpoint_sha256'] == receipt['checkpoint_sha256']
                == receipt['final_model_sha256'] == AUTHOR_CHECKPOINT_SHA
                and receipt['config_sha256'] == AUTHOR_CONFIG_SHA
                and receipt['author_configuration_seed'] == 3
                and receipt['downstream_seed'] == receipt['capture_noise_seed'] == seed
                and receipt['distinct_author_checkpoint_count'] == 1
                and receipt['capture_epoch'] == receipt['scheduler_steps_applied'] == 0
                and receipt['restored_temperature'] == receipt['restored_epsilon'] == 10
                and receipt['upstream_state_sha256'] == before['upstream']
                and receipt['original_GCN_state_sha256'] == before['teacher']
                and receipt['official_GAGA_loaded'] is False
                and receipt['upstream_optimizer_steps'] == 0
                and receipt['strict_state_dictionary_compatible'] is True
                and not {'best_epoch', 'completed_epochs', 'scheduler_restored_steps'}.intersection(receipt),
                'Official author receipt differs or fabricates formal training/scheduler history.')
    return dict(distinct_author_checkpoint_count=1, author_configuration_seed=3,
                repetition_seeds=list(SEEDS), shared_declared_state_sha256=common_states,
                checkpoint_sha256=AUTHOR_CHECKPOINT_SHA, config_sha256=AUTHOR_CONFIG_SHA,
                scope='One frozen author state; three downstream/capture-noise repetitions, not independently trained author models. State declarations are hash-bound but weights are not deserialized.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close(a, b):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9),
            'Saved numeric summary differs from independent recount.')


def predictions(logits, n):
    require(isinstance(logits, list) and len(logits) == n, 'Logit row count differs.')
    out = []
    for row in logits:
        require(isinstance(row, list) and len(row) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) for v in row), 'Nonfinite or nonbinary-class logits.')
        out.append(max(range(2), key=lambda index: row[index]))
    return out


def transitions(clean, attacked):
    require(len(clean) == len(attacked), 'Prediction lengths differ.')
    return dict(occurrences=len(clean), clean_wrong=sum(v != 1 for v in clean),
                attacked_wrong=sum(v != 1 for v in attacked),
                new_wrong=sum(a == 1 and b != 1 for a, b in zip(clean, attacked)),
                recoveries=sum(a != 1 and b == 1 for a, b in zip(clean, attacked)))


def quota_check(q, targets):
    T, C, B, E = (q[k] for k in ('target_count', 'candidate_count', 'injected_count', 'edge_budget'))
    require(all(type(v) is int for v in (T, C, B, E)) and T == targets and T > 0 and 0 <= C <= 128 and B > 0 and E > 0,
            'Invalid declared quota dimensions.')
    lo, hi, ideal = max(B, E - B*C), min(B*T, E), (E + 1)//2
    require(lo <= hi, 'Declared two-role quota is infeasible.')
    target = max(lo, min(ideal, hi))
    expected = dict(ideal_target_quota=ideal, ideal_candidate_quota=E-ideal,
                    lower_target_quota=lo, upper_target_quota=hi,
                    target_quota=target, candidate_quota=E-target,
                    projection_applied=target != ideal, exact_half=2*target == E,
                    odd_budget=bool(E % 2), target_capacity_limited=ideal > hi,
                    mandatory_or_candidate_capacity_limited=ideal < lo,
                    injected_to_injected_quota=0, total_budget_preserved=True)
    require(all(q.get(k) == value for k, value in expected.items()), 'Quota projection or flags differ.')
    for name, value in dict(target_fraction=target/E, candidate_fraction=(E-target)/E,
                            requested_target_fraction=.5).items():
        close(q[name], value)
    return target/E


def recount(run_root, protocol_path, input_hashes):
    root = Path(run_root).resolve(strict=True)
    def bound(name):
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe referenced artifact path.')
        path = (root/relative).resolve(strict=True)
        require(root in path.parents and path.is_file(), 'Referenced artifact escapes run root.')
        return path
    def file_hash(name, expected=None):
        actual = sha(bound(name))
        require(expected is None or actual == expected, 'Artifact byte hash differs: ' + name)
        input_hashes[name] = actual
        return actual
    def read(name):
        raw = bound(name).read_bytes()
        input_hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8'))
    protocol_path = Path(protocol_path).resolve(strict=True)
    raw_protocol = protocol_path.read_bytes()
    input_hashes['EXPLICIT_PROTOCOL'] = hashlib.sha256(raw_protocol).hexdigest()
    p = json.loads(raw_protocol.decode('utf-8'))
    require(p['schema'] == 'monti_author_staged_v1' and p['status'] == 'frozen'
            and p['owner_execution_authorized'] is True and p['final_TEST_allowed'] is True
            and tuple(p['arms']) == ARMS and tuple(p['seeds']) == SEEDS, 'Frozen full comparison protocol differs.')
    author_protocol_check(p)
    require(p['source_manifest']['files'].get('experiments/author_staged/audit_saved.py') == sha(__file__),
            'Registered source manifest does not bind the author logit auditor bytes.')
    require(p['logical_main_caps'] == dict(generator_forwards=5538, train_GCN_forwards=11244,
            evaluation_GCN_forwards=9090, evaluation_GAGA_forwards=9090, optimizer_updates=11244,
            autograd_calls=11244, API_calls=0, validation_queries=0), 'Registered logical query caps differ.')
    require(p['qualification'] == dict(primary='quota_learned-minus-native_monti_gaga',
            require_mean_gain_positive=True, require_all_three_seed_deltas_positive=True,
            require_nominal_paired_t95_lower_above_zero=True,
            random_comparison='secondary_only_not_a_new_selection_rule', controls_not_candidates=True),
            'Registered primary/secondary criterion differs.')
    groups = p['test_groups']; train = p['train_groups']
    require(len(groups) == len(set(groups)) == 909 and len(train) == len(set(train)) == 937
            and not set(groups).intersection(train), 'Registered group cohorts differ.')
    members = {int(key): value for key, value in p['test_members'].items()}
    require(set(members) == set(groups) and all(value and len(value) == len(set(value)) for value in members.values()),
            'Registered target membership is empty, duplicated or incomplete.')
    denominator = sum(map(len, members.values()))
    require(denominator == p['cohort']['test_target_occurrences'] == 10515, 'Complete target denominator differs.')
    result = read('result.json')
    require(result.get('status') == 'completed', 'Producer did not complete; partial results cannot pass.')
    unique_targets = len({node for group in groups for node in members[group]})
    require(unique_targets == p['cohort']['unique_test_targets'], 'Unique registered target count differs.')
    summary = read('summary.json'); seal = read('sealed_before_victim.json'); cache = read('upstream_cache_manifest.json')
    require('all_case_counts' not in summary and summary.get('status') == 'completed'
            and all(key in result and result[key] == value for key, value in summary.items()), 'Saved summary/result copies differ.')
    provenance = dict(source_commit=p['source_manifest']['source_commit'], runtime_commit=p['runtime_commit'],
                      protocol_sha256=input_hashes['EXPLICIT_PROTOCOL'])
    require(all(item['provenance'] == provenance for item in (result, seal, cache)), 'Source/protocol binding differs.')
    require(result['pre_victim_seal_sha256'] == input_hashes['sealed_before_victim.json']
            and seal['upstream_cache_sha256'] == input_hashes['upstream_cache_manifest.json']
            and seal['official_GAGA_loaded'] is False and seal['fitting_finished'] is True
            and cache['victim_loaded'] is False and cache['optimizer_updates_before_seal'] == 0,
            'Pre-victim seal declarations or byte bindings differ.')
    caches = {}
    expected_caches = {(s, phase, g) for s in SEEDS for phase, cohort in (('train', train), ('test', groups)) for g in cohort}
    for row in cache['cases']:
        key = row['seed'], row['phase'], row['group']
        require(key not in caches, 'Duplicate upstream cache record.')
        require(row['path'] == 's%d_%s_g%d.pt' % key, 'Unsafe or misbound upstream cache name.')
        file_hash('cache/' + row['path'], row['sha256'])
        selected = p['train_members'] if row['phase'] == 'train' else p['test_members']
        require(row['targets'] == len(selected[str(row['group'])]), 'Cached target count differs.')
        quota_check(row['quota'], row['targets']); caches[key] = row
    require(set(caches) == expected_caches, 'The5538 upstream cache records differ.')
    raw_cache = bound('raw/cache_progress.jsonl').read_bytes()
    input_hashes['raw/cache_progress.jsonl'] = hashlib.sha256(raw_cache).hexdigest()
    require([json.loads(line) for line in raw_cache.decode('utf-8').splitlines() if line.strip()] == cache['cases'],
            'Progress-cache manifest and completed manifest differ.')
    require(cache['receipts'] == result['upstream_receipts'], 'Upstream receipt copies differ.')
    require({row['seed'] for row in cache['receipts']} == set(SEEDS) and len(cache['receipts']) == 3,
            'Three upstream freeze declarations are required.')
    author_binding = author_receipts_check(cache['receipts'], p)
    require(result['downstream'] == seal['downstream_records'], 'Terminal downstream records differ.')
    require(len(result['downstream']) == 3 and {row['seed'] for row in result['downstream']} == set(SEEDS), 'Terminal seed inventory differs.')
    for row in result['downstream']:
        require(row['epochs'] == 4 and row['optimizer_updates'] > 0 and row['nonzero_gradient_cases'] > 0
                and row['initial_state_sha256'] != row['final_state_sha256']
                and row['parameter_count'] == 151872
                and row['checkpoint_selection'] == 'Terminal epoch4 only; no VALID or TEST selection.',
                'Declared downstream model, fitting or terminal selection differs.')
        wanted = seal['downstream_checkpoint_hashes'][str(row['seed'])]
        require(wanted == row['final_checkpoint_sha256'], 'Checkpoint hash declarations differ.')
        file_hash('downstream/s%d_final.pt' % row['seed'], wanted)
    raw_training = bound('raw/training.jsonl').read_bytes()
    input_hashes['raw/training.jsonl'] = hashlib.sha256(raw_training).hexdigest()
    training = [json.loads(line) for line in raw_training.decode('utf-8').splitlines() if line.strip()]
    expected_training = []
    for seed in SEEDS:
        for epoch in range(1, 5):
            order = list(train); random.Random(seed * 1000 + epoch).shuffle(order)
            expected_training.extend((seed, epoch, group) for group in order)
    require(len(training) == 11244 and
            [(row['seed'], row['epoch'], row['group']) for row in training] == expected_training,
            'Complete four-epoch TRAIN order or row inventory differs.')
    update_counts = {seed: 0 for seed in SEEDS}
    nonzero_counts = {seed: 0 for seed in SEEDS}
    skipped = 0
    for row in training:
        q = caches[row['seed'], 'train', row['group']]['quota']
        require(row['quota'] == q, 'Training quota differs from sealed cache.')
        has_choice = 0 < q['candidate_quota'] < q['injected_count'] * q['candidate_count']
        require(row['optimizer_update'] is has_choice, 'No-choice skip or trainable update differs.')
        if has_choice:
            require(math.isfinite(row['loss']) and math.isfinite(row['downstream_gradient_l2'])
                    and row['downstream_gradient_l2'] >= 0, 'Training loss/gradient is invalid.')
            update_counts[row['seed']] += 1
            nonzero_counts[row['seed']] += int(row['downstream_gradient_l2'] > 0)
        else:
            skipped += 1
    for row in result['downstream']:
        require(row['optimizer_updates'] == update_counts[row['seed']]
                and row['nonzero_gradient_cases'] == nonzero_counts[row['seed']],
                'Seed update/gradient census differs from raw training log.')
    plans = {}
    for row in seal['plans']:
        key = row['seed'], row['group']; require(key not in plans, 'Duplicate final plan record.')
        require(row['path'] == 's%d_g%d.pt' % key, 'Unsafe or misbound plan artifact name.')
        original = caches[key[0], 'test', key[1]]
        require(row['quota'] == original['quota'] and row['frozen_upstream_feature_sha256'] == original['upstream_feature_sha256']
                and row['frozen_upstream_target_block_sha256'] == original['upstream_target_block_sha256']
                and set(row['topology_sha256']) == set(ARMS), 'Matched plan/cache declarations differ.')
        file_hash('plans/' + row['path'], row['sha256']); plans[key] = row
    require(set(plans) == {(s, g) for s in SEEDS for g in groups}, 'The2727 final plan records differ.')
    raw_evaluation = bound('raw/evaluation.jsonl').read_bytes()
    input_hashes['raw/evaluation.jsonl'] = hashlib.sha256(raw_evaluation).hexdigest()
    rows = [json.loads(line) for line in raw_evaluation.decode('utf-8').splitlines() if line.strip()]
    require(len(rows) == 9090, 'Expected909clean plus8181attacked saved rows.')
    clean, attacked = {}, {}
    for row in rows:
        group = row['group']; require(group in members and row['node_ids'] == members[group], 'Logit target identity/order differs.')
        if row.get('stage') == 'clean':
            require(group not in clean, 'Duplicate clean record.')
            clean[group] = {d: predictions(row[d+'_logits'], len(members[group])) for d in ('gcn', 'gaga')}
        else:
            key = row['seed'], group, row['arm']; require(key not in attacked, 'Duplicate attacked record.')
            attacked[key] = row
    require(set(clean) == set(groups) and set(attacked) == {(s, g, a) for s in SEEDS for g in groups for a in ARMS}, 'Evaluation Cartesian cohort differs.')
    case_counts = {}; metrics = []
    for (seed, group, arm), row in attacked.items():
        plan = plans[seed, group]; q = plan['quota']; n = len(members[group])
        require(row['quota'] == q and row['targets'] == n and row['actual_nodes'] == q['injected_count']
                and row['actual_edges'] == q['edge_budget'] and row['topology_sha256'] == plan['topology_sha256'][arm]
                and row['feature_sha256'] == plan['frozen_upstream_feature_sha256'], 'Evaluation plan/hash/cost declarations differ.')
        if 'labels' in row:
            require(row['labels'] == [1]*n, 'Saved target labels differ from the known fraud-target assumption.')
        for detector in ('gcn', 'gaga'):
            require(row[detector] == transitions(clean[group][detector], predictions(row[detector+'_logits'], n)), 'Saved transition counts differ from logits.')
        case_counts[seed, group, arm] = {k: v for k, v in row.items() if k not in ('node_ids', 'gcn_logits', 'gaga_logits')}
    declared = {(r['seed'], r['group'], r['arm']): r for r in result['all_case_counts']}
    require(len(result['all_case_counts']) == len(declared) == 8181 and declared == case_counts, 'Producer case-count copy differs.')
    for seed in SEEDS:
        for arm in ARMS:
            selected = [case_counts[seed, g, arm] for g in groups]
            require(sum(r['targets'] for r in selected) == denominator, 'A denominator was filtered.')
            metrics.append(dict(seed=seed, arm=arm, groups=909, occurrences=denominator,
                           **{d+'_mr_percent': 100*sum(r[d]['attacked_wrong'] for r in selected)/denominator for d in ('gcn', 'gaga')}))
    require(len(result['seed_metrics']) == 9, 'Seed metric inventory differs.')
    for got, wanted in zip(result['seed_metrics'], metrics):
        require(all(got[k] == wanted[k] for k in ('seed', 'arm', 'groups', 'occurrences')), 'Metric keys/order differ.')
        for d in ('gcn', 'gaga'): close(got[d+'_mr_percent'], wanted[d+'_mr_percent'])
    means = {a: statistics.mean(r['gaga_mr_percent'] for r in metrics if r['arm'] == a) for a in ARMS}
    contrasts = {}
    for arm in ARMS[:-1]:
        diff = [next(r['gaga_mr_percent'] for r in metrics if r['seed'] == s and r['arm'] == 'quota_learned') -
                next(r['gaga_mr_percent'] for r in metrics if r['seed'] == s and r['arm'] == arm) for s in SEEDS]
        mean = statistics.mean(diff); radius = 4.302652729911275*statistics.stdev(diff)/math.sqrt(3)
        contrasts[arm] = dict(seed_differences_pp=diff, mean_pp=mean, nominal_paired_t95_pp=[mean-radius, mean+radius])
        saved = result['contrasts'][arm]
        for x, y in zip(saved['seed_differences_pp'], diff): close(x, y)
        close(saved['mean_pp'], mean)
        require(len(saved['nominal_paired_t95_pp']) == 2 and len(saved['seed_differences_pp']) == 3, 'Contrast shape differs.')
        for x, y in zip(saved['nominal_paired_t95_pp'], (mean-radius, mean+radius)): close(x, y)
    for arm in ARMS: close(result['gaga_mean_mr_percent'][arm], means[arm])
    primary = contrasts['native_monti']
    qualified = primary['mean_pp'] > 0 and all(v > 0 for v in primary['seed_differences_pp']) and primary['nominal_paired_t95_pp'][0] > 0
    require(result['qualification_met'] is qualified and result['primary'] == 'quota_learned'
            and result['controls_promoted'] is False and result['full_TEST_evaluated'] is True
            and result['frozen_teacher_unchanged'] is True and result['frozen_victim_unchanged'] is True, 'Frozen decision/declaration differs.')
    c = result['counters']
    require(c['upstream_generations'] == 5538 and c['upstream_optimizer_updates'] == c['API_calls'] == c['validation_queries'] == result['GPU_hours'] == 0
            and c['TEST_queries'] == 18180 and c['TEST_upstream_generations'] == c['TEST_downstream_policy_forwards'] == 2727
            and c['evaluation_GCN_forwards'] == c['official_GAGA_forwards'] == 9090
            and 0 < c['training_GCN_forwards'] == c['autograd_calls'] == c['downstream_optimizer_updates'] <= 11244
            and c['downstream_optimizer_updates'] + c['degenerate_training_cases'] == 11244
            and c['degenerate_training_cases'] == skipped
            and sum(r['optimizer_updates'] for r in result['downstream']) == c['downstream_optimizer_updates'], 'Logical execution counters differ.')
    return dict(provenance=provenance, author_upstream_binding=author_binding, saved_rows=9090, clean_groups=909, attacked_cases=8181,
                occurrence_denominator_per_arm_seed=denominator, unique_test_targets=unique_targets, seed_metrics=metrics,
                gaga_mean_mr_percent=means, contrasts=contrasts, qualification_met=qualified,
                producer_logical_counters=c, serialized_cache_files_hashed=5538,
                serialized_plan_files_hashed=2727, terminal_checkpoint_files_hashed=3,
                training_rows_recounted=len(training), training_updates_by_seed=update_counts,
                projected_TEST_cases=sum(r['quota']['projection_applied'] for r in plans.values()),
                target_fraction_range=[min(r['quota']['target_fraction'] for r in plans.values()), max(r['quota']['target_fraction'] for r in plans.values())])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv); start = time.perf_counter(); inputs = {}
    report = dict(schema='monti_author_staged_saved_logit_recount_v1', status='failed',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  auditor_source_sha256=sha(__file__), post_execution_source=False,
                  scope='Pre-registered independent saved-logit recount and declared artifact binding; not a new experiment or human verification.',
                  limitations=['Fraud target label1 is inherited from the registered task and producer checks, not reverified from dataset labels.',
                               'File-byte hashes and declared content hashes are checked without tensor deserialization; raw graph edges, feature tensors and model state hashes are not independently recomputed.',
                               'Matching sealed declarations does not independently establish chronological execution order or performance contribution of upstream conditioning.',
                               'Paired seed uncertainty is conditional on one shared author checkpoint, not variation across independently trained author models.'])
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        try:
            report['recount'] = recount(args.run_root, args.protocol, inputs); report['status'] = 'passed'
        except Exception as error:
            report['error_type'] = type(error).__name__; report['error'] = str(error)
        report.update(input_file_hashes=inputs, wall_seconds=time.perf_counter()-start,
                      model_queries=0, tensor_deserializations=0, checkpoint_deserializations=0,
                      optimizer_updates=0, API_calls=0, GPU_hours=0)
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False); stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'wall_seconds', 'model_queries', 'GPU_hours')}, sort_keys=True))
    if report['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__':
    main()
