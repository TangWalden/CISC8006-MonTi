"""Post-execution saved-logit recount; no tensor/model deserialization or queries."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ARMS = ('native_monti', 'quota_original', 'quota_random', 'quota_learned')
SEEDS = (3, 4, 5)


def require(condition, message):
    if not condition:
        raise ValueError(message)


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
    require(p['schema'] == 'monti_downstream_cooperation_v1' and p['status'] == 'frozen'
            and p['owner_execution_authorized'] is True and p['final_TEST_allowed'] is False
            and tuple(p['arms']) == ARMS and tuple(p['seeds']) == SEEDS, 'Frozen development protocol differs.')
    require(p['logical_main_caps'] == dict(generator_forwards=576, train_GCN_forwards=1536,
            validation_GCN_forwards=832, validation_GAGA_forwards=832, optimizer_updates=1536,
            autograd_calls=1536, API_calls=0, TEST_queries=0), 'Registered logical query caps differ.')
    groups = p['valid_groups']; train = p['train_groups']
    require(len(groups) == len(set(groups)) == 64 and len(train) == len(set(train)) == 128
            and not set(groups).intersection(train), 'Registered group cohorts differ.')
    members = {int(key): value for key, value in p['valid_members'].items()}
    require(set(members) == set(groups) and all(value and len(value) == len(set(value)) for value in members.values()),
            'Registered target membership is empty, duplicated or incomplete.')
    denominator = sum(map(len, members.values()))
    require(denominator == p['cohort']['valid_target_occurrences'] == 571, 'Complete target denominator differs.')
    result = read('result.json')
    require(result.get('status') == 'completed', 'Producer did not complete; partial results cannot pass.')
    unique_targets = len({node for group in groups for node in members[group]})
    require(unique_targets == p['cohort']['unique_valid_targets'] == 240, 'Unique registered target count differs.')
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
    expected_caches = {(s, phase, g) for s in SEEDS for phase, cohort in (('train', train), ('validation', groups)) for g in cohort}
    for row in cache['cases']:
        key = row['seed'], row['phase'], row['group']
        require(key not in caches, 'Duplicate upstream cache record.')
        require(row['path'] == 's%d_%s_g%d.pt' % key, 'Unsafe or misbound upstream cache name.')
        file_hash('cache/' + row['path'], row['sha256'])
        selected = p['train_members'] if row['phase'] == 'train' else p['valid_members']
        require(row['targets'] == len(selected[str(row['group'])]), 'Cached target count differs.')
        quota_check(row['quota'], row['targets']); caches[key] = row
    require(set(caches) == expected_caches, 'The576 upstream cache records differ.')
    require(cache['receipts'] == result['upstream_receipts'], 'Upstream receipt copies differ.')
    require({row['seed'] for row in cache['receipts']} == set(SEEDS) and len(cache['receipts']) == 3,
            'Three upstream freeze declarations are required.')
    for row in cache['receipts']:
        require(row['before'] == row['after'] and row['optimizer_updates'] == row['classifier_queries_during_generation'] == 0,
                'Declared frozen upstream or teacher changed.')
    require(result['downstream'] == seal['downstream_records'], 'Terminal downstream records differ.')
    require(len(result['downstream']) == 3 and {row['seed'] for row in result['downstream']} == set(SEEDS), 'Terminal seed inventory differs.')
    for row in result['downstream']:
        require(row['epochs'] == 4 and row['optimizer_updates'] > 0 and row['nonzero_gradient_cases'] > 0
                and row['initial_state_sha256'] != row['final_state_sha256'], 'No declared downstream learning evidence.')
        wanted = seal['downstream_checkpoint_hashes'][str(row['seed'])]
        require(wanted == row['final_checkpoint_sha256'], 'Checkpoint hash declarations differ.')
        file_hash('downstream/s%d_final.pt' % row['seed'], wanted)
    plans = {}
    for row in seal['plans']:
        key = row['seed'], row['group']; require(key not in plans, 'Duplicate final plan record.')
        require(row['path'] == 's%d_g%d.pt' % key, 'Unsafe or misbound plan artifact name.')
        original = caches[key[0], 'validation', key[1]]
        require(row['quota'] == original['quota'] and row['frozen_upstream_feature_sha256'] == original['upstream_feature_sha256']
                and row['frozen_upstream_target_block_sha256'] == original['upstream_target_block_sha256']
                and set(row['topology_sha256']) == set(ARMS), 'Matched plan/cache declarations differ.')
        file_hash('plans/' + row['path'], row['sha256']); plans[key] = row
    require(set(plans) == {(s, g) for s in SEEDS for g in groups}, 'The192 final plan records differ.')
    raw_evaluation = bound('raw/evaluation.jsonl').read_bytes()
    input_hashes['raw/evaluation.jsonl'] = hashlib.sha256(raw_evaluation).hexdigest()
    rows = [json.loads(line) for line in raw_evaluation.decode('utf-8').splitlines() if line.strip()]
    require(len(rows) == 832, 'Expected64clean plus768attacked saved rows.')
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
    require(len(result['all_case_counts']) == len(declared) == 768 and declared == case_counts, 'Producer case-count copy differs.')
    for seed in SEEDS:
        for arm in ARMS:
            selected = [case_counts[seed, g, arm] for g in groups]
            require(sum(r['targets'] for r in selected) == denominator, 'A denominator was filtered.')
            metrics.append(dict(seed=seed, arm=arm, groups=64, occurrences=denominator,
                           **{d+'_mr_percent': 100*sum(r[d]['attacked_wrong'] for r in selected)/denominator for d in ('gcn', 'gaga')}))
    require(len(result['seed_metrics']) == 12, 'Seed metric inventory differs.')
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
    qualified = primary['mean_pp'] >= 1 and all(v > 0 for v in primary['seed_differences_pp']) and primary['nominal_paired_t95_pp'][0] > 0 and means['quota_learned'] > max(means['quota_original'], means['quota_random'])
    require(result['qualification_met'] is qualified and result['primary'] == 'quota_learned'
            and result['controls_promoted'] is False and result['full_TEST_evaluated'] is False
            and result['frozen_teacher_unchanged'] is True and result['frozen_victim_unchanged'] is True, 'Frozen decision/declaration differs.')
    c = result['counters']
    require(c['upstream_generations'] == 576 and c['upstream_optimizer_updates'] == c['API_calls'] == c['TEST_queries'] == result['GPU_hours'] == 0
            and c['evaluation_GCN_forwards'] == c['official_GAGA_forwards'] == 832
            and 0 < c['training_GCN_forwards'] == c['autograd_calls'] == c['downstream_optimizer_updates'] <= 1536
            and c['downstream_optimizer_updates'] + c['degenerate_training_cases'] == 1536
            and sum(r['optimizer_updates'] for r in result['downstream']) == c['downstream_optimizer_updates'], 'Logical execution counters differ.')
    return dict(provenance=provenance, saved_rows=832, clean_groups=64, attacked_cases=768,
                occurrence_denominator_per_arm_seed=denominator, unique_valid_targets=unique_targets, seed_metrics=metrics,
                gaga_mean_mr_percent=means, contrasts=contrasts, qualification_met=qualified,
                producer_logical_counters=c, serialized_cache_files_hashed=576,
                serialized_plan_files_hashed=192, terminal_checkpoint_files_hashed=3,
                projected_validation_cases=sum(r['quota']['projection_applied'] for r in plans.values()),
                target_fraction_range=[min(r['quota']['target_fraction'] for r in plans.values()), max(r['quota']['target_fraction'] for r in plans.values())])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv); start = time.perf_counter(); inputs = {}
    report = dict(schema='monti_downstream_saved_logit_recount_v1', status='failed',
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  auditor_source_sha256=sha(__file__), post_execution_source=True,
                  scope='Separately authored post-hoc saved-logit recount and declared artifact binding; not a new experiment or human verification.',
                  limitations=['Fraud target label1 is inherited from the registered task and producer checks, not reverified from dataset labels.',
                               'File-byte hashes and declared content hashes are checked without tensor deserialization; raw graph edges, feature tensors and model state hashes are not independently recomputed.',
                               'Matching sealed declarations does not independently establish chronological execution order or performance contribution of upstream conditioning.'])
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
