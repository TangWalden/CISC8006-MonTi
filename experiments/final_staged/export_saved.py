"""Export completed formal evidence and bounded real graph examples, offline.

Every registered case remains in the public counts. Only the first two small
geometry-eligible TEST groups per seed include their exact added-edge subgraph.
This presentation filter cannot affect metrics or choose favourable outcomes.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time
import traceback

ARMS = ('native_monti', 'quota_random', 'quota_learned')
SEEDS = (0, 1, 2)
NAMES = dict(native_monti='Native MonTi', quota_random='Random candidate completion',
             quota_learned='Trained staged completion')
DEADLINE = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_time():
    if DEADLINE is not None and time.monotonic() >= DEADLINE:
        raise TimeoutError('The1800-second offline export wall cap was reached.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            check_time(); digest.update(block)
    return digest.hexdigest()


def opaque(seed, group):
    return hashlib.sha256(json.dumps([seed, group, 'final_staged_tensor_audit_v1']).encode('ascii')).hexdigest()[:20]


def prediction(logits, n):
    require(len(logits) == n and all(len(row) == 2 and all(type(v) in (int, float)
            and math.isfinite(v) for v in row) for row in logits), 'Invalid saved binary-class logits.')
    return [max(range(2), key=lambda index: row[index]) == 1 for row in logits]


def transition(clean, attacked):
    require(len(clean) == len(attacked), 'Saved prediction lengths differ.')
    return dict(occurrences=len(clean), clean_wrong=sum(not v for v in clean),
        attacked_wrong=sum(not v for v in attacked), new_wrong=sum(a and not b for a, b in zip(clean, attacked)),
        recoveries=sum(not a and b for a, b in zip(clean, attacked)))


def select_display_cases(protocol, geometry):
    """Metadata-only, registered-order selection; accepts no prediction inputs."""
    selected, eligible = set(), {}
    for seed in SEEDS:
        candidates = []
        for group in protocol['test_groups']:
            row = geometry[opaque(seed, group)]; q = row['planned_role_allocation']
            if q['injected_count'] <= 8 and q['target_count'] <= 32 and q['edge_budget'] <= 128 and q['candidate_quota'] > 0:
                candidates.append((seed, group))
        eligible[str(seed)] = len(candidates)
        selected.update(candidates[:2])
    return selected, eligible


def local_edges(matrix, targets, candidates, auxiliary):
    """Exact owned added edges using local role indices, without global IDs."""
    require(len(matrix) == auxiliary and all(len(row) == targets+candidates+auxiliary for row in matrix),
            'Display adjacency dimensions differ.')
    edges = []; counts = dict(target=0, candidate=0, auxiliary=0)
    for source, row in enumerate(matrix):
        for column, value in enumerate(row):
            require(value in (0, 1), 'Display adjacency is not binary.')
            if not value:
                continue
            if column < targets:
                role, destination = 'target', column
            elif column < targets+candidates:
                role, destination = 'candidate', column-targets
            else:
                role, destination = 'auxiliary', column-targets-candidates
                require(destination < source, 'Display auxiliary pair violates undirected ownership.')
            edges.append([source, role, destination]); counts[role] += 1
    return edges, counts


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    header = json.dumps([str(value.dtype), list(value.shape)]).encode('ascii')
    return hashlib.sha256(header+b'\0'+value.numpy().tobytes()).hexdigest()


def public_audit_summary(kind, record, raw_sha, protocol, post):
    """Publish audit conclusions without duplicating private file inventories."""
    require(record['status'] == 'passed', 'A failed audit cannot be summarized as passed.')
    if kind == 'saved_logit_recount':
        recount = record['recount']
        auditor_sha = record['auditor_source_sha256']
        require(auditor_sha == protocol['source_manifest']['files']['experiments/final_staged/audit_saved.py'],
                'Saved-logit auditor source differs from its scientific source registration.')
        source_commit = protocol['source_manifest']['source_commit']
        provenance = recount['provenance']
        counts = {name: recount[name] for name in ('saved_rows', 'clean_groups', 'attacked_cases',
            'occurrence_denominator_per_arm_seed', 'unique_test_targets', 'serialized_cache_files_hashed',
            'serialized_plan_files_hashed', 'terminal_checkpoint_files_hashed', 'training_rows_recounted')}
        counters = {name: record[name] for name in ('model_queries', 'tensor_deserializations',
            'checkpoint_deserializations', 'optimizer_updates', 'API_calls', 'GPU_hours')}
        limits = list(record['limitations'])
    else:
        require(kind == 'saved_tensor_geometry', 'Unknown public audit summary kind.')
        binding = record['audit_source_binding']
        auditor_sha = binding['auditor_source_sha256']
        require(auditor_sha == post['files']['experiments/final_staged/audit_tensors.py'],
                'Saved-tensor auditor source differs from its committed postprocessing registration.')
        source_commit, provenance = binding['source_commit'], record['provenance']
        counts = dict(cache_tensor_files_deserialized=record['counters']['cache_tensor_files_deserialized'],
            final_plan_files_deserialized=record['counters']['final_plan_files_deserialized'],
            TEST_geometry_rows=len(record['test_role_counts']))
        counters = record['counters']
        limits = [record[name] for name in ('physical_mapping_scope', 'shared_features_scope',
            'budget_scope', 'chronology_scope', 'independence')]
    summary = dict(schema='monti_public_audit_summary_v1', public_summary=True, status='passed', kind=kind,
        original_schema=record['schema'], scientific_provenance=provenance, full_raw_audit_sha256=raw_sha,
        auditor_source_sha256=auditor_sha, audit_source_commit=source_commit, core_counts=counts,
        counters=counters, limitations=limits, wall_seconds=record['wall_seconds'],
        process_cpu_seconds=record.get('process_cpu_seconds'), GPU_hours=record['GPU_hours'],
        retention='The complete raw audit, input hashes and tensor metadata remain in private run storage. This compact public file is not that raw audit and adds no verification calls.')
    if kind == 'saved_tensor_geometry': summary['quota_summary'] = record['quota_summary']
    return summary


def main(argv=None):
    global DEADLINE
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-root', 'protocol', 'saved-audit', 'tensor-audit', 'post-source-manifest', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--guardian', help='Optional completed main guardian JSON for exact command/outer cost.')
    args = parser.parse_args(argv); started, cpu_started = time.monotonic(), time.process_time()
    DEADLINE = started+1800
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    root = Path(args.run_root).resolve(strict=True); inputs = {}; generated = []
    receipt = dict(schema='monti_final_staged_export_receipt_v1', status='failed',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__), counters=dict(model_queries=0, optimizer_updates=0, API_calls=0,
        TEST_queries=0, dataset_loads=0, model_parameter_checkpoint_loads=0, selected_plan_tensor_loads=0), GPU_hours=0)
    armed = hasattr(signal, 'SIGALRM')
    def stopped(signum, frame): raise TimeoutError('The1800-second offline export wall cap was reached.')
    if armed:
        signal.signal(signal.SIGALRM, stopped); signal.alarm(1800)
    def read(path):
        path = Path(path).resolve(strict=True); raw = path.read_bytes()
        inputs[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw.decode('utf-8'))
    def bound(name):
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe relative run artifact path.')
        actual = (root/path).resolve(strict=True)
        require(root in actual.parents and actual.is_file(), 'Run artifact escapes its bound root.')
        return actual
    def write_json(name, value):
        with (output/name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=True, allow_nan=False, separators=(',', ':')); stream.write('\n')
        generated.append(name)
    def write_csv(name, rows, columns):
        with (output/name).open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
        generated.append(name)
    error = None
    try:
        post = read(args.post_source_manifest)
        require(re.fullmatch('[0-9a-f]{40}', post.get('source_commit', '')) is not None
                and post.get('files', {}).get('experiments/final_staged/export_saved.py') == receipt['source_sha256'],
                'Committed postprocessing source does not bind exporter bytes.')
        p = read(args.protocol); result = read(bound('result.json'))
        saved = read(args.saved_audit); geometry_audit = read(args.tensor_audit)
        require(p['schema'] == 'monti_final_staged_v1' and p['status'] == 'frozen'
                and p['owner_execution_authorized'] is True and p['final_TEST_allowed'] is True
                and tuple(p['arms']) == ARMS and tuple(p['seeds']) == SEEDS
                and len(p['train_groups']) == len(set(p['train_groups'])) == 937
                and len(p['test_groups']) == len(set(p['test_groups'])) == 909,
                'Wrong frozen full formal study.')
        provenance = dict(source_commit=p['source_manifest']['source_commit'], runtime_commit=p['runtime_commit'],
                          protocol_sha256=inputs[str(Path(args.protocol).resolve())])
        require(result['status'] == 'completed' and result['provenance'] == provenance
                and result['full_TEST_evaluated'] is True and result['GPU_hours'] == 0,
                'Only a completed same-protocol full TEST run can be exported.')
        require(saved['status'] == geometry_audit['status'] == 'passed'
                and saved['schema'] == 'monti_final_staged_saved_logit_recount_v1'
                and geometry_audit['schema'] == 'monti_final_staged_saved_tensor_geometry_audit_v1'
                and saved['recount']['provenance'] == geometry_audit['provenance'] == provenance,
                'Passed same-run independent logit/tensor audits are required.')
        audited = saved['recount']; seal = read(bound('sealed_before_victim.json'))
        cache_manifest = read(bound('upstream_cache_manifest.json'))
        result_sha = inputs[str(bound('result.json'))]; seal_sha = inputs[str(bound('sealed_before_victim.json'))]
        cache_sha = inputs[str(bound('upstream_cache_manifest.json'))]
        require(saved['input_file_hashes']['result.json'] == geometry_audit['inputs']['completed_result_sha256'] == result_sha
                and result['pre_victim_seal_sha256'] == seal_sha == geometry_audit['inputs']['sealed_plan_manifest_sha256']
                and seal['upstream_cache_sha256'] == cache_sha == geometry_audit['inputs']['upstream_cache_manifest_sha256']
                and seal['provenance'] == cache_manifest['provenance'] == provenance,
                'Completed result, audits and seal/cache byte bindings differ.')
        denominator = audited['occurrence_denominator_per_arm_seed']
        require(denominator == p['cohort']['test_target_occurrences'] == 10515
                and audited['attacked_cases'] == 8181 and audited['saved_rows'] == 9090,
                'Audited full denominator or matrix differs.')
        require(audited['seed_metrics'] == result['seed_metrics'] and audited['contrasts'] == result['contrasts']
                and audited['gaga_mean_mr_percent'] == result['gaga_mean_mr_percent']
                and audited['qualification_met'] == result['qualification_met'], 'Result summaries differ from audited values.')
        checkpoint_hashes = {}
        for seed in SEEDS:
            path = bound('downstream/s%d_final.pt' % seed); actual = sha(path); inputs[str(path)] = actual
            require(actual == seal['downstream_checkpoint_hashes'][str(seed)]
                    == geometry_audit['inputs']['terminal_checkpoint_byte_hashes'][str(seed)], 'Frozen downstream weight bytes differ.')
            checkpoint_hashes[str(seed)] = actual
        role_rows = geometry_audit['test_role_counts']
        geometry = {row['opaque_case_key']: row for row in role_rows}
        plans = {(row['seed'], row['group']): row for row in seal['plans']}
        require(len(role_rows) == len(geometry) == len(plans) == len(seal['plans']) == 2727
                and set(geometry) == {opaque(seed, group) for seed in SEEDS for group in p['test_groups']},
                'Audited TEST geometry coverage differs.')
        selected, eligible_counts = select_display_cases(p, geometry)
        raw_path = bound('raw/evaluation.jsonl'); raw_sha = sha(raw_path); inputs[str(raw_path)] = raw_sha
        require(raw_sha == saved['input_file_hashes']['raw/evaluation.jsonl'], 'Saved raw logit bytes changed after audit.')
        clean, predictions = {}, {}
        with raw_path.open('r', encoding='utf-8') as stream:
            for line in stream:
                check_time()
                if not line.strip(): continue
                row = json.loads(line); group = row['group']; members = p['test_members'][str(group)]; n = len(members)
                require(row['node_ids'] == members, 'Saved target order differs from frozen membership.')
                values = {d: prediction(row[d+'_logits'], n) for d in ('gcn', 'gaga')}
                if row.get('stage') == 'clean':
                    require(group not in clean, 'Duplicate clean prediction record.'); clean[group] = values
                else:
                    key = row['seed'], group, row['arm']
                    require(key not in predictions, 'Duplicate attacked prediction record.')
                    require(all(row[d] == transition(clean[group][d], values[d]) for d in ('gcn', 'gaga')),
                            'Saved raw transitions differ from their logits.')
                    predictions[key] = (values, {d: row[d] for d in ('gcn', 'gaga')})
        expected_keys = {(seed, group, arm) for seed in SEEDS for group in p['test_groups'] for arm in ARMS}
        require(set(clean) == set(p['test_groups']) and set(predictions) == expected_keys, 'Raw prediction matrix differs.')
        declared = {(r['seed'], r['group'], r['arm']): r for r in result['all_case_counts']}
        require(len(result['all_case_counts']) == len(declared) == 8181 and set(declared) == expected_keys,
                'Completed per-case record inventory differs.')
        metrics = []; case_counts = []; quota_rows = []; public_cases = []; torch = None
        for seed in SEEDS:
            for group in p['test_groups']:
                check_time(); key = opaque(seed, group); meta = geometry[key]; q = meta['planned_role_allocation']
                plan = plans[seed, group]; n = len(p['test_members'][str(group)])
                require(meta['seed'] == seed and meta['targets'] == n and plan['quota'] == q
                        and plan['frozen_upstream_feature_sha256'] == meta['frozen_upstream_feature_sha256']
                        and plan['frozen_upstream_target_block_sha256'] == meta['frozen_upstream_target_block_sha256'],
                        'Audited feature, quota or fixed target-block binding differs.')
                has_edges = (seed, group) in selected; edge_data = {}
                if has_edges:
                    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Display tensor decoding requires explicit CPU visibility.')
                    if torch is None:
                        import torch
                        require(not torch.cuda.is_available() and torch.cuda.device_count() == 0
                                and torch._C._cuda_getDeviceCount() == 0, 'No GPU may be visible during offline export.')
                        torch.set_num_threads(4)
                    require(plan['path'] == 's%d_g%d.pt' % (seed, group), 'Unsafe selected plan name.')
                    path = bound('plans/'+plan['path']); actual = sha(path); inputs[str(path)] = actual
                    require(actual == plan['sha256'], 'Selected plan bytes changed after sealing.')
                    matrices = torch.load(path, map_location='cpu'); receipt['counters']['selected_plan_tensor_loads'] += 1
                    require(set(matrices) == set(ARMS), 'Selected plan arm inventory differs.')
                    for arm in ARMS:
                        require(tensor_sha(matrices[arm]) == plan['topology_sha256'][arm], 'Selected plan content hash differs.')
                        edges, counts = local_edges(matrices[arm].tolist(), n, meta['candidates'], meta['injected_nodes'])
                        roles = meta['arms'][arm]
                        require(counts == dict(target=roles['target_edges'], candidate=roles['candidate_edges'],
                                auxiliary=roles['injected_to_injected_owned_edges']) and len(edges) == q['edge_budget'],
                                'Selected real edge roles differ from audited geometry.')
                        edge_data[arm] = edges
                    del matrices
                public = dict(seed=seed, group=key, targets=n, candidate_count=meta['candidates'],
                    auxiliary_count=meta['injected_nodes'], node_budget=q['injected_count'], edge_budget=q['edge_budget'],
                    target_quota=q['target_quota'], candidate_quota=q['candidate_quota'], actual_target_fraction=q['target_fraction'],
                    target_fraction_scope='Projected completion arms; native actual role fractions are recorded separately.',
                    projection_applied=q['projection_applied'], clean_correct=clean[group]['gaga'], edges_available=has_edges, arms={})
                quota = dict(opaque_case_key=key, seed=seed, targets=n, candidates=meta['candidates'],
                             node_budget=q['injected_count'], edge_budget=q['edge_budget'], **{
                    name: q[name] for name in ('ideal_target_quota', 'ideal_candidate_quota', 'target_quota', 'candidate_quota',
                    'target_fraction', 'candidate_fraction', 'projection_applied', 'exact_half', 'odd_budget')})
                for arm in ARMS:
                    values, counts = predictions[seed, group, arm]; source_row = declared[seed, group, arm]
                    require(all(counts[d] == source_row[d] for d in ('gcn', 'gaga'))
                            and source_row['quota'] == q and source_row['actual_edges'] == q['edge_budget']
                            and source_row['actual_nodes'] == q['injected_count']
                            and source_row['feature_sha256'] == meta['frozen_upstream_feature_sha256']
                            and source_row['topology_sha256'] == plan['topology_sha256'][arm], 'Audited case bindings changed.')
                    roles = meta['arms'][arm]; gaga = counts['gaga']
                    role_public = dict(target=roles['target_edges'], candidate=roles['candidate_edges'],
                        auxiliary=roles['injected_to_injected_owned_edges'], **{name: roles[name] for name in (
                        'target_edges', 'candidate_edges', 'injected_to_injected_owned_edges', 'total_owned_undirected_edges')})
                    public['arms'][arm] = dict(wrong=gaga['attacked_wrong'], total=n, new_flips=gaga['new_wrong'],
                        clean_wrong=gaga['clean_wrong'], recoveries=gaga['recoveries'], attacked_correct=values['gaga'],
                        actual_target_fraction=roles['target_fraction'], edge_roles=role_public, edges_available=has_edges)
                    if has_edges: public['arms'][arm]['edges'] = edge_data[arm]
                    row = dict(opaque_case_key=key, seed=seed, arm=arm, targets=n,
                        node_budget=q['injected_count'], edge_budget=q['edge_budget'])
                    for detector in ('gcn', 'gaga'):
                        row.update({detector+'_'+name: value for name, value in counts[detector].items() if name != 'occurrences'})
                    case_counts.append(row)
                    quota.update({arm+'_'+name: roles[name] for name in ('target_edges', 'candidate_edges',
                        'injected_to_injected_owned_edges', 'total_owned_undirected_edges', 'target_fraction',
                        'injected_physical_degree_min', 'injected_physical_degree_max', 'injected_physical_degree_sum')})
                public_cases.append(public); quota_rows.append(quota)
        for seed in SEEDS:
            for arm in ARMS:
                row = dict(seed=seed, arm=arm, groups=909, occurrences=denominator)
                selected_rows = [predictions[seed, group, arm][1] for group in p['test_groups']]
                for detector in ('gcn', 'gaga'):
                    for name in ('clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries'):
                        row[detector+'_'+name] = sum(v[detector][name] for v in selected_rows)
                    row[detector+'_mr_percent'] = 100*row[detector+'_attacked_wrong']/denominator
                original = next(v for v in audited['seed_metrics'] if v['seed'] == seed and v['arm'] == arm)
                require(all(math.isclose(row[d+'_mr_percent'], original[d+'_mr_percent'], abs_tol=1e-9)
                            for d in ('gcn', 'gaga')), 'Exported seed metrics differ from audited values.')
                metrics.append(row)
        selection = dict(rule='First two registered TEST groups per seed with B<=8, T<=32, E<=128 and candidate_quota>0; no outcome input.',
            selection_timing='Geometry-only display rule frozen before this run final TEST scoring; prior development and TRAIN preflight results were already known. Not a scientific cohort selection.',
            eligible_counts_by_seed=eligible_counts, included_counts_by_seed={str(s): sum(seed == s for seed, group in selected) for s in SEEDS},
            included_cases=[dict(seed=s, group=opaque(s, g)) for s in SEEDS for g in p['test_groups'] if (s, g) in selected],
            unavailable_case_count=2727-len(selected), all_evaluation_cases_retained=True,
            edge_scope='Exact added undirected-edge ownership only; original-to-original graph edges are not included.',
            unavailable_scope='Counts and target prediction outcomes only; no synthetic graph substitutes for omitted edges.')
        primary = audited['contrasts']['native_monti']; secondary = audited['contrasts']['quota_random']
        brief = lambda contrast: dict(mean_pp=contrast['mean_pp'], ci95_pp=contrast['nominal_paired_t95_pp'],
                                     seed_differences_pp=contrast['seed_differences_pp'])
        costs = dict(main_wall_seconds=result['wall_seconds'], main_process_cpu_seconds=result['process_cpu_seconds'],
            GPU_hours=0, counters=audited['producer_logical_counters'], stage_timings=result.get('stage_timings', []))
        guardian = None
        if args.guardian:
            guardian = read(args.guardian)
            require(guardian['status'] == 'completed' and guardian['exit_code'] == 0
                    and guardian['child_result_sha256'] == result_sha
                    and guardian['protocol_sha256'] == provenance['protocol_sha256'], 'Completed guardian binding differs.')
            costs['guardian'] = guardian
        public_protocol = dict(dataset='GossipCop-S', split=0, train_groups=937, test_groups=909, epochs=4,
            test_target_occurrences=denominator, unique_test_targets=audited['unique_test_targets'],
            requested_target_fraction=.5, quota_policy='closest_feasible', downstream_parameters=151872,
            native_scope='Fresh paired CPU replay of exact formal learned checkpoints; historical prediction identity is not assumed.')
        final_public = dict(schema='monti_final_staged_public_v1', status='completed', seeds=list(SEEDS),
            detector='Official GAGA', metric='MRpercent: incorrect fraud-target predictions / all target occurrences; higher means stronger attack.',
            arms={arm: dict(name=NAMES[arm], mean_mr=audited['gaga_mean_mr_percent'][arm],
                seed_mr=[next(r['gaga_mr_percent'] for r in audited['seed_metrics'] if r['seed'] == s and r['arm'] == arm) for s in SEEDS]) for arm in ARMS},
            primary=brief(primary), secondary_random=brief(secondary), qualification_met=audited['qualification_met'],
            protocol=public_protocol, geometry_display=selection, cases=public_cases,
            scientific_provenance=provenance, costs=costs,
            limitations=['Seed is the replication unit; repeated targets/groups are correlated.',
                'The full scheme changes role quotas and auxiliary-to-auxiliary edges; native contrast does not isolate learned scoring.',
                'No real fraud supply chain, first novelty, instructor approval or human review follows from these records.'])
        write_json('final_public.json', final_public)
        write_json('latest.json', dict(schema='monti_final_staged_compact_v1', status='completed', provenance=provenance,
            protocol=public_protocol, seeds=list(SEEDS), arms=list(ARMS), seed_metrics=audited['seed_metrics'],
            gaga_mean_mr_percent=audited['gaga_mean_mr_percent'], contrasts=audited['contrasts'],
            qualification_met=audited['qualification_met'], full_TEST_evaluated=True, controls_promoted=False,
            counters=audited['producer_logical_counters'], quota_summary=geometry_audit['quota_summary'], costs=costs,
            display_selection=selection, limitations=final_public['limitations']))
        write_csv('seed_metrics.csv', metrics, list(metrics[0]))
        write_csv('case_counts.csv', case_counts, list(case_counts[0]))
        write_csv('quota_by_case.csv', quota_rows, list(quota_rows[0]))
        raw_audit_hashes = dict(saved_logit_recount=inputs[str(Path(args.saved_audit).resolve())],
                               saved_tensor_geometry=inputs[str(Path(args.tensor_audit).resolve())])
        for name, kind, source, record in (
                ('audit_saved.json', 'saved_logit_recount', args.saved_audit, saved),
                ('audit_tensors.json', 'saved_tensor_geometry', args.tensor_audit, geometry_audit)):
            require(sha(source) == raw_audit_hashes[kind], 'A raw audit changed during export.')
            write_json(name, public_audit_summary(kind, record, raw_audit_hashes[kind], p, post))
        for seed in SEEDS:
            require(sha(bound('downstream/s%d_final.pt' % seed)) == checkpoint_hashes[str(seed)], 'Export mutated a frozen downstream checkpoint.')
        require(sha(bound('sealed_before_victim.json')) == seal_sha and sha(bound('upstream_cache_manifest.json')) == cache_sha,
                'Export mutated a seal or cache manifest.')
        require(sha(raw_path) == raw_sha, 'Saved raw logit bytes changed during export.')
        write_json('provenance.json', dict(schema='monti_final_staged_export_provenance_v1',
            scientific_provenance=provenance, postprocessing_source_commit=post['source_commit'],
            exporter_source_sha256=receipt['source_sha256'], input_file_sha256=inputs,
            full_raw_audit_sha256=raw_audit_hashes,
            payload_sha256={name: sha(output/name) for name in generated},
            original_run_root=str(root), command=guardian['command'] if guardian else None,
            command_scope='Exact model command only when a completed guardian was supplied; export command is recorded separately.',
            export_command=sys.argv if argv is None else ['export_saved.py']+list(argv),
            scope='Saved-output export; no new model calls, original-dataset or model-parameter deserialization, or training. Only bounded own topology plans are decoded.',
            full_raw_artifacts='All original caches/plans/logits remain retained outside this compact public export.',
            opaque_key_scheme='SHA256(JSON[seed, original_group, final_staged_tensor_audit_v1])[:20]',
            output_hash_scope='The external export receipt hashes this provenance and every payload; neither recursively hashes itself.'))
        receipt.update(status='completed', scientific_provenance=provenance,
            source_commit=post['source_commit'], included_graph_cases=len(selected), public_cases=2727,
            seed_metric_rows=9, case_count_rows=8181, quota_rows=2727,
            output_sha256={name: sha(output/name) for name in generated})
    except BaseException as caught:
        error = caught; receipt['error'] = repr(caught); receipt['traceback'] = traceback.format_exc(); traceback.print_exc()
    finally:
        if armed: signal.alarm(0)
        DEADLINE = None
        receipt.update(wall_seconds=time.monotonic()-started, process_cpu_seconds=time.process_time()-cpu_started)
        with (output/'export_receipt.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=True, allow_nan=False); stream.write('\n')
        print(json.dumps(dict(status=receipt['status'], public_cases=receipt.get('public_cases'), GPU_hours=0,
            export_receipt_sha256=sha(output/'export_receipt.json')), sort_keys=True), flush=True)
    if error is not None: raise SystemExit(1)


if __name__ == '__main__':
    main()
