"""Four-epoch downstream-only fit and one sealed development comparison."""
import argparse
import gc
import hashlib
import json
import pathlib
import random
import signal
import statistics
import time
import traceback

from common import (ARMS, SEEDS, verify_protocol, check_device, read_json,
                    write_json, sha, tensor_hash, state_hash, assert_frozen,
                    load_engine, capture_case, downstream_scores, candidate_mask,
                    adjacency, make_graph, canonical, softplus_loss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', required=True); args = parser.parse_args()
    start, cpu_start = time.monotonic(), time.process_time()
    p, provenance = verify_protocol(args.protocol)
    preflight = read_json(args.preflight)
    if preflight.get('status') != 'passed' or preflight.get('provenance') != provenance:
        raise RuntimeError('Passed same-source preflight is required.')
    output = pathlib.Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'downstream', 'plans', 'raw'): (output / name).mkdir()
    counts = dict(upstream_generations=0, upstream_optimizer_updates=0,
                  training_GCN_forwards=0, evaluation_GCN_forwards=0,
                  official_GAGA_forwards=0, autograd_calls=0,
                  downstream_optimizer_updates=0, degenerate_training_cases=0,
                  API_calls=0, TEST_queries=0)
    result = dict(status='failed', provenance=provenance, counters=counts,
                  GPU_hours=0, active_case=None)
    train_log = None
    limit = p['resource']['main_inner_seconds']
    def stopped(signum, frame): raise TimeoutError('Frozen downstream CPU phase interrupted or timed out.')
    signal.signal(signal.SIGALRM, stopped); signal.signal(signal.SIGTERM, stopped); signal.alarm(limit)
    def check():
        if time.monotonic() - start >= limit: raise TimeoutError('Frozen main wall limit reached.')
    def cache_path(seed, phase, group): return output / 'cache' / ('s%d_%s_g%d.pt' % (seed, phase, group))
    def plan_path(seed, group): return output / 'plans' / ('s%d_g%d.pt' % (seed, group))
    try:
        runtime = check_device()
        import torch
        from course.dual_surrogate.runtime import (gcn_logits, load_classifier,
                                                  load_original_view, transitions)
        from experiments.downstream_cooperation.model import DownstreamCandidateScorer, straight_through_mask
        write_json(output / 'registered.json', dict(provenance=provenance, runtime=runtime,
                   source_manifest=p['source_manifest'], settings=p['settings'],
                   train_groups=p['train_groups'], valid_groups=p['valid_groups'],
                   boundary='All upstream caches and capacities precede any downstream optimizer update. No victim load until final weights/topologies are sealed.'))
        cache_manifest = []; upstream_receipts = []
        engine = data = teacher = None; frozen_teacher = None
        # Prepare every seed/case first: a capacity error cannot follow any fit.
        for seed in SEEDS:
            check(); gc.collect()
            engine, data, receipt, frozen = load_engine(p, seed, output, start + limit)
            teacher = engine.teachers['gcn_base']; frozen_teacher = frozen['teacher']
            generation_queries = [0]
            def prohibit_teacher(module, inputs):
                generation_queries[0] += 1
                raise RuntimeError('Upstream generation must use its cached representations, without classifier queries.')
            handle = teacher.register_forward_pre_hook(prohibit_teacher)
            for phase, groups in (('train', p['train_groups']), ('validation', p['valid_groups'])):
                for index, group in enumerate(groups):
                    check(); result['active_case'] = dict(stage='cache', seed=seed, phase=phase, group=group)
                    case = capture_case(engine, data, seed, group, 'frozen_downstream_' + phase)
                    members = p['train_members'] if phase == 'train' else p['valid_members']
                    if case['ids'].tolist() != members[str(group)]: raise RuntimeError('Frozen group membership changed.')
                    counts['upstream_generations'] += 1
                    if counts['upstream_generations'] > 576: raise RuntimeError('Upstream generation cap exceeded.')
                    path = cache_path(seed, phase, group); torch.save(case, path)
                    cache_manifest.append(dict(seed=seed, phase=phase, group=group, path=path.name,
                        sha256=sha(path), quota=case['quota'], targets=len(case['ids']),
                        upstream_feature_sha256=case['upstream_feature_sha256'],
                        upstream_target_block_sha256=case['upstream_target_block_sha256']))
                    if (index + 1) % 16 == 0: print('CACHE', seed, phase, index + 1, flush=True)
                    del case
            handle.remove()
            assert_frozen(engine.model, frozen['upstream']); assert_frozen(teacher, frozen['teacher'])
            if engine.optimizer_steps or generation_queries[0]: raise RuntimeError('Upstream unexpectedly queried or updated.')
            upstream_receipts.append(dict(seed=seed, checkpoint_sha256=receipt['final_model_sha256'],
                before=frozen, after=dict(upstream=state_hash(engine.model), teacher=state_hash(teacher)),
                optimizer_updates=engine.optimizer_steps, classifier_queries_during_generation=generation_queries[0]))
        if counts['upstream_generations'] != 576: raise RuntimeError('Incomplete upstream cache cohort.')
        write_json(output / 'upstream_cache_manifest.json', dict(provenance=provenance, cases=cache_manifest,
                   receipts=upstream_receipts, optimizer_updates_before_seal=0, victim_loaded=False))
        cache_hash = sha(output / 'upstream_cache_manifest.json')
        def load_case(seed, phase, group):
            path = cache_path(seed, phase, group)
            wanted = next(row['sha256'] for row in cache_manifest if row['seed'] == seed and row['phase'] == phase and row['group'] == group)
            if sha(path) != wanted: raise RuntimeError('Frozen upstream cache changed.')
            case = torch.load(path, map_location='cpu')
            if tensor_hash(case['x_aux']) != case['upstream_feature_sha256'] or tensor_hash(case['target_adjacency']) != case['upstream_target_block_sha256']:
                raise RuntimeError('Cached features or target decisions changed.')
            return case
        def checked_gcn(graph, ids, training=False):
            key = 'training_GCN_forwards' if training else 'evaluation_GCN_forwards'
            counts[key] += 1
            if counts['training_GCN_forwards'] > 1536 or counts['evaluation_GCN_forwards'] > 832:
                raise RuntimeError('GCN query bound exceeded.')
            return gcn_logits(teacher, graph, ids, training=training)
        downstream_records = []; frozen_weights = {}; sealed_plans = []
        train_log = (output / 'raw' / 'training.jsonl').open('x', encoding='utf-8', newline='\n')
        for seed in SEEDS:
            check(); torch.manual_seed(20261111 + seed)
            lower = DownstreamCandidateScorer(data.feat_dim, 64, 64)
            initial = state_hash(lower)
            optimizer = torch.optim.Adam(lower.parameters(), lr=.001, weight_decay=.0001)
            optimizer_ids = {id(value) for group in optimizer.param_groups for value in group['params']}
            if optimizer_ids & {id(value) for model in (engine.model, teacher) for value in model.parameters()}:
                raise RuntimeError('Optimizer contains frozen parameters.')
            nonzero_gradient_cases = 0; seed_steps = 0
            for epoch in range(1, 5):
                lower.train(); order = list(p['train_groups']); random.Random(seed * 1000 + epoch).shuffle(order)
                epoch_loss = []; skipped = 0
                for index, group in enumerate(order):
                    check(); result['active_case'] = dict(stage='train', seed=seed, epoch=epoch, group=group)
                    case = load_case(seed, 'train', group); q = case['quota']['candidate_quota']
                    if q in (0, len(case['source_ids']) * len(case['candidate_ids'])):
                        counts['degenerate_training_cases'] += 1; skipped += 1
                        train_log.write(json.dumps(dict(seed=seed, epoch=epoch, group=group, quota=case['quota'], optimizer_update=False,
                                                       reason='No downstream edge choice under the registered quota.')) + '\n')
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    scores = downstream_scores(lower, case); hard = candidate_mask(case, scores)
                    weights = straight_through_mask(scores, hard, q)
                    graph = make_graph(data, case, adjacency(case, weights), hard=adjacency(case, hard))
                    logits = checked_gcn(graph, case['ids'], training=True); loss = softplus_loss(logits)
                    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite downstream loss.')
                    counts['autograd_calls'] += 1; loss.backward()
                    grads = [v.grad for v in lower.parameters() if v.grad is not None]
                    if not grads or not all(bool(torch.isfinite(value).all()) for value in grads):
                        raise RuntimeError('Missing or nonfinite downstream gradients.')
                    norm = float(sum(value.detach().double().square().sum() for value in grads).sqrt())
                    nonzero_gradient_cases += int(norm > 0)
                    torch.nn.utils.clip_grad_norm_(lower.parameters(), 5.)
                    optimizer.step(); counts['downstream_optimizer_updates'] += 1; seed_steps += 1
                    if any(value.grad is not None for value in teacher.parameters()): raise RuntimeError('Frozen classifier acquired parameter gradients.')
                    if tensor_hash(case['x_aux']) != case['upstream_feature_sha256']: raise RuntimeError('Upstream feature mutation.')
                    row = dict(seed=seed, epoch=epoch, group=group, quota=case['quota'], loss=float(loss.detach()),
                               downstream_gradient_l2=norm, optimizer_update=True)
                    train_log.write(json.dumps(row, allow_nan=False) + '\n'); epoch_loss.append(row['loss'])
                    del graph, logits, loss, weights, scores, hard, grads, case
                    if (index + 1) % 32 == 0: print('TRAIN', seed, epoch, index + 1, flush=True)
                train_log.flush()
                write_json(output / 'downstream' / ('s%d_epoch%d.json' % (seed, epoch)),
                           dict(seed=seed, epoch=epoch, groups=128, mean_loss=statistics.mean(epoch_loss) if epoch_loss else None,
                                no_choice_groups=skipped, cumulative_optimizer_updates=seed_steps, validation_performed=False))
            if seed_steps == 0 or nonzero_gradient_cases == 0 or state_hash(lower) == initial:
                raise RuntimeError('No demonstrated downstream training occurred.')
            assert_frozen(teacher, frozen_teacher)
            final_path = output / 'downstream' / ('s%d_final.pt' % seed)
            torch.save(lower.state_dict(), final_path)
            record = dict(seed=seed, epochs=4, optimizer_updates=seed_steps,
                          nonzero_gradient_cases=nonzero_gradient_cases, initial_state_sha256=initial,
                          final_state_sha256=state_hash(lower), final_checkpoint_sha256=sha(final_path),
                          checkpoint_selection='Terminal epoch4 only; no VALID selection.', parameter_count=sum(v.numel() for v in lower.parameters()))
            downstream_records.append(record); frozen_weights[str(seed)] = record['final_checkpoint_sha256']
            lower.eval()
            for group in p['valid_groups']:
                check(); case = load_case(seed, 'validation', group)
                with torch.no_grad(): learned = candidate_mask(case, downstream_scores(lower, case))
                plans = dict(native_monti=case['native_adjacency'], quota_original=adjacency(case, case['original_candidates']),
                             quota_random=adjacency(case, case['random_candidates']), quota_learned=adjacency(case, learned))
                if any(not bool(((value == 0) | (value == 1)).all()) or int(value.sum()) != case['quota']['edge_budget'] for value in plans.values()):
                    raise RuntimeError('A final plan changed total budget or binary geometry.')
                path = plan_path(seed, group); torch.save(plans, path)
                sealed_plans.append(dict(seed=seed, group=group, path=path.name, sha256=sha(path),
                    topology_sha256={arm: tensor_hash(value) for arm, value in plans.items()},
                    frozen_upstream_feature_sha256=case['upstream_feature_sha256'],
                    frozen_upstream_target_block_sha256=case['upstream_target_block_sha256'], quota=case['quota']))
        train_log.close()
        if sha(output / 'upstream_cache_manifest.json') != cache_hash: raise RuntimeError('Upstream manifest changed during fitting.')
        write_json(output / 'sealed_before_victim.json', dict(provenance=provenance, upstream_cache_sha256=cache_hash,
                   downstream_records=downstream_records, downstream_checkpoint_hashes=frozen_weights,
                   plans=sealed_plans, official_GAGA_loaded=False, fitting_finished=True))
        seal_hash = sha(output / 'sealed_before_victim.json')
        # No further optimizer or policy construction occurs beyond this gate.
        import numpy as np
        view = load_original_view(data); victim = load_classifier('gaga_victim', data); victim_hash = state_hash(victim)
        clean_sequences = np.load(pathlib.Path(p['asset_root']) / 'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',
                                  allow_pickle=False, mmap_mode='r')
        if (clean_sequences.shape != (data.graph.num_nodes(), 7, data.feat_dim)
                or not np.isfinite(clean_sequences).all()
                or not np.array_equal(clean_sequences[:, 0, :], data.features.numpy())):
            raise RuntimeError('Frozen clean GAGA sequence or self token is invalid.')
        for seed in SEEDS:
            if sha(output / 'downstream' / ('s%d_final.pt' % seed)) != frozen_weights[str(seed)]:
                raise RuntimeError('Downstream checkpoint changed after its pre-victim seal.')
        clean = {}; case_rows = []; raw_path = output / 'raw' / 'evaluation.jsonl'
        with raw_path.open('x', encoding='utf-8', newline='\n') as stream:
            for group in p['valid_groups']:
                check(); case = load_case(3, 'validation', group); ids = case['ids']
                with torch.no_grad():
                    gcn = checked_gcn(data.graph, ids)
                    counts['official_GAGA_forwards'] += 1
                    gaga = victim(torch.from_numpy(clean_sequences[ids.numpy()].copy()))
                clean[group] = dict(gcn=gcn.argmax(1), gaga=gaga.argmax(1))
                stream.write(json.dumps(dict(stage='clean', group=group, node_ids=ids.tolist(),
                    gcn_logits=gcn.tolist(), gaga_logits=gaga.tolist()), allow_nan=False) + '\n')
            for plan in sealed_plans:
                seed, group = plan['seed'], plan['group']; case = load_case(seed, 'validation', group)
                path = plan_path(seed, group)
                if sha(path) != plan['sha256']: raise RuntimeError('Final topology changed after sealing.')
                plans = torch.load(path, map_location='cpu'); labels = data.labels[case['ids']]
                for arm in ARMS:
                    check(); result['active_case'] = dict(stage='evaluate', seed=seed, group=group, arm=arm)
                    adj = plans[arm]
                    if arm != 'native_monti' and not torch.equal(adj[:, :len(case['ids'])], case['target_adjacency']):
                        raise RuntimeError('Matched arm changed upstream target decisions.')
                    with torch.no_grad():
                        graph = make_graph(data, case, adj)
                        gcn = checked_gcn(graph, case['ids'])
                        seq = canonical(data, view, graph, case)
                        counts['official_GAGA_forwards'] += 1
                        if counts['official_GAGA_forwards'] > 832: raise RuntimeError('Official GAGA cap exceeded.')
                        gaga = victim(seq)
                    row = dict(seed=seed, group=group, arm=arm, targets=len(labels), quota=case['quota'],
                               actual_nodes=len(case['source_ids']), actual_edges=int(adj.sum()),
                               topology_sha256=tensor_hash(adj), feature_sha256=case['upstream_feature_sha256'],
                               gcn=transitions(labels, clean[group]['gcn'], gcn.argmax(1)),
                               gaga=transitions(labels, clean[group]['gaga'], gaga.argmax(1)))
                    case_rows.append(row)
                    stream.write(json.dumps(dict(row, node_ids=case['ids'].tolist(), gcn_logits=gcn.tolist(),
                                                 gaga_logits=gaga.tolist()), allow_nan=False) + '\n'); stream.flush()
                    del graph, seq, gcn, gaga
                if len(case_rows) % 64 == 0: print('EVALUATED', len(case_rows), flush=True)
        assert_frozen(teacher, frozen_teacher); assert_frozen(victim, victim_hash)
        assert_frozen(engine.model, upstream_receipts[-1]['before']['upstream'])
        for seed in SEEDS:
            if sha(output / 'downstream' / ('s%d_final.pt' % seed)) != frozen_weights[str(seed)]:
                raise RuntimeError('Frozen downstream checkpoint changed during evaluation.')
        if sha(output / 'sealed_before_victim.json') != seal_hash: raise RuntimeError('Pre-victim seal changed.')
        metrics = []
        for seed in SEEDS:
            for arm in ARMS:
                rows = [r for r in case_rows if r['seed'] == seed and r['arm'] == arm]
                if len(rows) != 64: raise RuntimeError('Incomplete terminal VALID cohort.')
                denominator = sum(r['targets'] for r in rows)
                metrics.append(dict(seed=seed, arm=arm, groups=64, occurrences=denominator,
                    **{detector + '_mr_percent': 100 * sum(r[detector]['attacked_wrong'] for r in rows) / denominator for detector in ('gcn', 'gaga')}))
        means = {arm: statistics.mean(r['gaga_mr_percent'] for r in metrics if r['arm'] == arm) for arm in ARMS}
        contrasts = {}
        for control in ARMS[:-1]:
            diffs = [next(r['gaga_mr_percent'] for r in metrics if r['seed'] == seed and r['arm'] == 'quota_learned')
                     - next(r['gaga_mr_percent'] for r in metrics if r['seed'] == seed and r['arm'] == control) for seed in SEEDS]
            avg = statistics.mean(diffs); radius = 4.302652729911275 * statistics.stdev(diffs) / (3 ** .5)
            contrasts[control] = dict(seed_differences_pp=diffs, mean_pp=avg, nominal_paired_t95_pp=[avg - radius, avg + radius])
        primary = contrasts['native_monti']
        qualified = (primary['mean_pp'] >= 1.0 and all(value > 0 for value in primary['seed_differences_pp'])
                     and primary['nominal_paired_t95_pp'][0] > 0
                     and means['quota_learned'] > max(means['quota_original'], means['quota_random']))
        result.update(status='completed', active_case=None, seed_metrics=metrics, gaga_mean_mr_percent=means,
                      contrasts=contrasts, downstream=downstream_records, upstream_receipts=upstream_receipts,
                      pre_victim_seal_sha256=seal_hash, all_case_counts=case_rows,
                      frozen_teacher_unchanged=True, frozen_victim_unchanged=True,
                      primary='quota_learned', qualification_met=qualified, controls_promoted=False, full_TEST_evaluated=False,
                      interpretation='Development pilot under projected ratios. Repeated target occurrences are correlated; no independent population or novelty claim.')
        write_json(output / 'summary.json', {key: value for key, value in result.items() if key != 'all_case_counts'})
    except BaseException as error:
        result['error'] = repr(error); traceback.print_exc()
    finally:
        if train_log is not None and not train_log.closed: train_log.close()
        signal.alarm(0); result.update(wall_seconds=time.monotonic() - start, process_cpu_seconds=time.process_time() - cpu_start)
        write_json(output / 'result.json', result)
        print(json.dumps({key: result.get(key) for key in ('status', 'error', 'counters', 'gaga_mean_mr_percent', 'wall_seconds', 'GPU_hours')}, allow_nan=False), flush=True)
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
