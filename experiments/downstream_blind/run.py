"""Train only the registered action-blind control and reuse sealed comparators."""
import argparse
import json
import pathlib
import random
import signal
import statistics
import time
import traceback

from common import (SEEDS, verify_protocol, read_json, write_json, sha, old_helpers,
                    parent_protocol, parent_result, load_cached_case, blind_scores, summarize)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'output', 'preflight'): parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    p, provenance = verify_protocol(args.protocol)
    gate = read_json(args.preflight)
    if gate['status'] != 'passed' or gate['provenance'] != provenance: raise RuntimeError('Same-source passed preflight required.')
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=False)
    for name in ('downstream', 'plans', 'raw'): (out / name).mkdir()
    counts = dict(upstream_generations=0, upstream_optimizer_updates=0,
                  training_GCN_forwards=0, evaluation_GCN_forwards=0, official_GAGA_forwards=0,
                  downstream_optimizer_updates=0, autograd_calls=0, degenerate_training_cases=0,
                  summary_algebra_cases=0, API_calls=0, TEST_queries=0)
    result = dict(status='failed', provenance=provenance, counters=counts, GPU_hours=0, active_case=None)
    limit = p['resource']['main_inner_seconds']; handles = []; train_log = None
    def stopped(signum, frame): raise TimeoutError('Registered action-blind phase interrupted or timed out.')
    signal.signal(signal.SIGALRM, stopped); signal.signal(signal.SIGTERM, stopped); signal.alarm(limit)
    def check():
        if time.monotonic() - started >= limit: raise TimeoutError('Registered main deadline reached.')
    try:
        old = old_helpers(); result['runtime'] = old.check_device()
        import torch
        from experiments.downstream_blind.model import ActionBlindCandidateScorer
        from experiments.downstream_cooperation.model import straight_through_mask
        from course.dual_surrogate.runtime import gcn_logits, load_classifier, load_original_view, transitions
        parent = parent_result(p)
        engine, data, receipt, frozen = old.load_engine(parent_protocol(p), 5, out, started + limit)
        teacher = engine.teachers['gcn_base']
        if engine.victim_model is not None or set(engine.teachers) != {'gcn_base'}: raise RuntimeError('No victim may load during fitting.')
        def prohibit_generation(module, inputs): raise RuntimeError('No upstream regeneration is permitted.')
        handles.append(engine.model.register_forward_pre_hook(prohibit_generation))
        def count_teacher(module, inputs):
            name = 'training_GCN_forwards' if result.get('stage') == 'train' else 'evaluation_GCN_forwards'
            counts[name] += 1
            if counts['training_GCN_forwards'] > 1536 or counts['evaluation_GCN_forwards'] > 192: raise RuntimeError('GCN cap exceeded.')
        handles.append(teacher.register_forward_pre_hook(count_teacher))
        info = []
        # Algebra only on already saved inputs, before any fit; no model forward.
        for seed in SEEDS:
            for phase, groups in (('train', p['train_groups']), ('validation', p['valid_groups'])):
                for group in groups:
                    check(); case = load_cached_case(p, seed, phase, group, torch)
                    adj = case['target_adjacency']; targets = case['x_targets']; b, t = adj.shape
                    actual_count = adj.sum(1, keepdim=True)
                    actual_mean = adj.matmul(targets) / actual_count
                    neutral = torch.ones_like(adj)
                    neutral_mean = neutral.matmul(targets) / neutral.sum(1, keepdim=True)
                    neutral_count = torch.full_like(actual_count, case['quota']['target_quota'] / b)
                    same_mean = torch.equal(actual_mean, neutral_mean)
                    same_log_count = torch.equal(torch.log1p(actual_count), torch.log1p(neutral_count))
                    info.append(dict(seed=seed, phase=phase, group=group,
                        all_targets_attached=case['quota']['target_quota'] == b * t,
                        exact_same_mean=same_mean, exact_same_log_count=same_log_count,
                        exact_same_summary=same_mean and same_log_count,
                        maximum_mean_difference=float((actual_mean - neutral_mean).abs().max()),
                        maximum_log_count_difference=float((torch.log1p(actual_count) - torch.log1p(neutral_count)).abs().max())))
                    counts['summary_algebra_cases'] += 1
        write_json(out / 'conditioning_scope.json', dict(provenance=provenance, cases=info,
                   scope='Saved-input algebra before fitting; equality of direct summaries is not equality of separately trained policies. No classifier calls.'))
        downstream = []; plans = []; weights = {}
        train_log = (out / 'raw/training.jsonl').open('x', encoding='utf-8', newline='\n')
        result['stage'] = 'train'
        for seed in SEEDS:
            check(); torch.manual_seed(20261111 + seed)
            lower = ActionBlindCandidateScorer(data.feat_dim, 64, 64)
            initial = old.state_hash(lower)
            matched = next(row for row in parent['downstream'] if row['seed'] == seed)
            if initial != matched['initial_state_sha256'] or sum(v.numel() for v in lower.parameters()) != 151872:
                raise RuntimeError('Matched visible/blind initial parameters differ.')
            optimizer = torch.optim.Adam(lower.parameters(), lr=.001, weight_decay=.0001)
            if {id(v) for g in optimizer.param_groups for v in g['params']} & {id(v) for m in (engine.model, teacher) for v in m.parameters()}:
                raise RuntimeError('Frozen parameters entered the optimizer.')
            steps = nonzero = 0
            for epoch in range(1, 5):
                lower.train(); order = list(p['train_groups']); random.Random(seed * 1000 + epoch).shuffle(order)
                losses = []; skipped = 0
                for index, group in enumerate(order):
                    check(); result['active_case'] = dict(seed=seed, epoch=epoch, group=group)
                    case = load_cached_case(p, seed, 'train', group, torch); q = case['quota']['candidate_quota']
                    if q in (0, len(case['source_ids']) * len(case['candidate_ids'])):
                        skipped += 1; counts['degenerate_training_cases'] += 1
                        train_log.write(json.dumps(dict(seed=seed, epoch=epoch, group=group, optimizer_update=False, reason='No candidate choice.')) + '\n')
                        continue
                    optimizer.zero_grad(set_to_none=True)
                    scores = blind_scores(lower, case); hard = old.candidate_mask(case, scores)
                    weighted = straight_through_mask(scores, hard, q)
                    graph = old.make_graph(data, case, old.adjacency(case, weighted), hard=old.adjacency(case, hard))
                    logits = gcn_logits(teacher, graph, case['ids'], training=True); loss = old.softplus_loss(logits)
                    if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite TRAIN loss.')
                    loss.backward(); counts['autograd_calls'] += 1
                    grads = [v.grad for v in lower.parameters() if v.grad is not None]
                    if not grads or not all(bool(torch.isfinite(v).all()) for v in grads): raise RuntimeError('Missing/nonfinite gradient.')
                    norm = float(sum(v.detach().double().square().sum() for v in grads).sqrt()); nonzero += int(norm > 0)
                    torch.nn.utils.clip_grad_norm_(lower.parameters(), 5.)
                    optimizer.step(); steps += 1; counts['downstream_optimizer_updates'] += 1
                    if any(v.grad is not None for v in teacher.parameters()): raise RuntimeError('Frozen teacher acquired parameter gradients.')
                    if old.tensor_hash(case['x_aux']) != case['upstream_feature_sha256']: raise RuntimeError('Frozen feature mutation.')
                    value = float(loss.detach()); losses.append(value)
                    train_log.write(json.dumps(dict(seed=seed, epoch=epoch, group=group, optimizer_update=True,
                                                    loss=value, gradient_l2=norm), allow_nan=False) + '\n')
                    del graph, logits, loss, weighted, scores, hard, grads, case
                    if (index + 1) % 32 == 0: print('TRAIN', seed, epoch, index + 1, flush=True)
                train_log.flush()
                write_json(out / 'downstream' / ('s%d_epoch%d.json' % (seed, epoch)),
                    dict(seed=seed, epoch=epoch, groups=128, no_choice_groups=skipped,
                         cumulative_optimizer_updates=steps, mean_loss=statistics.mean(losses), validation_performed=False))
            if steps != matched['optimizer_updates'] or nonzero == 0 or old.state_hash(lower) == initial:
                raise RuntimeError('Blind updates differ from the matched fit or failed to train.')
            old.assert_frozen(teacher, frozen['teacher']); old.assert_frozen(engine.model, frozen['upstream'])
            path = out / 'downstream' / ('s%d_final.pt' % seed); torch.save(lower.state_dict(), path)
            row = dict(seed=seed, epochs=4, optimizer_updates=steps, parameter_count=151872,
                       nonzero_gradient_cases=nonzero, initial_state_sha256=initial, matches_visible_initial_state=True,
                       final_state_sha256=old.state_hash(lower), final_checkpoint_sha256=sha(path),
                       checkpoint_selection='Terminal epoch4; no VALID selection.')
            downstream.append(row); weights[str(seed)] = row['final_checkpoint_sha256']
            lower.eval()
            for group in p['valid_groups']:
                check(); case = load_cached_case(p, seed, 'validation', group, torch)
                with torch.no_grad(): selected = old.candidate_mask(case, blind_scores(lower, case))
                adj = old.adjacency(case, selected)
                if int(adj.sum()) != case['quota']['edge_budget'] or not torch.equal(adj[:, :len(case['ids'])], case['target_adjacency']):
                    raise RuntimeError('Blind plan changed quota or frozen target actions.')
                path = out / 'plans' / ('s%d_g%d.pt' % (seed, group)); torch.save(adj, path)
                plans.append(dict(seed=seed, group=group, path=path.name, sha256=sha(path), topology_sha256=old.tensor_hash(adj),
                    upstream_feature_sha256=case['upstream_feature_sha256'], target_block_sha256=case['upstream_target_block_sha256'], quota=case['quota']))
        train_log.close()
        write_json(out / 'sealed_before_victim.json', dict(provenance=provenance, downstream=downstream,
                   checkpoint_hashes=weights, plans=plans, fitting_finished=True, official_GAGA_loaded=False))
        seal_sha = sha(out / 'sealed_before_victim.json')
        result['stage'] = 'evaluation'; result['active_case'] = None
        view = load_original_view(data); victim = load_classifier('gaga_victim', data); victim_hash = old.state_hash(victim)
        def count_victim(module, inputs):
            counts['official_GAGA_forwards'] += 1
            if counts['official_GAGA_forwards'] > 192: raise RuntimeError('Official GAGA cap exceeded.')
        handles.append(victim.register_forward_pre_hook(count_victim))
        clean = {}
        with (pathlib.Path(p['runtime_root']) / p['parent_raw_evaluation']['path']).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('stage') == 'clean':
                    clean[row['group']] = row
        if len(clean) != 64: raise RuntimeError('Parent clean outcomes are incomplete.')
        rows = list(parent['all_case_counts']); new_rows = []
        with (out / 'raw/evaluation.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for plan in plans:
                check(); seed, group = plan['seed'], plan['group']; result['active_case'] = dict(seed=seed, group=group)
                case = load_cached_case(p, seed, 'validation', group, torch); path = out / 'plans' / plan['path']
                if sha(path) != plan['sha256']: raise RuntimeError('Post-seal blind plan mutation.')
                adj = torch.load(path, map_location='cpu'); labels = data.labels[case['ids']]
                if not bool((labels == 1).all()) or case['ids'].tolist() != clean[group]['node_ids']: raise RuntimeError('Target identity/order changed.')
                with torch.no_grad():
                    graph = old.make_graph(data, case, adj)
                    gcn = gcn_logits(teacher, graph, case['ids']); seq = old.canonical(data, view, graph, case); gaga = victim(seq)
                base = clean[group]
                row = dict(seed=seed, group=group, arm='quota_blind', targets=len(labels), actual_nodes=len(case['source_ids']),
                    actual_edges=int(adj.sum()), quota=case['quota'], topology_sha256=old.tensor_hash(adj), feature_sha256=case['upstream_feature_sha256'],
                    gcn=transitions(labels, torch.tensor(base['gcn_logits']).argmax(1), gcn.argmax(1)),
                    gaga=transitions(labels, torch.tensor(base['gaga_logits']).argmax(1), gaga.argmax(1)))
                rows.append(row); new_rows.append(row)
                stream.write(json.dumps(dict(row, node_ids=case['ids'].tolist(), gcn_logits=gcn.tolist(), gaga_logits=gaga.tolist()), allow_nan=False) + '\n'); stream.flush()
                del graph, seq, gcn, gaga
                if len(new_rows) % 32 == 0: print('EVALUATED', len(new_rows), flush=True)
        if len(new_rows) != 192 or sha(out / 'sealed_before_victim.json') != seal_sha: raise RuntimeError('Incomplete evaluation or seal mutation.')
        for seed in SEEDS:
            if sha(out / 'downstream' / ('s%d_final.pt' % seed)) != weights[str(seed)]: raise RuntimeError('Post-seal checkpoint mutation.')
        old.assert_frozen(teacher, frozen['teacher']); old.assert_frozen(victim, victim_hash); old.assert_frozen(engine.model, frozen['upstream'])
        verify_protocol(args.protocol)
        metrics, means, contrasts, criterion = summarize(rows)
        result.update(status='completed', stage='completed', active_case=None, seed_metrics=metrics, mean_mr_percent=means,
                      contrasts=contrasts, direct_action_summary_criterion_met=criterion,
                      all_case_counts=rows, downstream=downstream, pre_victim_seal_sha256=seal_sha,
                      conditioning_scope_sha256=sha(out / 'conditioning_scope.json'), frozen_models_unchanged=True,
                      reused_comparator_cases=768, reused_clean_groups=64, new_blind_cases=192,
                      full_TEST_evaluated=False, prior_qualification_unchanged=True,
                      interpretation='Post-hoc mechanism ablation on the inspected development cohort; no independent confirmation or novelty claim.')
        write_json(out / 'summary.json', {k: v for k, v in result.items() if k != 'all_case_counts'})
    except BaseException as error:
        result['error'] = repr(error); result['traceback'] = traceback.format_exc(); traceback.print_exc()
    finally:
        if train_log is not None and not train_log.closed: train_log.close()
        for handle in handles: handle.remove()
        signal.alarm(0)
        result.update(wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu_started)
        write_json(out / 'result.json', result)
        print(json.dumps({k: result.get(k) for k in ('status', 'error', 'counters', 'mean_mr_percent', 'wall_seconds')}, allow_nan=False), flush=True)
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
