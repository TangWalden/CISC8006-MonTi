"""Source-bound synthetic and four original-TRAIN action-blind gradient checks."""
import argparse
import io
import pathlib
import signal
import time
import traceback
import unittest

from common import (verify_protocol, read_json, write_json, sha, parent_protocol,
                    parent_result, old_helpers, load_cached_case, blind_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=False)
    counters = dict(upstream_generations=0, GCN_forwards=0, GAGA_forwards=0,
                    optimizer_updates=0, autograd_calls=0, TEST_queries=0, API_calls=0)
    result = dict(status='failed', counters=counters, GPU_hours=0, cases=[])
    def stopped(signum, frame): raise TimeoutError('Registered action-blind preflight interrupted.')
    signal.signal(signal.SIGALRM, stopped); signal.signal(signal.SIGTERM, stopped); signal.alarm(900)
    handles = []
    try:
        p, provenance = verify_protocol(args.protocol); result['provenance'] = provenance
        old = old_helpers(); result['runtime'] = old.check_device()
        import torch
        from experiments.downstream_blind.model import ActionBlindCandidateScorer
        from experiments.downstream_cooperation.model import straight_through_mask
        from course.dual_surrogate.runtime import gcn_logits
        modules = ['tests.test_downstream_quota',
                   'tests.test_downstream_model', 'tests.test_downstream_blind_model']
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        log = io.StringIO(); test_start = time.monotonic()
        checked = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        with (out / 'synthetic_tests.txt').open('x', encoding='utf-8', newline='\n') as stream: stream.write(log.getvalue())
        result['synthetic_tests'] = dict(tests_run=checked.testsRun, failures=len(checked.failures), errors=len(checked.errors),
                                         skipped=len(checked.skipped), wall_seconds=time.monotonic() - test_start)
        if not checked.wasSuccessful() or checked.skipped: raise RuntimeError('Registered synthetic checks must all run and pass.')
        engine, data, receipt, frozen = old.load_engine(parent_protocol(p), 3, out, started + 900)
        if engine.victim_model is not None or set(engine.teachers) != {'gcn_base'}:
            raise RuntimeError('No victim or extra surrogate may load for the TRAIN preflight.')
        teacher = engine.teachers['gcn_base']
        def prohibit_generation(module, inputs): raise RuntimeError('Upstream generation is not allowed in the cache-reuse ablation.')
        handles.append(engine.model.register_forward_pre_hook(prohibit_generation))
        def count_gcn(module, inputs):
            counters['GCN_forwards'] += 1
            if counters['GCN_forwards'] > 4: raise RuntimeError('TRAIN preflight query limit exceeded.')
        handles.append(teacher.register_forward_pre_hook(count_gcn))
        torch.manual_seed(20261111 + 3)
        model = ActionBlindCandidateScorer(data.feat_dim, 64, 64)
        initial = old.state_hash(model)
        expected = next(row for row in parent_result(p)['downstream'] if row['seed'] == 3)['initial_state_sha256']
        if initial != expected or sum(v.numel() for v in model.parameters()) != 151872:
            raise RuntimeError('Initial parameters differ from the matched visible policy.')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
        optimizer_ids = {id(v) for g in optimizer.param_groups for v in g['params']}
        if optimizer_ids & {id(v) for m in (engine.model, teacher) for v in m.parameters()}:
            raise RuntimeError('Frozen parameters entered the optimizer.')
        groups = p['preflight']['groups']
        if len(groups) != 4 or not set(groups) <= set(p['train_groups']): raise RuntimeError('Only four fixed TRAIN cache cases are permitted.')
        for group in groups:
            case = load_cached_case(p, 3, 'train', group, torch)
            before = {name: old.tensor_hash(v) for name, v in case.items() if torch.is_tensor(v)}
            optimizer.zero_grad(set_to_none=True)
            scores = blind_scores(model, case)
            rewired = dict(case, target_adjacency=case['target_adjacency'].roll(1, dims=1))
            with torch.no_grad(): changed = blind_scores(model, rewired)
            if not torch.equal(scores.detach(), changed): raise RuntimeError('Actual target decisions leaked into blind policy inputs.')
            q = case['quota']['candidate_quota']
            if q in (0, scores.numel()): raise RuntimeError('A gradient preflight case has no free candidate choice.')
            hard = old.candidate_mask(case, scores)
            weights = straight_through_mask(scores, hard, q)
            graph = old.make_graph(data, case, old.adjacency(case, weights), hard=old.adjacency(case, hard))
            logits = gcn_logits(teacher, graph, case['ids'], training=True)
            loss = old.softplus_loss(logits)
            if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite TRAIN preflight loss.')
            counters['autograd_calls'] += 1; loss.backward()
            grads = [v.grad for v in model.parameters() if v.grad is not None]
            if not grads or not all(bool(torch.isfinite(v).all()) for v in grads): raise RuntimeError('Missing or nonfinite gradient.')
            norm = float(sum(v.double().square().sum() for v in grads).sqrt())
            if norm <= 0: raise RuntimeError('The blind policy has no demonstrated training gradient.')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step(); counters['optimizer_updates'] += 1
            if any(old.tensor_hash(case[name]) != value for name, value in before.items()): raise RuntimeError('Cached upstream input mutated.')
            old.assert_frozen(engine.model, frozen['upstream']); old.assert_frozen(teacher, frozen['teacher'])
            result['cases'].append(dict(group=group, gradient_l2=norm, loss=float(loss.detach()),
                                        blind_scores_invariant_to_rewiring=True, cache_inputs_unchanged=True))
        if old.state_hash(model) == initial: raise RuntimeError('No blind parameter update occurred.')
        result.update(status='passed', initial_state_sha256=initial, matches_visible_initial_state=True,
                      final_state_sha256=old.state_hash(model), frozen_models_unchanged=True,
                      parameter_count=151872, upstream_checkpoint_sha256=receipt['final_model_sha256'])
    except BaseException as error:
        result['error'] = repr(error); result['traceback'] = traceback.format_exc(); traceback.print_exc()
    finally:
        for handle in handles: handle.remove()
        signal.alarm(0)
        result.update(wall_seconds=time.monotonic() - started, process_cpu_seconds=time.process_time() - cpu_started)
        write_json(out / 'result.json', result)
        print(__import__('json').dumps({k: result.get(k) for k in ('status', 'error', 'synthetic_tests', 'counters', 'wall_seconds')}, allow_nan=False), flush=True)
    if result['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
