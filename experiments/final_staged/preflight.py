"""Bounded original-TRAIN checks for the formal staged CPU comparison.

The synthetic suite is separate from the real-data counters. Real-data checks
use four registered TRAIN contexts, one frozen upstream and one frozen GCN.
No GAGA model or TEST inference is allowed in this phase.
"""

import argparse
import io
import pathlib
import signal
import sys
import time
import traceback
import unittest

from common import (verify_protocol, check_device, read_json, write_json, sha,
                    state_hash, assert_frozen, load_engine, capture_case,
                    downstream_scores, candidate_mask, adjacency, make_graph,
                    softplus_loss, tensor_hash, precheck_resources)


TEST_MODULES = ('tests.test_downstream_budget_audit', 'tests.test_downstream_quota',
                'tests.test_downstream_model')
TENSOR_INPUTS = ('ids', 'candidate_ids', 'x_aux', 'x_targets', 'x_candidates',
                 'h_candidates', 'native_adjacency', 'target_adjacency',
                 'original_candidates', 'random_candidates')


def immutable_case_hashes(case):
    return {name: tensor_hash(case[name]) for name in TENSOR_INPUTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    started, cpu_started = time.monotonic(), time.process_time()
    output = pathlib.Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    counters = dict(real_data_generator_forwards=0, real_data_GCN_forwards=0,
                    real_data_downstream_forwards=0, real_data_backward_calls=0,
                    real_data_optimizer_updates=0, real_data_GAGA_forwards=0,
                    TEST_queries=0, validation_queries=0, API_calls=0)
    result = dict(status='failed', counters=counters, real_data_cases=[],
                  synthetic_tests=None, GPU_hours=0)
    hooks = []
    engine = data = model = optimizer = None
    failed = None

    def interrupted(signum, frame):
        raise TimeoutError('Registered TRAIN preflight deadline or termination reached.')

    def bound():
        if time.monotonic() - started >= 900:
            raise TimeoutError('Frozen 900-second preflight bound reached.')

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(900)
    try:
        p, provenance = verify_protocol(args.protocol)
        result['provenance'] = provenance
        if (pathlib.Path(p['runtime_root']).resolve() / 'runs') not in output.parents:
            raise RuntimeError('Private preflight output must remain below runtime runs/.')
        groups = list(p['preflight']['groups'])
        original_train = set(p['train_groups'])
        if (len(groups) != 4 or len(set(groups)) != 4 or not set(groups) <= original_train
                or set(groups) & set(p['test_groups'])):
            raise RuntimeError('The four preflight contexts must be original TRAIN groups.')
        result['resource_admission'] = precheck_resources(output, p)
        result['device'] = check_device()
        import torch
        from experiments.downstream_cooperation.model import DownstreamCandidateScorer, straight_through_mask
        from course.dual_surrogate.runtime import gcn_logits

        test_log = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromNames(TEST_MODULES)
        tests_started = time.monotonic()
        checked = unittest.TextTestRunner(stream=test_log, verbosity=2).run(suite)
        with (output / 'synthetic_tests.log').open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(test_log.getvalue())
        result['synthetic_tests'] = dict(
            modules=list(TEST_MODULES), tests_run=checked.testsRun,
            failures=len(checked.failures), errors=len(checked.errors), skipped=len(checked.skipped),
            wall_seconds=time.monotonic() - tests_started,
            source='Only source-bound synthetic inputs; no real datasets or checkpoints in the suite.')
        if not checked.wasSuccessful() or checked.skipped:
            raise RuntimeError('Every registered synthetic test must run and pass without skips.')
        bound()
        seed = p['preflight']['seed']
        if seed != 0: raise RuntimeError('The frozen preflight uses formal seed0 only.')
        engine, data, receipt, frozen = load_engine(p, seed, output, started + 900)
        if engine.victim_model is not None or set(engine.teachers) != {'gcn_base'}:
            raise RuntimeError('Preflight must not load a GAGA or extra classifier branch.')
        teacher = engine.teachers['gcn_base']
        result['frozen_models_before'] = frozen
        result['upstream_checkpoint_sha256'] = receipt['final_model_sha256']
        result['real_data_groups'] = groups

        def counted_upstream(module, inputs):
            bound()
            counters['real_data_generator_forwards'] += 1
            if counters['real_data_generator_forwards'] > 4:
                raise RuntimeError('Preflight generation cap exceeded.')

        def counted_gcn(module, inputs):
            bound()
            counters['real_data_GCN_forwards'] += 1
            if counters['real_data_GCN_forwards'] > 12:
                raise RuntimeError('Preflight GCN forward cap exceeded.')

        hooks = [engine.model.register_forward_pre_hook(counted_upstream),
                 teacher.register_forward_pre_hook(counted_gcn)]
        torch.manual_seed(20261111 + seed)
        model = DownstreamCandidateScorer(data.feat_dim, 64, p['settings']['hidden_dimension'])
        optimizer = torch.optim.Adam(model.parameters(), lr=p['settings']['learning_rate'],
                                     weight_decay=p['settings']['weight_decay'])
        model.train()
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group['params']}
        own_ids = {id(parameter) for parameter in model.parameters()}
        forbidden_ids = {id(parameter) for frozen_model in (engine.model, teacher)
                         for parameter in frozen_model.parameters()}
        if optimizer_ids != own_ids or optimizer_ids & forbidden_ids:
            raise RuntimeError('Optimizer parameters are not exclusively downstream parameters.')
        result['downstream_parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        result['downstream_initial_state_sha256'] = state_hash(model)
        nonzero_gradient_cases = 0
        changed_weight_cases = 0
        with (output / 'progress.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            import json
            for group in groups:
                bound()
                case = capture_case(engine, data, seed, group, 'final_staged_preflight_train_v1')
                before_inputs = immutable_case_hashes(case)
                quota = case['quota']
                with torch.no_grad():
                    original = adjacency(case, case['original_candidates'])
                    graph = make_graph(data, case, original)
                    weighted = gcn_logits(teacher, graph, case['ids'], training=True)
                    ordinary = gcn_logits(teacher, graph, case['ids'], training=False)
                    maximum_difference = float((weighted - ordinary).abs().max())
                    class_agreement = bool(torch.equal(weighted.argmax(1), ordinary.argmax(1)))
                    if (not class_agreement
                            or not torch.allclose(weighted, ordinary, atol=1e-4, rtol=1e-5)):
                        raise RuntimeError('Unit hard-edge weights disagree with the same native hard graph.')
                del graph, original
                optimizer.zero_grad(set_to_none=True)
                previous_state = state_hash(model)
                counters['real_data_downstream_forwards'] += 1
                scores = downstream_scores(model, case)
                hard = candidate_mask(case, scores)
                weights = straight_through_mask(scores, hard, quota['candidate_quota'])
                if not torch.equal(weights.detach(), hard):
                    raise RuntimeError('Straight-through forward changed a binary hard connection.')
                graph = make_graph(data, case, adjacency(case, weights), hard=adjacency(case, hard))
                logits = gcn_logits(teacher, graph, case['ids'], training=True)
                loss = softplus_loss(logits)
                has_choice = 0 < quota['candidate_quota'] < scores.numel()
                gradient_maximum = 0.0
                changed = False
                if has_choice:
                    if not loss.requires_grad or not bool(torch.isfinite(loss)):
                        raise RuntimeError('The downstream TRAIN loss has no finite gradient graph.')
                    counters['real_data_backward_calls'] += 1
                    loss.backward()
                    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
                        raise RuntimeError('Downstream gradient is missing or nonfinite.')
                    gradient_maximum = max(float(value.abs().max()) for value in gradients)
                    if gradient_maximum > 0:
                        nonzero_gradient_cases += 1
                    optimizer.step()
                    counters['real_data_optimizer_updates'] += 1
                    changed = state_hash(model) != previous_state
                    changed_weight_cases += int(changed)
                assert_frozen(engine.model, frozen['upstream'])
                assert_frozen(teacher, frozen['teacher'])
                if immutable_case_hashes(case) != before_inputs:
                    raise RuntimeError('A frozen upstream input or comparison tensor changed.')
                row = dict(group=group, targets=len(case['ids']), quota=quota,
                           upstream_tensor_hashes=before_inputs,
                           weighted_native_logit_max_abs_difference=maximum_difference,
                           weighted_native_prediction_agreement=class_agreement,
                           loss=float(loss.detach()), candidate_choices_trainable=has_choice,
                           downstream_gradient_linf=gradient_maximum,
                           downstream_weights_changed=changed,
                           approximation='Selected hard-edge straight-through score derivative; hard topology and graph degrees are not differentiated.')
                result['real_data_cases'].append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                print('PREFLIGHT_TRAIN_GROUP', group, 'GRAD_MAX', gradient_maximum,
                      'WEIGHTS_CHANGED', changed, flush=True)
                del graph, logits, loss, scores, weights, hard, case
        if nonzero_gradient_cases < 1 or changed_weight_cases < 1:
            raise RuntimeError('No registered TRAIN case produced a nonzero gradient and a downstream weight update.')
        if (counters['real_data_generator_forwards'] != 4
                or counters['real_data_GCN_forwards'] != 12
                or counters['real_data_downstream_forwards'] != 4):
            raise RuntimeError('Incomplete registered preflight real-data matrix.')
        result.update(status='passed', nonzero_gradient_cases=nonzero_gradient_cases,
                      changed_downstream_weight_cases=changed_weight_cases,
                      downstream_final_state_sha256=state_hash(model),
                      frozen_models_after=dict(upstream=state_hash(engine.model), teacher=state_hash(teacher)),
                      outcome='Geometry, exact hard forward, same-graph GCN parity, conditional gradients and parameter freezing checked; no efficacy claim.')
    except BaseException as error:
        failed = error
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
        traceback.print_exc()
    finally:
        signal.alarm(0)
        for hook in hooks:
            hook.remove()
        result['wall_seconds'] = time.monotonic() - started
        result['process_cpu_seconds'] = time.process_time() - cpu_started
        write_json(output / 'result.json', result)
        print('PREFLIGHT_RESULT', result['status'], 'GCN_FORWARDS',
              counters['real_data_GCN_forwards'], flush=True)
    if failed is not None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
