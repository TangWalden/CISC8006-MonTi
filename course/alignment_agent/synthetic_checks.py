"""Meaningful tiny CPU checks; no real data, model checkpoint, GPU or API.

Includes analytic gradient/projection answers, real RNG forks and a tensor-only
fake-harness epoch exercising the actual batch-one training loop/counters.
"""
import json
import pathlib
import random
import tempfile
import unittest

import dgl
import numpy as np
import torch

from course.conditional_alignment.reference import alignment_penalty
from course.alignment_agent.diagnostics import gradient_norms, compare_gradients
from course.alignment_agent.experiment import (
    AgentAlignmentExperiment, attack_losses, training_objective,
    protect_alignment_gradients, protected_scalar_proxy, fork_generation_rng,
    generation_seed, model_state_sha256, transition_counts, MAX_GENERATIONS, MAX_GCN_FORWARDS)


def parameter(values):
    return torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))


class ProjectionChecks(unittest.TestCase):
    def test_hinge_flat_but_softplus_has_attack_gradient(self):
        value = parameter([8.0])
        logits = torch.stack((value, value*0.0), dim=1)
        margin, hinge, smooth = attack_losses(logits)
        gh, = torch.autograd.grad(hinge, value, retain_graph=True)
        gs, = torch.autograd.grad(smooth, value)
        self.assertEqual(float(hinge), 0.0)
        self.assertEqual(float(gh), 0.0)
        self.assertGreater(float(smooth), 0.0)
        self.assertLess(float(gs), 0.0)
        self.assertAlmostEqual(float(margin), 8.0)
        self.assertIsNone(value.grad)

    def test_analytic_opposing_component_is_removed(self):
        p = parameter([2.0, -3.0])
        named = (('linear_feat.weight', p),)
        ga = (torch.tensor([1.0, 0.0], dtype=torch.float64),)
        gq = (torch.tensor([-2.0, 1.0], dtype=torch.float64),)
        total, records = protect_alignment_gradients(named, ga, gq)
        torch.testing.assert_close(total[0], torch.tensor([1.0, 1.0], dtype=torch.float64))
        row = records['feature_head']
        self.assertTrue(row['projection_applied'])
        self.assertFalse(row['cap_applied'])
        self.assertAlmostEqual(row['protected_dot'], 0.0)
        self.assertGreaterEqual(row['protected_dot'], -1e-12)
        self.assertAlmostEqual(row['protected_alignment_norm'], 1.0)

    def test_orthogonal_large_component_is_capped_and_proxy_applies_gradient(self):
        p = parameter([2.0, -3.0])
        named = (('linear_feat.weight', p),)
        ga = (torch.tensor([1.0, 0.0], dtype=torch.float64),)
        gq = (torch.tensor([0.0, 3.0], dtype=torch.float64),)
        total, records = protect_alignment_gradients(named, ga, gq)
        self.assertAlmostEqual(records['feature_head']['cap_factor'], 1.0/3.0)
        self.assertGreaterEqual(records['feature_head']['protected_dot'], -1e-12)
        reported = torch.tensor(17.0, dtype=torch.float64, requires_grad=True)
        proxy = protected_scalar_proxy(reported, named, total)
        self.assertEqual(float(proxy.detach()), 17.0)
        proxy.backward()
        torch.testing.assert_close(p.grad, torch.tensor([1.0, 1.0], dtype=torch.float64))
        self.assertIsNone(reported.grad)

    def test_zero_attack_block_suppresses_alignment(self):
        p = parameter([2.0, -3.0])
        total, records = protect_alignment_gradients(
            (('linear_edge.weight', p),), (torch.zeros_like(p),),
            (torch.tensor([3.0, 4.0], dtype=torch.float64),))
        self.assertTrue(records['edge_head']['suppressed_attack_near_zero'])
        torch.testing.assert_close(total[0], torch.zeros_like(p))

    def test_none_support_preserved_and_zero_cosine_is_nullable(self):
        left, right, unused = parameter([1.0]), parameter([2.0]), parameter([3.0])
        named = (('linear_feat.weight', left), ('linear_feat.bias', right), ('linear_feat.unused', unused))
        ga = (torch.ones_like(left), None, None)
        gq = (None, -torch.ones_like(right), None)
        total, _ = protect_alignment_gradients(named, ga, gq)
        self.assertIsNone(total[2])
        protected_scalar_proxy(torch.tensor(1.0), named, total).backward()
        self.assertAlmostEqual(float(left.grad), 1.0)
        self.assertAlmostEqual(float(right.grad), -1.0)
        self.assertIsNone(unused.grad)
        zeros = (torch.zeros_like(left), None, None)
        zn, rn = gradient_norms(zeros, named), gradient_norms(gq, named)
        comparison = compare_gradients(zeros, gq, named, zn, rn)['all']
        self.assertIsNone(comparison['cosine'])
        self.assertIsNone(comparison['alignment_to_attack_norm_ratio'])

    def test_blocks_are_separate_even_if_global_dot_is_positive(self):
        a, b = parameter([1.0]), parameter([1.0])
        named = (('linear_feat.weight', a), ('linear_edge.weight', b))
        ga = (torch.ones_like(a), torch.ones_like(b))
        gq = (-torch.ones_like(a), torch.tensor([3.0], dtype=torch.float64))
        self.assertGreater(sum(float((x*y).sum()) for x,y in zip(ga,gq)), 0.0)
        total, records = protect_alignment_gradients(named, ga, gq)
        self.assertTrue(records['feature_head']['projection_applied'])
        torch.testing.assert_close(total[0], torch.ones_like(a))
        torch.testing.assert_close(total[1], torch.tensor([2.0], dtype=torch.float64))

    def test_off_returns_ordinary_tensor_and_leaves_alignment_only_parameter_unused(self):
        a, b = parameter([0.2]), parameter([2.0])
        base = a.sum()
        regularizer = b.square().sum()
        _, total = training_objective(base*2, base, regularizer, 'smooth_off', False)
        self.assertIs(total, base)
        optimizer = torch.optim.Adam([a,b], lr=0.1, weight_decay=0.1)
        before_b = b.detach().clone()
        total.backward()
        self.assertAlmostEqual(float(a.grad), 1.0)
        self.assertIsNone(b.grad)
        optimizer.step()
        self.assertTrue(torch.equal(before_b, b.detach()))

    def test_actual_nodewise_q90_uses_all_injected_nodes_as_denominator(self):
        graph = dgl.graph(([0,2,0,3], [2,0,3,0]), num_nodes=4)
        graph.ndata['x'] = torch.tensor([[1.0,0.0], [0.0,1.0],
                                        [1.0,0.0], [1.0,1.0]], dtype=torch.float64)
        ref = dict(degree_rule='floor_log2_nonself', buckets={
            '0':dict(center=[0.0,0.0], scale=[1.0,1.0], tolerance=0.5)})
        penalty, stats = alignment_penalty(graph, [2,3], ref, arm='gated', coefficient=0.1)
        self.assertEqual(stats['active_nodes'], 1)
        self.assertEqual(stats['injected_nodes'], 2)
        self.assertAlmostEqual(float(penalty), 0.05)
        _, total = training_objective(torch.tensor(0.2), torch.tensor(0.3), penalty,
                                      'smooth_threshold', True)
        self.assertAlmostEqual(float(total), 0.35, places=6)


class RNGChecks(unittest.TestCase):
    def test_state_hash_is_order_independent_and_records_dtype_shape_values_without_rng(self):
        state = {'b':torch.tensor([1.0,2.0]), 'a':torch.tensor(3,dtype=torch.long)}
        before = torch.random.get_rng_state().clone()
        expected = model_state_sha256(state)
        self.assertEqual(expected, model_state_sha256({'a':state['a'].clone(), 'b':state['b'].clone()}))
        self.assertNotEqual(expected, model_state_sha256(dict(state,b=state['b'].reshape(1,2))))
        self.assertNotEqual(expected, model_state_sha256(dict(state,b=state['b'].double())))
        self.assertNotEqual(expected, model_state_sha256(dict(state,b=torch.tensor([9.0,2.0]))))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_real_cpu_rng_fork_restores_all_streams(self):
        before_py = random.getstate()
        before_np = np.random.get_state()
        before_torch = torch.random.get_rng_state().clone()
        with fork_generation_rng(12345) as receipt:
            random.random();np.random.randn(3);torch.randn(3)
        self.assertTrue(receipt['rng_restored_exactly'])
        self.assertEqual(random.getstate(), before_py)
        after_np = np.random.get_state()
        self.assertTrue(np.array_equal(before_np[1], after_np[1]))
        self.assertEqual(before_np[2:], after_np[2:])
        self.assertTrue(torch.equal(before_torch, torch.random.get_rng_state()))

    def test_common_noise_ignores_arm_identity_and_phase_is_separated(self):
        seed = generation_seed(3, 'train', 2, 22)
        with fork_generation_rng(seed):
            first = torch.randn(8)
        torch.randn(7);random.random();np.random.randn(3)
        with fork_generation_rng(generation_seed(3, 'train', 2, 22)):
            second = torch.randn(8)
        with fork_generation_rng(generation_seed(3, 'validation', 2, 22)):
            other = torch.randn(8)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))


class FakeAssignment:
    def __getitem__(self, group):
        return self
    def nonzero(self):
        return np.array([0]), np.array([0])


class FakeData:
    target_assignment = FakeAssignment()
    def get_train_loader(self):
        for group in range(256):
            yield torch.tensor([group]), torch.tensor([1]), torch.tensor([1])


class FakeModel:
    def __init__(self):
        self.scheduler_steps = 0
    def train(self, enabled):
        pass
    def step(self):
        self.scheduler_steps += 1


class FakeFrozenSurrogate:
    def eval(self):
        pass


class EpochHarness(AgentAlignmentExperiment):
    def __init__(self, path):
        self.theta = parameter([1.0])
        self.evidence = pathlib.Path(path)
        (self.evidence/'raw').mkdir()
        self.model, self.surrogate_model = FakeModel(), FakeFrozenSurrogate()
        self.data, self.reference = FakeData(), {'gate_quantile':0.9}
        self.training_arm, self.outer_enabled = 'smooth_off', False
        self.current_epoch, self.applied_actions = 1, [False]
        self.probe_groups = set(range(8))
        self.named_parameters_for_probe = (('linear_feat.weight', self.theta),)
        self.gradient_calls = self.optimizer_steps = self.gcn_forwards = self.generator_forwards = 0
        self.last_generation_rng = {'rng_restored_exactly':True}
        self.optimizer = torch.optim.SGD([self.theta], lr=0.01)
        original = self.optimizer.step
        def counted():
            result = original()
            self.optimizer_steps += 1
            return result
        self.optimizer.step = counted
    def check_deadline(self):
        pass
    def get_perturbed_graph(self, *args, **kwargs):
        self.generator_forwards += 1
        return object()
    def get_logits(self, *args, **kwargs):
        self.gcn_forwards += 1
        return torch.stack((self.theta, -self.theta), dim=1)
    def consume_alignment(self, graph):
        q90 = self.theta.square().mean()*0.1
        return q90, dict(outside_tolerance_nodes=1, injected_nodes=1,
                         mean_distance=float(self.theta.detach().square().mean()), fallback_nodes=0)


class TrainingFlowChecks(unittest.TestCase):
    def test_same_condition_clean_baseline_separates_filter_harm_from_new_errors(self):
        values = transition_counts(torch.tensor([1,1,0,0]), torch.tensor([1,0,0,1]),
                                   torch.tensor([0,0,0,0]))
        self.assertEqual(values, dict(target_occurrences=4,clean_wrong=2,attacked_wrong=2,
                                     new_wrong=1,recoveries=1,retained_correct=1,retained_wrong=1))
        labels, filtered_attack = torch.tensor([1,1]), torch.tensor([0,0])
        filtered = transition_counts(labels, torch.tensor([0,1]), filtered_attack)
        wrong_baseline = transition_counts(labels, torch.tensor([1,1]), filtered_attack)
        self.assertEqual(filtered['new_wrong'], 1)
        self.assertEqual(filtered['retained_wrong'], 1)
        self.assertEqual(wrong_baseline['new_wrong'], 2)
        self.assertEqual(MAX_GENERATIONS, 2176)
        self.assertEqual(MAX_GCN_FORWARDS, 2562)

    def test_export_reports_actual_method_not_inherited_nodewise_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = object.__new__(AgentAlignmentExperiment)
            harness.evidence = pathlib.Path(directory)
            (harness.evidence/'raw').mkdir()
            harness.protocol = {'export_valid_groups':[957], '_protocol_sha256':'2'*64}
            harness.config = {'seed':3}
            harness.source_commit = '1'*40
            harness.phase, harness.current_epoch = 'validation', 8
            harness.arm, harness.training_arm = 'gated', 'smooth_agent'
            harness.data = type('Data', (), {'graph':dgl.graph(([],[]),num_nodes=2)})()
            harness.export_index = []
            harness._capture = dict(x_atk=torch.ones(1,2), adj_attack=torch.tensor([[1.0,0.0,0.0]]),
                                    target_ids=torch.tensor([0]), candidate_ids=torch.tensor([1]))
            harness._export_attack(957)
            payload = torch.load(harness.evidence/'raw/attack_epoch_008_group_957.pt', map_location='cpu')
            self.assertEqual(payload['arm'], 'smooth_agent')
            self.assertEqual(payload['nodewise_alignment_arm'], 'gated')
            self.assertEqual(payload['phase'], 'validation')
            self.assertEqual(payload['epoch'], 8)
            self.assertEqual(payload['source_commit'], '1'*40)
            self.assertEqual(payload['protocol_sha256'], '2'*64)
            self.assertEqual(harness.export_index[0]['arm'], 'smooth_agent')

    def test_real_epoch_loop_keeps_256_batch_one_steps_and_eight_same_forward_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = EpochHarness(directory)
            state = harness._train_one_epoch()
            self.assertEqual(harness.optimizer_steps, 256)
            self.assertEqual(harness.gcn_forwards, 256)
            self.assertEqual(harness.generator_forwards, 256)
            self.assertEqual(harness.gradient_calls, 16)
            self.assertEqual(harness.model.scheduler_steps, 1)
            self.assertEqual(state['train_groups'], 256)
            self.assertEqual(state['target_occurrences'], 256)
            self.assertEqual(state['gradient_probe_groups'], 8)
            self.assertEqual(state['prior_actions'], [False])
            rows = [json.loads(line) for line in (pathlib.Path(directory)/'raw/train_epoch_001_cases.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 256)
            self.assertEqual(sum(row['gradient_probe'] is not None for row in rows), 8)


if __name__ == '__main__':
    unittest.main()
