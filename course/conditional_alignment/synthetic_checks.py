"""Small CPU known-answer checks. No datasets, checkpoints or model forwards.

These tests are source preparation until a committed-source execution receipt
records an actual run. They establish no real-data alignment or attack efficacy.
"""
import unittest

import dgl
import torch

from course.conditional_alignment.reference import (
    alignment_penalty, build_reference, degree_buckets, matched_statistics,
    relationship_moments)


def tiny_graph():
    graph = dgl.graph(([0, 1, 0, 1], [1, 0, 0, 1]))
    features = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float64,
                            requires_grad=True)
    graph.ndata['x'] = features
    graph.ndata['y'] = torch.tensor([0, 1])
    graph.edata['weight'] = torch.ones(graph.num_edges(), dtype=torch.float64,
                                     requires_grad=True)
    return graph, features


def reference(tolerance=0.0):
    return dict(degree_rule='floor_log2_nonself', buckets={
        '0': dict(center=[0.0, 0.0], scale=[1.0, 1.0], tolerance=tolerance)})


class SyntheticChecks(unittest.TestCase):
    def test_known_relationship_moments_exclude_self_loops(self):
        graph, _ = tiny_graph()
        values, degrees = relationship_moments(graph, [1])
        self.assertEqual(degrees.tolist(), [1])
        torch.testing.assert_close(values, torch.tensor([[2.0**-0.5, 0.5]],
                                                         dtype=torch.float64))

    def test_degree_bins_and_lower_bucket_fallback(self):
        self.assertEqual(degree_buckets(torch.tensor([0, 1, 2, 3, 4, 7, 8])).tolist(),
                         [-1, 0, 1, 1, 2, 2, 3])
        ref = reference()
        ref['buckets']['2'] = dict(center=[1.0, 1.0], scale=[1.0, 1.0], tolerance=2.0)
        _, _, _, requested, chosen = matched_statistics(
            torch.tensor([2]), ref, dtype=torch.float32, device='cpu')
        self.assertEqual(requested, [1])
        self.assertEqual(chosen, [0])

    def test_gate_is_strict_at_exact_tolerance(self):
        graph, _ = tiny_graph()
        values, _ = relationship_moments(graph, [1])
        distance = float(values.square().mean().item())
        ref = reference(distance)
        loss, stats = alignment_penalty(graph, [1], ref, arm='gated')
        self.assertEqual(stats['outside_tolerance_nodes'], 0)
        self.assertEqual(stats['active_nodes'], 0)
        self.assertEqual(loss.item(), 0.0)
        always, _ = alignment_penalty(graph, [1], ref, arm='always')
        self.assertGreater(always.item(), 0.0)

    def test_alignment_has_finite_nonzero_feature_gradient(self):
        graph, features = tiny_graph()
        loss, stats = alignment_penalty(graph, [1], reference(), arm='gated', training=True)
        self.assertEqual(stats['active_nodes'], 1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(features.grad).all()))
        self.assertGreater(float(features.grad[1].abs().sum()), 0.0)
        self.assertIsNotNone(graph.edata['weight'].grad)

    def test_off_zero_gradient_and_all_arms_preserve_graph(self):
        graph, features = tiny_graph()
        original_edges = tuple(value.clone() for value in graph.edges())
        original_features = features.detach().clone()
        loss, stats = alignment_penalty(graph, [1], reference(), arm='off', training=True)
        loss.backward()
        self.assertEqual(stats['active_nodes'], 0)
        self.assertEqual(float(features.grad.abs().sum()), 0.0)
        for arm in ['off', 'always', 'gated']:
            alignment_penalty(graph, [1], reference(), arm=arm)
            self.assertEqual(graph.num_nodes(), 2)
            self.assertEqual(graph.num_edges(), 4)
            for before, after in zip(original_edges, graph.edges()):
                self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(original_features, graph.ndata['x'].detach()))

    def test_reference_ignores_nontrain_labels_and_consumes_no_rng(self):
        graph = dgl.graph(([0, 1, 2, 3], [1, 0, 3, 2]))
        graph.ndata['x'] = torch.tensor([[1.0, 0.0], [1.0, 1.0],
                                        [0.0, 1.0], [1.0, 0.0]])
        graph.ndata['y'] = torch.tensor([0, 0, 1, 0])
        before = torch.random.get_rng_state().clone()
        fitted = build_reference(graph, [0, 1], min_count=2)
        graph.ndata['y'][2:] = torch.tensor([0, 1])
        second = build_reference(graph, [0, 1], min_count=2)
        self.assertEqual(fitted, second)
        self.assertEqual(fitted['benign_train_nodes'], 2)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_invalid_arm_and_unconnected_injection_fail(self):
        graph, _ = tiny_graph()
        with self.assertRaises(ValueError):
            alignment_penalty(graph, [1], reference(), arm='unknown')
        isolated = dgl.graph(([], []), num_nodes=1)
        isolated.ndata['x'] = torch.ones(1, 2)
        with self.assertRaises(ValueError):
            alignment_penalty(isolated, [0], reference(), arm='gated')


if __name__ == '__main__':
    unittest.main()
