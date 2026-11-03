"""Synthetic CPU checks; no datasets, checkpoints, detector/model/API calls.

Source preparation only until root commits/freezes and records a test execution.
"""
import math
import unittest

import dgl
import torch

from course.alignment_agent.filter import (
    SCHEMA, FEATURE_NAMES, SCALE_FLOOR, CALIBRATION_QUANTILE,
    node_features, fit_reference, scores_from_features, score_nodes,
    isolate_flagged, apply_filter,
)


def tiny_graph():
    graph = dgl.graph(([0, 1, 1, 2, 0, 1, 2, 3], [1, 0, 2, 1, 0, 1, 2, 3]), num_nodes=4)
    graph.ndata['x'] = torch.tensor([[3., 4.], [0., 2.], [0., 0.], [-3., 0.]], dtype=torch.float64)
    graph.ndata['y'] = torch.tensor([1, 0, 1, 0])
    graph.edata['weight'] = torch.arange(graph.num_edges(), dtype=torch.float64)
    return graph


def simple_reference(threshold=0.0):
    return dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES), scale_floor=SCALE_FLOOR,
                calibration_quantile=CALIBRATION_QUANTILE,
                median=[0., 0., 0., 0.], scales=[1., 1., 1., 1.], threshold=threshold,
                flag_rule='strict_score_greater_than_threshold')


class FilterChecks(unittest.TestCase):
    def test_known_current_features_and_no_stale_degree_field(self):
        graph = tiny_graph()
        graph.ndata['degree'] = torch.full((4,), 999.0)
        result = node_features(graph, feature_chunk_size=2, edge_chunk_size=3)
        expected = torch.tensor([[math.log(2), 5., 4., math.log(3)],
                                 [math.log(3), 2., 2., math.log(2)],
                                 [math.log(2), 0., 0., math.log(3)],
                                 [0., 3., 3., 0.]], dtype=torch.float64)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result, node_features(graph, feature_chunk_size=4, edge_chunk_size=100))

    def test_exact_score_and_strict_threshold_tie(self):
        graph = tiny_graph()
        features = node_features(graph)
        expected = features.square().mean(1)
        torch.testing.assert_close(scores_from_features(features, simple_reference()), expected)
        reference = simple_reference(float(expected[0]))
        result = apply_filter(graph, reference)
        self.assertFalse(bool(result['flagged'][0]))
        torch.testing.assert_close(result['scores'], expected)

    def test_fit_train_split_mad_floor_labels_ignored_and_ties(self):
        nodes = torch.arange(20)
        graph = dgl.graph((nodes, nodes), num_nodes=20)
        graph.ndata['x'] = torch.tensor([[1., 0.]]).repeat(20, 1)
        graph.ndata['y'] = torch.zeros(20, dtype=torch.long)
        first = fit_reference(graph, nodes)
        graph.ndata['y'] = torch.ones(20, dtype=torch.long)
        second = fit_reference(graph, nodes.flip(0))
        self.assertEqual(first, second)
        self.assertEqual(first['fit_original_node_rates']['nodes'], 16)
        self.assertEqual(first['calibration_original_node_rates']['nodes'], 4)
        self.assertEqual(first['median'], [0., 1., 1., 0.])
        self.assertEqual(first['mad'], [0., 0., 0., 0.])
        self.assertEqual(first['scales'], [1e-3] * 4)
        self.assertEqual(first['threshold'], 0.)
        self.assertEqual(first['calibration_ties_at_threshold'], 4)
        self.assertEqual(first['all_train_original_node_rates']['flagged'], 0)
        self.assertFalse(bool(apply_filter(graph, first)['flagged'].any()))

    def test_isolation_preserves_nodes_loops_input_and_detaches_copies(self):
        graph = tiny_graph()
        original_src, original_dst = [value.clone() for value in graph.edges(order='eid')]
        original_features = graph.ndata['x'].clone()
        original_weight = graph.edata['weight'].clone()
        post, counters = isolate_flagged(graph, torch.tensor([False, True, False, False]))
        self.assertEqual(post.num_nodes(), 4)
        self.assertEqual(post.num_edges(), 4)
        self.assertEqual(counters['removed_nonself_incident_arcs'], 4)
        torch.testing.assert_close(post.in_degrees(), torch.ones(4, dtype=torch.int64))
        torch.testing.assert_close(graph.edges(order='eid')[0], original_src)
        torch.testing.assert_close(graph.edges(order='eid')[1], original_dst)
        torch.testing.assert_close(graph.ndata['x'], original_features)
        torch.testing.assert_close(graph.edata['weight'], original_weight)
        post.ndata['x'][0, 0] = 1000
        post.edata['weight'][0] = -1000
        torch.testing.assert_close(graph.ndata['x'], original_features)
        torch.testing.assert_close(graph.edata['weight'], original_weight)

    def test_all_none_and_no_origin_or_label_scoring_input(self):
        graph = tiny_graph()
        untouched, _ = isolate_flagged(graph, torch.zeros(4, dtype=torch.bool))
        for actual, expected in zip(untouched.edges(order='eid'), graph.edges(order='eid')):
            torch.testing.assert_close(actual, expected)
        isolated, _ = isolate_flagged(graph, torch.ones(4, dtype=torch.bool))
        self.assertEqual(isolated.num_nodes(), 4)
        self.assertEqual(isolated.num_edges(), 4)
        before = score_nodes(graph, simple_reference())
        graph.ndata['y'] = torch.tensor([0, 1, 0, 1])
        graph.ndata['injected_id_oracle'] = torch.tensor([1, 0, 0, 0])
        torch.testing.assert_close(before, score_nodes(graph, simple_reference()))
        with self.assertRaises(TypeError):
            apply_filter(graph, simple_reference(), injected_ids=[0])

    def test_fail_closed_invalid_inputs_and_memory_ceiling(self):
        graph = tiny_graph()
        with self.assertRaises(MemoryError):
            node_features(graph, max_working_bytes=1)
        with self.assertRaises(MemoryError):
            isolate_flagged(graph, torch.zeros(4, dtype=torch.bool), max_working_bytes=1)
        with self.assertRaises(ValueError):
            isolate_flagged(graph, torch.tensor([0, 1, 0, 0]))
        malformed = simple_reference()
        malformed['scales'][0] = 0
        with self.assertRaises(ValueError):
            score_nodes(graph, malformed)
        malformed = simple_reference(float('nan'))
        with self.assertRaises(ValueError):
            score_nodes(graph, malformed)
        graph.ndata['x'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            node_features(graph)
        graph = tiny_graph()
        with self.assertRaises(ValueError):
            fit_reference(graph, [0, 0, 1, 2])
        with self.assertRaises(ValueError):
            fit_reference(graph, [0., 1., 2., 3.])


if __name__ == '__main__':
    unittest.main()
