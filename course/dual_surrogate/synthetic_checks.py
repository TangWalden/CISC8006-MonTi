"""CPU-only known-answer checks for the isolated sequence adapter.

No real dataset, pretrained checkpoint, detector inference, optimizer or API.
Root executes only after committing/freezing this source and test registration.
"""
import random
import types
import unittest

import dgl
import numpy as np
import scipy.sparse as sp
import torch

from course.dual_surrogate.sequence_adapter import (
    updated_sequences, weighted_group_aggregate, estimate_workspace_bytes)


def neighbors(num_nodes, edges):
    result = [set() for _ in range(num_nodes)]
    for left, right in edges:
        if left != right:
            result[left].add(right)
    return result


def reach_two(adjacency, target):
    result = set()
    for middle in adjacency[target]:
        result.update(adjacency[middle])
    result.discard(target)
    return result


def independent_sums(features, edges, masks):
    """Python set oracle: count each reachable member once and mask self."""
    n, f = features.shape
    adjacency = neighbors(n, edges)
    sums, counts = torch.zeros((n, 7, f)), torch.zeros((n, 7, 1))
    for target in range(n):
        sums[target, 0] = features[target]
        counts[target, 0] = 1
        for hop, members in enumerate((adjacency[target], reach_two(adjacency, target))):
            for member in sorted(members):
                for group, mask in enumerate(masks):
                    if mask[member]:
                        sums[target, 1+3*hop+group] += features[member]
                        counts[target, 1+3*hop+group] += 1
            sums[target, 3+3*hop] += features[target]
            counts[target, 3+3*hop] += 1
    return sums, counts, adjacency


def view(features, undirected_edges, groups):
    n = len(features)
    features = torch.tensor(features, dtype=torch.float32)
    edges = [(a, b) for a, b in undirected_edges for a, b in ((a, b), (b, a))]
    masks = tuple(torch.tensor([int(g == index) for g in groups], dtype=torch.float32)
                  for index in range(3))
    sums, counts, adjacency = independent_sums(features, edges, masks)
    row, col = zip(*edges) if edges else ([], [])
    original = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    row2, col2 = [], []
    for target in range(n):
        for member in sorted(reach_two(adjacency, target)):
            row2.append(target); col2.append(member)
    original_two = sp.csr_matrix((np.ones(len(row2)), (row2, col2)), shape=(n, n))
    result = types.SimpleNamespace(features=features, feat_dim=features.shape[1], device=torch.device("cpu"),
        feat_sum=sums, d=counts, sequences=sums/torch.where(counts > 0, counts, torch.ones_like(counts)),
        adj=original, adj2=original_two, original_edges=edges,
        group_masks=dict(zip(("train_neg", "train_pos", "train_masked"), masks)))
    return result


def attacked(data, injected_features, undirected_edges=(), attack_weight=None):
    if not torch.is_tensor(injected_features):
        injected_features = torch.tensor(injected_features, dtype=torch.float32).reshape(-1, data.feat_dim)
    n, injected = len(data.features), len(injected_features)
    attack_edges = [(a, b) for a, b in undirected_edges for a, b in ((a, b), (b, a))]
    edges = list(data.original_edges) + attack_edges + [(node, node) for node in range(n+injected)]
    graph = dgl.graph(([a for a, _ in edges], [b for _, b in edges]), num_nodes=n+injected)
    graph.ndata["x"] = torch.cat((data.features, injected_features))
    weights = []
    for index in range(len(edges)):
        if len(data.original_edges) <= index < len(data.original_edges)+len(attack_edges) and attack_weight is not None:
            weights.append(attack_weight)
        else:
            weights.append(torch.tensor(1.0))
    graph.edata["weight"] = torch.stack(weights)
    for index, name in enumerate(("train_neg", "train_pos", "train_masked")):
        graph.ndata[name] = torch.cat((data.group_masks[name],
                                      torch.full((injected,), 1.0 if index == 2 else 0.0)))
    graph.ndata["y"] = torch.full((n+injected,), 99, dtype=torch.long)  # deliberately unusable
    return graph


def full_hard_oracle(data, graph, targets):
    src, dst = graph.edges(order="eid")
    masks = tuple(torch.cat((data.group_masks[name], torch.full(
        (graph.num_nodes()-len(data.features),), 1.0 if index == 2 else 0.0)))
        for index, name in enumerate(("train_neg", "train_pos", "train_masked")))
    sums, counts, _ = independent_sums(graph.ndata["x"].detach(), list(zip(src.tolist(), dst.tolist())), masks)
    return (sums/torch.where(counts > 0, counts, torch.ones_like(counts)))[targets]


def frozen_original(data, graph, targets, candidates, injections):
    # Use the frozen upstream implementation directly without constructing its
    # data handler or loading any graph/model files. Supply the matching view.
    from modules.data_handler import DataHandler
    proxy = types.SimpleNamespace(**data.__dict__)
    proxy.group_agg_tensor = types.MethodType(DataHandler.group_agg_tensor, proxy)
    proxy.feat_agg_tensor = types.MethodType(DataHandler.feat_agg_tensor, proxy)
    return DataHandler.get_updated_sequences(proxy, graph, torch.tensor(targets),
        None if candidates is None else torch.tensor(candidates), torch.tensor(injections), edge_weight=True)


class SequenceAdapterChecks(unittest.TestCase):
    def test_hard_forward_matches_frozen_update_and_independent_whole_graph_oracle(self):
        data = view([[1., 2.], [3., 1.], [4., -1.], [6., 5.], [10., 0.]],
                    [(0, 1), (0, 2), (1, 3), (2, 3)], [1, 0, 1, 2, 2])
        graph = attacked(data, [[8., 9.], [-2., 4.]], [(5, 0), (5, 2), (6, 4), (5, 6)])
        targets, candidates, injections = [4, 0], [3, 1, 2], [5, 6]
        actual, trace = updated_sequences(data, graph, targets, candidates, injections, return_trace=True)
        with torch.no_grad():
            reference = frozen_original(data, graph, targets, candidates, injections)
            oracle = full_hard_oracle(data, graph, targets)
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-6)
        torch.testing.assert_close(actual, oracle, rtol=0, atol=1e-6)
        self.assertFalse(trace["labels_read"])
        self.assertFalse(trace["full_graph_dense_matrix"])
        self.assertEqual(trace["selected_nodes"], 7)

    def test_onehop_weight_gradient_has_correct_numerator_and_denominator(self):
        data = view([[1.]], [], [1])
        weight = torch.tensor(1.0, requires_grad=True)
        generated = torch.tensor([[5.]], requires_grad=True)
        graph = attacked(data, generated, [(0, 1)], weight)
        sequence = updated_sequences(data, graph, [0], None, [1])
        self.assertAlmostEqual(float(sequence[0, 3, 0]), 3.0)
        gw, gx = torch.autograd.grad(sequence[0, 3, 0], (weight, generated))
        # (1+5w)/(1+w) at w=1: edge derivative +1; feature derivative 1/2.
        # The old unweighted numerator instead gives a negative edge derivative.
        torch.testing.assert_close(gw, torch.tensor(1.0))
        torch.testing.assert_close(gx, torch.tensor([[0.5]]))
        self.assertEqual(float(sequence[0, 2, 0]), 0.0)  # no target-label self leakage

    def test_twohop_st_has_binary_forward_and_known_nonzero_proxy_gradient(self):
        data = view([[1.], [2.]], [(0, 1)], [1, 0])
        weight = torch.tensor(1.0, requires_grad=True)
        generated = torch.tensor([[7.]], requires_grad=True)
        graph = attacked(data, generated, [(1, 2)], weight)
        sequence = updated_sequences(data, graph, [0], [1], [2])
        self.assertAlmostEqual(float(sequence[0, 6, 0]), 4.0)
        gw, gx = torch.autograd.grad(sequence[0, 6, 0], (weight, generated))
        # Hard mean=(1+7)/2; dmean/dreach=1.5 and rho'(1)=1/4.
        torch.testing.assert_close(gw, torch.tensor(0.375))
        torch.testing.assert_close(gx, torch.tensor([[0.5]]))

    def test_existing_twohop_members_are_not_counted_again_and_paths_are_deduplicated(self):
        data = view([[1.], [2.], [3.], [9.]], [(0, 1), (0, 2), (1, 3), (2, 3)], [2, 0, 0, 1])
        self.assertAlmostEqual(float(data.d[0, 5, 0]), 1.0)  # two paths to fraud node3, one member
        graph = attacked(data, [[5.]], [(0, 4), (4, 3)])
        actual = updated_sequences(data, graph, [0], [1, 2, 3], [4])
        torch.testing.assert_close(actual[0, 4:], data.sequences[0, 4:])
        torch.testing.assert_close(actual, full_hard_oracle(data, graph, [0]))

    def test_zero_attack_is_cache_identity_even_with_no_candidates(self):
        data = view([[1.], [3.], [8.]], [(0, 1)], [1, 0, 2])
        graph = attacked(data, [])
        actual = updated_sequences(data, graph, [2, 0], None, [])
        torch.testing.assert_close(actual, data.sequences[[2, 0]], rtol=0, atol=0)

    def test_fit_mask_view_is_authoritative_and_graph_labels_masks_are_not_read(self):
        data = view([[1.], [3.], [8.]], [(0, 1)], [2, 0, 2])
        graph = attacked(data, [[6.]], [(0, 3), (3, 2)])
        first = updated_sequences(data, graph, [0], [1, 2], [3])
        graph.ndata["train_neg"] = torch.zeros(4)
        graph.ndata["train_pos"] = torch.ones(4)
        graph.ndata["train_masked"] = torch.zeros(4)
        graph.ndata["y"] = torch.arange(4)
        second = updated_sequences(data, graph, [0], [1, 2], [3])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(second, full_hard_oracle(data, graph, [0]))

    def test_candidate_permutation_and_eval_weight_independence(self):
        data = view([[1.], [2.], [5.], [9.]], [(0, 1), (1, 2)], [2, 0, 1, 2])
        graph = attacked(data, [[7.]], [(0, 4), (4, 3)])
        first = updated_sequences(data, graph, [0], [1, 2, 3], [4])
        second = updated_sequences(data, graph, [0], [3, 2, 1], [4])
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        graph.edata["weight"] = graph.edata["weight"] * 0.5
        with self.assertRaises(ValueError):
            updated_sequences(data, graph, [0], [1, 2, 3], [4])
        hard_eval = updated_sequences(data, graph, [0], [1, 2, 3], [4], training=False)
        torch.testing.assert_close(hard_eval, first, rtol=0, atol=0)

    def test_input_and_rng_immutability_and_chunk_invariance(self):
        data = view([[1., 4.], [3., 2.], [8., 5.]], [(0, 1)], [2, 0, 1])
        graph = attacked(data, [[6., -1.]], [(0, 3), (3, 2)])
        before_x, before_w = graph.ndata["x"].clone(), graph.edata["weight"].clone()
        before_sum, before_degree = data.feat_sum.clone(), data.d.clone()
        before_src, before_dst = [v.clone() for v in graph.edges(order="eid")]
        py, np_state, cpu = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
        first = updated_sequences(data, graph, [0], [1, 2], [3], message_chunk=1)
        second = updated_sequences(data, graph, [0], [1, 2], [3], message_chunk=1024)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for left, right in ((graph.ndata["x"], before_x), (graph.edata["weight"], before_w),
                            (data.feat_sum, before_sum), (data.d, before_degree),
                            (graph.edges(order="eid")[0], before_src), (graph.edges(order="eid")[1], before_dst)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(random.getstate(), py)
        now = np.random.get_state()
        self.assertEqual(now[0], np_state[0]); self.assertTrue(np.array_equal(now[1], np_state[1]))
        self.assertEqual(now[2:], np_state[2:])
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))

    def test_memory_guard_and_invalid_roles_fail_before_dense_selected_work(self):
        data = view([[1.], [3.]], [(0, 1)], [2, 0])
        graph = attacked(data, [[6.]], [(0, 2)])
        reservation = estimate_workspace_bytes(1, 3, 1)
        self.assertGreater(reservation, 0)
        with self.assertRaises(MemoryError):
            updated_sequences(data, graph, [0], [1], [2], max_working_bytes=reservation-1)
        with self.assertRaises(MemoryError):
            updated_sequences(data, graph, [0], [1], [2], max_selected_nodes=2)
        with self.assertRaises(ValueError):
            updated_sequences(data, graph, [0], [0], [2])
        with self.assertRaises(ValueError):
            updated_sequences(data, graph, [0], [1], [1])


if __name__ == "__main__":
    unittest.main()
