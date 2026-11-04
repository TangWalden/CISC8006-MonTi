"""CPU synthetic fixed-topology feature tests; no real data or classifiers.

Run only after the source and execution test registration have been frozen.
"""
import types
import unittest

import dgl
import numpy as np
import scipy.sparse as sp
import torch

from course.feature_polish.affine import build_affine, features_from_unit_box
from course.dual_surrogate.sequence_adapter import updated_sequences


def undirected(pairs):
    return sorted(set((a, b) for left, right in pairs for a, b in ((left, right), (right, left)) if a != b))


def means(features, arcs, masks, targets):
    """Independent Python-set oracle for both complete hard neighborhoods."""
    neighbors = [set() for _ in range(len(features))]
    for left, right in arcs:
        if left != right:
            neighbors[left].add(right)
    sequences = features.new_zeros((len(targets), 7, features.shape[1]))
    sums, degrees = torch.zeros_like(sequences), features.new_zeros((len(targets), 7, 1))
    for row, target in enumerate(targets):
        sequences[row, 0] = sums[row, 0] = features[target]
        degrees[row, 0, 0] = 1
        one = neighbors[target]-{target}
        two = set()
        for node in one:
            two.update(neighbors[node])
        two.discard(target)
        for hop, members in enumerate((one, two)):
            for group, name in enumerate(('train_neg', 'train_pos', 'train_masked')):
                ids = sorted(node for node in members if masks[name][node])
                token = 1+3*hop+group
                if group == 2:
                    ids.append(target)  # Self is masked unknown irrespective of its label scope.
                degrees[row, token, 0] = len(ids)
                if ids:
                    sums[row, token] = features[ids].sum(0)
                    sequences[row, token] = sums[row, token]/len(ids)
    return sequences, sums, degrees


def case(features, original_pairs, node_classes, targets, attack_features, added_pairs):
    features = torch.tensor(features, dtype=torch.float32)
    injection = torch.tensor(attack_features, dtype=torch.float32)
    n = len(features)
    original_arcs = undirected(original_pairs)
    arcs = undirected(list(original_pairs)+list(added_pairs))
    masks = {name:torch.tensor([int(value == group) for value in node_classes], dtype=torch.float32)
             for group, name in enumerate(('train_neg', 'train_pos', 'train_masked'))}
    original_seq, sums, degrees = means(features, original_arcs, masks, list(range(n)))
    src, dst = zip(*original_arcs) if original_arcs else ([], [])
    adjacency = sp.csr_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
    two = (adjacency@adjacency).astype(bool).astype(np.int64)
    two.setdiag(0);two.eliminate_zeros()
    base = dgl.graph((list(src), list(dst)), num_nodes=n)
    base.ndata['x'] = features
    view = types.SimpleNamespace(graph=base, features=features, adj=adjacency, adj2=two,
                                 group_masks=masks, feat_sum=sums, d=degrees)
    joined = torch.cat((features, injection))
    src, dst = zip(*arcs) if arcs else ([], [])
    graph = dgl.add_self_loop(dgl.graph((list(src), list(dst)), num_nodes=len(joined)))
    graph.ndata['x'] = joined
    candidates = [node for node in range(n) if node not in targets]
    atk = list(range(n, len(joined)))
    augmented_masks = {name:torch.cat((values, torch.ones(len(atk)) if name == 'train_masked'
                                      else torch.zeros(len(atk)))) for name, values in masks.items()}
    seq, _, _ = means(joined, arcs, augmented_masks, targets)
    return view, graph, torch.tensor(targets), torch.tensor(candidates, dtype=torch.long), torch.tensor(atk), injection, arcs, augmented_masks, seq


class AffineChecks(unittest.TestCase):
    def single(self):
        return case([[1, 2], [3, 4]], [(0, 1)], [0, 2], [0], [[5, 6]], [(0, 2)])

    def multiple(self):
        # I1 has duplicate two-hop paths from target0 through original1 and I0.
        # Original2 becomes newly reachable through I1; its contribution stays
        # in the baseline while injected features change.
        return case([[1, 2], [3, 4], [7, 9]], [(0, 1)], [1, 0, 2], [0],
                    [[5, 6], [8, 10]], [(0, 3), (0, 4), (1, 4), (3, 4), (2, 4)])

    def test_single_known_coefficients_and_masked_self(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.single()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        expected = torch.zeros(1, 7, 1);expected[0, 3, 0] = 1/3
        torch.testing.assert_close(operator.coefficients, expected, atol=0, rtol=0)
        self.assertTrue(torch.equal(operator.sequence(x0), baseline))
        self.assertEqual(operator.trace['minimum_unknown_denominators'], [3., 1.])

    def test_multiple_injections_prior_injection_edges_and_duplicate_paths(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.multiple()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        expected = torch.zeros(1, 7, 2)
        expected[0, 3] = 1/3
        expected[0, 6] = 1/4
        torch.testing.assert_close(operator.coefficients, expected, atol=0, rtol=0)
        self.assertEqual(operator.trace['injected_hop_memberships'], [2, 2])

    def test_modified_features_match_independent_hard_means(self):
        for record in (self.single(), self.multiple()):
            view, graph, targets, candidates, atk, x0, arcs, masks, baseline = record
            operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
            for delta in (-.35, .2):
                with self.subTest(injected=len(atk), delta=delta):
                    changed = x0+delta*torch.arange(1, x0.numel()+1).reshape_as(x0)
                    expected, _, _ = means(torch.cat((view.features, changed)), arcs, masks, targets.tolist())
                    torch.testing.assert_close(operator.sequence(changed), expected, atol=2e-6, rtol=2e-6)
                    self.assertTrue(torch.equal(operator.sequence(changed)[:, [0, 1, 2, 4, 5]],
                                                baseline[:, [0, 1, 2, 4, 5]]))

    def test_canonical_baseline_and_fresh_updates_match(self):
        for record in (self.single(), self.multiple()):
            view, graph, targets, candidates, atk, x0, _, _, expected_baseline = record
            operator = build_affine(view, graph, targets, candidates, atk, x0)
            torch.testing.assert_close(operator.base_sequence, expected_baseline, atol=1e-6, rtol=1e-6)
            changed = x0*.8+.3
            modified = graph.local_var()
            modified.ndata['x'] = torch.cat((view.features, changed))
            expected = updated_sequences(view, modified, targets, candidates, atk, training=False)
            torch.testing.assert_close(operator.sequence(changed), expected, atol=2e-6, rtol=2e-6)

    def test_unknown_center_and_two_target_order(self):
        record = case([[1, 2], [3, 5], [6, 8], [9, 10]], [(0, 1), (1, 2)],
                      [2, 0, 1, 2], [2, 0], [[4, 7]], [(2, 4), (1, 4)])
        view, graph, targets, candidates, atk, x0, arcs, masks, baseline = record
        operator = build_affine(view, graph, targets, candidates, atk, x0)
        changed = x0+torch.tensor([[.5, -.25]])
        expected, _, _ = means(torch.cat((view.features, changed)), arcs, masks, targets.tolist())
        torch.testing.assert_close(operator.sequence(changed), expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(operator.base_sequence, baseline, atol=2e-6, rtol=2e-6)

    def test_zero_relation_degree_still_has_masked_self_and_no_feature_effect(self):
        record = case([[1, 2], [3, 4]], [], [2, 0], [0, 1], [[5, 6]], [(1, 2)])
        view, graph, targets, candidates, atk, x0, _, _, baseline = record
        operator = build_affine(view, graph, targets, candidates, atk, x0)
        self.assertEqual(int(torch.count_nonzero(operator.coefficients[0])), 0)
        changed = x0+100
        self.assertTrue(torch.equal(operator.sequence(changed)[0], baseline[0]))
        self.assertTrue(torch.equal(operator.base_sequence[0, 3], view.features[0]))
        self.assertTrue(torch.equal(operator.base_sequence[0, 6], view.features[0]))

    def test_exact_linear_feature_gradient(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.multiple()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        x = x0.double().requires_grad_(True)
        weights = torch.arange(14, dtype=torch.float64).reshape(1, 7, 2)/13
        loss = (operator.sequence(x)*weights).sum()
        gradient, = torch.autograd.grad(loss, x)
        expected = torch.zeros_like(x)
        for node in range(len(atk)):
            for feature in range(x.shape[1]):
                for target in range(len(targets)):
                    for token in range(7):
                        expected[node, feature] += float(operator.coefficients[target, token, node])*weights[target, token, feature]
        torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)

    def test_nonlinear_toy_binary_loss_finite_differences(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.multiple()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        weights = torch.arange(14, dtype=torch.float64).reshape(1, 7, 2)/71
        def objective(x):
            margin = (torch.tanh(operator.sequence(x))*weights).sum((1, 2))
            return torch.nn.functional.softplus(-margin).mean()
        x = (x0.double()/2).requires_grad_(True)
        gradient, = torch.autograd.grad(objective(x), x)
        for row in range(len(atk)):
            for col in range(x.shape[1]):
                plus, minus = x.detach().clone(), x.detach().clone()
                plus[row, col] += 1e-5;minus[row, col] -= 1e-5
                measured = float((objective(plus)-objective(minus))/(2e-5))
                self.assertAlmostEqual(float(gradient[row, col]), measured, places=8)

    def test_unit_box_zero_range_and_frozen_dimension_gradient(self):
        unit = torch.tensor([[.25, .75, .4]], requires_grad=True)
        lower, upper = torch.tensor([2., -3, 4]), torch.tensor([2., 1, 9])
        features = features_from_unit_box(unit, lower, upper)
        torch.testing.assert_close(features, torch.tensor([[2., 0, 6.]]), atol=0, rtol=0)
        features.sum().backward()
        self.assertTrue(torch.equal(unit.grad, torch.tensor([[0., 4, 5]])))
        self.assertTrue(torch.equal(features_from_unit_box(torch.zeros_like(unit), lower, upper), lower.unsqueeze(0)))
        self.assertTrue(torch.equal(features_from_unit_box(torch.ones_like(unit), lower, upper), upper.unsqueeze(0)))

    def test_builder_does_not_mutate_inputs_or_graph(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.multiple()
        original_features = graph.ndata['x'].clone()
        original_adjacency = view.adj.copy()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        operator.sequence(x0+.25)
        self.assertTrue(torch.equal(graph.ndata['x'], original_features))
        self.assertEqual((view.adj != original_adjacency).nnz, 0)
        self.assertTrue(torch.equal(operator.input_features, x0))

    def test_invalid_tail_and_mismatched_baseline_features_rejected(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.single()
        with self.assertRaises(ValueError):
            build_affine(view, graph, targets, candidates, torch.tensor([0]), x0, base_sequence=baseline)
        with self.assertRaises(ValueError):
            build_affine(view, graph, targets, candidates, atk, x0+1, base_sequence=baseline)
        with self.assertRaises(ValueError):
            build_affine(view, graph, targets, targets, atk, x0, base_sequence=baseline)

    def test_shape_dtype_and_feature_box_bounds_rejected(self):
        view, graph, targets, candidates, atk, x0, _, _, baseline = self.single()
        operator = build_affine(view, graph, targets, candidates, atk, x0, base_sequence=baseline)
        with self.assertRaises(ValueError):
            operator.sequence(x0[:, :1])
        with self.assertRaises(ValueError):
            features_from_unit_box(torch.zeros(1, 2), torch.tensor([2., 0]), torch.tensor([1., 1]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
