"""Synthetic CPU checks only; no real graph, dataset or checkpoint is loaded."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.downstream_cooperation.model import (
        DownstreamCandidateScorer, straight_through_mask,
    )


@unittest.skipIf(torch is None, "PyTorch is unavailable; these checks were not executed.")
class DownstreamModelTests(unittest.TestCase):
    def fixture(self, requires_grad=False):
        torch.manual_seed(117)
        values = [torch.randn(2, 3), torch.randn(3, 3),
                  torch.tensor([[1., 0., 0.], [0., 1., 1.]]),
                  torch.randn(4, 3), torch.randn(4, 2)]
        return [value.requires_grad_(requires_grad) for value in values]

    def test_known_answer_changed_upstream_decision_changes_scores_and_gradient(self):
        model = DownstreamCandidateScorer(2, 2, 2)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.target_projection.weight.copy_(torch.eye(2))
            model.candidate_projection.weight.copy_(torch.eye(2))
        auxiliary = torch.zeros(1, 2)
        targets = torch.tensor([[1., 0.], [0., 2.]])
        candidates = torch.eye(2)
        context = torch.zeros(2, 2)
        first = model(auxiliary, targets, torch.tensor([[1., 0.]]), candidates, context)
        second = model(auxiliary, targets, torch.tensor([[0., 1.]]), candidates, context)
        self.assertGreater(float(first[0, 0]), 0)
        self.assertEqual(float(first[0, 1]), 0)
        self.assertEqual(float(second[0, 0]), 0)
        self.assertGreater(float(second[0, 1]), 0)
        self.assertFalse(torch.equal(first, second))
        first[0, 0].backward()
        first_gradient = model.target_projection.weight.grad.clone()
        model.zero_grad()
        second[0, 1].backward()
        second_gradient = model.target_projection.weight.grad.clone()
        self.assertFalse(torch.equal(first_gradient, second_gradient))

    def test_count_conditioning_changes_even_when_target_feature_mean_is_equal(self):
        model = DownstreamCandidateScorer(2, 2, 2)
        targets = torch.ones(2, 2)
        first = model.conditioning_inputs(torch.zeros(1, 2), targets, torch.tensor([[1., 0.]]))
        second = model.conditioning_inputs(torch.zeros(1, 2), targets, torch.tensor([[1., 1.]]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertFalse(torch.equal(first[2], second[2]))

    def test_only_downstream_parameters_receive_gradients_and_optimizer_updates(self):
        model = DownstreamCandidateScorer(3, 2, 5)
        inputs = self.fixture(requires_grad=True)
        snapshots = [value.detach().clone() for value in inputs]
        parameter_before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scores = model(*inputs)
        coefficients = torch.arange(scores.numel(), dtype=scores.dtype).reshape_as(scores)
        (scores * coefficients).sum().backward()
        self.assertTrue(any(value.grad is not None and bool((value.grad != 0).any())
                            for value in model.parameters()))
        self.assertTrue(all(value.grad is None for value in inputs))
        optimizer.step()
        self.assertTrue(any(not torch.equal(value.detach(), parameter_before[name])
                            for name, value in model.named_parameters()))
        self.assertTrue(all(torch.equal(value.detach(), before)
                            for value, before in zip(inputs, snapshots)))
        parameter_ids = {id(value) for value in model.parameters()}
        self.assertEqual(parameter_ids, {id(value) for group in optimizer.param_groups
                                        for value in group['params']})

    def test_straight_through_forward_is_exact_binary_and_backward_is_finite(self):
        scores = torch.tensor([[0.1, 0.2, 0.5], [0.7, -0.4, 0.0]], requires_grad=True)
        hard = torch.tensor([[0., 0., 1.], [1., 0., 0.]])
        weights = straight_through_mask(scores, hard, 2)
        self.assertTrue(torch.equal(weights.detach(), hard))
        (weights * torch.arange(6.).reshape(2, 3)).sum().backward()
        self.assertTrue(bool(torch.isfinite(scores.grad).all()))
        self.assertGreater(float(scores.grad.abs().max()), 0)
        self.assertEqual(int(weights.detach().sum()), 2)

    def test_zero_and_full_quotas_have_no_trainable_selection(self):
        for quota in (0, 6):
            scores = torch.randn(2, 3, requires_grad=True)
            hard = torch.zeros_like(scores) if quota == 0 else torch.ones_like(scores)
            weights = straight_through_mask(scores, hard, quota)
            self.assertTrue(torch.equal(weights.detach(), hard))
            weights.sum().backward()
            self.assertTrue(torch.equal(scores.grad, torch.zeros_like(scores)))

    def test_model_construction_and_outputs_reproduce_with_same_seed(self):
        inputs = self.fixture()
        torch.manual_seed(911)
        first = DownstreamCandidateScorer(3, 2, 5)
        torch.manual_seed(911)
        second = DownstreamCandidateScorer(3, 2, 5)
        self.assertTrue(torch.equal(first(*inputs), second(*inputs)))

    def test_empty_candidates_are_valid_without_learning_a_connection(self):
        model = DownstreamCandidateScorer(3, 2, 5)
        inputs = self.fixture()
        inputs[3] = torch.empty(0, 3)
        inputs[4] = torch.empty(0, 2)
        scores = model(*inputs)
        self.assertEqual(tuple(scores.shape), (2, 0))
        self.assertEqual(tuple(straight_through_mask(scores, torch.empty(2, 0), 0).shape), (2, 0))

    def test_invalid_upstream_or_selection_inputs_are_rejected(self):
        model = DownstreamCandidateScorer(3, 2, 5)
        inputs = self.fixture()
        inputs[2] = torch.tensor([[0., 0., 0.], [0., 1., 1.]])
        with self.assertRaisesRegex(ValueError, "target attachment"):
            model(*inputs)
        inputs = self.fixture()
        inputs[2][0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            model(*inputs)
        for quota, hard, temperature in ((1, torch.zeros(2, 3), 1),
                                         (0, torch.zeros(2, 3), 0),
                                         (7, torch.zeros(2, 3), 1)):
            with self.assertRaises(ValueError):
                straight_through_mask(torch.zeros(2, 3), hard, quota, temperature)


if __name__ == '__main__':
    unittest.main()
