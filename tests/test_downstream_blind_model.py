"""Synthetic action-blind scorer checks; no real data or checkpoint reads."""

import inspect
import random
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from experiments.downstream_cooperation.model import DownstreamCandidateScorer
    from experiments.downstream_blind.model import ActionBlindCandidateScorer


@unittest.skipIf(torch is None, 'PyTorch unavailable; these checks were not executed.')
class DownstreamBlindModelTests(unittest.TestCase):
    @staticmethod
    def inputs(requires_grad=False):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(155)
        values = [torch.randn(2, 3, generator=generator),
                  torch.randn(4, 3, generator=generator),
                  torch.randn(5, 3, generator=generator),
                  torch.randn(5, 2, generator=generator)]
        return [value.requires_grad_(requires_grad) for value in values]

    @staticmethod
    def forward(model, inputs, quota=4):
        return model(inputs[0], inputs[1], quota, inputs[2], inputs[3])

    def test_exact_architecture_and_initial_state_parity(self):
        torch.manual_seed(918)
        visible = DownstreamCandidateScorer()
        after_visible = torch.random.get_rng_state().clone()
        torch.manual_seed(918)
        blind = ActionBlindCandidateScorer()
        self.assertEqual(sum(value.numel() for value in blind.parameters()), 151872)
        self.assertEqual(set(visible.state_dict()), set(blind.state_dict()))
        self.assertTrue(all(torch.equal(value, blind.state_dict()[name])
                            for name, value in visible.state_dict().items()))
        self.assertTrue(torch.equal(after_visible, torch.random.get_rng_state()))

    def test_known_answer_whole_list_mean_and_average_quota_count(self):
        model = ActionBlindCandidateScorer(2, 2, 2)
        auxiliary = torch.tensor([[7., 8.], [9., 10.]])
        targets = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        result = model.conditioning_inputs(auxiliary, targets, 3)
        self.assertTrue(torch.equal(result[0], auxiliary))
        self.assertTrue(torch.equal(result[1], torch.tensor([[3., 4.], [3., 4.]])))
        self.assertTrue(torch.equal(result[2], torch.full((2, 1), 1.5).log1p()))
        self.assertTrue(all(not value.requires_grad for value in result))

    def test_actual_attachment_matrix_is_not_a_policy_argument(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        names = tuple(inspect.signature(model.forward).parameters)
        self.assertEqual(names, ('x_aux', 'x_targets', 'target_quota',
                                 'x_candidates', 'h_candidates'))
        inputs = self.inputs()
        attachment = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]])
        with self.assertRaisesRegex(ValueError, 'integer target quota'):
            self.forward(model, inputs, attachment)
        with self.assertRaises(TypeError):
            model(x_aux=inputs[0], x_targets=inputs[1], adj_targets=attachment,
                  x_candidates=inputs[2], h_candidates=inputs[3])

    def test_target_list_permutation_does_not_change_blind_scores(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs()
        first = self.forward(model, inputs)
        permuted = [inputs[0], inputs[1][torch.tensor([2, 0, 3, 1])], inputs[2], inputs[3]]
        second = self.forward(model, permuted)
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-5))

    def test_full_target_attachment_is_a_matched_information_boundary(self):
        torch.manual_seed(512)
        visible = DownstreamCandidateScorer(3, 2, 5)
        torch.manual_seed(512)
        blind = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs()
        attachment = torch.ones(len(inputs[0]), len(inputs[1]))
        visible_result = visible(inputs[0], inputs[1], attachment, inputs[2], inputs[3])
        blind_result = self.forward(blind, inputs, int(attachment.numel()))
        self.assertTrue(torch.equal(visible_result, blind_result))

    def test_realistic_full_target_summary_scores_and_parameter_gradients_are_exact(self):
        generator = torch.Generator(device='cpu')
        generator.manual_seed(701)
        auxiliary = torch.randn(122, 768, generator=generator) * 0.1
        targets = torch.randn(7, 768, generator=generator) * 0.1
        candidates = torch.randn(11, 768, generator=generator) * 0.1
        context = torch.randn(11, 64, generator=generator) * 0.1
        attachment = torch.ones(122, 7)
        torch.manual_seed(702)
        visible = DownstreamCandidateScorer()
        torch.manual_seed(702)
        blind = ActionBlindCandidateScorer()
        visible_summary = visible.conditioning_inputs(auxiliary, targets, attachment)
        blind_summary = blind.conditioning_inputs(auxiliary, targets, 122 * 7)
        self.assertTrue(all(torch.equal(first, second)
                            for first, second in zip(visible_summary, blind_summary)))
        visible_scores = visible(auxiliary, targets, attachment, candidates, context)
        blind_scores = blind(auxiliary, targets, 122 * 7, candidates, context)
        self.assertTrue(torch.equal(visible_scores, blind_scores))
        coefficients = torch.randn(122, 11, generator=generator)
        (visible_scores * coefficients).sum().backward()
        (blind_scores * coefficients).sum().backward()
        blind_parameters = dict(blind.named_parameters())
        for name, parameter in visible.named_parameters():
            with self.subTest(parameter=name):
                counterpart = blind_parameters[name]
                self.assertIsNotNone(parameter.grad)
                self.assertIsNotNone(counterpart.grad)
                self.assertTrue(torch.equal(parameter.grad, counterpart.grad))

    def test_target_and_count_projections_receive_finite_nonzero_gradients(self):
        torch.manual_seed(613)
        model = ActionBlindCandidateScorer(3, 2, 5)
        scores = self.forward(model, self.inputs())
        coefficients = torch.arange(1, scores.numel() + 1, dtype=scores.dtype).reshape_as(scores)
        (scores * coefficients).sum().backward()
        for projection in (model.target_projection, model.count_projection):
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(bool(torch.isfinite(projection.weight.grad).all()))
            self.assertGreater(float(projection.weight.grad.abs().max()), 0)

    def test_only_model_parameters_update_and_upstream_inputs_are_detached(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs(requires_grad=True)
        snapshots = [value.detach().clone() for value in inputs]
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scores = self.forward(model, inputs)
        (scores * torch.arange(1, 11.).reshape(2, 5)).sum().backward()
        self.assertTrue(all(value.grad is None for value in inputs))
        optimizer.step()
        self.assertTrue(any(not torch.equal(value.detach(), before[name])
                            for name, value in model.named_parameters()))
        self.assertTrue(all(torch.equal(value.detach(), previous)
                            for value, previous in zip(inputs, snapshots)))
        self.assertEqual({id(value) for value in model.parameters()},
                         {id(value) for group in optimizer.param_groups for value in group['params']})

    def test_forward_does_not_mutate_inputs_or_rng(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs()
        snapshots = [value.clone() for value in inputs]
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        first = self.forward(model, inputs)
        second = self.forward(model, inputs)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(all(torch.equal(value, previous)
                            for value, previous in zip(inputs, snapshots)))

    def test_illegal_quotas_are_rejected(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs()
        for quota in (True, 0, 1, 9, 4.0, torch.tensor(4)):
            with self.subTest(quota=quota), self.assertRaisesRegex(ValueError, 'integer target quota'):
                self.forward(model, inputs, quota)
        for quota in (2, 8):
            self.assertTrue(bool(torch.isfinite(self.forward(model, inputs, quota)).all()))

    def test_nonfinite_features_and_empty_candidates(self):
        model = ActionBlindCandidateScorer(3, 2, 5)
        inputs = self.inputs()
        inputs[1][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.forward(model, inputs)
        inputs = self.inputs()
        inputs[2], inputs[3] = torch.empty(0, 3), torch.empty(0, 2)
        result = self.forward(model, inputs)
        self.assertEqual(tuple(result.shape), (2, 0))
        self.assertTrue(bool(torch.isfinite(result).all()))


if __name__ == '__main__':
    unittest.main()
