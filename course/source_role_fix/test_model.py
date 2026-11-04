"""CPU synthetic source-role checks; no real data/checkpoint/API/victim calls.

Run only after the new execution source and test registration are frozen.
"""
import types
import unittest
from unittest import mock

import dgl
import torch

from models.attack_model import MonTi
from modules.experiment_handler import ExperimentHandler
from course.source_role_fix.experiment import SourceExperiment
from course.source_role_fix.model import SourceRoleMonTi
from utils.utils import gumbel_softmax


DEVICE = types.SimpleNamespace(device=torch.device('cpu'))


def arguments(**changes):
    result = dict(feat_dim=4, min_feat=torch.zeros(4), max_feat=torch.ones(4),
                  discrete=False, feat_budget=2, emb_dim=4, ff_dim=8,
                  max_candidates=8, num_layers=1, num_heads=2, dropout=0.0,
                  initial_temp=2, initial_eps=1)
    result.update(changes)
    return result


def initialized(model_type=SourceRoleMonTi, mode='source_fixed', **changes):
    torch.manual_seed(73)
    args = arguments(**changes)
    if model_type is SourceRoleMonTi:
        args['source_mode'] = mode
    return model_type(**args)


class SourceRoleChecks(unittest.TestCase):
    def test_explicit_mode_contract(self):
        with self.assertRaises(ValueError):
            SourceRoleMonTi(**arguments(), source_mode='unknown')

    def test_parameter_initialization_and_rng_are_identical(self):
        original = initialized(MonTi)
        original_rng = torch.random.get_rng_state().clone()
        for mode in ('legacy', 'source_fixed'):
            variant = initialized(mode=mode)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
            self.assertEqual(list(original.state_dict()), list(variant.state_dict()))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, variant.state_dict()[name]), name)

    def test_experiment_constructor_matches_original_without_extra_initialization(self):
        config = dict(emb_dim=4, ff_dim=8, max_candidates=8, num_layers=1, num_heads=2,
                      initial_temp=2, min_temp=.01, temp_decay_rate=.63,
                      initial_eps=1, min_eps=.01, eps_decay_rate=.63, dropout=0,
                      shared_parameters=False, without_candidates=False,
                      pos_encoding=True, ada_budget=True)
        holder = SourceExperiment.__new__(SourceExperiment)
        holder.data = types.SimpleNamespace(features=torch.arange(24).reshape(6, 4).float()/24,
                                             discrete=False)
        holder.config, holder.device = config, torch.device('cpu')
        torch.manual_seed(81)
        original = ExperimentHandler.initialize_attack_model(holder)
        original_rng = torch.random.get_rng_state().clone()
        for mode in ('legacy', 'source_fixed'):
            holder.source_arm = mode
            torch.manual_seed(81)
            variant = holder.initialize_attack_model()
            self.assertEqual(variant.source_mode, mode)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), original_rng))
            for name, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, variant.state_dict()[name]), name)

    def test_legacy_edge_forward_matches_original(self):
        for training in (False, True):
            for adaptive in (False, True):
                with self.subTest(training=training, adaptive=adaptive):
                    original = initialized(MonTi, ada_budget=adaptive)
                    legacy = initialized(mode='legacy', ada_budget=adaptive)
                    original.train(training);legacy.train(training)
                    torch.manual_seed(811)
                    z = torch.randn(5, 4)
                    torch.manual_seed(65)
                    expected = original.edge_generation(z, DEVICE, 2, 1, 2, 4)
                    expected_rng = torch.random.get_rng_state().clone()
                    torch.manual_seed(65)
                    actual = legacy.edge_generation(z, DEVICE, 2, 1, 2, 4)
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))

    def test_legacy_edge_gradients_match_original(self):
        original, legacy = initialized(MonTi), initialized(mode='legacy')
        original.train();legacy.train()
        torch.manual_seed(17)
        initial = torch.randn(5, 4)
        weights = torch.arange(10).reshape(2, 5).float()/7
        results = []
        for model in (original, legacy):
            z = initial.clone().requires_grad_(True)
            torch.manual_seed(89)
            adjacency = model.edge_generation(z, DEVICE, 2, 1, 2, 4)
            (adjacency*weights).sum().backward()
            results.append((z.grad, model.linear_edge.weight.grad, model.linear_edge.bias.grad))
        for left, right in zip(*results):
            self.assertTrue(torch.equal(left, right))

    def test_legacy_complete_forward_matches_original(self):
        graph = dgl.graph(([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]), num_nodes=6)
        graph = dgl.add_self_loop(graph)
        graph.ndata['x'] = torch.arange(24).reshape(6, 4).float()/24
        graph.ndata['h'] = torch.arange(24).reshape(6, 4).float()/12
        graph.ndata['degree'] = graph.in_degrees().float()
        targets = torch.tensor([0, 1])
        for training in (False, True):
            with self.subTest(training=training):
                original, legacy = initialized(MonTi), initialized(mode='legacy')
                original.train(training);legacy.train(training)
                torch.manual_seed(59)
                expected = original(graph.clone(), targets, 2, 5)
                expected_rng = torch.random.get_rng_state().clone()
                torch.manual_seed(59)
                actual = legacy(graph.clone(), targets, 2, 5)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left, right))
                self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))

    def test_fixed_T2_C1_B2_known_raw_scores(self):
        model = initialized()
        model.eval()
        with torch.no_grad():
            model.linear_edge.weight.copy_(torch.eye(4));model.linear_edge.bias.zero_()
        z = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0],
                          [.8, .6, 0, 0], [.6, 0, .8, 0]])
        captured = []
        def observe(scores, *args, **kwargs):
            captured.append(scores.detach().clone())
            return gumbel_softmax(scores, *args, **kwargs)
        with mock.patch('course.source_role_fix.model.gumbel_softmax', side_effect=observe):
            model.edge_generation(z, DEVICE, 2, 1, 2, 4)
        # Row-major legal pairs: I0->{T0,T1,C0}; I1->{T0,T1,C0,I0}.
        expected = torch.tensor([.8, .6, 0, .6, 0, .8, .48])
        torch.testing.assert_close(captured[0], expected, atol=1e-7, rtol=1e-6)

    def test_fixed_single_injection_known_raw_scores(self):
        model = initialized();model.eval()
        with torch.no_grad():
            model.linear_edge.weight.copy_(torch.eye(4));model.linear_edge.bias.zero_()
        z = torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0], [.8, .6, 0, 0]])
        captured = []
        def observe(scores, *args, **kwargs):
            captured.append(scores.detach().clone())
            return gumbel_softmax(scores, *args, **kwargs)
        with mock.patch('course.source_role_fix.model.gumbel_softmax', side_effect=observe):
            result = model.edge_generation(z, DEVICE, 2, 1, 1, 1)
        torch.testing.assert_close(captured[0], torch.tensor([.8, .6, 0]), atol=1e-7, rtol=1e-6)
        self.assertTrue(torch.equal(result, torch.tensor([[1., 0, 0, 0]])))

    def test_last_injected_representation_gets_score_gradient(self):
        torch.manual_seed(41)
        z0 = torch.randn(5, 4)
        weights = torch.arange(10).reshape(2, 5).float()/9
        norms = {}
        for mode in ('legacy', 'source_fixed'):
            model = initialized(mode=mode);model.eval()
            z = z0.clone().requires_grad_(True)
            adjacency = model.edge_generation(z, DEVICE, 2, 1, 2, 4)
            (adjacency*weights).sum().backward()
            self.assertTrue(bool(torch.isfinite(z.grad).all()))
            self.assertGreater(float(model.linear_edge.weight.grad.norm()), 0)
            norms[mode] = float(z.grad[-1].norm())
        self.assertEqual(norms['legacy'], 0)
        self.assertGreater(norms['source_fixed'], 0)

    def test_adaptive_cardinality_target_incidence_and_original_legal_space(self):
        for t, c, b, budget in ((2, 1, 1, 1), (2, 1, 1, 3), (2, 1, 2, 2),
                                (2, 1, 2, 5), (2, 1, 2, 7), (1, 0, 3, 6)):
            with self.subTest(t=t, c=c, b=b, budget=budget):
                model = initialized();model.eval()
                torch.manual_seed(331)
                adjacency = model.edge_generation(torch.randn(t+c+b, 4), DEVICE, t, c, b, budget)
                self.assertTrue(bool(((adjacency == 0) | (adjacency == 1)).all()))
                self.assertEqual(int(adjacency.sum()), budget)
                self.assertTrue(bool((adjacency[:, :t].sum(1) >= 1).all()))
                # Explicit original legal destinations for row i are 0..T+C+i-1.
                for row in range(b):
                    self.assertEqual(int(adjacency[row, t+c+row:].sum()), 0)
        # A full legal edge budget must retain the original earlier-injection edge.
        model = initialized();model.eval()
        adjacency = model.edge_generation(torch.randn(5, 4), DEVICE, 2, 1, 2, 7)
        self.assertEqual(float(adjacency[1, 3]), 1)

    def test_nonadaptive_original_degree_allocation(self):
        model = initialized(ada_budget=False);model.eval()
        adjacency = model.edge_generation(torch.randn(5, 4), DEVICE, 2, 1, 2, 4)
        self.assertTrue(torch.equal(adjacency.sum(1), torch.tensor([2., 2.])))
        self.assertEqual(int(adjacency[0, 3:].sum()), 0)
        self.assertEqual(int(adjacency[1, 4:].sum()), 0)

    def test_random_edges_match_original_and_bypass_source_projection(self):
        original = initialized(MonTi)
        z = torch.arange(20).reshape(5, 4).float()/20
        torch.manual_seed(765)
        expected = original.edge_generation(z, DEVICE, 2, 1, 2, 4, random_edge=True)
        expected_rng = torch.random.get_rng_state().clone()
        for mode in ('legacy', 'source_fixed'):
            model = initialized(mode=mode)
            with mock.patch.object(model.linear_edge, 'forward', side_effect=AssertionError('unused')):
                torch.manual_seed(765)
                actual = model.edge_generation(z, DEVICE, 2, 1, 2, 4, random_edge=True)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
