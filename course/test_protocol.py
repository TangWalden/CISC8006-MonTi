import unittest
import pathlib
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import dgl
import numpy as np
import scipy.sparse as sp
import torch

from models.attack_model import MonTi
from modules.experiment_handler import ExperimentHandler
from course.runner import AuditedExperiment
from utils.utils import get_target_neighbor_size, misclassification_rate, set_seed


class ProtocolTests(unittest.TestCase):
    def test_weighted_metric_known_answer(self):
        # Group sizes 1 and 3: one wrong occurrence, NOT mean([1, 0]).
        self.assertAlmostEqual(misclassification_rate(np.ones(4), np.array([0, 1, 1, 1])), .25)

    def test_configuration_reaches_actual_selection_branch(self):
        graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 0])))
        for random_mode in (False, True):
            experiment = ExperimentHandler.__new__(ExperimentHandler)
            experiment.data = SimpleNamespace(graph=graph)
            experiment.config = dict(k_exp=1, masking=False, random_candidates=random_mode)
            experiment.model = Mock(return_value=(None, None, None))
            experiment.get_perturbed_graph(torch.tensor([0]), 1, 1)
            self.assertEqual(experiment.model.call_args.args[4], not random_mode)

    def make_model_graph(self, device):
        graph = dgl.graph(([0, 0, 1, 1, 2, 3, 4, 5], [2, 3, 4, 5, 0, 0, 1, 1]), num_nodes=6)
        graph = dgl.add_self_loop(graph).to(device)
        graph.ndata['x'] = torch.rand(6, 4, device=device)
        graph.ndata['h'] = torch.rand(6, 8, device=device)
        graph.ndata['degree'] = (graph.in_degrees() - 1).float()
        model = MonTi(4, torch.zeros(4, device=device), torch.ones(4, device=device),
                      False, 1, emb_dim=8, ff_dim=16, max_candidates=2,
                      num_layers=1, num_heads=2, dropout=0).to(device)
        return model, graph

    def test_selector_dispatch_and_injection_constraints(self):
        for device in (['cpu', 'cuda:0'] if torch.cuda.is_available() else ['cpu']):
            for learned in (False, True):
                set_seed(1729)
                model, graph = self.make_model_graph(device)
                calls = []
                handle = model.candidate_selector.register_forward_hook(lambda *args: calls.append(1))
                model.train()
                targets = torch.tensor([0, 1], device=device)
                attributes, adjacency, candidates = model(graph, targets, 2, 4, learned)
                handle.remove()
                self.assertEqual(len(calls), int(learned))
                self.assertEqual(len(set(candidates.tolist())), 2)
                self.assertTrue(set(candidates.tolist()).isdisjoint({0, 1}))
                self.assertEqual(tuple(attributes.shape), (2, 4))
                self.assertTrue(torch.all((attributes >= 0) & (attributes <= 1)).item())
                self.assertEqual(int(adjacency.detach().sum().item()), 4)
                self.assertTrue(torch.all(adjacency[:, :2].detach().sum(1) >= 1).item())
                self.assertEqual(float(adjacency[0, 4]), 0)
                self.assertEqual(float(adjacency[1, 5]), 0)
                (attributes.sum() + adjacency.sum()).backward()
                self.assertTrue(torch.isfinite(model.linear_feat.weight.grad).all().item())
                selector_gradients = [p.grad for p in model.candidate_selector.parameters()]
                self.assertEqual(any(g is not None for g in selector_gradients), learned)

    def test_documented_upstream_neighbor_count_difference(self):
        # Two targets share node 2: unique union has 3 nodes, source counts 4 incidences.
        graph = dgl.add_self_loop(dgl.graph(([0, 1, 2, 2], [2, 2, 0, 1]), num_nodes=3))
        assignments = sp.csr_matrix([[1, 1, 0]])
        self.assertEqual(int(get_target_neighbor_size(graph, assignments)[0]), 4)
        unique_neighbors = set(graph.successors(0).tolist()) | set(graph.successors(1).tolist())
        self.assertEqual(len(unique_neighbors), 3)

    def test_audited_evaluator_preserves_weighted_denominator(self):
        graph = dgl.graph(([], []), num_nodes=4)
        graph.ndata['y'] = torch.ones(4, dtype=torch.long)
        assignment = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 1, 1]])
        experiment = AuditedExperiment.__new__(AuditedExperiment)
        experiment.data = SimpleNamespace(graph=graph, target_assignment=assignment,
                                          get_validation_set=lambda: np.array([0, 1]))
        experiment.phase = 'pilot'
        experiment.evaluation_number = 0
        experiment.surrogate_type = experiment.victim_type = 'GCN'
        logits = torch.tensor([[2., 0.], [0., 2.], [0., 2.], [0., 2.]])
        experiment.get_logits = lambda graph, node_ids, *args, **kwargs: logits[node_ids]
        with tempfile.TemporaryDirectory() as directory:
            experiment.evidence = pathlib.Path(directory)
            result = experiment.check_performance(attack=False)
            self.assertAlmostEqual(result['misclf_rate'], .25)
            with self.assertRaisesRegex(RuntimeError, 'Pilot may not inspect'):
                experiment.check_performance(set_type='test')

    def test_failed_generation_cannot_silently_drop_group(self):
        graph = dgl.add_self_loop(dgl.graph(([0, 1], [1, 0])))
        experiment = AuditedExperiment.__new__(AuditedExperiment)
        experiment.data = SimpleNamespace(graph=graph)
        experiment.config = dict(k_exp=1, masking=False, random_candidates=False)
        experiment.model = Mock(return_value=(None, None, None))
        with self.assertRaisesRegex(RuntimeError, 'silently remove'):
            experiment.get_perturbed_graph(torch.tensor([0]), 1, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
