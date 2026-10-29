import copy
import json
import pathlib
import tempfile
import unittest

import dgl
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import WeightedRandomSampler

from course.improvements.policies import attach_coverage, coverage_anchors, group_sampling_weights
from course.improvements.sampling import SamplingDataHandler
from models.attack_model import MonTi
from utils.utils import set_seed


class PolicyTests(unittest.TestCase):
    def fixture(self, device):
        # 0,1 are targets; candidate 2 reaches both, 3 only target 0, 4 only target 1.
        graph = dgl.graph(([0, 1, 0, 1], [2, 2, 3, 4]), num_nodes=10).to(device)
        graph.ndata[dgl.NID] = torch.tensor([100, 101, 30, 20, 10, 40, 50, 60, 70, 80], device=device)
        graph.ndata['x'] = torch.randn(10, 6, device=device)
        graph.ndata['h'] = torch.randn(10, 8, device=device)
        graph.ndata['degree'] = graph.in_degrees().float()
        graph.ndata['beta'] = torch.bincount(graph.edges()[1], minlength=10).float()
        model = MonTi(6, torch.zeros(6, device=device), torch.ones(6, device=device),
                      False, 1, emb_dim=8, ff_dim=16, max_candidates=4,
                      num_layers=1, num_heads=2, dropout=0).to(device)
        return graph, model, torch.tensor([0, 1], device=device)

    def test_coverage_and_global_tie_break(self):
        graph = dgl.graph(([0, 1, 0, 1], [2, 2, 3, 3]), num_nodes=5)
        graph.ndata[dgl.NID] = torch.tensor([0, 1, 200, 100, 300])
        anchors = coverage_anchors(graph, torch.tensor([0, 1]), torch.tensor([2, 3, 4]))
        self.assertEqual(anchors.tolist(), [3])
        # Isolated candidates yield no arbitrary anchors.
        self.assertEqual(coverage_anchors(graph, torch.tensor([0, 1]), torch.tensor([4])).numel(), 0)

    def test_selection_gradient_devices_and_controls(self):
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                set_seed(42)
                graph, baseline, targets = self.fixture(device)
                original_state = set(baseline.state_dict())
                for mode in ['coverage', 'random_anchors']:
                    model = attach_coverage(copy.deepcopy(baseline), mode)
                    model.train()
                    gates, chosen = model.select_candidates(graph, targets)
                    self.assertEqual(len(chosen), 4)
                    self.assertEqual(len(set(chosen.tolist())), 4)
                    self.assertFalse(set(chosen.tolist()) & set(targets.tolist()))
                    self.assertEqual(set(model.state_dict()), original_state)
                    if mode == 'coverage':
                        self.assertEqual(chosen[0].item(), 2)
                    # A varying downstream signal must reach learned complement scores.
                    (gates * torch.arange(1, 5, device=device)).sum().backward()
                    gradient = sum(p.grad.abs().sum().item() for p in model.candidate_selector.parameters() if p.grad is not None)
                    self.assertGreater(gradient, 0)
                    self.assertEqual(model.candidate_selector.max_candidates, 4)

    def test_disabled_policy_and_small_pool_equivalence(self):
        graph, baseline, targets = self.fixture('cpu')
        baseline.eval()
        model = attach_coverage(copy.deepcopy(baseline), 'baseline')
        expected = baseline.select_candidates(graph, targets)
        actual = model.select_candidates(graph, targets)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        baseline.max_candidates = model.max_candidates = 20
        model = attach_coverage(model, 'coverage')
        self.assertTrue(torch.equal(baseline.select_candidates(graph, targets)[1], model.select_candidates(graph, targets)[1]))

    def test_size_sampling_and_uniform_control(self):
        weights = group_sampling_weights([1, 9], 'size_sampling')
        self.assertTrue(torch.allclose(weights, torch.tensor([.1, .9], dtype=torch.double)))
        sampler = WeightedRandomSampler(weights, 10000, replacement=True,
                                        generator=torch.Generator().manual_seed(7))
        draws = list(sampler)
        self.assertEqual(len(draws), 10000)
        self.assertAlmostEqual(np.mean(draws), .9, delta=.015)
        self.assertTrue(torch.equal(group_sampling_weights([1, 9], 'uniform_replacement'),
                                    torch.tensor([.5, .5], dtype=torch.double)))
        with self.assertRaises(ValueError):
            group_sampling_weights([0, 9], 'size_sampling')

    def test_actual_loader_preserves_update_count_and_records_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            data = SamplingDataHandler.__new__(SamplingDataHandler)
            data.sampling_evidence = pathlib.Path(folder)
            data.sampling_epoch, data.sampling_mode = 0, 'size_sampling'
            data.multi_target, data.batch_size = True, 1
            data.target_idx_train = np.arange(3)
            data.target_assignment = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1]])
            data.node_budgets, data.edge_budgets = np.array([1, 2, 3]), np.array([4, 5, 6])
            batches = list(data.get_train_loader())
            record = json.loads((pathlib.Path(folder) / 'sampling_epoch_001.json').read_text())
            self.assertEqual(len(batches), 3)
            self.assertTrue(record['complete'])
            self.assertEqual(record['groups'], [int(batch[0]) for batch in batches])
            for group, nodes, edges in batches:
                self.assertEqual(int(nodes), int(group)+1)
                self.assertEqual(int(edges), int(group)+4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
