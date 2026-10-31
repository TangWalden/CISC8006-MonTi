"""CPU-only, synthetic checks for unselected scoring adapters.

No dataset, saved checkpoint, detector, optimizer, or experiment runner is used.
The original unbound edge method is a regression reference on synthetic tensors.
Passing these checks is not approval or evidence of a real-data improvement.
"""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import MethodType, SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from models.attack_model import MonTi
from .scoring import (
    _mandatory_selection,
    bind_edge_generation,
    count_matched_random_mask,
    edge_generation,
    temporary_edge_generation,
)


class ConstantSchedule:
    def get_temperature(self):
        return 0.7

    def get_epsilon(self):
        return 0.3


class SyntheticHarness(nn.Module):
    edge_generation = MonTi.edge_generation

    def __init__(self, adaptive=True, training=False):
        super().__init__()
        self.linear_edge = nn.Identity()
        self.register_buffer("synthetic_marker", torch.tensor([13.0]))
        self.ada_budget = adaptive
        self.temp_scheduler = ConstantSchedule()
        self.eps_scheduler = ConstantSchedule()
        self.train(training)


CPU_GRAPH = SimpleNamespace(device=torch.device("cpu"))


def fixture(targets=2, candidates=2, nodes=1):
    # No random initialization, features, labels, data file or checkpoint.
    rows = targets + candidates + nodes
    return torch.linspace(-1.7, 2.3, rows * 4).reshape(rows, 4)


class SyntheticChecks(unittest.TestCase):
    def assert_same(self, first, second):
        self.assertTrue(torch.equal(first, second), (first, second))
        self.assertEqual(first.dtype, second.dtype)
        self.assertEqual(first.device, torch.device("cpu"))
        self.assertEqual(second.device, torch.device("cpu"))

    def test_default_output_gradient_rng_and_budget_mutation_parity(self):
        cases = [(True, 1, 1), (True, 2, 3), (True, 3, 6),
                 (False, 1, 2), (False, 2, 4), (False, 3, 6)]
        for adaptive, nodes, edges in cases:
            for training in (False, True):
                for random_edge in (False, True):
                    for tensor_budget in (False, True):
                        with self.subTest(adaptive=adaptive, nodes=nodes, edges=edges,
                                          training=training, random=random_edge,
                                          tensor_budget=tensor_budget):
                            model = SyntheticHarness(adaptive, training)
                            outputs, gradients, states, budgets = [], [], [], []
                            for method in (MonTi.edge_generation, edge_generation):
                                z = fixture(nodes=nodes).requires_grad_()
                                budget = torch.tensor(edges) if tensor_budget else edges
                                torch.manual_seed(20261031)
                                output = method(model, z, CPU_GRAPH, 2, 2, nodes,
                                                budget, random_edge)
                                states.append(torch.get_rng_state().clone())
                                outputs.append(output.detach().clone())
                                budgets.append(int(budget))
                                if not random_edge:
                                    weights = torch.arange(output.numel(), dtype=output.dtype)
                                    (output * weights.reshape_as(output)).sum().backward()
                                    gradients.append(z.grad.clone())
                            self.assert_same(*outputs)
                            self.assert_same(*states)
                            self.assertEqual(budgets[0], budgets[1])
                            if not random_edge:
                                self.assert_same(*gradients)
                                self.assertTrue(torch.isfinite(gradients[0]).all())

    def test_known_answer_source_role_and_direct_latent_gradient(self):
        # Target 0 is the legacy source. The injected source favors target 1.
        # These are independent synthetic latents, not a Transformer context.
        base = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
        results = {}
        for role, expected in (("legacy", 0), ("injected", 1)):
            z = base.clone().requires_grad_()
            trace = {}
            adj = edge_generation(SyntheticHarness(), z, CPU_GRAPH, 2, 0, 1, 1,
                                  scoring_role=role, trace=trace)
            self.assertEqual(adj.argmax(dim=-1).item(), expected)
            self.assertEqual(adj[0, 2].item(), 0.0)
            self.assertEqual(trace["start_atk"], 2)
            source = 0 if role == "legacy" else 2
            self.assert_same(trace["source_rows"], torch.full((2,), source))
            (adj * torch.tensor([[0.0, 1.0, 0.0]])).sum().backward()
            self.assertTrue(torch.isfinite(z.grad).all())
            results[role] = z.grad
            self.assert_same(z.detach(), base)
        self.assertEqual(results["legacy"][2].abs().sum().item(), 0.0)
        self.assertGreater(results["injected"][2].abs().sum().item(), 0.0)

    def test_aligned_projection_parameter_receives_gradient_without_update(self):
        model = SyntheticHarness()
        model.linear_edge = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.linear_edge.weight.copy_(torch.eye(2))
        before = model.linear_edge.weight.detach().clone()
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], requires_grad=True)
        adj = edge_generation(model, z, CPU_GRAPH, 2, 0, 1, 1, scoring_role="injected")
        (adj * torch.tensor([[0.0, 1.0, 0.0]])).sum().backward()
        gradient = model.linear_edge.weight.grad
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(gradient.abs().sum().item(), 0.0)
        self.assert_same(model.linear_edge.weight.detach(), before)

    def test_candidate_offset_and_distinct_injected_rows_known_answer(self):
        # T=2, C=1, B=2 distinguishes T+C+src from both src and T+src.
        z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8],
                          [0.8, 0.6], [-0.6, 0.8]], requires_grad=True)
        trace = {}
        adj = edge_generation(SyntheticHarness(), z, CPU_GRAPH, 2, 1, 2, 2,
                              scoring_role="injected", trace=trace)
        self.assert_same(trace["source_rows"], torch.tensor([3, 3, 3, 4, 4, 4, 4]))
        self.assert_same(trace["destination_rows"], torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        expected_scores = torch.tensor([0.8, 0.6, 0.96, -0.6, 0.8, 0.28, 0.0])
        self.assertTrue(torch.allclose(trace["raw_edge_scores"], expected_scores, atol=1e-7, rtol=0))
        self.assert_same(adj[:, :2].argmax(dim=-1), torch.tensor([0, 1]))
        weights = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0],
                                [2.0, 0.0, 0.0, 0.0, 0.0]])
        (adj * weights).sum().backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        self.assertTrue(bool((z.grad[3:].abs().sum(dim=-1) > 0).all()))

    def test_aligned_adaptive_budget_legal_edges_and_capacity(self):
        for nodes in (1, 2, 3, 5):
            total_slots = 4 + nodes
            legal = torch.tril(torch.ones(nodes, total_slots, dtype=torch.bool), diagonal=3)
            for edges in (nodes, nodes + 1, nodes + 3, 100):
                with self.subTest(nodes=nodes, edges=edges):
                    adj = edge_generation(SyntheticHarness(), fixture(nodes=nodes),
                                          CPU_GRAPH, 2, 2, nodes, edges,
                                          scoring_role="injected")
                    hard = adj > 0.5
                    self.assertEqual(int(hard.sum()), min(edges, int(legal.sum())))
                    self.assertFalse(bool(hard[~legal].any()))
                    self.assertTrue(bool((hard[:, :2].sum(dim=1) >= 1).all()))
                    self.assertTrue(bool(torch.isfinite(adj).all()))

    def test_random_branch_ignores_options_and_preserves_rng(self):
        model = SyntheticHarness()
        z = fixture(nodes=2)
        torch.manual_seed(100)
        expected = MonTi.edge_generation(model, z, CPU_GRAPH, 2, 2, 2, 3, True)
        state = torch.get_rng_state().clone()
        torch.manual_seed(100)
        actual = edge_generation(model, z, CPU_GRAPH, 2, 2, 2, 3, True,
                                 scoring_role="ignored", tie_policy="ignored",
                                 eligible_targets=["ignored"], target_global_ids=None)
        self.assert_same(expected, actual)
        self.assert_same(state, torch.get_rng_state())

    def test_mask_identity_fallback_and_nonapplicable_budget(self):
        for training in (False, True):
            model = SyntheticHarness(training=training)
            for role in ("legacy", "injected"):
                for mask in (torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)):
                    torch.manual_seed(101)
                    expected = edge_generation(model, fixture(), CPU_GRAPH, 2, 2, 1, 1,
                                               scoring_role=role)
                    state = torch.get_rng_state().clone()
                    torch.manual_seed(101)
                    actual = edge_generation(model, fixture(), CPU_GRAPH, 2, 2, 1, 1,
                                             scoring_role=role, eligible_targets=mask)
                    self.assert_same(expected, actual)
                    self.assert_same(state, torch.get_rng_state())
        for adaptive, nodes, edges in ((True, 2, 2), (True, 1, 2), (False, 1, 1)):
            model = SyntheticHarness(adaptive, True)
            torch.manual_seed(102)
            expected = edge_generation(model, fixture(nodes=nodes), CPU_GRAPH, 2, 2,
                                       nodes, edges, scoring_role="injected")
            state = torch.get_rng_state().clone()
            torch.manual_seed(102)
            actual = edge_generation(model, fixture(nodes=nodes), CPU_GRAPH, 2, 2,
                                     nodes, edges, scoring_role="injected",
                                     eligible_targets=torch.tensor([True, False]))
            self.assert_same(expected, actual)
            self.assert_same(state, torch.get_rng_state())

    def test_partial_mask_choice_finite_derivative_and_training_guard(self):
        mask = torch.tensor([False, True, False])
        z = fixture(targets=3, candidates=1).requires_grad_()
        trace = {}
        adj = edge_generation(SyntheticHarness(), z, CPU_GRAPH, 3, 1, 1, 1,
                              eligible_targets=mask, trace=trace)
        self.assertEqual(adj.argmax(dim=-1).item(), 1)
        self.assertEqual(int((adj > 0.5).sum()), 1)
        self.assertEqual(trace["eligibility_status"], "restricted")
        (adj * torch.arange(adj.numel()).reshape_as(adj)).sum().backward()
        self.assertTrue(bool(torch.isfinite(z.grad).all()))
        torch.manual_seed(103)
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            edge_generation(SyntheticHarness(training=True), z, CPU_GRAPH, 3, 1,
                            1, 1, eligible_targets=mask)
        self.assert_same(before, torch.get_rng_state())

    def test_global_id_exact_tie_and_permutation_consistency(self):
        z = torch.tensor([[1.0, 0.0]] * 4)
        ids = torch.tensor([90, 10, 50])
        for order in (torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])):
            adj = edge_generation(SyntheticHarness(), z, CPU_GRAPH, 3, 0, 1, 1,
                                  tie_policy="lowest_global_id", target_global_ids=ids[order])
            self.assertEqual(int(ids[order][adj[0, :3].argmax()]), 10)
        adj = edge_generation(SyntheticHarness(), z, CPU_GRAPH, 3, 0, 1, 1,
                              tie_policy="lowest_global_id", target_global_ids=ids,
                              eligible_targets=torch.tensor([True, False, True]))
        self.assertEqual(int(ids[adj[0, :3].argmax()]), 50)
        # One representable step is not an exact tie, even if the lower ID loses.
        p = torch.tensor([[0.50000006, 0.5]], requires_grad=True)
        selected, index = _mandatory_selection(p, None, torch.tensor([90, 10]), "lowest_global_id")
        self.assertEqual(index.item(), 0)
        (selected * torch.tensor([[2.0, 7.0]])).sum().backward()
        self.assert_same(p.grad, torch.tensor([[2.0, 7.0]]))

    def test_random_mask_cardinality_repeatability_and_rng_isolation(self):
        for original in (torch.zeros(7, dtype=torch.bool), torch.ones(7, dtype=torch.bool),
                         torch.tensor([True, False, True, False, False, True, False])):
            copy = original.clone()
            torch.manual_seed(104)
            random.seed(104)
            np.random.seed(104)
            torch_state = torch.get_rng_state().clone()
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            first = count_matched_random_mask(original, seed=555)
            second = count_matched_random_mask(original, seed=555)
            self.assert_same(first, second)
            self.assert_same(original, copy)
            self.assertEqual(int(first.sum()), int(original.sum()))
            self.assertEqual(first.numel(), original.numel())
            self.assertEqual(first.dtype, torch.bool)
            self.assert_same(torch_state, torch.get_rng_state())
            self.assertEqual(python_state, random.getstate())
            after = np.random.get_state()
            self.assertEqual(numpy_state[0], after[0])
            self.assertTrue(np.array_equal(numpy_state[1], after[1]))
            self.assertEqual(numpy_state[2:], after[2:])
            first[0] = ~first[0]
            self.assert_same(original, copy)

    def test_binding_nested_restore_exception_and_state_identity(self):
        model = SyntheticHarness()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        original = model.edge_generation.__func__
        restore = bind_edge_generation(model, scoring_role="injected")
        outer = model.edge_generation
        with self.assertRaisesRegex(ValueError, "synthetic interruption"):
            with temporary_edge_generation(model, scoring_role="legacy"):
                self.assertIsNot(model.edge_generation, outer)
                raise ValueError("synthetic interruption")
        self.assertIs(model.edge_generation, outer)
        restore()
        restore()
        self.assertIs(model.edge_generation.__func__, original)
        self.assertNotIn("edge_generation", model.__dict__)
        self.assertEqual(before.keys(), model.state_dict().keys())
        for key, value in model.state_dict().items():
            self.assert_same(before[key], value)
        override = MethodType(lambda current, *args: "synthetic override", model)
        model.edge_generation = override
        with temporary_edge_generation(model, scoring_role="legacy"):
            self.assertIsNot(model.edge_generation, override)
        self.assertIs(model.edge_generation, override)

    def test_invalid_active_options_fail_explicitly(self):
        for options, exception in (({"scoring_role": "unknown"}, ValueError),
                                   ({"tie_policy": "unknown"}, ValueError),
                                   ({"eligible_targets": [1, 0]}, TypeError),
                                   ({"eligible_targets": torch.tensor([True])}, ValueError),
                                   ({"tie_policy": "lowest_global_id"}, ValueError),
                                   ({"tie_policy": "lowest_global_id", "target_global_ids": [1, 1]}, ValueError)):
            with self.subTest(options=options):
                with self.assertRaises(exception):
                    edge_generation(SyntheticHarness(), fixture(), CPU_GRAPH, 2, 2, 1, 1, **options)
        with self.assertRaises(TypeError):
            count_matched_random_mask(torch.tensor([True, False]), seed=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            _mandatory_selection(torch.tensor([[float("nan"), 1.0]]), None,
                                 torch.tensor([1, 2]), "lowest_global_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON record; existing files are never overwritten")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES to the empty string for this CPU-only preparation check")
    output = Path(args.output)
    if output.exists():
        parser.error("The output record already exists; use a distinct attempt path")
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    source_files = ("course/outcome_v2/__init__.py", "course/outcome_v2/scoring.py",
                    "course/outcome_v2/synthetic_checks.py", "models/attack_model.py", "utils/utils.py")
    source_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_files}
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    tracked_changes = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                               cwd=str(root), text=True).strip()
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.perf_counter()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticChecks)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {
        "schema": "monti_outcome_v2_synthetic_checks_v1",
        "status": "passed" if result.wasSuccessful() else "failed",
        "scope": "Unselected adapter preparation; synthetic CPU tensors only, no efficacy evidence or owner approval",
        "start_utc": start_utc,
        "elapsed_seconds": time.perf_counter() - started,
        "source_commit": source_commit,
        "tracked_changes_before_checks": tracked_changes,
        "source_sha256": source_hashes,
        "command": "CUDA_VISIBLE_DEVICES= bash course/run.sh -m course.outcome_v2.synthetic_checks --output " + str(output),
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "torch_threads": 1},
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "trace": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "trace": trace} for test, trace in result.errors],
        "test_log": stream.getvalue(),
        "dataset_and_checkpoint_loading": "none; synthetic fixtures only",
        "training_runs": 0,
        "optimizer_steps": 0,
        "real_data_generator_calls": 0,
        "detector_calls": 0,
        "victim_test_queries": 0,
        "gpu_hours": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    sys.stdout.write(stream.getvalue())
    print("Synthetic preparation record: " + str(output))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
