"""Known-answer, optimality and immutable-information tests for quota rules."""

import itertools
import math
import random
import unittest

from methods.downstream_quota import (
    InfeasibleQuotaError, choose_edges, choose_target_edges, plan_quota,
)


class QuotaPlanTests(unittest.TestCase):
    def test_published_capacity_witness_projects_without_losing_budget(self):
        p = plan_quota(T=3, C=128, B=122, E=5734)
        self.assertEqual((p["ideal_target_quota"], p["target_quota"], p["candidate_quota"]),
                         (2867, 366, 5368))
        self.assertTrue(p["projection_applied"])
        self.assertEqual(p["target_quota"] + p["candidate_quota"], 5734)
        self.assertAlmostEqual(p["target_fraction"] + p["candidate_fraction"], 1)

    def test_mandatory_target_edges_project_when_E_equals_B(self):
        p = plan_quota(T=4, C=128, B=5, E=5)
        self.assertEqual((p["target_quota"], p["candidate_quota"]), (5, 0))

    def test_candidate_capacity_projects_to_targets(self):
        p = plan_quota(T=5, C=1, B=2, E=8)
        self.assertEqual((p["target_quota"], p["candidate_quota"]), (6, 2))
        self.assertTrue(p["mandatory_or_candidate_capacity_limited"])

    def test_impossible_total_or_mandatory_budget_raises(self):
        for values in ((1, 1, 2, 5), (4, 4, 3, 2)):
            with self.subTest(values=values), self.assertRaises(InfeasibleQuotaError):
                plan_quota(*values)

    def test_odd_and_zero_candidate_cases(self):
        p = plan_quota(T=4, C=4, B=1, E=3)
        self.assertEqual((p["target_quota"], p["candidate_quota"]), (2, 1))
        self.assertTrue(p["odd_budget"])
        self.assertFalse(p["projection_applied"])
        self.assertFalse(p["exact_half"])
        self.assertEqual(plan_quota(3, 0, 2, 4)["candidate_quota"], 0)

    def test_projection_is_nearest_legal_quota_for_small_exhaustive_cases(self):
        for T, C, B, E in itertools.product(range(1, 4), range(3), range(1, 4), range(1, 14)):
            legal = [t for t in range(E + 1) if B <= t <= B*T and E-t <= B*C]
            if not legal:
                with self.assertRaises(InfeasibleQuotaError):
                    plan_quota(T, C, B, E)
            else:
                p = plan_quota(T, C, B, E)
                self.assertIn(p["target_quota"], legal)
                self.assertEqual(abs(p["target_quota"] - math.ceil(E/2)),
                                 min(abs(t-math.ceil(E/2)) for t in legal))


class SelectionTests(unittest.TestCase):
    def test_global_top_with_required_pairs(self):
        edges = choose_edges([[9, 8, 7], [6, 5, 4]], 3, 3, 100, [20, 21], [1, 2, 3],
                             "candidate", requiredcoords=[(1, 2)])
        self.assertEqual(edges, [[1, 1, 0], [0, 0, 1]])

    def test_mandatory_target_selection_is_globally_optimal(self):
        scores = [[9, 8, 1], [7, 3, 2], [6, 5, 4]]
        edges = choose_target_edges(scores, 5, 3, 100, [20, 21, 22], [1, 2, 3])
        self.assertTrue(all(sum(row) >= 1 for row in edges))
        observed = sum(scores[r][c] for r in range(3) for c in range(3) if edges[r][c])
        legal_scores = [sum(scores[r][c] for r, c in selected)
                        for selected in itertools.combinations(itertools.product(range(3), repeat=2), 5)
                        if {r for r, c in selected} == {0, 1, 2}]
        self.assertEqual(observed, max(legal_scores))

    def test_physical_ties_reproduce_under_reordering_without_using_rng(self):
        state = random.getstate()
        a = choose_edges([[1, 1, 1], [1, 1, 1]], 3, 3, "group", [20, 21], [1, 2, 3], "candidate")
        b = choose_edges([[1, 1, 1], [1, 1, 1]], 3, 3, "group", [21, 20], [3, 1, 2], "candidate")
        pairs = lambda e, src, dst: {(src[r], dst[c]) for r in range(2) for c in range(3) if e[r][c]}
        self.assertEqual(pairs(a, [20, 21], [1, 2, 3]), pairs(b, [21, 20], [3, 1, 2]))
        self.assertEqual(random.getstate(), state)
        self.assertEqual(sum(map(sum, a)), 3)

    def test_empty_candidate_matrix_preserves_zero_quota(self):
        self.assertEqual(choose_edges([[], []], 0, 3, 1, [20, 21], [], "candidate"), [[], []])

    def test_ids_shape_and_nonfinite_scores_are_rejected(self):
        invalid = [([[1]], [20, 20], [1]), ([[1, 2]], [20], [1, 1]),
                   ([[1]], [20], [20]), ([[1, 2]], [20], [1]),
                   ([[float("nan")]], [20], [1]), ([[float("inf")]], [20], [1])]
        for scores, sources, destinations in invalid:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                choose_edges(scores, 1, 3, 1, sources, destinations, "candidate")

    def test_duplicate_required_or_insufficient_quota_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_edges([[1, 2]], 2, 3, 1, [20], [1, 2], "target", [(0, 0), (0, 0)])
        with self.assertRaises(InfeasibleQuotaError):
            choose_target_edges([[1], [2]], 1, 3, 1, [20, 21], [1])
        with self.assertRaises(InfeasibleQuotaError):
            choose_edges([[1]], 2, 3, 1, [20], [1], "candidate")


if __name__ == "__main__":
    unittest.main()
