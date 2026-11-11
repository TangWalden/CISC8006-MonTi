"""Known-answer and information-boundary checks for the metadata-only audit."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.downstream_cooperation.budget_audit import (
    audit_metadata_csv, classify_geometry, write_report_exclusive,
)


class BudgetGeometryTests(unittest.TestCase):
    def test_published_impossible_target_quota_known_answer(self):
        result = classify_geometry(targets=3, node_budget=122, edge_budget=5734)
        self.assertEqual(result["target_quota"], 2867)
        self.assertEqual(result["candidate_quota"], 2867)
        self.assertEqual(result["target_pair_capacity"], 366)
        self.assertEqual(result["failure_reasons"], ["target_quota_exceeds_legal_target_pairs"])

    def test_one_edge_rounds_to_target_and_preserves_mandatory_incidence(self):
        result = classify_geometry(targets=4, node_budget=1, edge_budget=1)
        self.assertEqual((result["target_quota"], result["candidate_quota"]), (1, 0))
        self.assertTrue(result["odd_budget"])
        self.assertFalse(result["necessary_failure"])
        self.assertIn("unverified", result["classification"])

    def test_target_quota_cannot_cover_every_injected_node(self):
        result = classify_geometry(targets=4, node_budget=3, edge_budget=3)
        self.assertEqual((result["target_quota"], result["candidate_quota"]), (2, 1))
        self.assertEqual(result["failure_reasons"], ["target_quota_below_one_target_edge_per_injected_node"])

    def test_candidate_count_is_only_an_upper_bound(self):
        result = classify_geometry(targets=300, node_budget=1, edge_budget=300)
        self.assertEqual(result["candidate_quota"], 150)
        self.assertEqual(result["candidate_pair_capacity_upper_bound"], 128)
        self.assertEqual(result["failure_reasons"], ["candidate_quota_exceeds_candidate_pair_upper_bound"])

    def test_total_two_head_capacity_does_not_include_injected_block(self):
        result = classify_geometry(targets=1, node_budget=1, edge_budget=260)
        self.assertEqual(result["combined_pair_capacity_upper_bound"], 129)
        self.assertIn("total_budget_exceeds_two_head_pair_upper_bound", result["failure_reasons"])

    def test_invalid_budget_types_are_rejected(self):
        for values in ((0, 1, 1), (1, 0, 1), (1, 1, 0), (True, 1, 1), (1, 1, 1.0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                classify_geometry(*values)


class MetadataAuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "cases.csv"
        self.fields = ["opaque_case_key", "seed", "arm", "targets", "node_budget", "edge_budget",
                       "gaga_attacked_wrong", "unrelated_payload"]

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def rows():
        return [dict(opaque_case_key=key, seed=seed, arm="baseline", targets=targets,
                     node_budget=nodes, edge_budget=edges, gaga_attacked_wrong="NOT_A_NUMBER",
                     unrelated_payload="outcome values must never be parsed")
                for seed in (3, 4, 5)
                for key, targets, nodes, edges in (("impossible", 3, 122, 5734), ("odd", 4, 1, 1))]

    def audit(self):
        return audit_metadata_csv(self.path, expected_groups_per_seed=2)

    def test_counted_shares_and_complete_failure_witnesses(self):
        rows = self.rows()
        rows.append(dict(opaque_case_key="ignored", seed="invalid", arm="random_target_fill",
                         targets="invalid", node_budget="invalid", edge_budget="invalid"))
        self.write_rows(rows)
        result = self.audit()
        self.assertEqual(result["cohort_geometry"]["seed_scoped_baseline_rows"], 6)
        self.assertEqual(result["cohort_geometry"]["necessary_failure_rows"], 3)
        self.assertEqual(result["cohort_geometry"]["necessary_failure_row_share"], 0.5)
        self.assertEqual(len(result["necessary_failure_witnesses"]), 3)
        self.assertEqual([item["necessary_failure_share"] for item in result["per_seed"]], [0.5] * 3)
        self.assertEqual(result["information_boundary"]["ignored_nonbaseline_rows"], 1)
        self.assertFalse(result["policy"]["actual_candidate_counts_known"])
        self.assertFalse(result["policy"]["injected_to_injected_edges_allowed"])
        self.assertEqual(result["execution"]["model_queries"], 0)
        self.assertFalse(result["execution"]["automatic_training_started"])

    def test_classifier_values_cannot_change_classification(self):
        rows = self.rows()
        self.write_rows(rows)
        before = self.audit()
        for index, row in enumerate(rows):
            row["gaga_attacked_wrong"] = str(10 ** (index + 3))
            row["unrelated_payload"] = "completely different predictions"
        self.write_rows(rows)
        after = self.audit()
        self.assertEqual(before["all_baseline_metadata_cases"], after["all_baseline_metadata_cases"])
        self.assertEqual(before["necessary_failure_witnesses"], after["necessary_failure_witnesses"])
        self.assertEqual(before["per_seed"], after["per_seed"])
        self.assertNotEqual(before["source_csv"]["sha256"], after["source_csv"]["sha256"])

    def test_changed_geometry_across_seeds_is_rejected(self):
        rows = self.rows()
        rows[-1]["node_budget"] = 2
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "geometry multiset differs across seeds"):
            self.audit()

    def test_seed_scoped_keys_are_not_cross_seed_identity(self):
        rows = self.rows()
        for row in rows:
            row["opaque_case_key"] += "_seed" + str(row["seed"])
        self.write_rows(rows)
        result = self.audit()
        self.assertTrue(result["cohort_geometry"]["same_geometry_multiset_across_seeds"])
        self.assertFalse(result["cohort_geometry"]["cross_seed_group_identity_verified"])

    def test_geometry_multiplicity_mismatch_is_rejected(self):
        rows = self.rows()
        rows[-1].update(targets=3, node_budget=122, edge_budget=5734)
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "geometry multiset differs across seeds"):
            self.audit()

    def test_duplicate_baseline_is_rejected(self):
        rows = self.rows()
        rows.append(dict(rows[0]))
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "Duplicate baseline"):
            self.audit()

    def test_missing_group_is_not_silently_dropped(self):
        self.write_rows(self.rows()[:-1])
        with self.assertRaisesRegex(ValueError, "group count differs"):
            self.audit()

    def test_report_is_exclusive_and_english_json(self):
        self.write_rows(self.rows())
        result = self.audit()
        output = Path(self.directory.name) / "receipt.json"
        write_report_exclusive(output, result)
        saved_bytes = output.read_bytes()
        self.assertEqual(json.loads(saved_bytes)["schema"], result["schema"])
        self.assertNotIn(b"\r\n", saved_bytes)
        with self.assertRaises(FileExistsError):
            write_report_exclusive(output, result)
        self.assertEqual(output.read_bytes(), saved_bytes)


if __name__ == "__main__":
    unittest.main()
