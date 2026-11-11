"""Audit necessary 50:50 edge-quota bounds from saved metadata only.

This module neither imports an ML runtime nor authorizes training.  In the
proposed two-head policy, target and candidate edges exhaust the edge budget;
there is no allowance for injected-to-injected edges.  Passing these necessary
bounds is not a proof that a realizable candidate graph or selection exists.
"""

import argparse
from collections import Counter
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import time


SCHEMA = "monti_downstream_half_quota_metadata_audit_v1"
EXPECTED_SEEDS = (3, 4, 5)
EXPECTED_GROUPS_PER_SEED = 128
CANDIDATE_COUNT_UPPER_BOUND = 128
METADATA_FIELDS = ("opaque_case_key", "seed", "targets", "node_budget", "edge_budget")
REQUIRED_FIELDS = ("arm",) + METADATA_FIELDS


def _integer(value, name):
    """Accept unsigned decimal metadata, rejecting floats and boolean values."""
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+", value.strip()):
        raise ValueError("Invalid unsigned integer metadata: " + name)
    return int(value.strip())


def classify_geometry(targets, node_budget, edge_budget):
    """Return necessary bounds; do not infer actual candidate availability."""
    values = (targets, node_budget, edge_budget)
    if any(type(value) is not int or value < 1 for value in values):
        raise ValueError("Targets and both structural budgets must be positive integers.")
    target_quota = (edge_budget + 1) // 2
    candidate_quota = edge_budget // 2
    target_capacity = node_budget * targets
    candidate_capacity_upper_bound = node_budget * CANDIDATE_COUNT_UPPER_BOUND
    reasons = []
    if target_quota > target_capacity:
        reasons.append("target_quota_exceeds_legal_target_pairs")
    if candidate_quota > candidate_capacity_upper_bound:
        reasons.append("candidate_quota_exceeds_candidate_pair_upper_bound")
    if target_quota < node_budget:
        reasons.append("target_quota_below_one_target_edge_per_injected_node")
    if edge_budget > target_capacity + candidate_capacity_upper_bound:
        reasons.append("total_budget_exceeds_two_head_pair_upper_bound")
    return {
        "targets": targets,
        "node_budget": node_budget,
        "edge_budget": edge_budget,
        "target_quota": target_quota,
        "candidate_quota": candidate_quota,
        "odd_budget": bool(edge_budget % 2),
        "target_pair_capacity": target_capacity,
        "candidate_pair_capacity_upper_bound": candidate_capacity_upper_bound,
        "combined_pair_capacity_upper_bound": target_capacity + candidate_capacity_upper_bound,
        "necessary_failure": bool(reasons),
        "failure_reasons": reasons,
        "classification": (
            "rejected_by_necessary_bounds" if reasons
            else "not_rejected_by_necessary_bounds_actual_feasibility_unverified"
        ),
    }


def audit_metadata_csv(input_path, expected_seeds=EXPECTED_SEEDS,
                       expected_groups_per_seed=EXPECTED_GROUPS_PER_SEED):
    """Read baseline metadata, compare geometry multisets, and return an audit.

    Other CSV columns, including all classifier outcomes, are never accessed.
    The whole-file hash identifies the source; it is not a use of its outcome
    values.  ``expected_*`` arguments permit small synthetic audit fixtures.
    """
    started = time.perf_counter()
    cpu_started = time.process_time()
    if (not expected_seeds or len(set(expected_seeds)) != len(expected_seeds)
            or any(type(seed) is not int or seed < 0 for seed in expected_seeds)
            or type(expected_groups_per_seed) is not int or expected_groups_per_seed < 1):
        raise ValueError("Expected seeds and group count must be explicit and valid.")
    expected_seeds = tuple(sorted(expected_seeds))
    input_path = Path(input_path).resolve(strict=True)
    raw = input_path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames
    if (fields is None or len(fields) != len(set(fields))
            or not set(REQUIRED_FIELDS).issubset(fields)):
        raise ValueError("CSV requires unique arm and baseline metadata columns.")
    by_seed = {seed: {} for seed in expected_seeds}
    ignored_rows = 0
    for row in reader:
        if row.get("arm") != "baseline":
            ignored_rows += 1
            continue
        metadata = {field: row.get(field) for field in METADATA_FIELDS}
        case_key = metadata["opaque_case_key"]
        if not isinstance(case_key, str) or not case_key.strip():
            raise ValueError("A baseline opaque case key is missing.")
        case_key = case_key.strip()
        seed = _integer(metadata["seed"], "seed")
        if seed not in by_seed:
            raise ValueError("Unexpected baseline seed: " + str(seed))
        if case_key in by_seed[seed]:
            raise ValueError("Duplicate baseline case key within a seed.")
        geometry = classify_geometry(*(
            _integer(metadata[field], field)
            for field in ("targets", "node_budget", "edge_budget")
        ))
        by_seed[seed][case_key] = geometry
    for seed in expected_seeds:
        if len(by_seed[seed]) != expected_groups_per_seed:
            raise ValueError("Baseline group count differs for seed " + str(seed))
    reference = by_seed[expected_seeds[0]]
    def geometry_multiset(cases):
        return Counter((case["targets"], case["node_budget"], case["edge_budget"])
                       for case in cases.values())
    for seed in expected_seeds[1:]:
        if geometry_multiset(by_seed[seed]) != geometry_multiset(reference):
            raise ValueError("Baseline geometry multiset differs across seeds.")

    per_seed = []
    witnesses = []
    all_cases = []
    for seed in expected_seeds:
        counts = {}
        failures = 0
        for case_key in sorted(by_seed[seed]):
            geometry = by_seed[seed][case_key]
            case = dict(opaque_case_key=case_key, seed=seed, **geometry)
            all_cases.append(case)
            if geometry["necessary_failure"]:
                failures += 1
                witnesses.append(case)
            for reason in geometry["failure_reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        per_seed.append({
            "seed": seed,
            "baseline_groups": expected_groups_per_seed,
            "necessary_failure_groups": failures,
            "necessary_failure_share": failures / expected_groups_per_seed,
            "not_rejected_groups_actual_feasibility_unverified": expected_groups_per_seed - failures,
            "failure_reason_counts": counts,
        })
    failure_cases = sum(item["necessary_failure_groups"] for item in per_seed)
    return {
        "schema": SCHEMA,
        "status": "completed",
        "result": (
            "strict_half_quota_necessary_failures_detected" if failure_cases
            else "no_necessary_failure_detected_actual_feasibility_unverified"
        ),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_csv": {"path": str(input_path), "sha256": hashlib.sha256(raw).hexdigest()},
        "information_boundary": {
            "row_filter_column": "arm",
            "used_metadata_columns": list(METADATA_FIELDS),
            "classifier_outcome_values_used": False,
            "other_column_values_used": False,
            "whole_file_hash_purpose": "Identify exact source bytes only.",
            "ignored_nonbaseline_rows": ignored_rows,
        },
        "policy": {
            "target_quota": "ceil(edge_budget / 2)",
            "candidate_quota": "floor(edge_budget / 2)",
            "candidate_count_upper_bound": CANDIDATE_COUNT_UPPER_BOUND,
            "actual_candidate_counts_known": False,
            "mandatory_target_incidence": "At least one target edge per injected node.",
            "injected_to_injected_edges_allowed": False,
            "edge_semantics": "Distinct undirected injected-to-original pairs; no duplicates or self-loops.",
            "quota_clamping_or_fallback_applied": False,
        },
        "cohort_geometry": {
            "seeds": list(expected_seeds),
            "baseline_rows_per_seed": expected_groups_per_seed,
            "same_geometry_multiset_across_seeds": True,
            "cross_seed_group_identity_verified": False,
            "opaque_case_key_scope": "Within-seed uniqueness only; CSV provides no cross-seed identity mapping.",
            "seed_scoped_baseline_rows": len(all_cases),
            "necessary_failure_rows": failure_cases,
            "necessary_failure_row_share": failure_cases / len(all_cases),
            "replication_limit": "Matching geometry multisets do not establish paired group identities or independent observations.",
        },
        "per_seed": per_seed,
        "necessary_failure_witnesses": witnesses,
        "all_baseline_metadata_cases": all_cases,
        "limitations": [
            "Passing necessary bounds does not establish actual candidate availability or a realizable selection.",
            "The candidate capacity uses an upper bound of128; actual candidate counts may be lower.",
            "Opaque keys cannot establish cross-seed group identity; only metadata geometry multisets are compared.",
            "No physical graph, selected endpoints, injected feature, checkpoint or model was read.",
            "This is a metadata audit, not an efficacy experiment, frozen training protocol or training authorization.",
        ],
        "execution": {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "model_imports": 0,
            "model_queries": 0,
            "training_steps": 0,
            "optimizer_updates": 0,
            "GPU_hours": 0,
            "automatic_training_started": False,
        },
    }


def write_report_exclusive(output_path, report):
    """Never overwrite a prior audit receipt."""
    with Path(output_path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Saved CSV; only baseline metadata is used.")
    parser.add_argument("--output", required=True, help="New exclusive JSON path in an existing directory.")
    args = parser.parse_args(argv)
    report = audit_metadata_csv(args.input)
    write_report_exclusive(args.output, report)
    print(json.dumps({
        "status": report["status"], "result": report["result"],
        "source_csv_sha256": report["source_csv"]["sha256"],
        "cohort_geometry": report["cohort_geometry"],
        "automatic_training_started": False,
        "GPU_hours": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
