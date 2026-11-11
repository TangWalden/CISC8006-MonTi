"""Exact, closest-feasible two-role quotas and deterministic hard selections.

Only injected-to-target and injected-to-candidate pairs are permitted.  Each
injected node needs a target edge.  No model outcome, label or RNG is used.
"""

import hashlib
import json
import math
from numbers import Real


class InfeasibleQuotaError(ValueError):
    """The full edge budget cannot fit the declared two-role constraints."""


def plan_quota(T, C, B, E):
    """Project ceil(E/2) onto all feasible integer target quotas, without loss."""
    if any(type(value) is not int for value in (T, C, B, E)):
        raise ValueError("Counts and budgets must be integers, not booleans.")
    if T < 1 or C < 0 or B < 1 or E < 1:
        raise ValueError("Require T,B,E >= 1 and C >= 0.")
    lower, upper = max(B, E - B * C), min(B * T, E)
    if lower > upper:
        raise InfeasibleQuotaError(
            "No full-budget target/candidate-only allocation: lower=%d upper=%d "
            "for T=%d C=%d B=%d E=%d." % (lower, upper, T, C, B, E))
    ideal = (E + 1) // 2
    target = min(max(ideal, lower), upper)
    return {
        "target_count": T, "candidate_count": C, "injected_count": B,
        "edge_budget": E,
        "ideal_target_quota": ideal, "ideal_candidate_quota": E - ideal,
        "lower_target_quota": lower, "upper_target_quota": upper,
        "target_quota": target, "candidate_quota": E - target,
        "target_fraction": target / E, "candidate_fraction": (E - target) / E,
        "requested_target_fraction": 0.5,
        "projection_applied": target != ideal,
        "exact_half": 2 * target == E, "odd_budget": bool(E % 2),
        "target_capacity_limited": ideal > upper,
        "mandatory_or_candidate_capacity_limited": ideal < lower,
        "injected_to_injected_quota": 0,
        "total_budget_preserved": target + (E - target) == E,
    }


def _validate(scores, seed, group, source_ids, dest_ids, purpose):
    source_ids, dest_ids = list(source_ids), list(dest_ids)
    if (not source_ids or any(type(value) is not int or value < 0
                              for value in source_ids + dest_ids)):
        raise ValueError("Physical source/destination IDs must be nonnegative integers.")
    if (len(set(source_ids)) != len(source_ids) or len(set(dest_ids)) != len(dest_ids)
            or set(source_ids).intersection(dest_ids)):
        raise ValueError("Physical source/destination IDs must be unique and disjoint.")
    if (type(seed) is not int or seed < 0 or type(group) not in (int, str)
            or (type(group) is int and group < 0) or group == ""
            or not isinstance(purpose, str) or not purpose):
        raise ValueError("A stable seed, group key and purpose are required.")
    matrix = [list(row) for row in scores]
    if len(matrix) != len(source_ids) or any(len(row) != len(dest_ids) for row in matrix):
        raise ValueError("Score matrix shape must equal physical source/destination counts.")
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(float(value)) for row in matrix for value in row):
        raise ValueError("All scores must be finite real numbers.")
    return matrix, source_ids, dest_ids


def _rank(score, row, column, seed, group, source_ids, dest_ids, purpose):
    payload = ["monti-downstream-quota-v1", seed, group,
               source_ids[row], dest_ids[column], purpose]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":"),
                                      ensure_ascii=True).encode("ascii")).digest()
    return -score, digest, source_ids[row], dest_ids[column]


def choose_edges(scores, k, seed, group, source_ids, dest_ids, purpose,
                 requiredcoords=()):
    """Select exactly k distinct pairs, with fixed required (row,column) pairs.

    Returns a rectangular list of zero/one integers.  The highest original
    score wins; only exact ties use a SHA key bound to physical node IDs.
    Required coordinates are never silently deduplicated or discarded.
    """
    matrix, source_ids, dest_ids = _validate(scores, seed, group, source_ids, dest_ids, purpose)
    B, D = len(source_ids), len(dest_ids)
    if type(k) is not int or not 0 <= k <= B * D:
        raise InfeasibleQuotaError("Requested quota exceeds distinct legal pair capacity.")
    required = []
    for coordinate in requiredcoords:
        if (not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)):
            raise ValueError("Required coordinates must be integer row/column pairs.")
        row, column = coordinate
        if not 0 <= row < B or not 0 <= column < D:
            raise ValueError("A required coordinate is outside the score matrix.")
        required.append((row, column))
    if len(set(required)) != len(required):
        raise ValueError("Duplicate required coordinates are forbidden.")
    if len(required) > k:
        raise InfeasibleQuotaError("Required pairs exceed the requested quota.")
    chosen = set(required)
    remaining = [(row, column) for row in range(B) for column in range(D)
                 if (row, column) not in chosen]
    remaining.sort(key=lambda pair: _rank(matrix[pair[0]][pair[1]], *pair,
                                         seed, group, source_ids, dest_ids, purpose))
    chosen.update(remaining[:k - len(chosen)])
    result = [[int((row, column) in chosen) for column in range(D)] for row in range(B)]
    if sum(map(sum, result)) != k:
        raise RuntimeError("Hard selection failed to preserve the exact quota.")
    return result


def choose_target_edges(scores, k, seed, group, source_ids, dest_ids,
                        purpose="fixed_upstream_target"):
    """Fix the best target in every row, then take remaining global top scores."""
    matrix, source_ids, dest_ids = _validate(scores, seed, group, source_ids, dest_ids, purpose)
    if not dest_ids:
        raise InfeasibleQuotaError("Mandatory target edges need at least one target.")
    required = [(row, min(range(len(dest_ids)),
                         key=lambda column: _rank(matrix[row][column], row, column,
                                                  seed, group, source_ids, dest_ids, purpose)))
                for row in range(len(source_ids))]
    return choose_edges(matrix, k, seed, group, source_ids, dest_ids, purpose, required)
