"""Bijective feature proposals for a frozen proxy's linearized objective.

No neural model, dataset, checkpoint, graph, label or API is accessed here.
Hungarian optimality concerns only the computed linear cost, not a nonlinear
classifier or transfer. The separate common acceptance rule uses caller-supplied
OWN teacher/GCN outputs and is not permission to use victim or TEST feedback.
"""
import hashlib
import json
import math
import random

import numpy as np
from scipy.optimize import linear_sum_assignment


ACCEPTANCE_TOLERANCE = 1e-7
CONTROL_NAMESPACE = 'feature-assignment-cycle-control-v1'


def _matrix(value, name):
    array = np.asarray(value)
    if (array.ndim != 2 or min(array.shape) < 1
            or array.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(array).all()):
        raise ValueError(name+' must be a finite nonempty float32/float64 matrix.')
    return array


def _permutation(value, count=None):
    array = np.asarray(value)
    if (array.ndim != 1 or array.dtype.kind not in 'iu' or not len(array)
            or (count is not None and len(array) != count)):
        raise ValueError('A nonempty integer permutation vector is required.')
    p = array.astype(np.int64, copy=True)
    if not np.array_equal(np.sort(p), np.arange(len(p), dtype=np.int64)):
        raise ValueError('Every source feature row must occur exactly once.')
    return p


def cycle_lengths(permutation):
    """Sorted full cycle signature, including fixed-point cycles of length1."""
    p = _permutation(permutation)
    visited = np.zeros(len(p), dtype=bool)
    lengths = []
    for start in range(len(p)):
        if visited[start]:continue
        current, length = start, 0
        while not visited[current]:
            visited[current] = True
            length += 1
            current = int(p[current])
        lengths.append(length)
    return sorted(lengths)


def apply_permutation(features, permutation):
    """Position i receives ORIGINAL source row permutation[i]; preserve dtype."""
    x = _matrix(features, 'Features')
    p = _permutation(permutation, len(x))
    return x[p].copy()


def _up(value):
    result = float(np.nextafter(np.float64(value), np.float64(np.inf)))
    if not math.isfinite(result):
        raise ValueError('Numerical scale is too large for a finite roundoff bound.')
    return result


def _positive_sum_upper(values):
    try:result = math.fsum(float(value) for value in values)
    except OverflowError as error:raise ValueError('Roundoff envelope overflowed.') from error
    return _up(result)


def _gamma(operations):
    unit = float(np.finfo(np.float64).eps)/2
    product = operations*unit
    if not 0 <= product < .5:
        raise ValueError('The dimension exceeds the supported float64 error envelope.')
    return _up(product/(1-product))


def _roundoff_bound(x, gradient, costs):
    """Uniform conservative FP64 error for a selected B-entry cost sum.

    For each cost, L=sum_f|g_if|(|x_jf|+|x_if|) <= 2*max|x|*max_i sum_f|g_if|.
    gamma_(3F+4)*L bounds subtraction/product/sum rounding conservatively under
    IEEE binary64 round-to-nearest. Positive envelopes use fsum and upward
    nextafter. An extra tiny*sum|g| term propagates subtraction underflow;
    (2F+4)*tiny covers product/accumulation underflow, using the NORMAL tiny
    constant conservatively even for flushed subnormal intermediates. B such
    entry errors plus gamma_(B+2)*B*max|computed cost| bound the final sum.
    Identity has exactly zero differences. A negative sum beyond this bound
    certifies only a negative ideal linearized delta for the supplied arrays.
    """
    b, f = x.shape
    row_sums = [_positive_sum_upper(abs(float(value)) for value in row) for row in gradient]
    maximum_gradient_sum = max(row_sums)
    maximum_feature = float(np.abs(x).max())
    scale = _up(_up(2*maximum_feature)*maximum_gradient_sum)
    tiny = float(np.finfo(np.float64).tiny)
    feature_gamma, sum_gamma = _gamma(3*f+4), _gamma(b+2)
    underflow = _positive_sum_upper((_up(tiny*maximum_gradient_sum), _up((2*f+4)*tiny)))
    per_entry = _positive_sum_upper((_up(feature_gamma*scale), _up((1+feature_gamma)*underflow)))
    maximum_cost = float(np.abs(costs).max())
    cost_sum_error = _up(_up(sum_gamma*b)*maximum_cost)
    total = _positive_sum_upper((_up(b*per_entry), cost_sum_error, _up((b+2)*tiny)))
    return total, dict(feature_operations_bound=3*f+4, selected_sum_operations_bound=b+2,
        maximum_absolute_feature=maximum_feature, maximum_row_gradient_absolute_sum=maximum_gradient_sum,
        uniform_absolute_dot_envelope=scale, per_entry_roundoff_bound=per_entry,
        computed_selected_sum_roundoff_bound=cost_sum_error,
        assumptions='Finite binary64 arithmetic, round-to-nearest; conservative normal-tiny underflow allowance; no nonlinear-model guarantee')


def propose_assignment(features, gradient):
    """Minimize sum_i dot(g_i,x[p_i]-x_i), with safe identity fallback.

    The returned ``permutation`` is the raw proposal BEFORE common nonlinear
    acceptance. ``solver_permutation`` preserves the solver output even when a
    non-certifiable tiny improvement falls back to identity. Match the random
    cycle control to ``permutation``, not a later accepted/fallback permutation.
    """
    original = _matrix(features, 'Features')
    raw_gradient = _matrix(gradient, 'Gradient')
    if raw_gradient.shape != original.shape:
        raise ValueError('Feature and gradient shapes must match exactly.')
    x, g = original.astype(np.float64, copy=True), raw_gradient.astype(np.float64, copy=True)
    b, f = x.shape
    identity = np.arange(b, dtype=np.int64)
    trivial = ('one_row' if b == 1 else 'identical_feature_rows' if np.all(x == x[0])
               else 'zero_gradient' if np.all(g == 0) else None)
    if trivial is not None:
        return dict(permutation=identity, solver_permutation=identity.copy(), linear_delta=0.,
            solver_linear_delta=0., roundoff_bound=0., roundoff_details=None,
            linear_improvement_certified=False, reason=trivial, cycle_lengths=[1]*b,
            moved_positions=0, before_common_acceptance=True)
    costs = np.empty((b, b), dtype=np.float64)
    try:
        with np.errstate(over='raise', invalid='raise', under='ignore'):
            for i in range(b):costs[i] = (x-x[i])@g[i]
    except FloatingPointError as error:
        raise ValueError('Float64 cost construction overflowed or became invalid.') from error
    if not np.isfinite(costs).all():raise ValueError('Assignment costs must remain finite.')
    rows, columns = linear_sum_assignment(costs)
    solver = np.empty(b, dtype=np.int64);solver[rows] = columns
    solver = _permutation(solver, b)
    delta = math.fsum(float(costs[i, solver[i]]) for i in range(b))
    if not math.isfinite(delta):raise ValueError('Selected linear objective is not finite.')
    bound, details = _roundoff_bound(x, g, costs)
    certified = bool(delta < -bound)
    chosen = solver.copy() if certified else identity
    return dict(permutation=chosen, solver_permutation=solver, linear_delta=delta if certified else 0.,
        solver_linear_delta=delta, roundoff_bound=bound, roundoff_details=details,
        linear_improvement_certified=certified,
        reason='certified_linear_improvement' if certified else 'improvement_not_above_roundoff_bound',
        cycle_lengths=cycle_lengths(chosen), moved_positions=int(np.count_nonzero(chosen != identity)),
        before_common_acceptance=True,
        limitation='Optimality is for the computed linearized proxy cost only; tie behavior is tied to the frozen SciPy runtime, not a nonlinear optimum or cross-runtime identity promise')


def cycle_matched_random(primary_permutation, *, seed, group_key):
    """Seeded random relabeling conjugates the PRIMARY RAW proposal cycles.

    If r maps old labels to relabeled positions, q[r[i]]=r[p[i]]. This preserves
    the complete cycle lengths and moved count before common acceptance. The
    accepted control/primary permutations may move different numbers of rows.
    A conjugate can equal the primary, including singleton conjugacy classes;
    no outcome-dependent rejection sampling forces a difference.
    """
    p = _permutation(primary_permutation)
    if type(seed) is not int or seed < 0 or type(group_key) is not int or group_key < 0:
        raise ValueError('Seed and group key must be nonnegative Python integers.')
    key = json.dumps([seed, group_key, p.tolist(), CONTROL_NAMESPACE], separators=(',', ':')).encode('ascii')
    digest = hashlib.sha256(key).hexdigest()
    labels = list(range(len(p)));random.Random(int(digest, 16)).shuffle(labels)
    relabel = np.asarray(labels, dtype=np.int64)
    q = np.empty(len(p), dtype=np.int64);q[relabel] = relabel[p]
    q = _permutation(q, len(p))
    moved = int(np.count_nonzero(p != np.arange(len(p))))
    if cycle_lengths(q) != cycle_lengths(p) or int(np.count_nonzero(q != np.arange(len(q)))) != moved:
        raise AssertionError('Conjugation failed exact raw cycle/movement matching.')
    return dict(permutation=q, relabeling=relabel, key_sha256=digest, cycle_lengths=cycle_lengths(q),
        moved_positions=moved, before_common_acceptance=True,
        matches='Raw primary proposal cycles/moved count only; no final accepted-count or nonlinear-loss matching claim')


def accept_if_improved(teacher_baseline_logits, teacher_candidate_logits,
                       gcn_baseline_logits, gcn_candidate_logits):
    """Common OWN-teacher loss and pointwise original-GCN class0 guard."""
    arrays = [np.asarray(value) for value in (teacher_baseline_logits, teacher_candidate_logits,
                                             gcn_baseline_logits, gcn_candidate_logits)]
    if any(value.ndim != 2 or value.shape[1] != 2 or not len(value)
           or value.dtype.kind not in 'fiu' for value in arrays) or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError('Four parallel nonempty binary-logit matrices are required.')
    if not all(np.isfinite(value).all() for value in arrays):
        return dict(accepted=False, finite=False, reason='nonfinite_logits', tolerance=ACCEPTANCE_TOLERANCE,
            teacher_baseline_softplus=None, teacher_candidate_softplus=None, teacher_improvement=None,
            protected_GCN_count=None, lost_protected_GCN_count=None)
    base, candidate, gcn_base, gcn_candidate = [value.astype(np.float64, copy=False) for value in arrays]
    with np.errstate(over='ignore', invalid='ignore'):
        base_margin = base[:, 1]-base[:, 0];candidate_margin = candidate[:, 1]-candidate[:, 0]
        base_loss = float(np.logaddexp(0., base_margin).mean())
        candidate_loss = float(np.logaddexp(0., candidate_margin).mean())
    if not math.isfinite(base_loss) or not math.isfinite(candidate_loss):
        return dict(accepted=False, finite=False, reason='nonfinite_loss', tolerance=ACCEPTANCE_TOLERANCE,
            teacher_baseline_softplus=None, teacher_candidate_softplus=None, teacher_improvement=None,
            protected_GCN_count=None, lost_protected_GCN_count=None)
    protected = np.argmax(gcn_base, axis=1) == 0
    lost = protected & (np.argmax(gcn_candidate, axis=1) != 0)
    improvement = base_loss-candidate_loss
    accepted = bool(improvement > ACCEPTANCE_TOLERANCE and not np.any(lost))
    reason = ('accepted' if accepted else 'lost_protected_GCN_class0' if np.any(lost)
              else 'teacher_improvement_not_above_fixed_tolerance')
    return dict(accepted=accepted, finite=True, reason=reason, tolerance=ACCEPTANCE_TOLERANCE,
        teacher_baseline_softplus=base_loss, teacher_candidate_softplus=candidate_loss,
        teacher_improvement=improvement, protected_GCN_count=int(np.count_nonzero(protected)),
        lost_protected_GCN_count=int(np.count_nonzero(lost)))
