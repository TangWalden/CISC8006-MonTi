"""Independent saved-record/tensor audit of fixed-graph feature assignment.

Freeze this source before model execution. Run only after completion.
Allowed tensor deserialization is restricted to the new captured artifacts.
Original datasets and classifier/checkpoint weights are never deserialized;
there is no neural model construction, inference, gradient or optimizer call.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time
import traceback


ARMS = ('baseline', 'relation_assignment', 'gcn_assignment',
        'matched_random_permutation')
PROPOSALS = ARMS[1:]
SEEDS = (3, 4, 5)
FIELDS = ('occurrences', 'clean_wrong', 'attacked_wrong', 'new_wrong', 'recoveries')
EPS64 = 2.220446049250313e-16
ACCEPTANCE_EPSILON = 1e-7
RUNTIME_COMMIT = 'ca093d8b12ea153a076b02bbdbc4ede58508560c'
TEACHER_MANIFEST_SHA = 'de1bb484c201a2e60c7e2b63047587e8e1123ee465ba5eb72ddfe226ab60f111'
TEACHER_CHECKPOINT_SHA = '1952387acec4473bef7246aa0db5ef44f9971b727d89a44086af701989e4c3fd'
PREFLIGHT_CAPS = dict(generations=4,generator_completed_forwards=4,
    proposal_generation_GCN_forwards=0,native_canonical_updates=8,
    own_GAGA_float32_forwards=16,own_GCN_float32_forwards=8,
    own_GAGA_float64_forwards=12,own_GCN_float64_forwards=12,
    autograd_calls=16,GCN_block_topologies=4,classifier_float64_clones=2,
    official_GAGA_forwards=0,optimizer_updates=0,API_calls=0,TEST_queries=0)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compact_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'),
        ensure_ascii=True).encode('ascii')).hexdigest()


def write_json(path, value):
    with pathlib.Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


class Audit:
    def __init__(self, roots):
        self.roots = {key: pathlib.Path(value).resolve() for key, value in roots.items()}
        self.check_counts = collections.Counter()
        self.files = {}

    def check(self, value, label):
        self.check_counts[label] += 1
        if not bool(value):
            raise AssertionError('Independent assignment audit failed: ' + label)

    def file(self, path, expected=None, root=None):
        raw = pathlib.Path(path)
        resolved = raw.resolve() if raw.is_absolute() else (self.roots[root] / raw).resolve()
        self.check(any(base in resolved.parents for base in self.roots.values()),
            'registered_input_root_boundary')
        self.check(resolved.is_file(), 'registered_input_file_exists')
        actual = sha256(resolved)
        key = str(resolved)
        self.check(self.files.setdefault(key, actual) == actual, 'bytes_unchanged_during_audit')
        if expected is not None:
            self.check(actual == expected, 'actual_input_SHA256')
        return resolved

    def json(self, path, expected=None, root=None):
        return json.loads(self.file(path, expected, root).read_text(encoding='utf-8'))

    def rows(self, path, expected=None, root=None):
        with self.file(path, expected, root).open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]

    def close(self, actual, expected, label, atol=1e-10, rtol=1e-9):
        self.check(math.isfinite(actual) and math.isfinite(expected)
            and abs(actual - expected) <= atol + rtol * abs(expected), label)


def tensor_hash(value):
    tensor = value.detach().cpu().contiguous()
    header = json.dumps([str(tensor.dtype), list(tensor.shape)],
                        separators=(',', ':')).encode('ascii')
    return hashlib.sha256(len(header).to_bytes(8, 'big') + header
                          + tensor.numpy().tobytes()).hexdigest()


def random_fill_independently(adjacency, targets, candidates, injected, seed, group):
    """Replay the pre-existing strong random-target-fill rule, not its code."""
    t, c = len(targets), len(candidates)
    rows = [list(map(int, row)) for row in adjacency]
    transformed = [row[:] for row in rows]
    for i, row in enumerate(rows):
        donors = [node for j, node in enumerate(candidates) if row[t+j]]
        missing_t = [node for j, node in enumerate(targets) if not row[j]]
        missing_c = [node for j, node in enumerate(candidates) if not row[t+j]]
        count = min(len(donors), len(missing_t), len(missing_c))
        def key(purpose):
            value = [seed, str(group), injected[i], purpose, None,
                     'candidate-target-reallocation-v1']
            return int(compact_hash(value), 16)
        removed = random.Random(key('candidate-donors')).sample(sorted(donors), count)
        added = random.Random(key('random-target-addition')).sample(sorted(missing_t), count)
        for node in removed:
            transformed[i][t+candidates.index(node)] = 0
        for node in added:
            transformed[i][targets.index(node)] = 1
    return transformed


def as_logits(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value


def tensor_inventory(value, prefix=''):
    """Hash the actual saved tensors without importing producer hash helpers."""
    found = {}
    if hasattr(value, 'detach'):
        found[prefix] = tensor_hash(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.update(tensor_inventory(item, prefix+'.'+str(key) if prefix else str(key)))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            found.update(tensor_inventory(item, prefix+'.'+str(index)))
    return found


def multiset_hash(value):
    rows = value.detach().cpu().contiguous().numpy()
    row_hashes = sorted(hashlib.sha256(row.tobytes()).digest() for row in rows)
    return hashlib.sha256(b''.join(row_hashes)).hexdigest()


def permutation(audit, value, size, label):
    result = list(value)
    audit.check(len(result) == size and all(type(v) is int for v in result)
        and sorted(result) == list(range(size)), label)
    return result


def cycle_lengths(value):
    seen, lengths = set(), []
    for start in range(len(value)):
        if start in seen:
            continue
        cursor, length = start, 0
        while cursor not in seen:
            seen.add(cursor)
            length += 1
            cursor = value[cursor]
        if cursor != start:
            raise AssertionError('Invalid cycle decomposition.')
        lengths.append(length)
    return sorted(lengths)


def conjugate_permutation(value, relabel):
    """Independent cycle-preserving relabeling q p q^-1."""
    inverse = [0] * len(relabel)
    for old, new in enumerate(relabel):
        inverse[new] = old
    return [relabel[value[inverse[position]]] for position in range(len(value))]


def raw_classes(audit, values, expected_rows, label):
    audit.check(len(values) == expected_rows, label + '_complete_row_count')
    result = []
    for row in values:
        audit.check(len(row) == 2 and all(isinstance(v, (int, float))
            and math.isfinite(v) for v in row), label + '_finite_binary_logits')
        result.append(0 if row[0] >= row[1] else 1)
    return result


def softplus_mean(values):
    """Stable float64 scalar recount from raw logits; no tensor/model call."""
    terms = []
    for benign, fraud in values:
        margin = float(fraud) - float(benign)
        terms.append((margin + math.log1p(math.exp(-margin))) if margin >= 0
                     else math.log1p(math.exp(margin)))
    return math.fsum(terms) / len(terms)


def accept_proposal(audit, baseline, candidate, size):
    before = raw_classes(audit, as_logits(baseline['gcn']), size, 'baseline_own_GCN')
    after = raw_classes(audit, as_logits(candidate['gcn']), size, 'proposal_own_GCN')
    raw_classes(audit, as_logits(baseline['gaga']), size, 'baseline_own_GAGA')
    raw_classes(audit, as_logits(candidate['gaga']), size, 'proposal_own_GAGA')
    guard = all(c != 0 or z == 0 for c, z in zip(before, after))
    original_loss = softplus_mean(as_logits(baseline['gaga']))
    proposal_loss = softplus_mean(as_logits(candidate['gaga']))
    improves = proposal_loss < original_loss - ACCEPTANCE_EPSILON
    return dict(accepted=bool(improves and guard), gcn_class0_preserved=guard,
        own_gaga_loss_improved=improves, baseline_gaga_loss=original_loss,
        proposal_gaga_loss=proposal_loss, loss_delta=proposal_loss-original_loss)


def transitions(labels, clean, attacked):
    return dict(occurrences=len(labels),
        clean_wrong=sum(c != y for c, y in zip(clean, labels)),
        attacked_wrong=sum(z != y for z, y in zip(attacked, labels)),
        new_wrong=sum(c == y and z != y for c, z, y in zip(clean, attacked, labels)),
        recoveries=sum(c != y and z == y for c, z, y in zip(clean, attacked, labels)))


def independent_hungarian(np, costs):
    """Dense square primal/dual Hungarian solver, independent of SciPy.

    One-based augmenting paths with explicit row/column potentials. NumPy only
    performs vector arithmetic. No producer function or assignment solver is
    imported. Tie handling selects the first lowest remaining column.
    """
    n = costs.shape[0]
    if costs.shape != (n, n) or n == 0 or not np.isfinite(costs).all():
        raise ValueError('Finite nonempty square costs required.')
    u, v = np.zeros(n+1, dtype=np.float64), np.zeros(n+1, dtype=np.float64)
    matched = np.zeros(n+1, dtype=np.int64)
    way = np.zeros(n+1, dtype=np.int64)
    for row in range(1, n+1):
        matched[0] = row
        minimum = np.full(n+1, np.inf, dtype=np.float64)
        used = np.zeros(n+1, dtype=np.bool_)
        column = 0
        while True:
            used[column] = True
            row0 = int(matched[column])
            available = np.flatnonzero(~used[1:]) + 1
            reduced = costs[row0-1, available-1] - u[row0] - v[available]
            changed = reduced < minimum[available]
            destination = available[changed]
            minimum[destination] = reduced[changed]
            way[destination] = column
            next_column = int(available[int(np.argmin(minimum[available]))])
            delta = float(minimum[next_column])
            used_indices = np.flatnonzero(used)
            u[matched[used_indices]] += delta
            v[used_indices] -= delta
            minimum[~used] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while True:
            previous = int(way[column])
            matched[column] = matched[previous]
            column = previous
            if column == 0:
                break
    assignment = np.empty(n, dtype=np.int64)
    for column in range(1, n+1):
        assignment[matched[column]-1] = column-1
    return assignment.tolist(), u[1:], v[1:]


def assignment_certificate(audit, np, costs, proposed, construction_error=0.):
    """Re-solve independently and check dual feasibility and objective gap."""
    n = len(proposed)
    alternative, u, v = independent_hungarian(np, costs)
    magnitude = float(np.max(np.abs(costs)))
    # Conservative double-arithmetic bound, scaled to the actual costs; no
    # arbitrary unit-size floor that could conceal tiny assignment differences.
    tolerance = 128 * max(1, n*n) * EPS64 * magnitude + float(np.nextafter(0., 1.))
    slack = u[:, None] + v[None, :] - costs
    original = math.fsum(float(costs[i, i]) for i in range(n))
    proposed_cost = math.fsum(float(costs[i, proposed[i]]) for i in range(n))
    independent_cost = math.fsum(float(costs[i, alternative[i]]) for i in range(n))
    dual = math.fsum(map(float, u)) + math.fsum(map(float, v))
    total_tolerance = (n+2) * tolerance
    audit.check(float(np.max(slack)) <= tolerance, 'independent_Hungarian_dual_feasibility')
    audit.check(abs(independent_cost-dual) <= total_tolerance,
                'independent_Hungarian_primal_dual_equality')
    audit.check(abs(proposed_cost-independent_cost) <= total_tolerance+2*n*construction_error,
                'producer_permutation_is_linear_assignment_optimum')
    audit.check(proposed_cost <= original+total_tolerance+2*n*construction_error,
                'raw_assignment_linear_cost_not_above_identity')
    return dict(identity_cost=original, proposed_cost=proposed_cost,
        independent_optimum_cost=independent_cost, independent_dual_cost=dual,
        maximum_dual_violation=float(np.max(slack)), per_entry_tolerance=tolerance,
        objective_tolerance=total_tolerance+2*n*construction_error,
        producer_vs_independent_cost_entry_error_bound=construction_error,
        caveat='Optimality concerns the saved first-order assignment costs, not the nonlinear attack objective.')


def assignment_costs_and_error(np, x, gradient):
    """Independent FP64 sum ordering and conservative entry error envelope."""
    rows, width = x.shape
    values = np.empty((rows, rows), dtype=np.float64)
    absolute_sum = 0.
    for i in range(rows):
        differences = x-x[i]
        values[i] = np.einsum('jf,f->j', differences, gradient[i], optimize=False)
        absolute_sum = max(absolute_sum,
            float(np.max(np.einsum('jf,f->j', np.abs(differences),
                                  np.abs(gradient[i]), optimize=False))))
    unit = EPS64/2
    gamma = ((3*width+4)*unit)/(1-(3*width+4)*unit)
    normal_tiny = float(np.finfo(np.float64).tiny)
    gradient_sum = max(math.fsum(abs(float(v)) for v in row) for row in gradient)
    feature_scale = 2*float(np.max(np.abs(x)))*gradient_sum
    # This also covers the producer's subtraction-first matrix construction.
    error = 4*gamma*max(absolute_sum, feature_scale)
    error += (4*width+8)*normal_tiny*(1+gradient_sum)
    return values, float(np.nextafter(error, np.inf))


def assignment_recount(audit, np, x, gradient, metadata):
    rows = len(x)
    solver = permutation(audit, metadata['solver_permutation'], rows,
                         'raw_solver_permutation_bijection')
    chosen = permutation(audit, metadata['permutation'], rows,
                         'raw_roundoff_filtered_permutation_bijection')
    identity = list(range(rows))
    trivial = ('one_row' if rows == 1 else 'identical_feature_rows'
               if bool(np.all(x == x[0])) else 'zero_gradient'
               if bool(np.all(gradient == 0)) else None)
    audit.check(metadata['before_common_acceptance'] is True,
                'linear_proposal_precedes_common_nonlinear_acceptance')
    if trivial is not None:
        audit.check(solver == chosen == identity and metadata['reason'] == trivial
            and metadata['linear_delta'] == metadata['solver_linear_delta']
            == metadata['roundoff_bound'] == 0
            and metadata['roundoff_details'] is None
            and metadata['linear_improvement_certified'] is False,
            'independent_trivial_assignment_identity')
        certificate = dict(trivial=trivial, identity_cost=0., proposed_cost=0.)
    else:
        costs, error = assignment_costs_and_error(np, x, gradient)
        certificate = assignment_certificate(audit, np, costs, solver, error)
        independent_delta = math.fsum(float(costs[i, solver[i]]) for i in range(rows))
        audit.check(math.isfinite(metadata['solver_linear_delta'])
            and abs(metadata['solver_linear_delta']-independent_delta)
            <= rows*error+certificate['objective_tolerance'],
            'saved_solver_delta_agrees_with_independent_gradient_feature_costs')
        bound = metadata['roundoff_bound']
        details = metadata['roundoff_details']
        audit.check(math.isfinite(bound) and bound > 0
            and details['feature_operations_bound'] == 3*x.shape[1]+4
            and details['selected_sum_operations_bound'] == rows+2,
            'frozen_dimension_dependent_roundoff_envelope_present')
        # Recount the nonnegative envelope components from the saved supplied
        # arrays; these do not depend on validation outcomes or model replay.
        maximum_feature = float(np.max(np.abs(x)))
        gradient_sum = max(math.fsum(abs(float(v)) for v in row) for row in gradient)
        audit.close(details['maximum_absolute_feature'], maximum_feature,
                    'roundoff_maximum_feature', atol=0., rtol=1e-14)
        audit.close(details['maximum_row_gradient_absolute_sum'], gradient_sum,
                    'roundoff_gradient_absolute_sum', atol=float(np.finfo(np.float64).tiny), rtol=1e-12)
        unit = EPS64/2
        feature_gamma = ((3*x.shape[1]+4)*unit)/(1-(3*x.shape[1]+4)*unit)
        sum_gamma = ((rows+2)*unit)/(1-(rows+2)*unit)
        tiny = float(np.finfo(np.float64).tiny)
        per_entry = feature_gamma*(2*maximum_feature*gradient_sum)
        per_entry += (1+feature_gamma)*(tiny*gradient_sum+(2*x.shape[1]+4)*tiny)
        audit.check(details['per_entry_roundoff_bound'] >= per_entry*(1-1e-12),
                    'roundoff_entry_bound_covers_independent_forward_error')
        maximum_cost = float(np.max(np.abs(costs)))
        minimum_sum_error = sum_gamma*rows*max(0., maximum_cost-error)
        maximum_sum_error = sum_gamma*rows*(maximum_cost+error)
        audit.check(minimum_sum_error*(1-1e-12)
            <= details['computed_selected_sum_roundoff_bound']
            <= maximum_sum_error*(1+1e-12)+(rows+2)*tiny,
            'roundoff_selected_sum_bound_recount')
        reconstructed = rows*details['per_entry_roundoff_bound']
        reconstructed += details['computed_selected_sum_roundoff_bound']
        reconstructed += (rows+2)*float(np.finfo(np.float64).tiny)
        audit.close(bound, reconstructed, 'registered_roundoff_envelope_component_sum',
                    atol=(rows+2)*float(np.finfo(np.float64).tiny), rtol=1e-12)
        certified = metadata['solver_linear_delta'] < -bound
        audit.check(metadata['linear_improvement_certified'] == certified
            and chosen == (solver if certified else identity)
            and metadata['linear_delta'] == (metadata['solver_linear_delta'] if certified else 0.)
            and metadata['reason'] == ('certified_linear_improvement' if certified
                                      else 'improvement_not_above_roundoff_bound'),
            'registered_roundoff_identity_fallback_applied_without_outcome_selection')
    audit.check(metadata['cycle_lengths'] == cycle_lengths(chosen)
        and metadata['moved_positions'] == sum(i != j for i,j in enumerate(chosen)),
        'raw_assignment_cycle_signature_and_moved_indices')
    return chosen, certificate


def paired_interval(differences):
    if len(differences) != 3:
        raise ValueError('Exactly3registered generator seeds required.')
    mean, sd = statistics.fmean(differences), statistics.stdev(differences)
    critical = math.sqrt(2) * .95 / math.sqrt(1-.95**2)
    half = critical * sd / math.sqrt(3)
    return dict(seed_differences_pp=differences, mean_difference_pp=mean,
        sample_sd_pp=sd, paired_t95_pp=[mean-half, mean+half], attack_seeds=3, df=2)


def cohort_recount(audit, protocol):
    selection = protocol['selection']
    population = selection['original_population']
    exclusions = selection['excluded_ordered_cohorts']
    audit.check(len(population) == len(set(population)) == 592
        and all(type(v) is int and v >= 0 for v in population),
        'saved_original_VALID_population592_exact_unique_IDs')
    audit.check(len(exclusions) == 3
        and all(len(v) == len(set(v)) == 128 for v in exclusions),
        'three_exact_ordered_prior128_context_lists')
    excluded = {v for row in exclusions for v in row}
    audit.check(len(excluded) == 384 and excluded <= set(population),
        'three_disjoint_prior_context_lists_not_news_disjointness')
    eligible = sorted(set(population)-excluded)
    selected = random.Random(20261111).sample(eligible, 128)
    audit.check(selection['seed'] == 20261111 and len(eligible) == 208
        and selection['eligible_sorted'] == eligible
        and selection['selected_ordered'] == selected == protocol['valid_groups']
        and selection['no_size_label_outcome_selection'] is True
        and selection['original_lists_unchanged'] is True,
        'independent_ID_only_fourth_context_draw_replay')
    cohort = protocol['cohort']
    members = {int(g): list(v) for g,v in cohort['members'].items()}
    audit.check(set(members) == set(selected) and all(v for v in members.values())
        and all(len(v) == len(set(v)) and v == sorted(v)
                and all(type(node) is int and node >= 0 for node in v)
                for v in members.values()), 'complete_unchanged_selected_target_membership_lists')
    union = {v for row in members.values() for v in row}
    occurrences = sum(map(len, members.values()))
    audit.check(cohort['groups'] == 128 and cohort['target_occurrences'] == occurrences
        and cohort['unique_targets'] == len(union), 'full_new_cohort_occurrence_and_identity_denominators')
    previous = set(selection['excluded_union_target_ids'])
    audit.check(len(previous) == len(selection['excluded_union_target_ids'])
        and selection['excluded_union_target_ids'] == sorted(previous)
        and cohort['overlap_with_previous_three_cohorts'] == len(union & previous)
        and cohort['new_unique_relative_to_previous_contexts'] == len(union-previous),
        'saved_ID_union_overlap_recount_not_an_original_dataset_replay')
    multiplicities = collections.Counter(v for row in members.values() for v in row)
    return selected, members, dict(groups_per_seed=128,
        target_occurrences_per_seed=occurrences, unique_targets_per_seed=len(union),
        correlated_occurrences_all_seeds=3*occurrences,
        target_identities_repeated_across_groups=sum(v > 1 for v in multiplicities.values()),
        maximum_group_memberships=max(multiplicities.values()),
        overlap_with_previous_three_cohorts=len(union & previous),
        new_unique_relative_to_previous_contexts=len(union-previous),
        original_population_membership_verified_from_saved_metadata_only=True)


def raw_pair_counts(labels, clean, before, after):
    result = collections.Counter()
    delta = sum(z != y for z,y in zip(after,labels))-sum(z != y for z,y in zip(before,labels))
    result['group_wins' if delta > 0 else 'group_losses' if delta < 0 else 'group_ties'] = 1
    for y,c,b,z in zip(labels, clean, before, after):
        result['occurrences'] += 1
        result['gained_wrong'] += b == y and z != y
        result['lost_wrong'] += b != y and z == y
        if c == y:
            result['new_success_gained'] += b == y and z != y
            result['baseline_new_success_lost'] += b != y and z == y
        else:
            result['additional_recovery'] += b != y and z == y
            result['previous_recovery_lost'] += b == y and z != y
    return result


def metric_summary(audit, selected, members, labels, clean_classes, attack_classes):
    """Full-cohort metrics from caller-normalized saved classes only."""
    occurrence_count = sum(map(len, members.values()))
    identities = len({v for row in members.values() for v in row})
    metrics = []
    lookup = {}
    for seed in SEEDS:
        for arm in ARMS:
            for detector in ('gaga', 'gcn'):
                totals = collections.Counter()
                macro = []
                for group in selected:
                    count = transitions(labels[group], clean_classes[group,detector],
                                        attack_classes[seed,group,arm,detector])
                    totals.update(count)
                    macro.append(100*count['attacked_wrong']/count['occurrences'])
                row = dict(seed=seed, arm=arm, detector=detector,
                    **{field:totals[field] for field in FIELDS},
                    mr_percent=100*totals['attacked_wrong']/occurrence_count,
                    group_macro_mr_percent=statistics.fmean(macro), unique_targets=identities)
                audit.check(row['occurrences'] == occurrence_count
                    and row['attacked_wrong'] == row['clean_wrong']+row['new_wrong']-row['recoveries'],
                    'allcase_clean_new_recovery_transition_identity')
                metrics.append(row)
                lookup[seed,arm,detector] = row
    contrasts = {}
    pairs = [(arm,'baseline') for arm in PROPOSALS]
    pairs += [('relation_assignment', arm) for arm in ('gcn_assignment','matched_random_permutation')]
    for candidate, control in pairs:
        for detector in ('gaga','gcn'):
            values = [lookup[s,candidate,detector]['mr_percent']
                      -lookup[s,control,detector]['mr_percent'] for s in SEEDS]
            result = paired_interval(values)
            counts = collections.Counter()
            for seed in SEEDS:
                for group in selected:
                    counts.update(raw_pair_counts(labels[group], clean_classes[group,detector],
                        attack_classes[seed,group,control,detector],
                        attack_classes[seed,group,candidate,detector]))
            audit.check(counts['occurrences'] == 3*occurrence_count
                and sum(counts[key] for key in ('group_wins','group_ties','group_losses')) == 384,
                'all384_group_seed_pairs_no_case_filtering')
            result['paired_counts'] = dict(counts)
            contrasts[candidate+'-minus-'+control+'_'+detector] = result
    primary = contrasts['relation_assignment-minus-baseline_gaga']
    controls = {arm:contrasts['relation_assignment-minus-'+arm+'_gaga']['mean_difference_pp']
                for arm in ('gcn_assignment','matched_random_permutation')}
    criterion = dict(primary='relation_assignment',
        mean_gain_pp=primary['mean_difference_pp'],
        all_three_seed_deltas_positive=all(v > 0 for v in primary['seed_differences_pp']),
        nominal_paired_t95_pp=primary['paired_t95_pp'],
        mean_advantages_over_controls_pp=controls,
        control_paired_t95_pp={arm:contrasts['relation_assignment-minus-'+arm+'_gaga']['paired_t95_pp']
                              for arm in ('gcn_assignment','matched_random_permutation')},
        criterion_met=bool(primary['mean_difference_pp'] >= 1
            and all(v > 0 for v in primary['seed_differences_pp'])
            and primary['paired_t95_pp'][0] > 0 and all(v > 0 for v in controls.values())),
        controls_promoted=False, final_TEST_allowed=False,established_control_superiority_claimed=False,
        uncertainty='Conditional3generator-seed/df2 interval; fixed teachers and repeated/adaptively inspected development contexts, not IID-news or confirmatory uncertainty.')
    return metrics, contrasts, criterion


def tensor_case(audit, np, torch, protocol, record, payload, members):
    required = {'capture','random_fill_adjacency','gradients','raw_permutations',
        'final_permutations','raw_features','features','raw_canonical_sequences',
        'canonical_sequences','own_logits_raw','own_logits_final'}
    audit.check(set(payload) == required, 'complete_captured_tensor_case_schema')
    inventory = tensor_inventory(payload)
    audit.check(inventory == record['tensor_hashes'], 'all_actual_captured_tensor_hashes_bound')
    seed, group = record['seed'], record['group']
    cap = payload['capture']
    audit.check(set(cap) == {'x_atk','adj_attack','target_ids','candidate_ids'},
                'original_generated_capture_four_tensor_schema')
    x0, old_adj, adjacency = cap['x_atk'], cap['adj_attack'], payload['random_fill_adjacency']
    targets, candidates = cap['target_ids'].tolist(), cap['candidate_ids'].tolist()
    n, width = x0.shape
    t, c = len(targets), len(candidates)
    original = record['original_node_count']
    audit.check(x0.dtype == old_adj.dtype == adjacency.dtype == torch.float32
        and width == 768 and n > 0 and tuple(old_adj.shape) == tuple(adjacency.shape) == (n,t+c+n)
        and cap['target_ids'].dtype == cap['candidate_ids'].dtype == torch.int64
        and bool(torch.isfinite(x0).all()) and bool(torch.isfinite(old_adj).all())
        and bool(torch.isfinite(adjacency).all()), 'finite_fixed_dtype_generated_features_and_owned_graph')
    audit.check(targets == members[group] == record['node_ids']
        and len(record['labels']) == t and all(v == 1 for v in record['labels'])
        and len(set(targets+candidates)) == t+c
        and all(type(v) is int and 0 <= v < original for v in targets+candidates),
        'complete_known_fraud_target_lists_and_original_ID_order')
    audit.check(bool(((old_adj == 0)|(old_adj == 1)).all())
        and bool(((adjacency == 0)|(adjacency == 1)).all()), 'exact_binary_owned_adjacencies')
    legal = torch.tril(torch.ones_like(adjacency,dtype=torch.bool), diagonal=t+c-1)
    audit.check(not bool((old_adj[~legal] != 0).any())
        and not bool((adjacency[~legal] != 0).any())
        and bool((old_adj[:,:t].sum(1) >= 1).all()),
        'strict_undirected_injection_ownership_and_mandatory_old_target_edge')
    rebuilt = random_fill_independently(old_adj.to(torch.int64).tolist(), targets,
        candidates, list(range(original,original+n)), seed, group)
    audit.check(adjacency.to(torch.int64).tolist() == rebuilt,
                'independent_strong_random_target_fill_reconstruction')
    audit.check(tensor_hash(adjacency[:,t+c:]) == tensor_hash(old_adj[:,t+c:])
        and torch.equal(adjacency.sum(1),old_adj.sum(1))
        and not bool(((old_adj[:,:t] == 1)&(adjacency[:,:t] != 1)).any()),
        'strong_baseline_retains_old_targets_injected_block_and_each_row_cost')
    matrix = adjacency.to(torch.int64).tolist()
    degrees = [sum(row)+sum(matrix[j][t+c+i] for j in range(i+1,n))
               for i,row in enumerate(matrix)]
    audit.check(degrees == record['physical_injected_degrees']
        and n == record['node_budget'] == record['actual_injected_nodes']
        and int(adjacency.sum()) == record['edge_budget'] == record['actual_injected_edges'],
        'actual_node_edge_and_physical_injected_degree_costs')
    identity = list(range(n))
    features64 = x0.numpy().astype(np.float64)
    audit.check(set(payload['gradients']) == {'gaga','gcn'}
        and set(payload['raw_permutations']) == set(PROPOSALS)
        and set(payload['final_permutations']) == set(ARMS)
        and all(set(payload[name]) == set(ARMS) for name in
                ('raw_features','features','raw_canonical_sequences','canonical_sequences',
                 'own_logits_raw','own_logits_final')),
        'all_four_arms_and_two_gradient_channels_retained')
    certificates = {}
    proposals = {}
    for arm, role in (('relation_assignment','gaga'),('gcn_assignment','gcn')):
        gradient = payload['gradients'][role]
        audit.check(gradient.dtype == torch.float32 and tuple(gradient.shape) == tuple(x0.shape)
            and bool(torch.isfinite(gradient).all())
            and tensor_hash(gradient) == record['gradients_hashes'][role],
            'finite_actual_saved_baseline_feature_gradient')
        proposed, certificate = assignment_recount(audit, np, features64,
            gradient.numpy().astype(np.float64), record['proposal_metadata'][arm])
        proposals[arm] = proposed
        certificates[arm] = certificate
    random_meta = record['proposal_metadata']['matched_random_permutation']
    digest = compact_hash([seed,group,proposals['relation_assignment'],
                           'feature-assignment-cycle-control-v1'])
    relabel = list(range(n))
    random.Random(int(digest,16)).shuffle(relabel)
    random_perm = conjugate_permutation(proposals['relation_assignment'],relabel)
    audit.check(random_meta['key_sha256'] == digest
        and random_meta['relabeling'] == relabel
        and random_meta['permutation'] == random_perm
        and random_meta['cycle_lengths'] == cycle_lengths(random_perm)
        == cycle_lengths(proposals['relation_assignment'])
        and random_meta['moved_positions'] == sum(i != j for i,j in enumerate(random_perm))
        and random_meta['before_common_acceptance'] is True,
        'independent_SHA_keyed_raw_primary_cycle_conjugation')
    proposals['matched_random_permutation'] = random_perm
    baseline_pair = payload['own_logits_raw']['baseline']
    accepted = {}
    losses = {}
    finite_boundary = []
    raw_perm = {'baseline':identity}
    final_perm = {'baseline':identity}
    for arm in PROPOSALS:
        raw_perm[arm] = proposals[arm]
        audit.check(payload['raw_permutations'][arm].tolist() == proposals[arm],
                    'actual_raw_permutation_equals_independent_proposal')
        decision = accept_proposal(audit, baseline_pair, payload['own_logits_raw'][arm], t)
        saved = record['acceptance'][arm]
        # Exact source-contract arithmetic is checked independently with NumPy;
        # scalar fsum recount above additionally bounds reduction-order effects.
        z0 = np.asarray(as_logits(baseline_pair['gaga']),dtype=np.float64)
        z1 = np.asarray(as_logits(payload['own_logits_raw'][arm]['gaga']),dtype=np.float64)
        l0 = float(np.logaddexp(0.,z0[:,1]-z0[:,0]).mean())
        l1 = float(np.logaddexp(0.,z1[:,1]-z1[:,0]).mean())
        protected = [v == 0 for v in raw_classes(audit,as_logits(baseline_pair['gcn']),t,'cached_own_guard_baseline')]
        candidate_class = raw_classes(audit,as_logits(payload['own_logits_raw'][arm]['gcn']),t,'cached_own_guard_candidate')
        lost = sum(flag and value != 0 for flag,value in zip(protected,candidate_class))
        use = (l0-l1 > ACCEPTANCE_EPSILON) and lost == 0
        expected_reason = ('accepted' if use else 'lost_protected_GCN_class0' if lost
                           else 'teacher_improvement_not_above_fixed_tolerance')
        audit.check(saved['accepted'] == use and saved['finite'] is True
            and saved['reason'] == expected_reason and saved['tolerance'] == ACCEPTANCE_EPSILON
            and saved['protected_GCN_count'] == sum(protected)
            and saved['lost_protected_GCN_count'] == lost,
            'common_native_own_logits_acceptance_and_pointwise_GCN_guard')
        for key,value in (('teacher_baseline_softplus',l0),('teacher_candidate_softplus',l1),('teacher_improvement',l0-l1)):
            audit.close(saved[key],value,'common_filter_saved_float64_loss_recount')
        audit.close(decision['baseline_gaga_loss'],l0,'independent_scalar_baseline_softplus_recount')
        audit.close(decision['proposal_gaga_loss'],l1,'independent_scalar_candidate_softplus_recount')
        rounding = 64*max(1,t)*EPS64*(abs(l0)+abs(l1)+float(np.finfo(np.float64).tiny))
        if decision['accepted'] != use:
            audit.check(abs((l0-l1)-ACCEPTANCE_EPSILON) <= rounding,
                        'filter_predicate_difference_only_at_declared_float64_reduction_boundary')
            finite_boundary.append(arm)
        accepted[arm] = bool(use)
        final_perm[arm] = proposals[arm] if use else identity
        losses[arm] = dict(baseline=l0,raw_candidate=l1,improvement=l0-l1,
                          protected_GCN_count=sum(protected),lost_protected_GCN_count=lost)
    base_multiset = multiset_hash(x0)
    base_sequence = payload['raw_canonical_sequences']['baseline']
    features_by_key = {}
    cache_expected = []
    changed = {}
    for arm in ARMS:
        final = payload['final_permutations'][arm].tolist()
        audit.check(final == final_perm[arm], 'whole_case_identity_fallback_or_accepted_permutation')
        for prefix, mapping, invariant_key in (('raw_',raw_perm,'raw_invariants'),('',final_perm,'final_invariants')):
            feature = payload[prefix+'features'][arm]
            sequence = payload[prefix+'canonical_sequences'][arm]
            expected = x0.index_select(0,torch.tensor(mapping[arm],dtype=torch.long))
            audit.check(feature.dtype == torch.float32 and tuple(feature.shape) == tuple(x0.shape)
                and tensor_hash(feature) == tensor_hash(expected)
                and multiset_hash(feature) == base_multiset,
                'exact_original_feature_row_bytes_and_whole_multiset_preserved')
            audit.check(sequence.dtype == torch.float32 and tuple(sequence.shape) == (t,7,768)
                and bool(torch.isfinite(sequence).all())
                and tensor_hash(sequence[:,0]) == tensor_hash(base_sequence[:,0]),
                'finite_native_canonical_shape_and_original_SELF_token_invariance')
            sequence_meta = 'raw_sequence_hashes' if prefix else 'final_sequence_hashes'
            audit.check(tensor_hash(sequence) == record[sequence_meta][arm],
                        'actual_native_raw_and_final_sequence_hashes')
            receipt = record[invariant_key][arm]
            moved = sum(i != j for i,j in enumerate(mapping[arm]))
            feature_changed = int((feature != x0).any(1).sum())
            audit.check(receipt['multiset_sha256'] == base_multiset
                and receipt['feature_sha256'] == tensor_hash(feature)
                and receipt['moved_positions'] == moved
                and receipt['changed_feature_rows'] == feature_changed,
                'index_movement_distinguished_from_changed_duplicate_feature_values')
            audit.close(receipt['raw_displacement_linf'],float((feature-x0).abs().max()),'actual_feature_Linf_displacement')
            audit.close(receipt['raw_displacement_l2'],float((feature-x0).double().norm()),'actual_feature_L2_displacement')
        key = tensor_hash(payload['raw_features'][arm])
        pair = payload['own_logits_raw'][arm]
        audit.check(set(pair) == {'gaga','gcn'},'parallel_own_raw_model_channels')
        for role in ('gaga','gcn'):
            audit.check(as_logits(pair[role]) == record['own_logits_raw'][arm][role],
                        'saved_raw_own_logits_tensor_JSON_identity')
            expected_pair = pair if arm == 'baseline' or accepted[arm] else baseline_pair
            audit.check(tensor_hash(payload['own_logits_final'][arm][role]) == tensor_hash(expected_pair[role])
                and as_logits(payload['own_logits_final'][arm][role]) == record['own_logits_final'][arm][role],
                'final_own_logits_follow_actual_common_acceptance')
        raw_seq_hash = tensor_hash(payload['raw_canonical_sequences'][arm])
        if arm != 'baseline':
            cache_expected.append(dict(arm=arm,feature_sha256=key,reused=key in features_by_key))
        if key in features_by_key:
            previous = features_by_key[key]
            audit.check(raw_seq_hash == previous['sequence']
                and all(tensor_hash(pair[role]) == previous[role] for role in ('gaga','gcn')),
                'cached_own_logits_bind_to_identical_actual_native_input')
        else:
            features_by_key[key] = dict(sequence=raw_seq_hash,
                **{role:tensor_hash(pair[role]) for role in ('gaga','gcn')})
        accepted_arm = arm == 'baseline' or accepted[arm]
        expected_sequence = payload['raw_canonical_sequences'][arm] if accepted_arm else base_sequence
        audit.check(tensor_hash(payload['canonical_sequences'][arm]) == tensor_hash(expected_sequence),
                    'final_sealed_canonical_sequence_follows_common_acceptance')
        changed[arm] = dict(raw_moved_positions=sum(i != j for i,j in enumerate(raw_perm[arm])),
            final_moved_positions=sum(i != j for i,j in enumerate(final_perm[arm])),
            final_changed_feature_rows=int((payload['features'][arm] != x0).any(1).sum()),
            final_canonical_changed_coordinates=int((payload['canonical_sequences'][arm] != base_sequence).sum()))
    audit.check(record['cache_receipts'] == cache_expected
        and record['logical_own_forward_requests'] == 8 and record['logical_autograd_calls'] == 2,
        'actual_duplicate_cache_receipts_and_eight_logical_own_requests')
    physical = len(features_by_key)
    return dict(seed=seed,group=group,labels=record['labels'],targets=t,
        node_budget=n,edge_budget=int(adjacency.sum()),physical_degrees=degrees,
        accepted=accepted,losses=losses,changes=changed,certificates=certificates,
        own_physical_forwards_per_role=physical,own_cache_hits=2*(4-physical),
        native_input_unique_features=physical,acceptance_float64_boundary_arms=finite_boundary,
        final_own_classes={(arm,role):raw_classes(audit,
            as_logits(payload['own_logits_final'][arm][role]),t,'final_cached_own_logits')
            for arm in ARMS for role in ('gaga','gcn')})


def preflight_recount(audit, protocol, gate, provenance):
    audit.check(gate['status'] == 'passed' and gate['provenance'] == provenance
        and gate['official_victim_loaded'] is False
        and gate['no_VALID_or_TEST_queries'] is True,
        'same_source_protocol_TRAIN_only_passed_preflight')
    settings = protocol['preflight']
    audit.check(settings == dict(groups=[263,238,84,125],sequence_atol=1e-5,
        logit_atol=1e-4,FD_epsilon=1e-4,FD_atol=1e-6,FD_rtol=5e-3)
        and set(settings['groups']) <= set(protocol['train_groups'])
        and not set(settings['groups']) & set(protocol['valid_groups']),
        'fixed_preflight_groups_and_unrelaxed_numerical_tolerances')
    expected = dict(PREFLIGHT_CAPS,dataset_loads=1,generator_models_loaded=1,
                    own_GAGA_models_loaded=1,proposal_GCN_models_loaded=1)
    audit.check(gate['counter_caps'] == PREFLIGHT_CAPS and gate['counters'] == expected,
                'all_preflight_model_derivative_and_zero_victim_counters')
    audit.check(gate['immutable_content_before'] == gate['immutable_content_after']
        and gate['model_states_before'] == gate['model_states_after']
        and gate['actual_teacher_manifest_sha256'] == TEACHER_MANIFEST_SHA,
        'preflight_classifier_and_original_feature_immutability_receipts')
    audit.check(len(gate['cases']) == 4 and [v['group'] for v in gate['cases']] == settings['groups'],
                'all_four_predeclared_TRAIN_cases_no_substitution')
    informative = collections.Counter()
    for case in gate['cases']:
        audit.check(case['status'] == 'passed' and case['feature_width'] == 768
            and max(case['sequence_absolute_errors']) <= settings['sequence_atol']
            and max(case['logit_absolute_errors']) <= settings['logit_atol'],
            'recorded_native_affine_sequence_logit_parity')
        audit.check(set(case['derivatives']) == {'gaga','gcn'},'both_frozen_model_feature_derivative_checks')
        for role, value in case['derivatives'].items():
            auto32, auto64 = value['float32_directional_autograd'], value['float64_directional_autograd']
            estimate = (value['plus_float64_loss']-value['minus_float64_loss'])/(2e-4)
            tolerance = 1e-6+5e-3*abs(auto64)
            error = abs(auto64-estimate)
            precision_error = abs(auto32-auto64)
            audit.close(value['central_float64_difference'],estimate,'preflight_raw_plus_minus_FD_recount')
            audit.close(value['finite_difference_absolute_error'],error,'preflight_FD_error_recount')
            audit.close(value['float32_float64_derivative_absolute_error'],precision_error,'preflight_derivative_precision_error_recount')
            audit.close(value['acceptance_tolerance'],tolerance,'preflight_original_absolute_relative_tolerance')
            audit.check(value['role'] == role and value['gradient_point'] == 'baseline_x0'
                and value['epsilon'] == 1e-4 and value['original_atol'] == 1e-6
                and value['original_rtol'] == 5e-3 and error <= tolerance
                and precision_error <= tolerance and value['finite_difference_passed'] is True
                and value['float32_float64_derivative_passed'] is True
                and value['zero_directional_derivative'] == (auto64 == 0.)
                and value['direction_informative'] == (abs(auto64) > 1e-6),
                'preflight_actual_softplus_derivative_flags_no_changed_objective')
            key = compact_hash([3,case['group'],role,'assignment-preflight-direction-v1'])
            audit.check(value['direction_key_sha256'] == key,
                        'fixed_TRAIN_numerical_direction_key')
            informative[role+'_informative_directions'] += value['direction_informative']
            informative[role+'_zero_directions'] += value['zero_directional_derivative']
    return dict(counters=expected,direction_counts=dict(informative),
        process_cpu_seconds=gate['process_cpu_seconds'],wall_seconds=gate['wall_seconds'],
        scope='Independent saved numerical recount only; no derivative/model replay.')


def chronology_and_guardians(audit, protocol, stage, main, seal, provenance):
    registered = audit.json(stage/'main_attempt1'/'registered.json')
    state = audit.json(stage/'phase_state.json')
    audit.check(state['provenance'] == provenance and state['automatic_retry'] is False
        and len(state['rows']) == 2 and [v['phase'] for v in state['rows']] == ['preflight','run'],
        'single_preflight_and_main_dispatch_no_automatic_retry')
    def parsed(value):
        timestamp = datetime.datetime.fromisoformat(value)
        audit.check(timestamp.tzinfo is not None,'recorded_UTC_timestamp_has_timezone')
        return timestamp
    last_end = parsed(state['phase_started_utc'])
    for row, phase, expected_status in zip(state['rows'],('preflight','run'),('passed','completed')):
        guardian = audit.json(stage/(phase+'_guardian.json'))
        child = stage/('preflight_attempt1' if phase == 'preflight' else 'main_attempt1')/'result.json'
        child_data = audit.json(child,row['child_result_sha256'])
        audit.check(guardian == row and row['status'] == 'completed' and row['exit_code'] == 0
            and child_data['status'] == expected_status and child_data['provenance'] == provenance
            and row['selected_device'] == protocol['device']
            and row['protocol_sha256'] == provenance['protocol_sha256'],
            'guardian_result_hash_and_phase_source_device_binding')
        inner = protocol['resource']['preflight_inner_seconds' if phase == 'preflight' else 'main_inner_seconds']
        audit.check(row['inner_seconds'] == inner
            and row['outer_seconds'] == inner+60 and row['outer_wall_seconds'] >= child_data['wall_seconds']
            and child_data['wall_seconds'] <= inner+15
            and row['outer_wall_seconds'] <= row['outer_seconds']+30,
            'actual_guardian_child_clocks_and_frozen_phase_deadlines')
        begin, end = parsed(row['started_utc']), parsed(row['finished_utc'])
        audit.check(last_end <= begin <= end,'sequential_receipt_chronology_including_review_wait')
        audit.close(row['whole_phase_wall_seconds_including_review_wait'],
            (end-parsed(state['phase_started_utc'])).total_seconds(),
            'whole_phase_wall_receipt_includes_preflight_and_review_gap',atol=1e-4)
        expected_hours = row['whole_phase_wall_seconds_including_review_wait']/3600 if protocol['device']['kind'] == 'gpu' else 0
        audit.close(row['whole_phase_GPU_hours'],expected_hours,'whole_phase_GPU_charge_receipt')
        last_end = end
    resource_key = 'GPU_phase_seconds' if protocol['device']['kind'] == 'gpu' else 'CPU_phase_seconds'
    phase_wall = state['rows'][-1]['whole_phase_wall_seconds_including_review_wait']
    audit.check(phase_wall <= protocol['resource'][resource_key], 'whole_phase_cap_includes_all_wait_and_both_jobs')
    audit.check(registered['provenance'] == provenance
        and registered['valid_groups'] == protocol['valid_groups']
        and registered['arms'] == list(ARMS)
        and registered['native_inputs_seal_before_official_victim'] is True
        and registered['cached_GCN_attacked_channel'] is True,
        'registered_native_seal_and_declared_cached_secondary_channel')
    audit.file(registered['preflight']['path'],registered['preflight']['sha256'])
    audit.check(seal['status'] == 'sealed' and seal['provenance'] == provenance
        and seal['official_GAGA_loaded'] is False and seal['official_GAGA_forwards'] == 0
        and seal['cases'] == 384 and seal['final_canonical_sequences'] == 1536
        and main['geometry_sealed_before_victim_load'] is True
        and parsed(seal['completed_utc']) < parsed(main['official_victim_load_utc'])
        <= parsed(main['completed_utc']) <= parsed(main['finished_utc']),
        'all_final_inputs_sealed_before_recorded_official_victim_loading')
    return dict(phase_wall_seconds=phase_wall,
        charged_GPU_hours=state['rows'][-1]['whole_phase_GPU_hours'],
        no_double_sum_of_cumulative_guardian_charges=True,
        scope='Source-bound saved clocks/dispatch records, not independent process attestation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--audit-source-sha256',required=True)
    args = parser.parse_args()
    protocol_path = pathlib.Path(args.protocol).resolve()
    stage = pathlib.Path(args.stage).resolve()
    output = pathlib.Path(args.output).resolve()
    if output.exists():
        raise RuntimeError('Independent audit output must be a new directory.')
    output.mkdir(parents=True,exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    audit = None
    operations = dict(captured_tensor_loads=0,original_dataset_deserializations=0,
        original_partition_array_loads=0,classifier_weight_deserializations=0,
        model_constructions=0,model_forwards=0,autograd_calls=0,optimizer_updates=0,
        API_calls=0,TEST_prediction_decodes=0,GPU_hours=0)
    result = dict(schema='independent_feature_assignment_saved_tensor_audit_v1',status='failed',
        independent_of_method_implementation_authorship=True,human_verification=False,
        saved_record_recount=True,neural_model_replay=False)
    try:
        if sha256(__file__) != args.audit_source_sha256:
            raise RuntimeError('Auditor differs from its externally frozen source hash.')
        if sha256(protocol_path) != args.protocol_sha256:
            raise RuntimeError('Protocol differs from its externally frozen hash.')
        write_json(output/'registered_scope.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            audit_source_sha256=args.audit_source_sha256,protocol_sha256=args.protocol_sha256,
            scientific_records_opened=False,
            allowed='Hash-bound source/metadata/guardian JSON; opaque original input/checkpoint byte hashes; exactly384new captured tensor artifacts; new own/official VALID logits.',
            forbidden='Original graph/partition/weight deserialization, model construction or forward/backward/optimizer, APIs, TEST prediction decoding/queries or visible CUDA.',
            distinction='Input byte hashing is not dataset/model deserialization; saved derivative/parity receipts are recounted, never replayed.'))
        p = json.loads(protocol_path.read_text(encoding='utf-8'))
        roots = dict(source=p['source_root'],runtime=p['runtime_root'],asset=p['asset_root'],
                     stage=stage,registration=protocol_path.parent)
        audit = Audit(roots)
        audit.file(protocol_path,args.protocol_sha256)
        audit.check((pathlib.Path(p['asset_root']).resolve()/'runs') in stage.parents
            and (pathlib.Path(p['asset_root']).resolve()/'runs') in output.parents,
            'private_stage_and_fresh_audit_output_below_ignored_runs')
        audit.check(p['schema'] == 'monti_feature_assignment_validation_v1'
            and p['status'] == 'frozen' and p['owner_execution_authorized'] is True
            and p['post_hoc'] is True and p['final_TEST_allowed'] is False
            and p['runtime_commit'] == RUNTIME_COMMIT and tuple(p['arms']) == ARMS
            and tuple(p['seeds']) == SEEDS,
            'fixed_owner_authorized_posthoc_four_arm_no_TEST_protocol')
        audit.check(p['settings'] == dict(acceptance_improvement=1e-7,iterations=1,
            loss='uniform_mean_softplus_fraud_minus_benign',gradient_point='baseline_x0'),
            'one_shot_uniform_objective_and_fixed_common_acceptance')
        audit.check(p['qualification'] == dict(primary='relation_assignment',mean_gain_at_least_pp=1,
            all_three_seed_deltas_positive=True,paired_t95_lower_positive=True,
            mean_above_both_controls=True,controls_not_candidates=True),
            'primary_qualification_requires_both_controls_no_promotion')
        audit.check(p['logical_main_caps'] == dict(generations=384,own_forward_requests=3072,
            autograd_calls=768,official_GAGA_forwards=1664,clean_GCN_forwards=128,
            final_canonical_sequences=1536,optimizer_updates=0,API_calls=0,TEST_queries=0),
            'fixed_logical_main_call_and_zero_training_TEST_API_caps')
        resource = p['resource']
        device_kind = p['device']['kind']
        audit.check(device_kind in ('cpu','gpu') and resource == dict(GPU_phase_seconds=7200,
            CPU_phase_seconds=21600,main_inner_seconds=6000 if device_kind == 'gpu' else 18000,
            preflight_inner_seconds=900 if device_kind == 'gpu' else 1800,
            outer_extra_seconds=60,threads=8,automatic_retry=False),
            'frozen_device_specific_phase_and_inner_resource_limits')
        for name,wanted in p['source_manifest']['files'].items():
            audit.file(name,wanted,root='source')
        source_names = set(p['source_manifest']['files'])
        audit.check({'methods/feature_assignment.py','methods/target_reallocation.py'}
            | {'experiments/feature_assignment/'+v for v in ('runtime.py','prepare.py','preflight.py','run.py','driver.py')}
            <= source_names,'all_scientific_proposal_runtime_and_guardian_source_bytes_frozen')
        for ref in p['input_refs']:
            audit.file(ref['path'],ref['sha256'],root=ref['root'])
        actual_tree = subprocess.check_output(['git','rev-parse',RUNTIME_COMMIT+'^{tree}'],
            cwd=p['runtime_root'],universal_newlines=True).strip()
        audit.check(actual_tree == p['runtime_tree'],'historical_public_runtime_tree_binding')
        scope = audit.json(protocol_path.parent/'metadata_scope.json',p['metadata_scope_sha256'])
        audit.check(scope['new_ID_array_reads_started'] is False
            and scope['runtime_commit'] == RUNTIME_COMMIT
            and scope['source_manifest'] == p['source_manifest'],
            'source_manifest_scope_recorded_before_ID_metadata_reads')
        audit.check(p['own_teacher_manifest']['sha256'] == TEACHER_MANIFEST_SHA
            and p['own_teacher_manifest']['checkpoint_sha256'] == TEACHER_CHECKPOINT_SHA,
            'same_independently_full_TRAIN_fitted_teacher_opaque_artifact_identity')
        selected,members,cohort = cohort_recount(audit,p)
        provenance = dict(study_source_commit=p['source_manifest']['source_commit'],
            runtime_commit=RUNTIME_COMMIT,protocol_sha256=args.protocol_sha256)
        main_dir = stage/'main_attempt1'
        main_result = audit.json(main_dir/'result.json')
        audit.check(main_result['status'] == 'completed'
            and main_result['provenance'] == provenance and main_result['full_case_count'] == 384
            and main_result['official_victim_loaded'] is True,
            'completed_same_source_full_matrix_before_saved_audit')
        gate = audit.json(stage/'preflight_attempt1'/'result.json')
        preflight = preflight_recount(audit,p,gate,provenance)
        seal = audit.json(main_dir/'geometry_sealed.json',main_result['geometry_sealed_sha256'])
        accounting = chronology_and_guardians(audit,p,stage,main_result,seal,provenance)
        cases = audit.rows(main_dir/'cases.jsonl',seal['cases_jsonl_sha256'])
        audit.check(len(cases) == 384
            and [(v['seed'],v['group']) for v in cases] == [(s,g) for s in SEEDS for g in selected]
            and [v['artifact'] for v in cases] == seal['artifacts'],
            'all384_ordered_seed_group_cases_and_sealed_artifact_manifest')
        for case in cases:
            audit.check(case['artifact']['path'] == 'artifacts/seed%d_group%d.pt'%(case['seed'],case['group'])
                and case['ordinal'] == selected.index(case['group'])
                and case['original_node_count'] == main_result['original_node_count'],
                'strict_capture_paths_and_registered_group_order')
            audit.file(main_dir/case['artifact']['path'],case['artifact']['sha256'])
            audit.check(datetime.datetime.fromisoformat(main_result['started_utc'])
                <= datetime.datetime.fromisoformat(case['final_inputs_created_utc'])
                <= datetime.datetime.fromisoformat(seal['completed_utc']),
                'every_case_final_input_receipt_precedes_global_victim_gate')
        write_json(output/'authorized_input_manifest.json',
            [dict(path=k,sha256=v) for k,v in sorted(audit.files.items())])
        audit.check(os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                    'auditor_explicitly_CPU_only_before_Torch_import')
        import numpy as np
        import torch
        audit.check(sys.version_info[:3] == (3,8,20) and np.__version__ == '1.24.4'
            and torch.__version__ == '1.13.1+cu117', 'locked_saved_tensor_audit_runtime')
        audit.check(not torch.cuda.is_available()
            and torch.cuda.device_count() == torch._C._cuda_getDeviceCount() == 0,
            'auditor_zero_visible_Torch_and_native_CUDA')
        torch.set_num_threads(8)
        summaries = {}
        physical = cache_hits = native_updates = 0
        acceptance_counts = collections.Counter()
        counter_boundaries = []
        labels = {}
        actual_geometry = collections.Counter()
        expected_cached_logits = {}
        case_by_key = {(v['seed'],v['group']):v for v in cases}
        for case in cases:
            path = audit.file(main_dir/case['artifact']['path'],case['artifact']['sha256'])
            payload = torch.load(path,map_location='cpu')
            operations['captured_tensor_loads'] += 1
            summary = tensor_case(audit,np,torch,p,case,payload,members)
            key = (case['seed'],case['group'])
            summaries[key] = summary
            audit.check(labels.setdefault(case['group'],case['labels']) == case['labels'],
                        'same_truth_and_target_order_across_all_generator_seeds')
            physical += summary['own_physical_forwards_per_role']
            cache_hits += summary['own_cache_hits']
            native_updates += summary['native_input_unique_features']
            for arm in PROPOSALS:
                acceptance_counts[arm+'_accepted'] += summary['accepted'][arm]
                acceptance_counts[arm+'_rejected'] += not summary['accepted'][arm]
            counter_boundaries.extend(dict(seed=case['seed'],group=case['group'],arm=arm)
                for arm in summary['acceptance_float64_boundary_arms'])
            actual_geometry[case['seed'],'nodes'] += summary['node_budget']
            actual_geometry[case['seed'],'edges'] += summary['edge_budget']
            for arm in ARMS:
                expected_cached_logits[case['seed'],case['group'],arm] = as_logits(payload['own_logits_final'][arm]['gcn'])
            del payload
        expected_counters = dict(dataset_loads=3,generator_models_loaded=3,proposal_GCN_models_loaded=3,own_GAGA_models_loaded=1,
            generations=384,generator_forward_dispatches=384,generator_completed_forwards=384,proposal_generation_GCN_forwards=0,
            own_forward_requests=3072,own_GAGA_physical_forwards=physical,own_GCN_physical_forwards=physical,
            own_GAGA_completed_forwards=physical,own_GCN_completed_forwards=physical,own_cache_hits=cache_hits,
            native_sequence_cache_hits=1536-native_updates,autograd_calls=768,native_canonical_updates=native_updates,
            final_canonical_sequences=1536,sealed_cases=384,official_GAGA_models_loaded=1,official_GAGA_forwards=1664,
            clean_GCN_forwards=128,cached_attacked_GCN_requests=1536,artifact_tensor_loads=384,
            optimizer_updates=0,API_calls=0,TEST_queries=0)
        audit.check(main_result['counters'] == expected_counters
            and 2*physical+cache_hits == 3072,'actual_physical_forward_cache_and_full_logical_counter_reconciliation')
        before_evaluation = dict(expected_counters,official_GAGA_models_loaded=0,
            official_GAGA_forwards=0,clean_GCN_forwards=0,cached_attacked_GCN_requests=0,artifact_tensor_loads=0)
        audit.check(seal['counters'] == before_evaluation,
                    'zero_official_GAGA_calls_before_all_inputs_seal')
        audit.check(main_result['immutable_content_before'] == main_result['immutable_content_after']
            == seal['immutable_original_content']
            and main_result['official_victim_state_before'] == main_result['official_victim_state_after']
            and main_result['own_teacher']['state_sha256'] == main_result['own_teacher_state_after']
            == gate['model_states_before']['own_teacher']
            and main_result['own_teacher']['state_sha256'] != main_result['official_victim_state_before']
            and main_result['own_teacher']['manifest_sha256'] == TEACHER_MANIFEST_SHA
            and main_result['own_model_filter_not_victim_feedback'] is True
            and main_result['cached_GCN_not_independent_neural_replay'] is True,
            'recorded_frozen_models_original_features_and_information_boundary')
        for seed in SEEDS:
            before = audit.json(main_dir/'engine_receipts'/('seed%d_before.json'%seed))
            after = audit.json(main_dir/'engine_receipts'/('seed%d_after.json'%seed))
            audit.check(before['generator_state'] == after['generator_state']
                and before['gcn_state'] == after['gcn_state']
                and before['original_content'] == after['original_content'] == seal['immutable_original_content']
                and before['scheduler'] == after['scheduler'] and before['scheduler']['restoration_ticks'] == 8
                and after['engine_generator_forwards'] == 128,
                'all_three_generator_GCN_scheduler_and_data_before_after_receipts')
            if seed == 5:
                audit.check(after['gcn_state'] == main_result['final_original_GCN_state_after'],
                            'final_cached_GCN_state_matches_last_frozen_engine')
        expected_eval_names = {'clean.jsonl'}|{arm+'_seed%d.jsonl'%seed for seed in SEEDS for arm in ARMS}
        audit.check(set(main_result['evaluation_files']) == expected_eval_names,
                    'all13_clean_and_complete_attack_evaluation_streams_retained')
        for name,wanted in main_result['evaluation_files'].items():
            audit.file(main_dir/'evaluation'/name,wanted)
        clean_rows = audit.rows(main_dir/'evaluation'/'clean.jsonl')
        clean = {row['group']:row for row in clean_rows}
        audit.check(len(clean) == len(clean_rows) == 128 and set(clean) == set(selected),
                    'all128_original_clean_cases_no_denominator_filter')
        clean_classes = {}
        for group,row in clean.items():
            audit.check(row['node_ids'] == members[group] and row['labels'] == labels[group],
                        'clean_rows_bind_full_original_member_truth_lists')
            for detector in ('gcn','gaga'):
                clean_classes[group,detector] = raw_classes(audit,row[detector+'_logits'],len(members[group]),'fresh_clean_'+detector)
        attack_classes = {}
        case_csv = []
        for seed in SEEDS:
            for arm in ARMS:
                rows = audit.rows(main_dir/'evaluation'/(arm+'_seed%d.jsonl'%seed))
                keyed = {row['group']:row for row in rows}
                audit.check(len(keyed) == len(rows) == 128 and set(keyed) == set(selected),
                            'complete_registered_attack_group_stream')
                for group,row in keyed.items():
                    source = case_by_key[seed,group]
                    summary = summaries[seed,group]
                    audit.check(row['seed'] == seed and row['arm'] == arm
                        and row['node_ids'] == members[group] and row['labels'] == labels[group]
                        and row['node_budget'] == summary['node_budget'] and row['edge_budget'] == summary['edge_budget']
                        and row['artifact_sha256'] == source['artifact']['sha256']
                        and row['canonical_sequence_sha256'] == source['final_sequence_hashes'][arm]
                        and row['feature_sha256'] == source['final_invariants'][arm]['feature_sha256']
                        and row['gcn_logits'] == expected_cached_logits[seed,group,arm],
                        'official_prediction_sealed_native_input_and_declared_cached_GCN_binding')
                    counts = {}
                    for detector in ('gaga','gcn'):
                        predicted = raw_classes(audit,row[detector+'_logits'],len(members[group]),'saved_attack_'+detector)
                        attack_classes[seed,group,arm,detector] = predicted
                        counts[detector] = transitions(labels[group],clean_classes[group,detector],predicted)
                        audit.check(row[detector] == counts[detector],
                                    'all_raw_logit_argmax_clean_new_recovery_counts')
                    case_csv.append(dict(seed=seed,group=group,arm=arm,targets=len(members[group]),
                        node_budget=summary['node_budget'],edge_budget=summary['edge_budget'],
                        accepted=False if arm == 'baseline' else summary['accepted'][arm],
                        **summary['changes'][arm],
                        **{detector+'_'+field:value for detector,count in counts.items() for field,value in count.items()}))
        metrics,contrasts,criterion = metric_summary(audit,selected,members,labels,clean_classes,attack_classes)
        audit.check(operations['captured_tensor_loads'] == 384
            and all(value == 0 for key,value in operations.items() if key != 'captured_tensor_loads'),
            'only384_new_captured_tensor_deserializations_no_model_dataset_or_TEST_work')
        with (output/'all_case_counts.csv').open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(case_csv[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(case_csv)
        result.update(status='passed',provenance=provenance,cohort=cohort,metrics=metrics,paired_contrasts=contrasts,
            primary_qualification=criterion,qualified_candidate='relation_assignment' if criterion['criterion_met'] else None,
            acceptance_counts=dict(acceptance_counts),acceptance_float64_boundary_cases=counter_boundaries,
            physical_cost_per_seed=[dict(seed=s,summed_injected_nodes=actual_geometry[s,'nodes'],
                summed_nonself_undirected_edges=actual_geometry[s,'edges']) for s in SEEDS],
            producer_counters=expected_counters,preflight=preflight,accounting=accounting,
            main_process_cpu_seconds=main_result['process_cpu_seconds'],main_wall_seconds=main_result['wall_seconds'],
            case_counts_csv_sha256=sha256(output/'all_case_counts.csv'),
            limitations=[
                'Gradients, native canonical construction and model states are source-bound producer captures/receipts; the independent audit never constructs or executes a model.',
                'Linear assignment optimality uses independently reconstructed FP64 costs and a separate primal/dual Hungarian solver within a declared arithmetic error envelope, not a nonlinear optimum.',
                'Common acceptance uses actual saved native own-GAGA logits; the pointwise guard protects only the own GCN, not the official victim.',
                'Cycle matching preserves raw permutation cycles, not actual displacement, accepted dose, conditional camouflage or article realizability.',
                'The GCN attacked channel is cached from the shared-filter proposal phase; official GAGA is fresh after sealing. Neither agent work nor this saved audit is human verification.',
                'All384cases remain;128ID-sampled contexts can repeat original news. Conditional3seed/df2 uncertainty does not make this untouched or IID validation or a final TEST result.',
                'Original inputs/checkpoints are opaque-byte hashed only. Original metadata membership and seal chronology are saved/source assertions, not independent dataset/model replay.'])
    except BaseException as error:
        traceback.print_exc()
        result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(audit_operation_counters=operations,process_cpu_seconds=time.process_time()-cpu_started,
            wall_seconds=time.monotonic()-started)
        if audit is not None:
            result.update(check_counts=dict(audit.check_counts),byte_hashed_files=len(audit.files))
            write_json(output/'input_hashes.json',[dict(path=k,sha256=v) for k,v in sorted(audit.files.items())])
        write_json(output/'result.json',result)
        print(json.dumps(dict(status=result['status'],output=str(output),
            checks=sum(result.get('check_counts',{}).values()),process_cpu_seconds=result['process_cpu_seconds'],
            wall_seconds=result['wall_seconds']),allow_nan=False),flush=True)
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
