"""Fixed CPU feature transforms without labels, models or topology changes.

The primary transform doubles the logit of a box-normalized coordinate. These
coordinates are generated numeric features, not calibrated class probabilities.
The shuffled control matches the primary PRECLIP normalized displacement norm;
clipping and the float32 raw-feature map can change its final displacement.
"""
import hashlib
import json
import random

import torch


ARMS = ('baseline', 'logit_sharp', 'linear_contrast', 'shuffled_delta')
BETA = 2.0
LINEAR_SLOPE = 1.5
BASELINE_BOX_ATOL = 1e-6
SHUFFLE_ROLE = 'shuffled_delta'
SHUFFLE_NAMESPACE = 'feature-logit-sharpen-v1'


def _identity(value, baseline):
    old = baseline.contiguous().view(torch.uint8).reshape(*baseline.shape, 4)
    new = value.contiguous().view(torch.uint8).reshape(*value.shape, 4)
    changed = (old != new).any(-1)
    return dict(exact_equals_baseline=not bool(changed.any()),
                changed_rows=int(changed.any(1).sum()), changed_coordinates=int(changed.sum()),
                rows=baseline.shape[0], coordinates=baseline.numel(),
                comparison='Exact logical row-major float32 bytes')


def _row_norm(value):
    return value.square().sum(1).sqrt()


def _stats(value, baseline, raw_unit, minimum, maximum, movable, preclip_delta):
    width = maximum.to(torch.float64)-minimum.to(torch.float64)
    raw_delta = value.to(torch.float64)-baseline.to(torch.float64)
    postclip_delta = raw_delta[:, movable]/width[movable]
    pre = _row_norm(preclip_delta)
    post = _row_norm(postclip_delta)
    raw_l2 = _row_norm(raw_delta)
    raw_linf = raw_delta.abs().max(1).values
    terminal = value[:, movable]
    lower = terminal == minimum[movable]
    upper = terminal == maximum[movable]
    before = raw_unit[:, movable]+preclip_delta
    below, above = before < 0, before > 1
    rows = [dict(row=i, preclip_unit_l2=float(pre[i]), postclip_unit_l2=float(post[i]),
                 raw_delta_l2=float(raw_l2[i]), raw_delta_linf=float(raw_linf[i]),
                 terminal_lower_endpoint_coordinates=int(lower[i].sum()),
                 terminal_upper_endpoint_coordinates=int(upper[i].sum()),
                 preclip_below_zero_coordinates=int(below[i].sum()),
                 preclip_above_one_coordinates=int(above[i].sum()))
            for i in range(len(baseline))]
    return dict(rows=rows, preclip_unit_l2_frobenius=float(pre.square().sum().sqrt()),
                postclip_unit_l2_frobenius=float(post.square().sum().sqrt()),
                raw_delta_frobenius=float(raw_delta.square().sum().sqrt()),
                raw_delta_linf=float(raw_delta.abs().max()),
                terminal_lower_endpoint_coordinates=int(lower.sum()),
                terminal_upper_endpoint_coordinates=int(upper.sum()),
                preclip_below_zero_coordinates=int(below.sum()),
                preclip_above_one_coordinates=int(above.sum()),
                norm_scope='Movable coordinates for unit norms; all raw coordinates for raw norms',
                preclip_definition=('Actual stored primary float32 displacement divided by float64 widths; '
                                    'within-row permutation for shuffled_delta; stored displacement for other arms'))


def transform_features(x0, minimum, maximum, *, seed, group, original_count):
    """Return the four fixed arms and complete identity/displacement metadata.

    ``baseline`` is an exact byte copy, including accepted box-roundoff of at
    most 1e-6. All other arms obey the raw coordinate box. Zero-width coordinates
    retain baseline bits. Shuffling uses a private Python RNG per injected node;
    neither Python's nor Torch's global RNG is read or changed by that operation.
    The recorded key and permutation hashes identify its locked-runtime mapping.
    """
    if (not torch.is_tensor(x0) or x0.device.type != 'cpu' or x0.dtype != torch.float32
            or x0.ndim != 2 or min(x0.shape) < 1 or not bool(torch.isfinite(x0.detach()).all())):
        raise ValueError('Injected features must be a finite nonempty float32 CPU matrix.')
    width_count = x0.shape[1]
    if any(not torch.is_tensor(value) or value.device.type != 'cpu' or value.dtype != torch.float32
           or value.shape != (width_count,) or not bool(torch.isfinite(value.detach()).all())
           for value in (minimum, maximum)):
        raise ValueError('Bounds must be finite float32 CPU vectors matching the feature width.')
    if not bool((maximum >= minimum).all()):
        raise ValueError('Raw coordinate bounds are reversed.')
    if any(type(value) is not int or value < 0 for value in (seed, group, original_count)) or original_count < 1:
        raise ValueError('Seed/group must be nonnegative Python integers; original_count must be positive.')
    with torch.no_grad():
        baseline = x0.detach().clone()
        lo, hi = minimum.detach().to(torch.float64), maximum.detach().to(torch.float64)
        width = hi-lo
        movable = width > 0
        violation = torch.maximum((lo-baseline.to(torch.float64)).clamp_min(0),
                                  (baseline.to(torch.float64)-hi).clamp_min(0))
        maximum_violation = float(violation.max())
        if maximum_violation > BASELINE_BOX_ATOL:
            raise ValueError('Baseline lies outside the original box beyond its frozen1e-6 tolerance.')
        if not torch.equal(baseline[:, ~movable], minimum[~movable].expand(len(baseline), -1)):
            raise ValueError('A zero-width coordinate differs from its original constant.')
        raw_unit = torch.zeros_like(baseline, dtype=torch.float64)
        raw_unit[:, movable] = (baseline[:, movable].to(torch.float64)-lo[movable])/width[movable]
        unit = raw_unit.clamp(0, 1)

        def raw_from_unit(value):
            result = (lo+value*width).to(torch.float32)
            result = torch.maximum(minimum, torch.minimum(maximum, result))
            result = torch.where(movable.unsqueeze(0), result, baseline)
            if not bool(torch.isfinite(result).all() and ((result >= minimum)&(result <= maximum)).all()):
                raise ValueError('A transformed feature violates the original finite raw box.')
            return result.detach().clone()

        squared = unit.square()
        sharpened_unit = squared/(squared+(1-unit).square())
        sharp = raw_from_unit(sharpened_unit)
        linear = raw_from_unit((.5+LINEAR_SLOPE*(unit-.5)).clamp(0, 1))
        # Match the ACTUAL stored primary displacement rather than the ideal
        # unrounded formula. Flat coordinates never enter the permutation pool.
        sharp_delta = (sharp[:, movable].to(torch.float64)-baseline[:, movable].to(torch.float64))/width[movable]
        shuffled_delta = torch.empty_like(sharp_delta)
        receipts = []
        for row in range(len(baseline)):
            key = [seed, group, original_count+row, SHUFFLE_ROLE, SHUFFLE_NAMESPACE]
            encoded = json.dumps(key, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
            digest = hashlib.sha256(encoded).hexdigest()
            permutation = list(range(int(movable.sum())))
            random.Random(int(digest, 16)).shuffle(permutation)
            shuffled_delta[row] = sharp_delta[row, permutation]
            permutation_bytes = json.dumps(permutation, separators=(',', ':')).encode('ascii')
            receipts.append(dict(row=row, physical_node_id=original_count+row,
                                 key_sha256=digest, permutation_sha256=hashlib.sha256(permutation_bytes).hexdigest()))
        shuffled_unit = raw_unit.clone()
        shuffled_unit[:, movable] = (raw_unit[:, movable]+shuffled_delta).clamp(0, 1)
        shuffled = raw_from_unit(shuffled_unit)
        arms = dict(baseline=baseline, logit_sharp=sharp,
                    linear_contrast=linear, shuffled_delta=shuffled)
        if tuple(arms) != ARMS:
            raise AssertionError('Fixed feature-transform arm order changed.')
        preclip = dict(baseline=torch.zeros_like(sharp_delta), logit_sharp=sharp_delta,
                       linear_contrast=(linear[:, movable].to(torch.float64)-baseline[:, movable].to(torch.float64))/width[movable],
                       shuffled_delta=shuffled_delta)
        stats = {arm:_stats(value, baseline, raw_unit, minimum, maximum, movable, preclip[arm])
                 for arm,value in arms.items()}
        norm_gap = (_row_norm(sharp_delta)-_row_norm(shuffled_delta)).abs()
        return dict(arms=arms, identity={arm:_identity(value, baseline) for arm,value in arms.items()},
                    stats=stats, shuffle_receipts=receipts, beta=BETA, linear_slope=LINEAR_SLOPE,
                    zero_width_dimensions=int((~movable).sum()), movable_dimensions=int(movable.sum()),
                    baseline_box_max_violation=maximum_violation, baseline_box_atol=BASELINE_BOX_ATOL,
                    baseline_exact_box=maximum_violation == 0,
                    maximum_recorded_preclip_unit_l2_gap=float(norm_gap.max()),
                    shuffle_matches_preclip_unit_l2_by_construction=True,
                    shuffle_key_scheme='compact JSON [seed,group,original_count+row,role,namespace]; SHA256 hexadecimal -> private random.Random integer seed',
                    shuffle_role=SHUFFLE_ROLE, shuffle_namespace=SHUFFLE_NAMESPACE,
                    no_terminal_or_raw_norm_matching_claim=True,
                    zero_width_coordinates_exactly_frozen=True, topology_changed=False,
                    labels_used=False, model_outputs_used=False, global_rng_changed=False,
                    no_row_l2_normalization=True, class_confidence_interpretation=False)
