"""CPU-only TRAIN references and a fixed half-mean raw-feature blend.

Only TRAIN-indexed label values select reference pools. All original unlabeled
feature rows determine the same transductive coordinate box as frozen MonTi.
No model, graph, topology, neighbor label, API, normalization, fitting or RNG
operation is used. A numeric benign prototype is not a realizable news article.
"""
import torch


ARMS = ('baseline', 'benign_mix', 'all_train_mix', 'fraud_mix')
ALPHA = .5
BASELINE_BOX_ATOL = 1e-6


def _ids(values, count, name):
    try:
        raw = torch.as_tensor(values)
    except (TypeError, ValueError) as error:
        raise ValueError(name+' must contain numeric original node IDs.') from error
    if (raw.device.type != 'cpu' or raw.ndim != 1 or raw.dtype == torch.bool
            or raw.is_complex()):
        raise ValueError(name+' must be a one-dimensional CPU vector of exact integral IDs.')
    if raw.numel() and (not bool(torch.isfinite(raw).all())
                       or bool((raw < 0).any()) or bool((raw >= count).any())):
        raise ValueError(name+' contains a nonfinite or out-of-range original node ID.')
    ids = raw.to(torch.long)
    if (not torch.equal(raw.to(torch.float64), ids.to(torch.float64))
            or ids.numel() != torch.unique(ids).numel()):
        raise ValueError(name+' must be exactly integral and unique; repeated group memberships are not reference samples.')
    return ids.detach().clone()


def selectreference(features, labels, train_ids, valid_ids, test_ids):
    """Return immutable-by-convention reference tensors and explicit read scope.

    Reference means use float64 accumulation and are then cast to float32.
    ``labels`` is inspected only for shape/device before ``labels[train_ids]``;
    held-out values, including invalid sentinels, never enter a value operation.
    Partition IDs are unique and mutually disjoint. Empty held-out partitions
    are accepted for synthetic checks; TRAIN must contain both binary classes.
    """
    if (not torch.is_tensor(features) or features.device.type != 'cpu'
            or features.dtype != torch.float32 or features.ndim != 2
            or min(features.shape) < 1 or not bool(torch.isfinite(features.detach()).all())):
        raise ValueError('Original features must be a finite nonempty float32 CPU matrix.')
    n, width = features.shape
    if not torch.is_tensor(labels) or labels.device.type != 'cpu' or labels.shape != (n,):
        raise ValueError('Original labels must have one CPU entry per original node.')
    train, valid, test = (_ids(train_ids, n, 'TRAIN'), _ids(valid_ids, n, 'VALID'), _ids(test_ids, n, 'TEST'))
    if not train.numel():raise ValueError('The original TRAIN reference partition is empty.')
    joined = torch.cat((train, valid, test))
    if torch.unique(joined).numel() != joined.numel():
        raise ValueError('Original TRAIN, VALID and TEST node partitions must be disjoint.')
    with torch.no_grad():
        # This is the only label-value read. Do not validate, summarize or index
        # the complete label vector: held-out labels are not reference inputs.
        selected_labels = labels.detach().index_select(0, train).clone()
        if (selected_labels.is_complex() or not bool(torch.isfinite(selected_labels).all())
                or not bool(((selected_labels == 0)|(selected_labels == 1)).all())):
            raise ValueError('Only the indexed TRAIN labels must be finite binary class0/class1 values.')
        selected_labels = selected_labels.to(torch.long)
        benign, fraud = train[selected_labels == 0], train[selected_labels == 1]
        if not benign.numel() or not fraud.numel():
            raise ValueError('Both benign class0 and fraud class1 must have positive original TRAIN reference counts.')
        detached = features.detach()
        minimum, maximum = detached.min(0).values, detached.max(0).values
        movable = maximum > minimum
        def mean(ids):
            value = detached.index_select(0, ids).to(torch.float64).mean(0).to(torch.float32)
            value = torch.maximum(minimum, torch.minimum(maximum, value))
            value = torch.where(movable, value, minimum)
            if not bool(torch.isfinite(value).all()):raise ValueError('A TRAIN prototype is nonfinite.')
            return value.detach().clone()
        return dict(minimum=minimum.detach().clone(), maximum=maximum.detach().clone(),
            benign_mean=mean(benign), all_train_mean=mean(train), fraud_mean=mean(fraud),
            train_ids=train, train_labels=selected_labels.detach().clone(),
            benign_ids=benign.detach().clone(), fraud_ids=fraud.detach().clone(),
            counts=dict(original_nodes=n, feature_dimension=width, train_nodes=train.numel(),
                benign_train_nodes=benign.numel(), fraud_train_nodes=fraud.numel(),
                valid_nodes=valid.numel(), test_nodes=test.numel(), label_values_read=train.numel()),
            labels_read_scope='Exactly original TRAIN label entries; no VALID/TEST label values read',
            reference_scope='Unweighted distinct original TRAIN nodes; float64 coordinate means cast to float32',
            feature_box_scope='All original unlabeled transductive feature rows; same coordinate box as frozen MonTi',
            row_l2_normalization=False, continuous_l0_constraint=False)


def _identity(candidate, baseline):
    # Logical row-major float32 bytes detect even a signed-zero bit change.
    old = baseline.detach().contiguous().view(torch.uint8).reshape(*baseline.shape, 4)
    new = candidate.detach().contiguous().view(torch.uint8).reshape(*candidate.shape, 4)
    changed = (old != new).any(-1)
    return dict(exact_equals_baseline=not bool(changed.any()),
        changed_rows=int(changed.any(1).sum()), changed_coordinates=int(changed.sum()),
        rows=baseline.shape[0], coordinates=baseline.numel(), comparison='Exact logical row-major float32 bytes')


def mixfeatures(x0, reference):
    """Return all four fixed arms; baseline bytes are copied without clipping.

    Other arms use exactly 0.5*x0+0.5*TRAIN_mean in the raw feature space,
    followed only by the original coordinate-box projection for roundoff.
    No row normalization, small-Lp constraint, topology change or outcome gate.
    """
    if (not torch.is_tensor(x0) or x0.device.type != 'cpu' or x0.dtype != torch.float32
            or x0.ndim != 2 or min(x0.shape) < 1 or not bool(torch.isfinite(x0.detach()).all())):
        raise ValueError('Saved injected features must be a finite nonempty float32 CPU matrix.')
    width = x0.shape[1]
    bounds = [reference[name] for name in ('minimum', 'maximum')]
    if any(not torch.is_tensor(value) or value.device.type != 'cpu' or value.dtype != torch.float32
           or value.shape != (width,) or not bool(torch.isfinite(value).all()) for value in bounds):
        raise ValueError('Reference coordinate bounds must be finite float32 CPU vectors of matching width.')
    minimum, maximum = bounds
    if not bool((maximum >= minimum).all()):raise ValueError('Reference box bounds are reversed.')
    movable = maximum > minimum
    with torch.no_grad():
        baseline = x0.detach().clone()
        lower_error = (minimum-baseline).clamp_min(0)
        upper_error = (baseline-maximum).clamp_min(0)
        maximum_violation = float(torch.maximum(lower_error, upper_error).max())
        if maximum_violation > BASELINE_BOX_ATOL:
            raise ValueError('Saved baseline is outside the original feature box beyond its frozen1e-6 tolerance.')
        if not torch.equal(baseline[:, ~movable], minimum[~movable].expand(len(baseline), -1)):
            raise ValueError('A zero-width original feature coordinate differs from its fixed value.')
        arms = dict(baseline=baseline)
        for arm, key in (('benign_mix', 'benign_mean'), ('all_train_mix', 'all_train_mean'), ('fraud_mix', 'fraud_mean')):
            prototype = reference[key]
            if (not torch.is_tensor(prototype) or prototype.device.type != 'cpu'
                    or prototype.dtype != torch.float32 or prototype.shape != (width,)
                    or not bool(torch.isfinite(prototype).all())
                    or not bool(((prototype >= minimum)&(prototype <= maximum)).all())):
                raise ValueError('A fixed TRAIN prototype is malformed or outside the original box.')
            value = baseline*ALPHA+prototype.unsqueeze(0)*ALPHA
            value = torch.maximum(minimum, torch.minimum(maximum, value))
            # Copy zero-width coordinates exactly, including baseline signed-zero
            # bits. They remain the same numeric global constant.
            value = torch.where(movable.unsqueeze(0), value, baseline)
            if not bool(torch.isfinite(value).all() and ((value >= minimum)&(value <= maximum)).all()):
                raise ValueError('A fixed half-mean feature blend violates the original raw box.')
            arms[arm] = value.detach().clone()
        if tuple(arms) != ARMS:raise AssertionError('Fixed prototype arm order changed.')
        return dict(arms=arms, identity={arm:_identity(value, baseline) for arm,value in arms.items()},
            alpha=ALPHA, zero_width_dimensions=int((~movable).sum()),
            baseline_box_max_violation=maximum_violation, baseline_box_atol=BASELINE_BOX_ATOL,
            baseline_exact_box=maximum_violation == 0,
            no_row_l2_normalization=True, zero_width_coordinates_exactly_frozen=True,
            topology_changed=False, model_outputs_used=False)
