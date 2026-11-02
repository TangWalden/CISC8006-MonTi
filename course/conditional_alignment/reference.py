"""Train-reference relationship moments and a non-learnable tolerance gate.

Reference centers have known benign TRAIN labels. Their neighbors use the same
original transductive graph features as MonTi; neighbor/evaluation labels and
predictions are never used. This is moment alignment, not a claim of equality of
complete distributions or a first-in-literature defense/attack invention.
"""
import collections
import hashlib
import json

import torch
import torch.nn.functional as F


def degree_buckets(degrees):
    degrees = torch.as_tensor(degrees, dtype=torch.long)
    if bool((degrees < 0).any()):
        raise ValueError('Non-self degrees must be nonnegative.')
    buckets = torch.floor(torch.log2(degrees.clamp_min(1).to(torch.float64))).long()
    return torch.where(degrees > 0, buckets, torch.full_like(buckets, -1))


def relationship_moments(graph, node_ids, *, chunk_size=4096, use_weights=False):
    """Return mean(cos), mean(cos**2), and actual non-self degree per node.

Only incident edges already present in the generated graph are used. Training
can differentiate through generated features and existing ST edge weights. The
discrete degree/bin lookup is detached. This does not create gradients for edges
absent from the graph, matching the upstream graph-building limitation.
"""
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive.')
    ids = torch.as_tensor(node_ids, dtype=graph.idtype, device=graph.device)
    if ids.ndim != 1 or ids.numel() == 0 or torch.unique(ids).numel() != ids.numel():
        raise ValueError('node_ids must be a nonempty unique vector.')
    if bool((ids < 0).any()) or bool((ids >= graph.num_nodes()).any()):
        raise ValueError('node_ids are outside this graph.')
    features = graph.ndata['x']
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError('Expected floating point node features.')
    source, destination, edge_ids = graph.out_edges(ids, form='all')
    keep = source != destination
    source, destination, edge_ids = source[keep], destination[keep], edge_ids[keep]
    positions = torch.full((graph.num_nodes(),), -1, dtype=torch.long, device=graph.device)
    positions[ids] = torch.arange(ids.numel(), device=graph.device)
    rows = positions[source]
    degrees = torch.bincount(rows, minlength=ids.numel())
    sums = features.new_zeros((ids.numel(), 2))
    for start in range(0, source.numel(), chunk_size):
        end = min(source.numel(), start + chunk_size)
        cosine = F.cosine_similarity(features[source[start:end]],
                                     features[destination[start:end]], dim=1, eps=1e-8)
        moments = torch.stack((cosine, cosine.square()), dim=1)
        if use_weights:
            if 'weight' not in graph.edata:
                raise ValueError('Training graph has no ST edge weights.')
            weight = graph.edata['weight'][edge_ids[start:end]].reshape(-1, 1)
            moments = moments * weight
        sums = sums.index_add(0, rows[start:end], moments)
    signatures = sums / degrees.clamp_min(1).to(features.dtype).unsqueeze(1)
    if not bool(torch.isfinite(signatures).all()):
        raise ValueError('Nonfinite relationship signature.')
    return signatures, degrees


def build_reference(graph, train_node_ids, *, min_count=8, gate_quantile=0.9,
                    scale_floor=0.05, chunk_size=4096):
    """Fit all supported log2-degree buckets from original benign TRAIN centers.

    Zero-degree centers are counted and excluded because injected nodes have a
    mandatory physical connection. No sampling and no RNG calls are made.
    """
    if min_count < 2 or not 0 < gate_quantile < 1 or scale_floor <= 0:
        raise ValueError('Invalid reference settings.')
    ids = torch.as_tensor(train_node_ids, dtype=graph.idtype, device=graph.device)
    if ids.ndim != 1 or ids.numel() == 0 or torch.unique(ids).numel() != ids.numel():
        raise ValueError('Training node IDs must be unique and nonempty.')
    # The only labels consulted are indexed by the registered TRAIN node list.
    benign = ids[graph.ndata['y'][ids].reshape(-1) == 0]
    if benign.numel() < min_count:
        raise ValueError('Not enough benign TRAIN reference nodes.')
    with torch.no_grad():
        signatures, degrees = relationship_moments(graph, benign, chunk_size=chunk_size)
        buckets = degree_buckets(degrees)
        fitted = {}
        counts = {}
        for bucket in sorted(set(buckets.cpu().tolist())):
            mask = buckets == bucket
            count = int(mask.sum().item())
            counts[str(bucket)] = count
            if bucket < 0 or count < min_count:
                continue
            values = signatures[mask].to(dtype=torch.float64, device='cpu')
            center = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False).clamp_min(scale_floor)
            distance = ((values-center)/scale).square().mean(dim=1)
            fitted[str(bucket)] = dict(count=count, center=center.tolist(),
                                      scale=scale.tolist(),
                                      tolerance=float(torch.quantile(distance, gate_quantile)))
    if not fitted:
        raise ValueError('No supported positive-degree reference bucket.')
    private_ids = benign.detach().cpu().tolist()
    return dict(version=1, kind='benign TRAIN degree-conditioned relationship moments',
                signature=['mean_neighbor_raw_feature_cosine', 'mean_squared_neighbor_raw_feature_cosine'],
                train_nodes=int(ids.numel()), benign_train_nodes=int(benign.numel()),
                zero_degree_reference_nodes=counts.get('-1', 0),
                reference_node_ids_sha256=hashlib.sha256(
                    json.dumps(private_ids, separators=(',', ':')).encode()).hexdigest(),
                neighborhood='original transductive graph; raw features; no neighbor labels or predictions',
                degree_rule='floor_log2_nonself', unsupported_rule='nearest_supported_bucket_tie_lower',
                min_count=min_count, gate_quantile=gate_quantile, scale_floor=scale_floor,
                bucket_counts=counts, buckets=fitted,
                interpretation='Two relationship moments, not complete-distribution equality. No efficacy claim.')


def matched_statistics(degrees, reference, *, dtype, device):
    if reference.get('degree_rule') != 'floor_log2_nonself':
        raise ValueError('Unexpected frozen degree rule.')
    requested = degree_buckets(degrees).detach().cpu().tolist()
    supported = sorted(int(key) for key in reference['buckets'])
    if not supported:
        raise ValueError('Empty reference.')
    chosen = [min(supported, key=lambda key: (abs(key-value), key)) for value in requested]
    entries = [reference['buckets'][str(key)] for key in chosen]
    center = torch.tensor([entry['center'] for entry in entries], dtype=dtype, device=device)
    scale = torch.tensor([entry['scale'] for entry in entries], dtype=dtype, device=device)
    tolerance = torch.tensor([entry['tolerance'] for entry in entries], dtype=dtype, device=device)
    if center.shape != (len(requested), 2) or scale.shape != center.shape:
        raise ValueError('Malformed reference dimensions.')
    if not bool(torch.isfinite(center).all() and torch.isfinite(scale).all()
                and torch.isfinite(tolerance).all()) or bool((scale <= 0).any()) or bool((tolerance < 0).any()):
        raise ValueError('Malformed reference statistics.')
    return center, scale, tolerance, requested, chosen


def alignment_penalty(graph, injected_ids, reference, *, arm, coefficient=0.1,
                      chunk_size=4096, training=False):
    if arm not in ('off', 'always', 'gated') or coefficient < 0:
        raise ValueError('Unknown alignment arm or negative coefficient.')
    signature, degrees = relationship_moments(graph, injected_ids, chunk_size=chunk_size,
                                               use_weights=training)
    if bool((degrees < 1).any()):
        raise ValueError('An injected node has no physical non-self neighbor.')
    center, scale, tolerance, requested, chosen = matched_statistics(
        degrees, reference, dtype=signature.dtype, device=signature.device)
    distance = ((signature-center)/scale).square().mean(dim=1)
    # The gate is fixed by TRAIN reference tolerance, not an optimized parameter.
    exceed = distance.detach() > tolerance
    active = (torch.zeros_like(exceed) if arm == 'off' else
              torch.ones_like(exceed) if arm == 'always' else exceed)
    # Average across ALL injected nodes, not only active nodes.
    penalty = coefficient * (distance * active.to(distance.dtype)).mean()
    if not bool(torch.isfinite(penalty)):
        raise ValueError('Nonfinite alignment penalty.')
    stats = dict(arm=arm, injected_nodes=int(degrees.numel()),
                 active_nodes=int(active.sum().item()), outside_tolerance_nodes=int(exceed.sum().item()),
                 mean_distance=float(distance.detach().mean().item()),
                 max_distance=float(distance.detach().max().item()),
                 mean_tolerance=float(tolerance.mean().item()),
                 alignment_penalty=float(penalty.detach().item()),
                 mean_signature=signature.detach().mean(dim=0).cpu().tolist(),
                 physical_degree_min=int(degrees.min().item()), physical_degree_max=int(degrees.max().item()),
                 requested_buckets=dict(collections.Counter(str(x) for x in requested)),
                 matched_buckets=dict(collections.Counter(str(x) for x in chosen)),
                 fallback_nodes=sum(left != right for left, right in zip(requested, chosen)))
    return penalty, stats
