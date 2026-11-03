"""TRAIN-fitted anomaly diagnostic and isolation filter; evaluation only.

Injected authenticity is different from fraud class. No fraud labels, current
injected IDs, target IDs, clean differences or detector predictions enter scoring.
Descriptors are not the optimized alignment moments, but are not statistically
independent of graph degree/normalization. This is not a learned defender or an
adaptive-robustness claim. Dependencies are imported only when functions run.
"""
import hashlib
import json
import math

SCHEMA = 'monti_independent_anomaly_filter_v1'
SPLIT_SEED = 20261103
SCALE_FLOOR = 1e-3
CALIBRATION_QUANTILE = 0.99
MAX_WORKING_BYTES = 1024 ** 3
FEATURE_NAMES = (
    'log1p_nonself_in_degree',
    'raw_feature_l2_norm',
    'raw_feature_max_abs',
    'incoming_neighbor_mean_log1p_nonself_in_degree',
)
FEATURE_DEFINITIONS = (
    'Natural log(1 + current nonself incoming-arc count). Parallel arcs count separately.',
    'L2 norm of the current raw node feature vector; no pretrained embedding.',
    'Maximum absolute coordinate of the current raw feature vector.',
    'Mean log1p nonself in-degree of incoming nonself neighbors, weighted by arc multiplicity; zero if none.',
)


def _digest_ids(ids):
    return hashlib.sha256(json.dumps([int(x) for x in ids], separators=(',', ':')).encode()).hexdigest()


def _hash_split(ids):
    ordered = sorted((int(x) for x in ids), key=lambda node: (
        hashlib.sha256(('%s|independent-filter-train|%s' % (SPLIT_SEED, node)).encode()).hexdigest(), node))
    count = int(0.8 * len(ordered))
    if count < 2 or len(ordered) - count < 2:
        raise ValueError('The fixed 80/20 TRAIN split needs at least two nodes in each subset.')
    return ordered[:count], ordered[count:]


def _check_graph(graph):
    if not graph.is_homogeneous:
        raise ValueError('The fixed filter supports one homogeneous graph only.')
    if graph.num_nodes() < 1 or 'x' not in graph.ndata:
        raise ValueError('A nonempty graph with raw x features is required.')
    values = graph.ndata['x']
    if values.ndim != 2 or values.shape[0] != graph.num_nodes() or values.shape[1] < 1 or not values.is_floating_point():
        raise ValueError('Expected a floating point node-by-raw-feature matrix.')


def _memory_guard(estimate, maximum):
    if type(maximum) is not int or maximum < 1:
        raise ValueError('A positive explicit working-memory ceiling is required.')
    if estimate > maximum:
        raise MemoryError('Estimated filter workspace %d exceeds frozen ceiling %d; stop rather than expand it.' %
                          (estimate, maximum))


def node_features(graph, *, feature_chunk_size=4096, edge_chunk_size=262144,
                  max_working_bytes=MAX_WORKING_BYTES):
    """Vectorized/chunked current-graph descriptors for ALL nodes, without labels.

    This uses incoming arcs; on the symmetric MonTi graph these are ordinary
    neighbor degrees. Self loops are excluded regardless of their multiplicity.
    The memory ceiling is an estimated workspace guard, not an OS-RSS guarantee.
    """
    import torch
    _check_graph(graph)
    if feature_chunk_size < 1 or edge_chunk_size < 1:
        raise ValueError('Chunk sizes must be positive.')
    nodes, edges = graph.num_nodes(), graph.num_edges()
    width = graph.ndata['x'].shape[1]
    estimate = edges * 24 + nodes * 80 + min(nodes, feature_chunk_size) * width * 8
    _memory_guard(estimate, max_working_bytes)
    with torch.no_grad():
        source, destination = graph.edges(order='eid')
        degree = graph.in_degrees().to(dtype=torch.long).clone()
        for start in range(0, edges, edge_chunk_size):
            src, dst = source[start:start+edge_chunk_size], destination[start:start+edge_chunk_size]
            loops = dst[src == dst].long()
            if loops.numel():
                degree -= torch.bincount(loops, minlength=nodes)
        if bool((degree < 0).any()):
            raise RuntimeError('Nonself degree calculation is inconsistent.')
        log_degree = torch.log1p(degree.to(torch.float64))
        neighbor_sum = torch.zeros(nodes, dtype=torch.float64, device=graph.device)
        for start in range(0, edges, edge_chunk_size):
            src, dst = source[start:start+edge_chunk_size], destination[start:start+edge_chunk_size]
            keep = src != dst
            neighbor_sum.index_add_(0, dst[keep].long(), log_degree[src[keep].long()])
        features = torch.empty((nodes, 4), dtype=torch.float64, device=graph.device)
        features[:, 0] = log_degree
        features[:, 3] = neighbor_sum / degree.clamp_min(1).to(torch.float64)
        raw = graph.ndata['x'].detach()
        for start in range(0, nodes, feature_chunk_size):
            block = raw[start:start+feature_chunk_size].to(torch.float64)
            if not bool(torch.isfinite(block).all()):
                raise ValueError('Nonfinite raw graph features; no scores are returned.')
            features[start:start+len(block), 1] = torch.linalg.vector_norm(block, dim=1)
            features[start:start+len(block), 2] = block.abs().amax(dim=1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite anomaly descriptors.')
        return features.detach()


def _reference_tensors(reference, *, device):
    import torch
    if (reference.get('schema') != SCHEMA or reference.get('feature_names') != list(FEATURE_NAMES)
            or reference.get('scale_floor') != SCALE_FLOOR
            or reference.get('calibration_quantile') != CALIBRATION_QUANTILE
            or reference.get('flag_rule') != 'strict_score_greater_than_threshold'):
        raise ValueError('Reference does not match the fixed filter definition.')
    for key in ('median', 'scales'):
        value = reference.get(key)
        if type(value) is not list or len(value) != 4 or any(
                type(x) not in (int, float) or not math.isfinite(float(x)) for x in value):
            raise ValueError('Malformed reference: ' + key)
    if any(float(x) < SCALE_FLOOR for x in reference['scales']):
        raise ValueError('Reference scale is below the frozen floor.')
    threshold = reference.get('threshold')
    if type(threshold) not in (int, float) or not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError('Invalid nonnegative anomaly threshold.')
    center = torch.tensor(reference['median'], dtype=torch.float64, device=device)
    scale = torch.tensor(reference['scales'], dtype=torch.float64, device=device)
    return center, scale, float(threshold)


def scores_from_features(features, reference):
    """No origin/label inputs; score is mean squared robust z across four axes."""
    import torch
    if features.ndim != 2 or features.shape[1] != 4 or not bool(torch.isfinite(features).all()):
        raise ValueError('Expected finite four-feature descriptors.')
    center, scale, _ = _reference_tensors(reference, device=features.device)
    with torch.no_grad():
        scores = ((features.to(torch.float64) - center) / scale).square().mean(dim=1)
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('Nonfinite anomaly scores.')
        return scores.detach()


def fit_reference(graph, original_train_nodes, *, max_working_bytes=MAX_WORKING_BYTES):
    """CPU only: hash 80% fit/20% calibration of ALL original TRAIN nodes.

    Caller registers original graph and TRAIN identities. Fraud labels are never
    read: authenticity of an original node is not its fraud/benign class. Only
    TRAIN descriptors set medians/scales/threshold; full transductive topology
    determines each descriptor. No model, API or prediction is used.
    """
    import torch
    _check_graph(graph)
    if graph.device.type != 'cpu':
        raise ValueError('Reference fitting is a single CPU process; no device substitution.')
    ids = torch.as_tensor(original_train_nodes)
    if ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('TRAIN node IDs must be integer identities, not labels or probabilities.')
    if ids.ndim != 1 or ids.numel() < 4 or torch.unique(ids).numel() != ids.numel():
        raise ValueError('TRAIN identities must be a unique nonempty one-dimensional list.')
    ids = ids.cpu().long()
    if bool((ids < 0).any()) or bool((ids >= graph.num_nodes()).any()):
        raise ValueError('TRAIN identity outside the original graph.')
    fit_ids, calibration_ids = _hash_split(ids.tolist())
    features = node_features(graph, max_working_bytes=max_working_bytes)
    fit = features[fit_ids]
    with torch.no_grad():
        median = torch.quantile(fit, 0.5, dim=0, interpolation='linear')
        mad = torch.quantile((fit - median).abs(), 0.5, dim=0, interpolation='linear')
        scales = mad.clamp_min(SCALE_FLOOR)
        scores = ((features - median) / scales).square().mean(dim=1)
        calibration_scores = scores[calibration_ids]
        threshold = float(torch.quantile(calibration_scores, CALIBRATION_QUANTILE,
                                         interpolation='linear'))
        flagged = scores > threshold
    def rates(selected):
        count = int(flagged[selected].sum())
        return dict(nodes=len(selected), flagged=count, false_deletion_rate=count / len(selected))
    return dict(schema=SCHEMA, feature_names=list(FEATURE_NAMES),
        feature_definitions=dict(zip(FEATURE_NAMES, FEATURE_DEFINITIONS)),
        population='ALL original TRAIN nodes; fraud labels ignored',
        graph_context='Original full transductive graph; no neighbor labels or predictions',
        split_seed=SPLIT_SEED, split_rule='SHA256(seed|independent-filter-train|nodeID); first floor(0.8*n) fit',
        fit_fraction=0.8, fit_node_ids_sha256=_digest_ids(fit_ids),
        calibration_node_ids_sha256=_digest_ids(calibration_ids),
        original_train_node_ids_sha256=_digest_ids(sorted(ids.tolist())),
        original_graph_nodes=graph.num_nodes(), original_graph_arcs=graph.num_edges(),
        median=median.tolist(), mad=mad.tolist(), scales=scales.tolist(), scale_floor=SCALE_FLOOR,
        scale_definition='Unscaled coordinatewise median absolute deviation, floored at 1e-3',
        score_definition='Mean of squared (descriptor - median) / floored MAD over four descriptors',
        calibration_quantile=CALIBRATION_QUANTILE, quantile_interpolation='linear', threshold=threshold,
        flag_rule='strict_score_greater_than_threshold',
        calibration_ties_at_threshold=int((calibration_scores == threshold).sum()),
        fit_original_node_rates=rates(fit_ids), calibration_original_node_rates=rates(calibration_ids),
        all_train_original_node_rates=rates(ids.tolist()),
        nominal_calibration_false_deletion_rate=0.01,
        generalization='No guarantee of 1% deletion on validation or attacked graphs; report actual rates.',
        relation_to_alignment='Unoptimized descriptor family, not statistically independent of graph normalization.',
        max_working_bytes=max_working_bytes, fraud_labels_used=False, model_calls=0, api_calls=0)


def score_nodes(graph, reference, *, max_working_bytes=MAX_WORKING_BYTES):
    """Score ALL nodes in the received graph without current origin identities."""
    features = node_features(graph, max_working_bytes=max_working_bytes)
    return scores_from_features(features, reference)


def isolate_flagged(graph, flagged, *, max_working_bytes=MAX_WORKING_BYTES):
    """New graph: suppress nonself arcs incident to ANY flagged node.

    All node IDs, features and exactly one existing self loop per node remain.
    Source graph/tensors are never mutated; cloned attributes prevent aliasing.
    Actual original-node collateral counts belong in the offline evaluator.
    """
    import dgl
    import torch
    _check_graph(graph)
    nodes, edges = graph.num_nodes(), graph.num_edges()
    node_bytes = sum(value.numel() * value.element_size() for value in graph.ndata.values())
    edge_bytes = sum(value.numel() * value.element_size() for value in graph.edata.values())
    _memory_guard(edges * 80 + nodes * 32 + node_bytes + edge_bytes, max_working_bytes)
    mask = torch.as_tensor(flagged, device=graph.device)
    if mask.dtype != torch.bool or mask.ndim != 1 or len(mask) != nodes:
        raise ValueError('Flag mask must be Boolean and cover every current node.')
    with torch.no_grad():
        src, dst = graph.edges(order='eid')
        loops = src == dst
        loop_counts = torch.bincount(dst[loops].long(), minlength=nodes)
        if not bool((loop_counts == 1).all()):
            raise ValueError('Isolation requires exactly one source self loop per node; normalize explicitly before this stage.')
        keep = loops | ~(mask[src.long()] | mask[dst.long()])
        ids = keep.nonzero(as_tuple=True)[0]
        post = dgl.graph((src[ids], dst[ids]), num_nodes=nodes, idtype=graph.idtype, device=graph.device)
        for key, value in graph.ndata.items():
            post.ndata[key] = value.detach().clone()
        for key, value in graph.edata.items():
            post.edata[key] = value[ids].detach().clone()
        removed = int((~keep).sum())
        counters = dict(input_nodes=nodes, output_nodes=post.num_nodes(),
            flagged_nodes=int(mask.sum()), flagged_fraction=float(mask.double().mean()),
            input_arcs=edges, output_arcs=post.num_edges(), self_loops_preserved=nodes,
            input_nonself_arcs=edges-nodes, output_nonself_arcs=post.num_edges()-nodes,
            removed_nonself_incident_arcs=removed,
            original_node_or_injected_detection_counts='Not available to scoring; offline evaluator only',
            input_node_edge_counts_preserved=(graph.num_nodes() == nodes and graph.num_edges() == edges))
        if post.num_nodes() != nodes or post.num_edges() != edges - removed:
            raise RuntimeError('Isolation changed the registered case/node denominator unexpectedly.')
        return post, counters


def apply_filter(graph, reference, *, max_working_bytes=MAX_WORKING_BYTES):
    """Evaluation-only result with scores, flags, isolated graph and counters.

    This function receives no labels, target list, origin mask or injected IDs.
    Its no-gradient design is not evidence of robustness to adaptive attacks.
    """
    scores = score_nodes(graph, reference, max_working_bytes=max_working_bytes)
    _, _, threshold = _reference_tensors(reference, device=scores.device)
    flagged = scores > threshold
    post, counters = isolate_flagged(graph, flagged, max_working_bytes=max_working_bytes)
    counters.update(threshold=threshold, flag_rule=reference['flag_rule'],
                    score_feature_names=list(FEATURE_NAMES), fraud_labels_used=False,
                    current_injected_id_or_origin_input=False)
    return dict(scores=scores, flagged=flagged, post_graph=post, counters=counters)
