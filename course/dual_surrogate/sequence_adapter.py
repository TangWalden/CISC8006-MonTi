"""Isolated GAGA sequence update with hard-forward/ST-backward semantics.

For binary generated graphs the forward is the original DataHandler update:
add only newly reachable one/two-hop members to immutable original caches,
deduplicate two-hop paths, remove self relations, and retain cached masked-self
tokens. The authoritative masks/caches belong to the supplied data VIEW, not to
the possibly different masks stored on the attack graph. No labels are read.

One-hop weighted sums use weight * feature in numerator and denominator.
Two-hop reachability is discrete. During training only, its backward uses
rho(P)=P/(1+P), P=A[:T]@A, through hard+(rho-rho.detach()). This is an explicit
straight-through gradient approximation, NOT an exact discrete derivative.
Only selected T+C+I adjacency is dense; full-graph adjacency stays CPU sparse.
No sampler, detector, model, RNG, optimizer or network operation is used here.
"""
import dgl
import torch


SCHEMA = "gaga_sequence_hard_forward_st_rational_twohop_v1"
MASK_NAMES = ("train_neg", "train_pos", "train_masked")
DEFAULT_MAX_WORKING_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_SELECTED_NODES = 2048
DEFAULT_MESSAGE_CHUNK = 1024


def estimate_workspace_bytes(num_targets, num_selected, feat_dim, element_size=4,
                             message_chunk=DEFAULT_MESSAGE_CHUNK):
    """Conservative tensor-workspace reservation, excluding input/model storage.

    Includes selected subgraph/index buffers, local adjacency and its backward,
    target-row two-hop intermediates, aggregation buffers and two message chunks.
    This is a pre-allocation bound for this algorithm, not CUDA allocator telemetry
    or a bound on arbitrary native-library/model storage outside the adapter.
    """
    values = (num_targets, num_selected, feat_dim, element_size, message_chunk)
    if any(type(value) is not int or value < 0 for value in values) or min(values[2:]) < 1:
        raise ValueError("Invalid workspace dimensions.")
    t, n, f, b, chunk = values
    return int(16*b*n*n + 32*b*t*n + 32*b*t*f + 8*b*n*f + 2*b*chunk*f)


def _check_binary(value, description):
    detached = value.detach()
    if not bool(torch.isfinite(detached).all()) or not bool(((detached == 0) | (detached == 1)).all()):
        raise ValueError(description + " must have finite binary hard-forward values.")


def _ids(values, device, dtype, description):
    result = torch.as_tensor(values, device=device, dtype=dtype).reshape(-1)
    if len(result) != len(torch.unique(result)):
        raise ValueError(description + " contains duplicate IDs.")
    return result


def _authoritative_masks(data, old_ids, injected_count, original_count, device, dtype):
    supplied = getattr(data, "group_masks", None)
    if supplied is None:
        base_graph = getattr(data, "graph", None)
        if base_graph is None:
            raise ValueError("The cache view needs group_masks or an original graph with masks.")
        supplied = {name: base_graph.ndata[name] for name in MASK_NAMES}
        source = "data_view.graph.ndata"
    else:
        source = "data_view.group_masks"
    outputs = []
    for index, name in enumerate(MASK_NAMES):
        value = supplied[name]
        if not torch.is_tensor(value) or value.ndim != 1 or len(value) != original_count:
            raise ValueError("An authoritative group-mask dimension differs from the original graph.")
        _check_binary(value, "Authoritative group mask")
        selected = value.detach()[old_ids.to(value.device)].to(device=device, dtype=dtype)
        injected = torch.full((injected_count,), 1.0 if index == 2 else 0.0,
                              device=device, dtype=dtype)
        outputs.append(torch.cat((selected, injected)))
    if not bool((sum(outputs) == 1).all()):
        raise ValueError("Authoritative group masks must form a disjoint complete partition.")
    return tuple(outputs), source


def _original_support(matrix, targets_cpu, originals_cpu, width, device):
    """Read only selected rows/columns from an immutable CPU sparse cache."""
    row, col = matrix[targets_cpu.numpy()][:, originals_cpu.numpy()].nonzero()
    support = torch.zeros((len(targets_cpu), width), dtype=torch.bool, device=device)
    if len(row):
        support[torch.as_tensor(row, dtype=torch.long, device=device),
                torch.as_tensor(col, dtype=torch.long, device=device)] = True
    return support


def weighted_group_aggregate(adjacency, features, group_masks, *, message_chunk=DEFAULT_MESSAGE_CHUNK):
    """Weighted group sums/degrees with bounded messages and differentiable weights.

    Adjacency has binary hard-forward support in adapter use. Support is discrete;
    gradients concern values on this selected support, not nonexistent graph arcs.
    A detached chunked sum preserves original row-major accumulation. Its
    backward is the algebraically equivalent matrix product on the same hard
    support, avoiding autograd retaining O(nonzero_pairs * feature_dim) gathered
    message tensors across chunks. Only adjacency/features are saved by matmul.
    """
    if adjacency.ndim != 2 or features.ndim != 2 or adjacency.shape[1] != features.shape[0]:
        raise ValueError("Weighted aggregation dimensions differ.")
    if adjacency.device != features.device or adjacency.dtype != features.dtype:
        raise ValueError("Weighted aggregation device/dtype differs.")
    if len(group_masks) != 3 or type(message_chunk) is not int or message_chunk < 1:
        raise ValueError("Invalid group masks or message chunk.")
    sums, degrees = [], []
    for mask in group_masks:
        if mask.shape != (features.shape[0],):
            raise ValueError("A selected mask has another length.")
        weights = adjacency * mask.unsqueeze(0)
        support = weights.detach() != 0
        src, dst = support.nonzero(as_tuple=True)
        with torch.no_grad():
            hard_sum = features.new_zeros((adjacency.shape[0], features.shape[1]))
            for start in range(0, len(src), message_chunk):
                left, right = src[start:start+message_chunk], dst[start:start+message_chunk]
                hard_sum.index_add_(0, left, features[right] * weights[left, right].unsqueeze(-1))
        proxy_sum = (weights*support.to(weights.dtype)) @ features
        total = hard_sum+(proxy_sum-proxy_sum.detach())
        sums.append(total)
        degrees.append(weights.sum(dim=-1, keepdim=True))
    return torch.cat(sums, dim=-1), torch.cat(degrees, dim=-1)


def updated_sequences(data, graph_perturbed, targets, cand_ids, atk_ids, *, training=True,
                      max_working_bytes=DEFAULT_MAX_WORKING_BYTES,
                      max_selected_nodes=DEFAULT_MAX_SELECTED_NODES,
                      message_chunk=DEFAULT_MESSAGE_CHUNK, return_trace=False):
    """Return target-ordered (T,7,F) sequences, optionally paired with aggregate trace.

    ``data`` exposes original features (or graph.ndata['x']), CPU sparse adj/adj2,
    CPU feat_sum(N,7,F), d(N,7,1), and authoritative group_masks or graph masks.
    Different FIT-TRAIN and official-TRAIN views MUST use their matching caches.
    ``training=True`` consumes graph edge weights with binary hard-forward values.
    ``training=False`` ignores weights, as the frozen original hard evaluation does.
    Existing original nodes/features are never modified; injected IDs are appended.
    """
    if type(training) is not bool or type(return_trace) is not bool:
        raise ValueError("Training/trace flags must be Boolean.")
    if type(max_working_bytes) is not int or max_working_bytes < 1:
        raise ValueError("A positive workspace reservation is required.")
    if type(max_selected_nodes) is not int or max_selected_nodes < 1:
        raise ValueError("A positive selected-node ceiling is required.")
    base_features = getattr(data, "features", None)
    if base_features is None:
        base_features = data.graph.ndata["x"]
    if not torch.is_tensor(base_features) or base_features.ndim != 2:
        raise ValueError("The original feature view must be a matrix.")
    original_count, feat_dim = base_features.shape
    device, idtype = graph_perturbed.device, graph_perturbed.idtype
    target_ids = _ids(targets, device, idtype, "Targets")
    candidates = _ids([] if cand_ids is None else cand_ids, device, idtype, "Candidates")
    injected = _ids(atk_ids, device, idtype, "Injected nodes")
    if not len(target_ids):
        raise ValueError("At least one original target is required.")
    originals = torch.cat((target_ids, candidates))
    selected = torch.cat((originals, injected))
    if len(torch.unique(selected)) != len(selected):
        raise ValueError("Target, candidate and injection roles overlap.")
    if not bool(((originals >= 0) & (originals < original_count)).all()):
        raise ValueError("Targets/candidates are outside the original node view.")
    expected = torch.arange(original_count, original_count+len(injected), device=device, dtype=idtype)
    if graph_perturbed.num_nodes() != original_count+len(injected) or not torch.equal(injected, expected):
        raise ValueError("Injected IDs must be the appended original-count suffix, in order.")
    t, n = len(target_ids), len(selected)
    dtype = graph_perturbed.ndata["x"].dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Floating point raw features are required.")
    element_size = torch.empty((), dtype=dtype).element_size()
    workspace = estimate_workspace_bytes(t, n, feat_dim, element_size, message_chunk)
    if n > max_selected_nodes or workspace > max_working_bytes:
        raise MemoryError("Selected-subgraph sequence workspace exceeds its frozen reservation.")
    if tuple(data.adj.shape) != (original_count, original_count) or tuple(data.adj2.shape) != tuple(data.adj.shape):
        raise ValueError("Original sparse-cache dimensions differ from the feature view.")
    if tuple(data.feat_sum.shape) != (original_count, 7, feat_dim) or tuple(data.d.shape) != (original_count, 7, 1):
        raise ValueError("Original feature-sum/degree caches have another shape.")
    local = dgl.remove_self_loop(dgl.node_subgraph(graph_perturbed, selected))
    if not torch.equal(local.ndata[dgl.NID], selected):
        raise RuntimeError("DGL changed the selected target/candidate/injection order.")
    features = local.ndata["x"]
    if features.shape != (n, feat_dim) or not bool(torch.isfinite(features.detach()).all()):
        raise ValueError("Selected raw feature dimensions/values differ.")
    # Injection changes features only for appended nodes. Cache matching assumes
    # original feature values, and fails instead of silently using stale caches.
    if not torch.equal(features[:len(originals)].detach(),
                       base_features.detach()[originals.to(base_features.device)].to(device=device, dtype=dtype)):
        raise ValueError("Original features changed relative to the supplied cache view.")
    src, dst = local.edges(order="eid")
    if len(src) > n*n:
        raise MemoryError("Parallel selected arcs exceed the simple-graph workspace reservation.")
    if training:
        if "weight" not in local.edata:
            raise ValueError("Training sequences require explicit graph edge weights.")
        edge_values = local.edata["weight"].to(dtype=dtype)
        if edge_values.ndim != 1 or len(edge_values) != len(src):
            raise ValueError("Training edge weights have another shape.")
        _check_binary(edge_values, "Training edge weights")
    else:
        edge_values = features.new_ones(len(src))
    adjacency = features.new_zeros((n, n))
    adjacency[src.long(), dst.long()] = edge_values
    masks, mask_source = _authoritative_masks(data, originals, len(injected), original_count, device, dtype)
    target_cpu, original_cpu = target_ids.detach().cpu().long(), originals.detach().cpu().long()
    original_one = _original_support(data.adj, target_cpu, original_cpu, n, device)
    original_two = _original_support(data.adj2, target_cpu, original_cpu, n, device)
    new_one = adjacency[:t] * (~original_one).to(dtype)
    paths = adjacency[:t] @ adjacency
    self_mask = torch.ones((t, n), dtype=torch.bool, device=device)
    self_mask[torch.arange(t, device=device), torch.arange(t, device=device)] = False
    hard_reachability = ((paths.detach() > 0) & self_mask).to(dtype)
    if training:
        soft_reachability = paths / (1+paths)
        reachability = hard_reachability + (soft_reachability-soft_reachability.detach())
    else:
        reachability = hard_reachability
    new_two = reachability * ((~original_two) & self_mask).to(dtype)
    feat_one, degree_one = weighted_group_aggregate(new_one, features, masks, message_chunk=message_chunk)
    feat_two, degree_two = weighted_group_aggregate(new_two, features, masks, message_chunk=message_chunk)
    zero_features = features.new_zeros((t, feat_dim))
    feat_delta = torch.cat((zero_features, feat_one, feat_two), dim=-1).reshape(t, 7, feat_dim)
    degree_delta = torch.cat((features.new_zeros((t, 1)), degree_one, degree_two), dim=-1).reshape(t, 7, 1)
    cached_sum = data.feat_sum[target_cpu.to(data.feat_sum.device)].to(device=device, dtype=dtype)
    cached_degree = data.d[target_cpu.to(data.d.device)].to(device=device, dtype=dtype)
    numerator, degree = cached_sum+feat_delta, cached_degree+degree_delta
    denominator = torch.where(degree > 0, degree, torch.ones_like(degree))
    sequences = numerator/denominator
    if not bool(torch.isfinite(sequences.detach()).all()):
        raise RuntimeError("Nonfinite updated group sequence.")
    if not return_trace:
        return sequences
    trace = dict(schema=SCHEMA, targets=t, candidates=len(candidates), injected_nodes=len(injected),
                 selected_nodes=n, feature_dim=feat_dim, reserved_workspace_bytes=workspace,
                 max_working_bytes=max_working_bytes, message_chunk=message_chunk,
                 authoritative_masks=mask_source, labels_read=False,
                 hard_forward="Original binary newly-reachable union with cached group means; no self/two-hop multiplicity.",
                 gradient=("Weighted selected-support one-hop; ST rho(P)=P/(1+P) two-hop approximation."
                           if training else "Hard unweighted evaluation; no training edge-weight path."),
                 full_graph_dense_matrix=False)
    return sequences, trace
