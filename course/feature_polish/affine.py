"""Exact feature dependence of hard GAGA means on a fixed attack graph.

All topology and group memberships stay fixed. Injected nodes are unknown, so
only unknown-group one/two-hop means (tokens3/6) change with their raw features.
The canonical baseline retains every original-node contribution, including
original nodes newly reached through an injection. No straight-through topology
derivative, detector call, model call or parameter update occurs in this module.
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from course.dual_surrogate.sequence_adapter import updated_sequences


SCHEMA = 'gaga_fixed_hard_topology_injected_feature_affine_v1'


def _ids(value, description):
    raw = torch.as_tensor([] if value is None else value).detach().cpu()
    if raw.ndim != 1 or (raw.numel() and raw.dtype not in
                         (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError(description+' must be a one-dimensional integer ID list.')
    result = raw.to(torch.long).numpy()
    if len(np.unique(result)) != len(result):
        raise ValueError(description+' contains duplicate IDs.')
    return result


@dataclass(frozen=True)
class FixedFeatureAffine:
    base_sequence: torch.Tensor
    coefficients: torch.Tensor
    input_features: torch.Tensor
    trace: dict

    def sequence(self, features):
        if (not torch.is_tensor(features) or features.shape != self.input_features.shape
                or features.device != self.input_features.device
                or features.dtype not in (torch.float32, torch.float64)):
            raise ValueError('Refined features must retain the injected-feature shape/device and floating dtype.')
        if not bool(torch.isfinite(features.detach()).all()):
            raise ValueError('Refined features must be finite.')
        baseline = self.input_features.to(dtype=features.dtype)
        weights = self.coefficients.to(dtype=features.dtype)
        constant = self.base_sequence.to(dtype=features.dtype)
        return constant+torch.einsum('tki,if->tkf', weights, features-baseline)


def features_from_unit_box(unit, minimum, maximum):
    """Map normalized coordinates to the unchanged raw feature box.

    Zero-width dimensions are exactly frozen by multiplication with zero.
    Projection of optimizer steps into [0,1] belongs to the refinement caller.
    The function itself stays affine and supports interior finite differences.
    """
    if not torch.is_tensor(unit) or unit.ndim != 2 or unit.dtype not in (torch.float32, torch.float64):
        raise ValueError('Normalized feature coordinates must be a floating matrix.')
    lower = torch.as_tensor(minimum, device=unit.device, dtype=unit.dtype)
    upper = torch.as_tensor(maximum, device=unit.device, dtype=unit.dtype)
    if (lower.shape != (unit.shape[1],) or upper.shape != lower.shape
            or not bool(torch.isfinite(unit.detach()).all() and torch.isfinite(lower).all()
                        and torch.isfinite(upper).all() and (upper >= lower).all())):
        raise ValueError('Finite ordered feature-box bounds must match the feature width.')
    return lower+unit*(upper-lower)


def build_affine(view, graph, targets, candidates, atk, x0, *, base_sequence=None):
    """Build S(X)=S0+W@(X-X0) for this immutable hard attack topology.

    ``view`` supplies original CSR adjacency without self loops, original raw
    features and authoritative original unknown masks. With no external S0 it
    must additionally supply the complete canonical sequence-cache view.
    The only original-original topology assumption is the existing capture
    contract: those arcs have not changed from ``view.adj``. Injected incident
    arcs are read from the supplied graph; duplicated in/out queries coalesce.
    """
    original_features = getattr(view, 'features', None)
    if original_features is None:
        original_features = view.graph.ndata['x']
    if not torch.is_tensor(original_features) or original_features.ndim != 2:
        raise ValueError('Original raw features must be a matrix.')
    n, width = original_features.shape
    target_ids, candidate_ids, injected_ids = (_ids(targets, 'Targets'), _ids(candidates, 'Candidates'),
                                               _ids(atk, 'Injected nodes'))
    count, injected = len(target_ids), len(injected_ids)
    if count < 1 or injected < 1:
        raise ValueError('At least one target and injected node are required.')
    original_ids = np.concatenate((target_ids, candidate_ids))
    if (len(np.unique(original_ids)) != len(original_ids)
            or np.any(original_ids < 0) or np.any(original_ids >= n)):
        raise ValueError('Targets/candidates overlap or fall outside the original graph.')
    if graph.num_nodes() != n+injected or not np.array_equal(injected_ids, np.arange(n, n+injected)):
        raise ValueError('Injected IDs must be the complete appended tail in order.')
    if (not torch.is_tensor(x0) or x0.shape != (injected, width) or x0.dtype != torch.float32
            or x0.device != graph.device or not bool(torch.isfinite(x0.detach()).all())):
        raise ValueError('Baseline injected features must be finite float32 on the graph device.')
    graph_features = graph.ndata['x']
    if (graph_features.shape != (n+injected, width)
            or not torch.equal(graph_features[n:].detach(), x0.detach())
            or not torch.equal(graph_features[:n].detach(), original_features.detach().to(graph.device))):
        raise ValueError('Graph raw features differ from the frozen original/injected inputs.')

    adjacency = sp.csr_matrix(view.adj, copy=True)
    if adjacency.shape != (n, n):
        raise ValueError('Original adjacency dimensions differ.')
    adjacency.sum_duplicates();adjacency.eliminate_zeros()
    if adjacency.data.size and (not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0)):
        raise ValueError('Original adjacency must have finite nonnegative support.')
    adjacency.setdiag(0);adjacency.eliminate_zeros()
    adjacency.data = np.ones(adjacency.nnz, dtype=np.int64)
    adjacency = adjacency.astype(np.int64)

    masks = getattr(view, 'group_masks', None)
    if masks is None:
        masks = {name:view.graph.ndata[name] for name in ('train_neg', 'train_pos', 'train_masked')}
    unknown = masks['train_masked']
    if not torch.is_tensor(unknown) or unknown.shape != (n,):
        raise ValueError('The original unknown mask has another shape.')
    unknown = unknown.detach().cpu().numpy()
    if not np.isfinite(unknown).all() or not np.isin(unknown, [0, 1]).all():
        raise ValueError('The original unknown mask must be finite binary.')
    unknown = np.concatenate((unknown.astype(np.int64), np.ones(injected, dtype=np.int64)))

    query_ids = torch.as_tensor(injected_ids, device=graph.device, dtype=graph.idtype)
    in_src, in_dst = graph.in_edges(query_ids)
    out_src, out_dst = graph.out_edges(query_ids)
    src = np.concatenate((in_src.detach().cpu().numpy(), out_src.detach().cpu().numpy())).astype(np.int64)
    dst = np.concatenate((in_dst.detach().cpu().numpy(), out_dst.detach().cpu().numpy())).astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if np.any(src < 0) or np.any(dst < 0) or np.any(src >= n+injected) or np.any(dst >= n+injected):
        raise ValueError('An injected incident arc has an invalid endpoint.')
    original_endpoints = np.concatenate((src[src < n], dst[dst < n]))
    if not set(map(int, original_endpoints)) <= set(map(int, original_ids)):
        raise ValueError('An injected arc reaches an original node outside targets/candidates.')
    incident = sp.coo_matrix((np.ones(len(src), dtype=np.int64), (src, dst)), shape=(n+injected, n+injected)).tocsr()
    augmented = sp.block_diag((adjacency, sp.csr_matrix((injected, injected), dtype=np.int64)), format='csr')+incident
    augmented.sum_duplicates();augmented.eliminate_zeros()
    augmented.data.fill(1)

    one = augmented[target_ids].tocsr()
    two = (one@augmented).tocoo()
    # Reachability is a set, including when two distinct paths reach one node.
    keep = two.col != target_ids[two.row]
    two = sp.coo_matrix((np.ones(int(keep.sum()), dtype=np.int64), (two.row[keep], two.col[keep])),
                        shape=one.shape).tocsr()
    two.sum_duplicates();two.eliminate_zeros();two.data.fill(1)
    degrees = [np.asarray(reach@unknown).reshape(-1).astype(np.float64)+1 for reach in (one, two)]
    # Both unknown tokens always include the masked center itself once.
    coefficients = np.zeros((count, 7, injected), dtype=np.float32)
    memberships = []
    for token, reach, denominator in zip((3, 6), (one, two), degrees):
        membership = reach[:, n:].toarray().astype(np.float64)
        coefficients[:, token, :] = (membership/denominator[:, None]).astype(np.float32)
        memberships.append(int(membership.sum()))

    baseline_supplied = base_sequence is not None
    if base_sequence is None:
        base_sequence = updated_sequences(view, graph, targets, candidates, atk, training=False)
    if (not torch.is_tensor(base_sequence) or base_sequence.shape != (count, 7, width)
            or base_sequence.dtype != torch.float32 or base_sequence.device != graph.device
            or not bool(torch.isfinite(base_sequence.detach()).all())):
        raise ValueError('Canonical baseline sequence must be finite float32 (T,7,F) on the graph device.')
    weights = torch.as_tensor(coefficients, dtype=torch.float32, device=graph.device)
    return FixedFeatureAffine(base_sequence=base_sequence.detach().clone(), coefficients=weights,
        input_features=x0.detach().clone(), trace=dict(schema=SCHEMA, targets=count,
            candidates=len(candidate_ids), injected_nodes=injected, feature_dim=width,
            active_tokens=[3, 6], injected_hop_memberships=memberships,
            minimum_unknown_denominators=[float(value.min()) for value in degrees],
            maximum_unknown_denominators=[float(value.max()) for value in degrees],
            masked_self_count_per_hop=1, canonical_baseline_supplied=baseline_supplied,
            original_original_edges_assumed_unchanged=True,
            topology='Fixed binary hard support; deduplicated two-hop reachability, center removed.',
            derivative='Exact fixed-topology mean derivative for injected raw features, up to floating arithmetic.',
            model_calls=0, optimizer_steps=0))
