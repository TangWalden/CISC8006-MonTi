"""Candidate scores conditional on frozen upstream features and target edges.

This module has no classifier, label, prediction or checkpoint interface.
Upstream tensors are detached at the boundary. The attachment summary consists
of its actual target-feature mean and connection count; it is not a lossless
encoding of every possible attachment matrix.

The optional straight-through weights are a biased optimization surrogate.
They do not differentiate discrete graph topology or hard-graph degree changes,
and are not a claim of an exact gradient of the discrete attack objective.
"""

import math

import torch
from torch import nn


class DownstreamCandidateScorer(nn.Module):
    """Score B-by-C candidate pairs after observing B-by-T target decisions."""

    def __init__(self, feature_dim=768, context_dim=64, hidden_dim=64):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (feature_dim, context_dim, hidden_dim)):
            raise ValueError("All model dimensions must be positive integers.")
        self.feature_dim = feature_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.aux_projection = nn.Linear(feature_dim, hidden_dim)
        self.target_projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.count_projection = nn.Linear(1, hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(feature_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        self.row_bias = nn.Linear(hidden_dim, 1, bias=False)
        self.candidate_bias = nn.Linear(hidden_dim, 1, bias=False)

    def conditioning_inputs(self, x_aux, x_targets, adj_targets):
        """Return detached features, actual attachment means, and log counts."""
        if (x_aux.ndim != 2 or x_targets.ndim != 2
                or x_aux.shape[0] < 1 or x_targets.shape[0] < 1
                or x_aux.shape[1] != self.feature_dim
                or x_targets.shape[1] != self.feature_dim
                or adj_targets.shape != (len(x_aux), len(x_targets))):
            raise ValueError("Upstream feature or target-attachment shapes differ.")
        if not x_aux.is_floating_point() or not x_targets.is_floating_point():
            raise ValueError("Raw upstream features must be floating point.")
        if x_aux.device != x_targets.device or adj_targets.device != x_aux.device:
            raise ValueError("All upstream tensors must share one device.")
        if x_aux.dtype != x_targets.dtype:
            raise ValueError("Raw upstream feature dtypes must match.")
        x_aux = x_aux.detach()
        x_targets = x_targets.detach()
        adjacency = adj_targets.detach()
        if (not bool(torch.isfinite(x_aux).all() and torch.isfinite(x_targets).all()
                     and torch.isfinite(adjacency).all())
                or not bool(((adjacency == 0) | (adjacency == 1)).all())):
            raise ValueError("Upstream features must be finite and attachments binary.")
        adjacency = adjacency.to(dtype=x_aux.dtype)
        counts = adjacency.sum(dim=1, keepdim=True)
        if bool((counts < 1).any()):
            raise ValueError("Each auxiliary node needs an upstream target attachment.")
        attached_mean = adjacency.matmul(x_targets) / counts
        return x_aux, attached_mean, torch.log1p(counts)

    def forward(self, x_aux, x_targets, adj_targets, x_candidates, h_candidates):
        auxiliary, target_mean, log_count = self.conditioning_inputs(
            x_aux, x_targets, adj_targets)
        if (x_candidates.ndim != 2 or x_candidates.shape[1] != self.feature_dim
                or h_candidates.shape != (len(x_candidates), self.context_dim)):
            raise ValueError("Candidate feature and cached-context shapes differ.")
        if (not x_candidates.is_floating_point() or not h_candidates.is_floating_point()
                or x_candidates.dtype != auxiliary.dtype
                or h_candidates.dtype != auxiliary.dtype
                or x_candidates.device != auxiliary.device
                or h_candidates.device != auxiliary.device):
            raise ValueError("Candidate inputs must match upstream floating dtype and device.")
        candidates = x_candidates.detach()
        context = h_candidates.detach()
        if not bool(torch.isfinite(candidates).all() and torch.isfinite(context).all()):
            raise ValueError("Candidate features and cached contexts must be finite.")
        row = torch.tanh(self.aux_projection(auxiliary)
                         + self.target_projection(target_mean)
                         + self.count_projection(log_count))
        column = torch.tanh(self.candidate_projection(candidates)
                            + self.context_projection(context))
        scores = row.matmul(column.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        scores = scores + self.row_bias(row) + self.candidate_bias(column).transpose(0, 1)
        if not bool(torch.isfinite(scores).all()):
            raise RuntimeError("Nonfinite downstream candidate scores.")
        return scores


def straight_through_mask(scores, hard_mask, quota, temperature=1.0):
    """Keep an exact hard forward mask with a scaled-softmax backward surrogate.

    Hard choices are made outside this function by the frozen deterministic
    selector. A zero or full quota has no free choice and therefore zero
    gradient. The graph builder may retain only hard-selected physical edges;
    doing so is an explicitly biased selected-edge gradient estimator.
    """
    if (scores.ndim != 2 or not scores.is_floating_point()
            or type(quota) is not int or quota < 0 or quota > scores.numel()
            or hard_mask.shape != scores.shape or hard_mask.device != scores.device
            or isinstance(temperature, bool) or not math.isfinite(float(temperature))
            or float(temperature) <= 0):
        raise ValueError("Invalid scores, quota, hard-mask geometry or temperature.")
    if (not bool(torch.isfinite(scores).all() and torch.isfinite(hard_mask).all())
            or not bool(((hard_mask == 0) | (hard_mask == 1)).all())
            or int(hard_mask.detach().sum()) != quota):
        raise ValueError("Hard selection must be finite, binary and have its exact quota.")
    hard = hard_mask.detach().to(dtype=scores.dtype)
    if quota == 0 or quota == scores.numel():
        return hard + scores * 0
    soft = torch.softmax(scores.reshape(-1) / float(temperature), dim=0)
    soft = soft.reshape_as(scores) * quota
    return hard + (soft - soft.detach())
