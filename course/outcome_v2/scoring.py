"""Isolated, unselected edge-scoring variants for later controlled testing.

The baseline operations are derived from MonTi.edge_generation in the frozen
models/attack_model.py. The repository's upstream license and attribution apply.
This module loads no data/checkpoints and does not execute a model or experiment.
No option is evidence of a successful or approved improvement.
"""

from contextlib import contextmanager
import operator
from types import MethodType

import torch

from utils.utils import gumbel_softmax, st_top_k


def _bool_mask(value, count=None, device=None):
    """Validate an eligibility mask without interpreting labels as eligibility."""
    mask = torch.as_tensor(value, device=device)
    if mask.dtype != torch.bool:
        raise TypeError("eligible_targets must contain Boolean values, not labels or scores")
    if mask.ndim != 1 or (count is not None and mask.numel() != count):
        raise ValueError("eligible_targets must be a one-dimensional mask of target length")
    return mask


def _global_ids(value, count, device):
    if value is None:
        raise ValueError("lowest_global_id requires target_global_ids")
    ids = torch.as_tensor(value, device=device)
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if ids.dtype not in integer_types:
        raise TypeError("target_global_ids must contain integer node IDs")
    if ids.ndim != 1 or ids.numel() != count:
        raise ValueError("target_global_ids must be a one-dimensional vector of target length")
    ids = ids.to(dtype=torch.long)
    if bool((ids < 0).any()) or torch.unique(ids).numel() != count:
        raise ValueError("target_global_ids must be nonnegative and unique")
    return ids


def count_matched_random_mask(eligible_targets, *, seed):
    """Return a uniformly drawn mask with the same size and number of True values.

    Use an explicitly seeded private CPU generator. Neither the global CPU/CUDA
    torch RNG nor Python/NumPy RNG is consumed. A tensor input retains its device;
    a Boolean sequence produces a CPU tensor. The input mask is never modified.
    All-True and all-False masks are returned as clones without a random draw.
    """
    mask = _bool_mask(eligible_targets)
    if isinstance(seed, bool):
        raise TypeError("seed must be an explicit integer, not a Boolean")
    seed = operator.index(seed)
    size, count = mask.numel(), int(mask.sum().item())
    if count == 0 or count == size:
        return mask.clone()
    private_generator = torch.Generator(device="cpu")
    private_generator.manual_seed(seed)
    positions = torch.randperm(size, generator=private_generator, device="cpu")[:count]
    result = torch.zeros_like(mask, dtype=torch.bool)
    result[positions.to(device=mask.device)] = True
    return result


def _mandatory_selection(probabilities, eligible, target_global_ids, tie_policy):
    """Select one target per injected row, retaining the original ST derivative."""
    if eligible is None and tie_policy == "native":
        # Preserve the exact upstream call for all inactive/default policies.
        return st_top_k(probabilities, 1, dim=-1)

    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("active eligibility/tie policies require finite target probabilities")
    selection_scores = probabilities.detach()
    if eligible is not None:
        selection_scores = selection_scores.masked_fill(~eligible.unsqueeze(0), float("-inf"))

    if tie_policy == "native":
        indices = selection_scores.topk(1, dim=-1).indices
    else:
        ids = _global_ids(target_global_ids, probabilities.size(-1), probabilities.device)
        maxima = selection_scores == selection_scores.max(dim=-1, keepdim=True).values
        # IDs are unique; sorting them requires neither a score tolerance nor a
        # sentinel ID that could collide with a legal maximum int64 identifier.
        by_global_id = torch.argsort(ids)
        first_tie_in_id_order = maxima[:, by_global_id].to(dtype=torch.long).argmax(dim=-1)
        indices = by_global_id[first_tie_in_id_order].unsqueeze(-1)

    hard = torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)
    # Never insert -inf into this expression: -(-inf) + (-inf) produces NaNs.
    # Masking affects only the hard decision. B is inference-only; C retains the
    # same identity straight-through derivative with respect to finite scores.
    selection = hard - probabilities.detach() + probabilities
    return selection, indices


def edge_generation(model, z, subgraph, num_targets, num_candidates,
                    node_budget, edge_budget, random_edge=False, *,
                    scoring_role="legacy", eligible_targets=None,
                    target_global_ids=None, tie_policy="native", trace=None):
    """Generate the original adjacency tensor with opt-in draft policies.

    A: scoring_role='injected' changes only normalized source rows from src to
    num_targets + num_candidates + src. 'legacy' retains the frozen expression.

    B: a supplied Boolean eligibility mask restricts the existing mandatory
    target choice only when ada_budget is True and both budgets equal one.
    All-True/all-False masks disable B; other budget regimes and non-adaptive
    allocation ignore B. Explicit A/C options remain active. A partial mask raises during
    model.training before scoring or Gumbel RNG use. No labels are accepted.

    C: tie_policy='lowest_global_id' selects the lowest supplied GLOBAL target
    ID among exact maxima of the mandatory target vector. It changes no near
    ties, remaining-edge choices, or non-adaptive allocation. 'native' is default.

    random_edge=True follows the original random branch and ignores A/B/C.
    Default legacy/no-B/native follows the original numerical operations and RNG
    calls. Original dtype/device choices and edge_budget tensor mutation semantics
    are retained. Return type remains the adjacency tensor. Optional mutable
    trace receives detached snapshots/positions; it performs no file writes.
    """
    adj_attack_shape = (node_budget, z.size(0))
    adj_attack = torch.zeros(adj_attack_shape, device=subgraph.device)
    mask_tril = torch.tril(torch.ones(adj_attack_shape, dtype=torch.bool),
                           diagonal=num_targets + num_candidates - 1).to(subgraph.device)

    if random_edge:
        rand = torch.rand_like(adj_attack)
        _, indices = rand[mask_tril].topk(edge_budget)
        src, dst = mask_tril.nonzero(as_tuple=True)
        adj_attack[src[indices], dst[indices]] = 1
        if trace is not None:
            trace.update(random_edge=True, scoring_role_applied=None,
                         eligibility_status="ignored_random_edge", tie_policy_applied=None)
        return adj_attack

    if scoring_role not in ("legacy", "injected"):
        raise ValueError("scoring_role must be 'legacy' or 'injected'")
    if tie_policy not in ("native", "lowest_global_id"):
        raise ValueError("tie_policy must be 'native' or 'lowest_global_id'")

    eligible = None
    eligibility_status = "not_supplied"
    if eligible_targets is not None:
        applicable = model.ada_budget and node_budget == 1 and edge_budget == 1
        if not applicable:
            eligibility_status = "ignored_nonapplicable"
        else:
            supplied = _bool_mask(eligible_targets, num_targets, subgraph.device)
            count = int(supplied.sum().item())
            if count == 0:
                eligibility_status = "empty_fallback"
            elif count == num_targets:
                eligibility_status = "all_fallback"
            else:
                if model.training:
                    raise RuntimeError("a restricted eligible_targets mask is inference-only")
                eligible = supplied
                eligibility_status = "restricted"

    e = model.linear_edge(z)
    e = torch.nn.functional.normalize(e, dim=-1)

    src, dst = mask_tril.nonzero(as_tuple=True)
    source_rows = src if scoring_role == "legacy" else num_targets + num_candidates + src
    edge_scores = torch.sum(e[source_rows] * e[dst], dim=-1)
    if trace is not None:
        trace.update(random_edge=False, scoring_role_applied=scoring_role,
                     start_atk=num_targets + num_candidates,
                     source_rows=source_rows.detach().clone(), destination_rows=dst.detach().clone(),
                     raw_edge_scores=edge_scores.detach().clone(),
                     eligibility_status=eligibility_status,
                     applied_eligible_targets=None if eligible is None else eligible.detach().clone(),
                     tie_policy_applied=tie_policy if model.ada_budget else None)

    temperature = model.temp_scheduler.get_temperature()
    epsilon = model.eps_scheduler.get_epsilon()
    edge_scores = gumbel_softmax(edge_scores, temperature, epsilon, dim=0, randomness=model.training)

    edge_scores_mat = torch.zeros(adj_attack_shape, device=subgraph.device)
    edge_scores_mat[src, dst] = edge_scores
    if model.ada_budget:
        mandatory, indices = _mandatory_selection(
            edge_scores_mat[:, :num_targets], eligible, target_global_ids, tie_policy)
        adj_attack[:, :num_targets] = mandatory
        if trace is not None:
            trace.update(mandatory_probabilities=edge_scores_mat[:, :num_targets].detach().clone(),
                         mandatory_indices=indices.detach().clone())

        edge_budget -= node_budget
        if edge_budget > 0:
            mask_empty = (adj_attack == 0)
            mask = mask_empty & mask_tril
            edge_budget = min(edge_budget, mask.sum())
            adj_attack_remaining, _ = st_top_k(edge_scores_mat[mask], edge_budget, dim=0)
            adj_attack[mask] = adj_attack_remaining
    else:
        assert edge_budget % node_budget == 0
        degree_budget = edge_budget // node_budget
        adj_attack, _ = st_top_k(edge_scores_mat, degree_budget, dim=-1)

    return adj_attack


def bind_edge_generation(model, **options):
    """Bind the draft to one model instance and return an idempotent restore.

    No class or source file is modified. Existing instance overrides are retained
    and restored; otherwise restoration removes the temporary instance attribute
    so the original class method is visible again. Nested bindings must be
    restored in last-in/first-out order. Masks/IDs are bound by reference; use a
    fresh per-group binding when those inputs change.
    """
    if not callable(getattr(model, "edge_generation", None)):
        raise TypeError("model must expose a callable edge_generation method")
    sentinel = object()
    previous = model.__dict__.get("edge_generation", sentinel)
    fixed_options = dict(options)

    def bound(current, z, subgraph, num_targets, num_candidates,
              node_budget, edge_budget, random_edge=False):
        return edge_generation(current, z, subgraph, num_targets, num_candidates,
                               node_budget, edge_budget, random_edge, **fixed_options)

    model.edge_generation = MethodType(bound, model)
    active = True

    def restore():
        nonlocal active
        if not active:
            return
        if previous is sentinel:
            delattr(model, "edge_generation")
        else:
            model.edge_generation = previous
        active = False

    return restore


@contextmanager
def temporary_edge_generation(model, **options):
    """Temporarily bind the draft and always restore the previous method."""
    restore = bind_edge_generation(model, **options)
    try:
        yield model
    finally:
        restore()
