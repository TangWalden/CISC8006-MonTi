# Unselected draft scoring adapters

**Preparation only. No direction has been selected or approved.** This folder
does not load datasets/checkpoints, launch experiments, or modify frozen source
files. The code has not been executed or validated by its author. Separate
synthetic tests, review, an owner decision, and a frozen protocol are required
before any real experiment. No improvement or novelty is claimed.

The baseline operations are copied from the frozen `MonTi.edge_generation` in
`models/attack_model.py`, using the same `gumbel_softmax` and `st_top_k` primitives
from `utils/utils.py`. Upstream attribution and the repository license apply.

## Functional interface

`course.outcome_v2.edge_generation(model, z, subgraph, num_targets,
num_candidates, node_budget, edge_budget, random_edge=False, *,
scoring_role='legacy', eligible_targets=None, target_global_ids=None,
tie_policy='native', trace=None)` returns the adjacency tensor, as the original
method does. `trace`, when supplied as a mutable dictionary, receives detached
score/mask/index snapshots and does not change the return type or write files.

- **A, source role:** `scoring_role='injected'` changes only the source gather
  from `e[src]` to `e[num_targets + num_candidates + src]`. The default is
  `'legacy'`. Masks, projections, normalization, Gumbel scheduling/randomness,
  mandatory targets, remaining budgets, non-adaptive allocation, and the random
  branch retain the original operations. A consistent A study would require
  retraining; this module does not perform it.
- **B, supplied eligibility:** `eligible_targets` must be a one-dimensional
  Boolean mask in target order, not labels or scores. It applies only to the
  existing mandatory-target stage with adaptive allocation and both budgets
  equal to one. A partially eligible set is inference-only and raises if
  `model.training` is true, before scoring or Gumbel draws. All-True and all-False
  masks disable B and are permitted in training; explicitly selected A/C options
  remain active, so identity is relative to the same A/C configuration.
  "Empty" means an all-False mask of target length, not a length-zero vector.
  Other budget regimes and non-adaptive allocation ignore B. Eligibility must
  be supplied by a separately reviewed caller; this module computes no clean
  predictions and consumes no labels.
- **C, exact ties:** `tie_policy='lowest_global_id'` chooses the lowest supplied
  GLOBAL target ID among exact maxima of the mandatory-target probability
  vector. IDs must be unique, nonnegative integers in target order. There is no
  tolerance or perturbation of near ties. The default `'native'` preserves the
  original top-k call. C does not alter remaining-edge or non-adaptive choices.
- **Random edges:** `random_edge=True` takes the unchanged random branch and
  ignores A/B/C. Its random draw and mask semantics are retained.

Only hard index selection uses an eligibility mask containing negative infinity.
The straight-through expression uses the finite original probabilities, avoiding
`inf - inf` NaNs. C retains the original identity ST derivative. Active partial B
is not a training policy; any gradient through an evaluation call is not evidence
of a supported B training procedure.

The default legacy/no-B/native path calls the original primitives with the same
arguments and preserves their dtype/device choices. In particular, it does not
silently switch adjacency tensors to `z.dtype`, change temperature behavior in
evaluation, alter native tie rules, or remove the original in-place mutation
when a tensor `edge_budget` reaches the adaptive remaining-budget operation.
These are intended compatibility properties, pending regression tests.

The caller must supply valid original-method budgets and dimensions. This is
not a general input or budget validator: for example, the inherited adaptive
path assigns one mandatory edge per injected node even if an invalid caller
supplies fewer edges than nodes. Constant-degree capacity behavior is also
inherited. Synthetic legality checks cover registered valid-budget cases;
preserving the frozen method does not establish universal budget safety.

## Independent random-mask control

`count_matched_random_mask(eligible_targets, *, seed)` returns a uniformly drawn
Boolean mask with the same target length and eligible count. It uses an explicitly
seeded private CPU `torch.Generator`, retaining a tensor input's device. It does
not consume global torch CPU/CUDA, NumPy, or Python random state. All/empty masks
return clones. The caller must register how run/group identifiers determine the
explicit seed; the helper does not invent that policy.

## Reversible runtime binding

`bind_edge_generation(model, **options)` binds the function to that one instance
and returns an idempotent restore callback. `temporary_edge_generation(model,
**options)` is a context manager that restores on normal or exceptional exit.
Existing instance overrides are restored; class methods are never edited. Use
last-in/first-out restoration for nested bindings. Masks and IDs are bound by
reference, so group-specific inputs require a fresh binding or careful caller
management. The helper does not patch module-global primitives.

## Review and test handoff

The root agent owns tests and protocol/status records. The required synthetic
checks should establish default output/gradient/RNG parity with the frozen
method; injected-source row gradients; legal masks and node/edge budgets with
remaining-edge allocation; non-adaptive and random-branch parity; all/empty and
nonapplicable eligibility identity; rejection of active B during training;
eligibility without NaNs; exact global-ID tie selection and unchanged unique
maxima; private-RNG mask counts/repeatability without global-state consumption;
and restoration after exceptions/nested bindings. Real data, checkpoints,
training, and inference are outside this preparation task.
