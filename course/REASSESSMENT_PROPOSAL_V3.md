# Evidence-based research reassessment, version 3

Status: proposal for owner review, 2 November 2026. No new direction, pilot,
modified-group evaluation, training or GAGA test is approved by this document.
The owner requested ideas grounded in actual experiments/code/data, questioned
the public value of attack-cost reduction, and challenged initial target-group
selection fairness. B is retained as an untested historical hypothesis and is
not recommended here as the main contribution. A remains a conformance control.
The earlier negative pilots and every frozen scientific result remain unchanged.

## A new observation from the same saved predictions

The [reassessment record](../evidence/reassessment_20261102_v1.json) recomputes
published GCN validation counts only. Both summaries below use the same 592
groups and the same three previously selected checkpoints in each arm.

| Summary of saved validation outcomes | Learned candidates MR % | Random candidates MR % | Learned minus random, pp |
|---|---:|---:|---:|
| Registered target-occurrence weighting | 97.05823091 | 97.21270737 | -0.15447646 |
| Every target group receives equal weight | 97.14085253 | 96.96550067 | +0.17535186 |

The small mean ordering reverses when the estimand changes. This is descriptive
sensitivity, not statistical significance, a newly selected winner, evidence of
author misconduct or a reason to replace the original metric. These are selected
surrogate-validation snapshots, not fresh checkpoints selected by group means.
The frozen GAGA reproduction conclusion remains inconclusive.

The complete saved metadata also records 598 unique validation target nodes
across 4,963 group-node occurrences per checkpoint. This count comes from a
single previously generated bin covering all 592 groups, not from adding
overlapping bin counts. Repeated occurrence represents a node under different
group attacks; it is not inherently an error or leakage. It does change the
population being weighted. A unique-node average would require a stated way of
combining its different attack contexts and full membership/prediction records;
the public top-ten-node table cannot provide that metric.

Both old pilot variants remain negative under the additional group-equal
summary: coverage -0.02570009 pp and size sampling -0.21378968 pp. Their original
occurrence-weighted differences remain -0.18331806 and -0.54995417 pp. No negative
result is being hidden by selecting a favorable metric.

## Candidate 1: group-and-budget-controlled evaluation robustness

Research question: how stable are comparative conclusions to grouping and
evaluation weights when target exposure and structural resources are controlled?

This would improve the risk-evaluation procedure. It does not seek cheaper fraud
or claim to have improved a detector. The contribution would be an auditable
stress-test implementation and a bounded characterization of robustness, not the
invention of group means, occurrence means or randomization methods.

### Module 1: explicit estimands and clean/attacked transitions

Preserve registered occurrence-MR as the main reproduction measure. Add a
predefined group-equal summary and, only when complete records exist, a separately
defined node-balanced sensitivity summary. Report effects with their units and
do not choose the most favorable summary after measurement.

For the same fixed groups, report clean-correct to attacked-wrong, clean-wrong to
attacked-correct, unchanged-wrong and unchanged-correct counts. Require
`attacked_errors - clean_errors = new_errors - recoveries` with an unchanged
denominator. The existing selected GPU diagnosis already gives 57 clean errors,
39 new errors and zero recoveries, hence 96 errors in 243 occurrences. It is not
a whole-validation estimate. Completing the missing paired validation evidence
would require a separately registered and approved inference protocol.

No filtering out initially wrong targets from the primary MR, dropping difficult
groups, redefining old results or treating overlapping occurrences as independent
seeds is permitted. Benign-node false positives and overall defensive benefit
are not measured by the available target-only records.

### Module 2: matched target-group perturbations

Proposed code/data mechanism, not implemented or evaluated:

1. Keep the original published graph, targets, split and original-group result
   separate and intact. Begin with validation-only sensitivity at fixed attacker
   checkpoints; do not change training lists and call that a retraining result.
2. Represent group membership as a binary group-by-node matrix. A legal two-group
   membership swap can preserve every group size and every node's occurrence
   count. Perform it only within the same split and forbid duplicate membership.
3. Restrict swaps by exact relevant degree/budget contributions, then independently
   recompute and require equality of every group's registered node/edge budgets.
   If exact matching is infeasible, report that outcome; do not quietly relax it.
4. Preserve each group's original first target and its sequence position in the
   primary grouping contrast. The legacy scorer uses the first target as source,
   so changing that identity would confound membership with score-source role.
   A separate explicitly labeled order contrast can investigate that factor.
5. Freeze perturbation seeds, attempts, accepted swaps, stopping and reporting
   rules before any model evaluation. Record unchanged and changed memberships,
   overlap, internal edges, shared neighborhoods and candidate-pool geometry.

Holding the first target's identity does not hold its contextualized Transformer
representation fixed when other members change. This contrast measures the
end-to-end effect of grouping with that source identity controlled; it cannot
isolate an effect independent of the legacy scorer's representation mechanism.

Budget equality must be checked against the retained implementation. Its
incidence-based helper differs from the paper's unique-neighbor union, which may
change after swaps even when target degrees match. Do not claim both formulas
are invariant without checking both. A paper-budget comparison must also define
whether and over which complete group population its average neighbor quantity
is recomputed. Structural budgets and extra inference cost are separate quantities.

Equal budget caps do not establish equal realized attack size. Audit actual
injected-node counts and added non-self undirected-edge counts, because a changed
candidate pool or legal edge space can cause an underfilled budget. Preserve and
report all cases with such mismatches under the registered denominator; do not
drop them after querying a model. If realized sizes differ, describe the contrast
as cap-matched rather than equal-cost. Before freezing an order control, verify
that sparse-matrix/DGL processing preserves the intended source identity and order.

Such controls retain node exposure and group size, but do not retain real-world
collusion semantics or all graph geometry. They are controlled null scenarios,
not more truthful target lists. A constrained finite swap chain is not proven
uniform, well mixed, or composed of independent samples. Degree-preserving swap
methods have substantial prior work; for example [Erdos et al. 2018](https://doi.org/10.1371/journal.pone.0201995)
analyzes mixing under particular degree-sequence conditions. This proposal does
not establish those conditions for these additional constraints or claim a new
sampling theorem.

Falsification: if comparative directions/effects remain stable across registered
summaries and matched perturbations, the hypothesis of material sensitivity is
not supported. If directions change, report conditional sensitivity and its
magnitude, not proof that authors selected groups to favor MonTi. Three model
seeds remain the training replication units; perturbation draws are nested.

## Candidate 2: input-order and execution stability as a control

New saved-pair decomposition: of 48 CPU/GPU generator pairs, 12 have the same
candidate set and order, yet 9 of those have different chosen endpoints. Among
the other 36 pairs, 26 endpoints differ. Candidate substitution therefore is not
necessary for every observed difference. This does not identify the remaining
cause: numerical features/representations still differ within those 12 pairs.

All 96 captured scores use the first target representation; all 48 original
device pairs use the same target order. A pure target-order intervention has
not been measured. The edge scorer in `models/attack_model.py` uses injected-relative
`src` to index the full sequence; this establishes an order-dependent source
identity in the legacy expression, not an observed change in attack performance.

Proposed controlled test: keep the target set, graph, weights, budgets and mapped
random draws fixed; change only input serialization order. Map outputs back to
global identities. Contrast the legacy expression with a locked-original-source
control; paper-role alignment and exact-tie conventions remain separate controls.
Existing retained tensors may first support an algebra-only probe, but that would
not be a rerun of the Transformer or evidence of end-to-end invariance.

Record endpoint/feature/prediction disagreements, not just MR. Canonical ordering
and ID tie rules are engineering controls, not established algorithmic novelty.
ID-based stability under serialization does not establish invariance to renaming
node IDs, numerical device identity or improved security. This direction is best
used to protect Candidate 1's comparisons from execution confounding.

## Ideas rejected or downgraded by actual evidence

- Coverage anchors and size sampling already had negative three-seed pilots.
  Neither is revived as a newly justified positive method.
- Simple loss reweighting is not currently recommended. The hinge already zeros
  a currently misclassified target's own loss term. This does not permanently
  mask a target that was wrong on the clean graph.
- Every formal/pilot training configuration uses batch size 1. Replacing the
  batch mean with total node loss divided by the current batch's node count
  would equal the existing loss. It would not be a new training method.
- The old size-proportional replacement sampler already targets an
  occurrence-weighted hinge expectation at fixed parameters. Its negative pilot
  is relevant counterevidence; it is not identical to a zero-one MR objective.
  A separate uniform-replacement training control was not run in those nine
  pilots, despite being contemplated by earlier policies/protocol wording.
- A new defense is not proposed from target-only errors alone. Normal-node false
  positives, failure mechanisms and a fair protection/utility comparison are
  missing. No detector benefit is claimed.

## Recommendation, limits and next checkpoint

Recommended for discussion: Candidate 1 as the main research question, with
Candidate 2 as a comparison-validity control. This is a recommendation, not owner
selection. Original-list construction code was not found in the inspected current
official tree; that gap motivates auditability, not an accusation or proof of bias.

The next deliverable, if the direction is accepted, would be a concrete bounded
validation-only protocol and independently checked matching/order controls before
any real-data inference. It must state fixed inputs, seeds, estimands, invariants,
failure rules and aggregate cost. Full GAGA testing and final freeze retain their
separate owner checkpoints. Teacher acceptance of an evaluation-method improvement
as the course's required extension remains unconfirmed; two approved pilots, a
complete improvement/control evaluation and human verification are not fulfilled
by this memo. No new full experiment, group perturbation or defense was run.

All new arithmetic is post hoc and uses saved GCN validation records. Preliminary
numbers were inspected before the reproducible script was written, as disclosed
in its output. Earlier reports/slides remain dated drafts of their original
evidence snapshots; this memo does not silently update their scientific inputs.
