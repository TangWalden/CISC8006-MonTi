# Method-level alternatives for discussion

Status: 2 November 2026, unselected and untested. The owner declined the v3
grouping/evaluation route and requested other improvement methods. Preserve that
proposal and its post-hoc arithmetic as historical analysis, not an active plan.
No new training, real-data method evaluation, detector queries or final test is
authorized by this memo. Existing scientific files and negative results stand.

## Budget-conditioned injection representation

Saved evidence: 432 of 438 residual target occurrences at the selected learned
GCN validation snapshots lie in the lowest training-derived nodes-per-target
bin, which contains 18.32% of exposures. This association is confounded by degree,
geometry and budgets. It does not establish a missing-input cause.

Static code: [MonTi.forward](../models/attack_model.py) constructs injected seeds
and projects `x_atk` before calling `edge_generation`. The edge budget is read
only in the subsequent edge-generation path. Holding graph, targets, candidates,
node budget and random inputs fixed, changing only the edge budget therefore
does not change the original generated features. Degree, beta, target context
and injected-token count already provide indirect resource information; do not
describe the original model as entirely budget-blind.

Proposed change: encode two independent training-normalized quantities, such as
nodes per target and edges per injected node, with a small MLP and add its output
to existing injected seed tokens. Preserve the original random seeds for multiple
injections, token count/positions, candidate rule, loss, sampling and actual
node/edge caps. Shared Transformer states drive both features and edges, so this
is a representation extension, not a strictly feature-only intervention.

This differs from `ada_budget`, which allocates output edges, and from the old
size sampler, which changes training exposure. The feature/edge output constraints
remain unchanged. It does not promise cheaper attacks or a protective effect.

Essential controls, to be resolved before a freeze: the original model, the same
added MLP receiving a fixed training-derived input, and a same-capacity degree-only
input. Use matched initialization/training resources and paired seeds. The fixed
input arm must not be a zero-input, no-bias branch with no gradients. Keep the
original score-source and tie policy in the first contrast; do not combine A/C
into this candidate and attribute the result to one change. Any later corrected-
role study needs its own paired control. Audit actual attack sizes and added
compute; equal structural caps do not mean equal computation.

Falsification: if real budget input cannot outperform the matched constant and
degree-only controls, the claimed additional informational value is unsupported.
Changed features or fewer score ties alone are not efficacy. Report full primary
MR as well as any predefined resource strata, including negative results.

Important counterexample: in the 16 one-node/one-edge groups these ratios vary
only with group size. Groups 1285/1185/1188 share size four and budgets one/one
but have different outcomes. A budget token alone cannot distinguish those cases.
No first-novelty claim follows from this bounded source review.

## Neighborhood-robust detection as a different project direction

The selected GPU reconstruction contains 243 target occurrences: 57 already
incorrect, 39 newly incorrect after attack, zero recoveries and 96 incorrect
afterward. All 39 new flips were at the selected direct endpoint. This selected
cohort motivates testing sensitivity to one added neighbor; it does not prove
that all graph attacks act locally or that self features are sufficient.

Proposed detector change: a separate self-feature prediction branch and a graph
neighborhood branch, with a learned gate that can retain self evidence when the
current neighborhood is unreliable. A low-degree indicator and disagreement
between branches are possible gate inputs. No injected-node IDs, clean-graph
oracle, victim correctness or evaluation labels may be supplied at deployment.

This changes the detector, rather than extending the MonTi attacker. Compare
against the unchanged detector, a fixed-gate same-capacity model, self-only and
graph-only branches, and an established defense if the direction is accepted.
Measure clean utility and benign false positives as well as attacked fraud
recall. Include an attack that adapts to the modified detector: success only
against a fixed old attacker is insufficient evidence of robust protection.

Neighborhood reweighting already has prior art, including
[GNNGuard](https://papers.neurips.cc/paper_files/paper/2020/hash/690d83983a63aa1818423fd6edd3bfdb-Abstract.html).
Neither a self branch nor a gate is claimed as a new invention. The available
target-only records do not establish benign utility, a defense benefit or exact
originality. Training a new detector also changes the scope and resource needs;
it cannot silently replace the frozen reproduction victim or count as an
already-approved MonTi improvement.

## Selection boundary and provenance

Budget conditioning is the more contained MonTi method extension. The detector
alternative addresses the owner's concern about direct protective value but is
a larger change of project direction. Neither is selected by this memo. If a
direction is accepted, first prepare a concrete bounded validation-only protocol
and applicable controls; full GAGA testing and final submission freeze retain
their distinct checkpoints. Course acceptance and genuine human review remain
unconfirmed.

Root and two Codex agents reviewed the static dependency and controls. Agent
checks are not human verification. Existing [resource diagnostics](../docs/RESULTS_FIRST_DIAGNOSIS.md),
[one-edge results](../docs/ONE_EDGE_DIAGNOSIS.md) and
[device/role evidence](../docs/SCORE_ROLE_REPLAY.md) were reused. A bounded primary-
source search checked existing budget/injection and neighborhood-defense work;
it was not an exhaustive novelty review. No datasets/checkpoints were loaded,
no model ran, and no new GPU use, API review or artifact rebuild occurred.
