# Outcomes-first pilot protocol — DRAFT

**DRAFT / NOT APPROVED / NOT FROZEN / NOT EXECUTED**

Prepared for owner review on 2026-10-31. This document proposes measurements; it authorizes no training, inference, launch, source freeze, or method selection. Neither A nor B has been selected. Instructor approval and genuine team checks remain separate requirements. Existing scientific files, protocols, results, failed attempts, and reproduction conclusions remain unchanged.

## What the proposed pilot would answer

We would train two matched versions: a fresh native MonTi baseline and a version using the paper's injected-node source role for edge scores. Separately, we would test a small endpoint-filtering rule on the same fresh baseline checkpoints, with controls that distinguish useful information from merely changing the number of choices. The comparison would use GCN validation data only. A negative or unclear result would be retained; the procedure would not automatically select the least-negative method.

The observed score-role discrepancy and device-sensitive selections motivate these questions. They do not establish that either proposed change improves attack performance. Historical prediction agreement is not identity of the historical generated features or graphs. The earlier coverage full evaluation remains on hold.

## Proposed common experimental setting

| Item | Proposed setting, subject to approval and final freeze |
|---|---|
| Dataset and graph | The same full GossipCop-S graph, split 0, fixed preparation, target assignment, feature bounds, and caches |
| Detector | The same fixed official GCN surrogate; no GAGA or other victim model loading or queries |
| Training subset | The same 256 training target groups for every trained arm |
| Validation | All 592 original GCN validation groups, with target-occurrence-weighted MR |
| Training seeds | Paired seeds 3, 4, and 5 |
| Pilot training limit | Proposed maximum 10 epochs, validation every epoch, patience 5 |
| Other settings | Original fixed node/edge budget functions, feature constraints, model dimensions, loss, optimizer, learning rate, weight decay, and parameter structure |
| Resource maximum | At most 2 aggregate GPU wall-clock hours for the entire approved pilot, including checks and failures |

The exact 256 training IDs, input/source hashes, device identity, RNG seeds, run matrix, and resolved settings are not supplied by this draft. They must be reviewed and frozen before any run. In particular, the earlier 256-group subset is a possible reuse option, not an authorized default.

## Arms and fair controls

### Fresh matched training: native baseline and A

| Arm | Proposed intervention | New training |
|---|---|---:|
| Native baseline | Current audited native score-role behavior | Three runs, seeds 3/4/5 |
| A: paper-role alignment | Use the projected, normalized injected-node representation as each injected row's edge-score source, in training and inference | Three runs, seeds 3/4/5 |

The proposed A definition applies the source-role change to all injected rows, not only one-edge groups. It changes no edge budget, legal endpoint set, model parameter count, or unrelated behavior. The precise implementation and scope must be approved and tested before freeze. This is a paper-conformance variant, not a claim of first-in-literature novelty or a demonstrated repair benefit.

Within each seed, both arms would start with identical parameter values and receive the same training groups, update opportunity, optimizer settings, maximum epochs, and checkpoint criterion. Record actual group order, epochs, costs, and any differences. Equal maximum training budgets do not imply equal elapsed time or equal completed epochs under early stopping.

### Inference-only B and its controls

These conditions would share each fresh native baseline's selected checkpoint. They would not train additional attackers or use A's checkpoints.

| Condition | Proposed endpoint policy for groups with node budget = edge budget = 1 |
|---|---|
| Native reference | Preserve the captured native attack without postprocessing |
| All-target mask control | Run the postprocessing path with every target eligible |
| B: predicted-fraud mask | Permit target endpoints predicted fraudulent by the fixed GCN on the clean graph |
| Random count-matched mask | Uniformly sample, without replacement, the same number of eligible targets as B |

For each checkpoint, group, and evaluation draw, generate the native feature and score values once. Hold them fixed across postprocessing conditions; change only the allowed sole target endpoint. Preserve one injected node, one logical undirected edge, and the original self-loop convention. Other validation groups retain their native captured outputs. Include every validation group meeting the 1/1 budget condition, rather than selecting favorable residual cases.

B would use clean GCN **predictions**, not ground-truth labels or attacked outcomes, to define eligibility. The proposed empty-mask behavior is to fall back to all targets; this behavior must be explicitly confirmed before freeze. Random masks must match the resulting effective cardinality, including this fallback. Their draws and tie handling must also be fixed in advance.

The all-target control must traverse the same postprocessing code path. Do not assume it reproduces native selection: a copied or compacted tied tensor can change a native top-k choice. Record native-versus-control endpoint, feature, prediction, and count agreement. B outperforming a degraded procedural control alone would not establish improvement over native MonTi.

## Checkpoints and paired evaluation randomness

Proposed checkpoint selection uses the earliest epoch attaining the highest size-weighted GCN validation MR under one fixed selection draw; strict improvement resets patience. The criterion and tie behavior must match both trained arms.

Primary comparisons would reevaluate each selected checkpoint using three separately registered evaluation draws, paired across methods by training seed, target group, and draw. Average draws within a training seed before computing the three paired seed differences. Draws, overlapping nodes, and target occurrences are not additional independent training replications.

Reset declared evaluation RNG streams consistently before attacker generation, and isolate validation RNG from the subsequent training RNG state. Use a separate declared stream for random masks so drawing a mask cannot change the captured attack feature. Preserve the original group/query ordering and record the actual backend policy; do not tune numerical settings to obtain favorable agreement. The precise seed derivation, selection/comparison draw IDs, state-isolation mechanism, and reference device remain blocking choices before freeze.

Primary evaluations should use one fixed reference GPU/backend policy. The training-device schedule and any parallelism must be registered rather than confounded with arm identity.

## Clean-query and information fairness

Clean GCN queries are additional information and cost. Give the postprocessing conditions the same permitted clean-query budget and report the actual shared calls and each method's standalone requirements. Controls may ignore the clean predictions; do not invent or execute unnecessary dummy queries to equalize time.

Cache clean outputs per eligible **target group**, with the same target ordering and query semantics as the frozen evaluator, and reuse them across attacker checkpoints and draws. Do not replace this with a global-node prediction cache unless equivalence has first been established: batching and normalization semantics matter. No GAGA loading, victim queries, test prediction reads, test-stratum analysis, or test-based parameter selection belongs in this pilot.

Public artifacts would remain English aggregate tables with source/configuration/RNG/cost provenance. Raw node identities, per-node outputs, generated features, and detailed traces would remain in the established ignored private evidence locations.

## Comparisons, interpretation, and progression

- **A primary efficacy comparison:** paired full-validation MR difference, A minus the fresh native baseline.
- **B primary efficacy comparison:** paired full-validation MR difference, B minus that same fresh native reference.
- **B mechanism controls:** B minus all-target masking and B minus the random count-matched mask. These test procedural and eligibility-information explanations separately.
- Report all 592-group results as the proposed primary pilot metric. Report the affected 1/1 subset separately, with its own denominator; do not present its percentage-point change as a whole-validation gain.

Report each seed, mean/sample SD, paired differences, and appropriately labeled descriptive uncertainty. Checkpoint selection and pilot comparison use validation evidence, so these are exploratory screens rather than held-out confirmation. Do not claim family-wise error control from several unadjusted intervals.

A proposed progression screen is a positive mean primary paired MR difference with nonnegative differences in at least two of three seeds. B would additionally need positive mean differences against both mask controls. These proposed thresholds require owner approval; they are not already adopted rules. Integrity, query, and resource requirements must pass independently of performance.

If neither efficacy candidate qualifies, report **no qualifying efficacy candidate**. Do not automatically promote the less-negative result or reopen parameter search. The next step would be an owner decision: further diagnosis, a revised question, or an explicitly approved full study that may produce a negative result. The course requirement to fully evaluate one improvement is not satisfied by this draft or by a technical pilot alone.

### Separate reproducibility control C

An exact-tie global-node-ID rule is a possible separate inference-only control, not a third trained efficacy arm or an authorized addition to this matrix. It must not add epsilon to scores or treat near-equal values as exact ties. Its scope, comparison with the all-target control, device matrix, and cost would need explicit approval and registration.

If included, its relevant outcome is actual cross-device endpoint/class disagreement. Identical-vector algebra agreement alone is not evidence of whole-pipeline reproducibility, because candidate sets, features, and nonidentical floating-point scores may still differ. Do not rank C as an MR winner or combine it silently with A or B.

## Old checkpoints and the validation-size tradeoff

Fixed GCN weights and verified caches are reusable. Earlier attacker checkpoints selected on 128 validation groups under the earlier evaluation-RNG behavior cannot supply published-score controls for this new regime. They may support a separately labeled within-checkpoint sensitivity analysis only if every compared condition is reevaluated under identical data and RNG rules. Such results must remain separate from the fresh trained A comparison.

An inference-only role change to old attacker weights is a mechanistic ablation, not the proposed trained A variant. Fine-tuning old weights would likewise require a new matched warm-start design, not an unreported substitution for fresh training.

Full 592-group checkpoint validation is the proposed common design. A lower-cost alternative would select checkpoints using a fixed 128-group subset and then perform a common 592-group comparison. That alternative must be chosen **before execution**, applied to both trained arms, and separately frozen. Do not switch validation scope after observing scores or splice partial results from incompatible selection regimes.

## Resource and failure proposal

| Proposed reservation | Aggregate GPU wall-clock hours |
|---|---:|
| Six trained runs | 1.4 |
| Selected-checkpoint evaluation and approved controls | 0.4 |
| Preflight checks and contingency | 0.2 |
| Total hard maximum | 2.0 |

These are ceilings, not runtime estimates or permission to launch. Exact per-run limits, forward/update caps, scheduling, stopping, cleanup, and retry rules must be resolved before freeze. Any C work must fit an explicitly approved allocation rather than expanding the total. Charge concurrent devices additively and include failed attempts and checks.

Use exclusive attempt IDs, record exact commands/configurations/hashes before execution, and flush partial evidence as it is produced. Preserve failures and negative results. An incomplete three-seed matrix cannot support a complete paired comparison; no survivor-only analysis, replacement seed, favorable retry, or budget expansion is implicit. Freeze a deterministic valid-attempt rule and any permitted infrastructure repair policy before results exist.

## Blocking decisions before a final freeze

1. Owner confirmation of the research direction and which proposed A/B/control scope, if any, should proceed. No A/B choice is recorded here.
2. Exact A source-role intervention and tested adapter; B eligibility/fallback, masking operation, random-mask draws, and shared tie behavior.
3. Exact 256 training IDs; choice of full-592 versus the separately described 128-selection alternative; hashes of all fixed data, models, caches, and sources.
4. Reference device/backend and training schedule; selection/comparison RNG seeds, draw count, training-RNG isolation, and checkpoint selection rule.
5. Final efficacy comparisons and progression rule; whether C is excluded or separately included with a bounded reproducibility question.
6. Complete cost/forward/update limits, failure and repair rules, attempt disposition, raw-evidence/privacy checks, and executable tests.
7. Explicit review of the concrete final protocol before any launch. Instructor/course approval and actual team checks must retain their true status rather than being inferred from owner authorization.

Until these decisions are resolved, this file remains **DRAFT / NOT APPROVED / NOT FROZEN / NOT EXECUTED**. It contains no runnable launch configuration.
