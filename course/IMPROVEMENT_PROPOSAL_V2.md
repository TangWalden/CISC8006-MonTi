# Outcome-Based Improvement Options, Version 2

**Owner direction confirmation is pending.** This proposal selects no method and authorizes no new training or final test evaluation. The earlier coverage follow-up remains on hold; its negative pilots and original selection rule are retained. The frozen baseline conclusion remains inconclusive.

## Evidence that changes the question

The [crossed-device diagnostic](../docs/SCORE_ROLE_REPLAY.md) follows the saved-outcome analysis and finite one-edge experiment. It uses the original model and budgets, rather than an implemented correction.

| Actual observation | Implication for the next question |
|---|---|
| All 96 captured edge-score calls use the first target slot as the source; the [published definition](PAPER_EDGE_SCORE_CHECK.md) uses the injected slot. | Investigate role alignment, with a separate controlled implementation. |
| GPU-generated attacks reproduce all 48 historical prediction vectors; CPU generation reproduces 17. Detector CPU/GPU classes agree for each captured attack. | Keep the original GPU execution path for the primary comparison and distinguish generation from detector effects. |
| Actual target probabilities tie at the maximum in 63/96 captures; identical-vector top-k reference choices differ on devices in 32/96. Candidate sets also differ in 36/48 generator pairs. | A tie policy is worth testing, but edge-only tie handling cannot guarantee whole-generator portability. |
| Six GPU-reconstructed choices spend the only edge on an already clean-GCN-incorrect target despite another clean-correct target being available. | Test whether scarce-budget prioritization uses that opportunity more effectively. |

These are post-hoc validation observations, not independent confirmation or final improvement results. Historical prediction agreement does not identify the original generated feature or graph.

## A. Align the score source with the injected-node role

Use the injected representation at sequence position `T+C+src` when scoring an injected-relative adjacency row `src`. Preserve the architecture, candidate rule, masks, node/edge budgets, loss, training limits and top-k behavior. Implement the variant in an isolated adapter; do not change frozen scientific files.

**Question:** does training with the aligned score role change attack behavior and subsequent performance under the frozen setting? This is a formulation/implementation alignment study, not an established novel algorithm or a demonstrated performance fix.

Retrain the variant. The old weights were optimized under the original expression; their mathematical injected-role reference still has raw maximum ties in 36/96 cases. An inference-only substitution cannot stand in for consistently trained A.

Primary comparison: original-role training/evaluation versus aligned-role training/evaluation, with paired seeds and equal resources. A crossed training/evaluation-role ablation can distinguish immediate substitution from adaptation through retraining. Do not combine a new tie policy into the primary A contrast.

## B. Prioritize an effective target when only one edge is available

Restrict this policy to `node_budget=edge_budget=1`. Query the fixed surrogate on the clean target group. If any targets are predicted as the known fraud class, let the existing mandatory-edge score/top-k choose within that eligible set. Otherwise retain the all-target fallback. Keep the generated feature, injected-node count and edge count fixed.

**Question:** does spending the sole connection on a target the surrogate still detects increase useful attack effects? Use surrogate predictions available to the attacker, never victim outcomes or validation-label-based oracle choices. Correctness on GCN does not imply correctness on GAGA; transfer may worsen.

Essential controls: an all-target mask must reproduce the baseline, and a uniformly random mask with the same eligible count must separate informative prioritization from simply shrinking the choice set. Record the extra clean-surrogate computation and provide the same information/computation to the appropriate controls. The earlier optimistic endpoint oracle is motivation, not the performance of B.

## C. Explicit exact-tie handling as a reproducibility control

For exactly tied eligible maxima, use a documented stable global-node-ID order, with no labels or victim queries. Test identical-vector selection stability and report real generator differences separately. Candidate selection and changing score values may still differ across devices. This is an engineering/control option, not a promised MR gain or a guarantee of CPU/GPU generator identity.

## Recommended preparation and later gates

Prepare validation-only pilots for **A and B**, with C as a functional/reproducibility control. Reserve at most **2 aggregate GPU-hours** for the next pilot stage, within the remaining 12-hour additional-research limit. Prepare a known-answer source-role test, gradient and budget checks, mask/fallback identity checks, fixed configurations, seeds, query accounting and a pilot decision rule before execution. No source variant has been trained or evaluated by this proposal.

Freeze the concrete pilot protocol before any approved pilot runs. Keep all null/negative results; do not force a winner merely because one negative score is less negative. Any later decision to fully investigate a null result needs a stated scientific question. Final method selection, the complete full-evaluation protocol and final submission freeze retain their owner checkpoints. GAGA test evaluation remains prohibited until its separately frozen protocol is confirmed.

Course approval, the actual UMMoodle freeze and genuine team technical checks remain unconfirmed. Planned responsibilities and agent-generated work are not completed human contributions. The value of this study is its auditable diagnosis and controlled evaluation; novelty and positive results must not be invented.

Preparation update, 2026-10-31: the owner requested a beginner explanation before
deciding. [Isolated adapters](outcome_v2/README.md) now have 12 passing synthetic
CPU checks; this uses no datasets, saved checkpoints, detector calls, training or
GPU work and establishes no efficacy. The [pilot draft](outcome_v2/PILOT_PROTOCOL_DRAFT.md)
lists unresolved choices and is not approved or frozen. The existing full-run
and final-freeze checkpoints remain in force.
