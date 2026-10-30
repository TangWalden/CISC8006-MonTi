# Results-First Diagnosis of Saved Validation Outcomes

**Status:** exploratory diagnosis, not a confirmed failure mechanism or selected improvement. The earlier coverage full-evaluation proposal is on hold. Its registered protocol and negative pilots remain preserved. The primary reproduction conclusion remains inconclusive; no new improvement or course approval is claimed.

## Scope and evidence checks

Following the [diagnostic plan](../course/RESULTS_FIRST_DIAGNOSIS_PLAN.md), extraction covered six full-baseline runs and nine pilots: 15 runs, five conditions, and 160 recorded epochs. It used saved GCN validation predictions and fixed graph, split, and budget metadata. No new training, inference, or test-prediction reads occurred. CPU-only extraction took 8.25 seconds elapsed and used zero GPU-hours. [Registration and summary](../evidence/results_first_diagnosis_v1/) identify the inputs and information boundary.

An independent **post-extraction** [server audit](../evidence/results_first_diagnosis_audit_v1.json) passed 398 input-manifest hash checks and 29 frozen-input/scientific-source checks, reconciling 1,568 group-condition rows and 165 feature-bin rows. This establishes checked identity and consistency after extraction, not a claim that the metadata audit happened before execution.

A residual is a target occurrence still classified correctly by GCN after attack: the complement of attacked MR. The occurrence key is (target group, node); the same node can participate in different group attacks. Counts accumulated across three seeds are **seed-occurrences**, not independent examples. Without corresponding clean GCN predictions, these are not established clean-to-attacked flips or proof of intrinsic difficulty.

## Observed result: residuals concentrate in low-allocation strata

For the learned full baseline, selected snapshots contain 438 residuals among 14,889 seed-occurrences. The following strata use cutoffs derived from training metadata:

| Learned-baseline stratum | Residuals / exposures | Residual rate | Share of all 438 residuals |
|---|---:|---:|---:|
| Injected nodes per target < 5/7 | 432 / 2,727 | 15.84% | 98.63% |
| Remaining node-allocation strata | 6 / 12,162 | 0.0493% | 1.37% |
| Allocated edges per target < 9.3333 | 432 / 2,664 | 16.22% | 98.63% |

The first stratum accounts for 18.32% of exposures but 98.63% of residuals. These overlapping stratifications are not independent confirmations. [All feature-bin rows](../evidence/results_first_diagnosis_v1/training_derived_bins.csv), including empty bins, and [exact boundaries](../evidence/results_first_diagnosis_v1/bin_boundaries.json) remain available across all conditions; the report does not retain only favorable strata.

![Selected GCN validation residual rates by allocated nodes per target and target-group size](figures/results_first/validation_residual_associations.png)

The learned residual rates across increasing group-size bins are 2.29%, 3.38%, 2.81%, and 2.96%; there is no monotone large-group pattern. The figure uses different panel scales and shows exact exposure counts. Its [plotted data](figures/results_first/plotted_data.csv) and [provenance](figures/results_first/provenance.json) retain source, script and output checksums.

**Candidate explanation:** scarce injection allocation may constrain attacks on some groups. **Missing evidence:** budgets, target degree, group size, and graph geometry are structurally confounded; allocated budgets do not identify actual target-edge incidence or margins.

A supplementary, post-hoc edge_budget=1 subset contains 16 groups, 243 exposures, and 147 residuals: 60.49%. It is not a prespecified primary bin. These groups have one injected node and one edge. The [edge-generation code](../models/attack_model.py) first allocates an edge from each injected node to a target; no remaining edge budget is available here. Candidate endpoints therefore cannot receive an injected edge in these cases, although candidate context can still influence features and target choice.

This is not a sufficient budget-only explanation: groups 1285, 1185, and 1188 share displayed size 4, mean target degree 1, and node/edge budgets 1, yet have respectively 0/12, 9/12, and 6/12 residuals. Their [group outcomes](../evidence/results_first_diagnosis_v1/group_outcomes.csv) require explanation beyond those displayed metadata.

## Observed result: the pilots did not improve total validation MR

In matched pilot comparisons, coverage produced net changes of -3, -2, and -1 misclassified occurrences for seeds 3/4/5; size sampling produced -7, -9, and -2. Negative values mean weaker attacks. [Paired changes](../evidence/results_first_diagnosis_v1/paired_changes.csv) retain gains and losses, rather than only net totals.

Coverage reduced persistent residual keys—correctly detected in all three seeds—from 18 to 14, but increased total residual seed-occurrences from 76 to 82. This [persistence counterexample](../evidence/results_first_diagnosis_v1/persistence.csv) shows why improvement in one persistence summary cannot substitute for the primary weighted MR. The old selection rule's preference for coverage does not establish a supported improvement. These are GCN validation observations, not demonstrated GAGA test gains.

## Observed result: early best epochs do not establish collapse

Full-baseline selected epochs range from 1 to 5, with training ending exactly ten epochs later, consistent with the registered patience rule. Final MR is lower than selected-best MR, but the trajectories are not uniformly monotone: learned seed 0 attains its same best MR at epochs 4 and 9. [Run summaries](../evidence/results_first_diagnosis_v1/runs.csv) and [individual curves](../evidence/results_first_diagnosis_v1/validation_curves.csv) preserve this distinction.

The selected best is the maximum on these validation trajectories, so final-minus-best MR is non-positive by construction. Its sign alone does not establish degradation. Selected GCN validation MR is also slightly lower for learned than random in every full-baseline seed, by 4, 13, and 6 misclassified occurrences. This surrogate-validation comparison does not replace the frozen victim-test result or its inconclusive conclusion.

Weights, training schedules, and evaluation randomness change together. Recorded temperature differences alone do not establish a scheduler-mismatch cause: [gumbel_softmax](../utils/utils.py) ignores temperature and epsilon when randomness=False. No scheduler fix is justified by these observations.

## Proposed controlled check, not a proposed fix

The next useful measurement is actual injected-edge incidence at each target, remaining targets with zero direct injected edges, and clean/attacked GCN margins on recorded training/validation groups. A separately registered probe should hold checkpoints, target groups, random seeds, and injection budgets fixed, report the complete strata, and inspect whether incidence and margins distinguish matched-metadata counterexamples.

That probe has not been performed here. Any additional inference needs its own information-boundary review; GAGA test predictions remain excluded. Only after measurement should improvement options and a controlled comparison be proposed for owner confirmation.

Selected-best validation snapshots were chosen on these same rows, so their residual patterns are not independent confirmation. Overlapping occurrences, early-stopping survivor effects, confounding, and potentially tie-sensitive node rankings further limit interpretation. The present evidence narrows the diagnostic question; it does not select or validate a remedy.
