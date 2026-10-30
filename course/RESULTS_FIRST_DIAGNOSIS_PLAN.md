# Results-first diagnosis before choosing an improvement

Recorded2026-10-30, before this diagnostic extraction. The owner requires improvement selection to start from actual experimental outcomes. No new improvement is selected in this plan. The earlier coverage full-evaluation proposal is on hold; original protocols, pilots, negative results and baseline conclusions remain unchanged.

## Phase1: saved evidence and fixed input metadata only

Inputs: all six completed full baseline runs (learned/random; seeds0/1/2), all nine completed technical pilots (common baseline/coverage/size sampling; seeds3/4/5), their saved GCN validation predictions and epoch summaries, recorded best/completed epochs, and fixed graph/split/budget metadata. Read result.json only for run state, best/completed epochs and checkpoint identity; do not use its GAGA test score to select an improvement. Do not read test prediction files or perform new model inference.

Questions to answer before proposing changes:

1. How do validation MR and loss evolve, where is the selected epoch, how much does the last epoch differ, and how often do several epochs tie for the best MR? These are observed trajectories, not proof of a particular cause.
2. Which validation target occurrences remain detector-correct after attack, and how consistently across the three seeds? Use `(target group, node)` as the occurrence key because a node can appear in several different group attacks.
3. Are residual outcomes associated with group size, node budget, edge budget, nodes/edges per target occurrence, edges per injected node, or mean/minimum/maximum target degree? Report the complete feature list and bins whose cutoffs come from training metadata, not from favorable validation outcomes. Use the unique25th/50th/75th-percentile training cutoffs, left-closed/right-open bins, and retain empty bins. These bins need not contain equal counts when features have ties.
4. Are residuals concentrated in a few groups or repeatedly occurring nodes? Report exposure denominators as well as error counts; repeated appearances do not establish intrinsic difficulty.
5. In the matched pilot subsets, which groups and occurrences account for the observed losses of coverage and size sampling? Keep pilot and full-baseline blocks separate; their seeds, subsets and training limits differ.
6. Do selected and final epochs use different recorded scheduler stages? Summarize the observable epoch/state relationship without assuming that it changes hard decisions or causes a performance gap.

## Metric and inference boundaries

The frozen MR definition and decision rule remain unchanged. Here, a residual is a target occurrence the detector still classifies correctly after attack; it is the complement of attacked MR, not a newly defined conditional attack success rate. Without corresponding clean GCN predictions, do not call these proven correct-to-wrong or wrong-to-correct transitions. Report exact0/3,1/3,2/3,3/3 residual frequencies within each condition/cohort. Cross-condition gains/losses are descriptive changes between attacked outcomes, not clean-to-attacked flips.

Validate saved prediction hashes, group membership/labels/budgets, denominators and selected-epoch indexing. Best epochs were selected on these same validation rows, so their residual patterns are selected validation performance rather than independent confirmation. Retain individual run curves; late-epoch averages are subject to early-stopping survivor bias. Describe group associations and cross-seed patterns; do not treat overlapping target occurrences as independent experimental replications or infer causes from correlations. Budget, degree and group size are structurally confounded, and recorded allocations do not establish how many edges were actually used. Full-pool versus fixed-K coverage is not a matched-size quality comparison. Training curves also mix changing weights, schedulers and evaluation noise.

GAGA test predictions remain excluded from diagnosis and tuning. No new victim queries are added. Additional validation-only measurements, if needed, require a separate recorded diagnostic question and information-boundary review. The primary reproduction conclusion stays inconclusive.

## Required deliverable and next checkpoint

Publish English tables and a short factual report. Separate **observed result**, **candidate explanation**, **missing evidence/counterexample**, and **proposed controlled check**. Only then propose improvement options with direct evidence links and ask the owner to confirm the direction. Preserve all previous negative and failed runs and all pending instructor/human-review requirements.
