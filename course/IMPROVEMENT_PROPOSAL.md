# Three improvement proposals and technical pilots

Owner steering on2026-10-30 places the coverage-first follow-up on hold. The existing pilots and their negative results remain unchanged. Analyze actual training/validation outcomes before selecting or revising an improvement; see RESULTS_FIRST_DIAGNOSIS_PLAN.md. The historical pilot selection rule does not constitute approval to launch the earlier full-evaluation proposal.

The motivation comes from source review and the planned training/validation diagnosis, not from choosing parameters using test results. The baseline's three-seed CI crossing zero is not direct evidence of an algorithmic defect. Course approval has not been confirmed.

## A: Structural-coverage candidates

The current per-candidate scores and mean target-group representation have no explicit coverage constraint. With K fixed at 128, test whether reserving up to 32 greedy coverage anchors and filling the remaining slots with the original learned ST top-k improves target-occurrence-weighted MR. Candidate count, parameter count, and injected-node/edge budgets remain unchanged; retain every candidate in small pools. Break ties by global node ID and stop when coverage gain reaches zero. The learned fill must retain gradients; an entirely hard selection cannot be described as a learned selector.

Ablations: reuse the frozen learned baseline with the added mechanism disabled; also replace greedy coverage with random anchors of the **same anchor count**, separating reserved slots from structural coverage. If diagnosis reveals only a small coverage gap, report that the hypothesis has weak support.

## B: Sampling by target-group size

The current implementation averages hinge loss within each group and traverses groups uniformly without replacement, whereas the primary metric weights by group size. Sample with replacement using `p(g)=|T_g|/sum|T|`, retaining 937 group updates per epoch, so expected training weights correspond to the evaluation objective. Do not simply multiply the loss: with batch_size=1, Adam partly cancels pure scale changes.

Ablation: uniform sampling with replacement isolates the effect of replacement itself; the frozen learned baseline uses uniform traversal without replacement. Record actual group sequences, update counts, processed target occurrences, and GPU-hours. More frequent large groups do not imply strictly identical computation.

## C: Budget-aware candidate count (fallback)

Determine K, capped at 128, from each group's injected-node/edge budgets. Test whether fewer candidates maintain MR while reducing candidate count and attention sequence size. The exact function is not yet frozen, and no speedup can be claimed. Shared-GPU runtime alone cannot support an efficiency conclusion. Implement A and B first.

## Diagnosis and pilot design

First fix seed 1729 and sample 64 training and 64 validation groups. Report group-size distributions, the difference between uniform-group and target-weighted distributions, pool size, baseline direct-neighbor coverage, and full-pool achievable coverage. Diagnosis only replays an already trained attacker. No new test inference or test-stratified selection is permitted.

The technical pilots were [frozen separately before execution](improvements/pilot_protocol_v1.json): the common baseline, A, and B each use seeds 3/4/5, fixed subsets of 256 training groups and 128 validation groups, a maximum of 10 epochs, and patience=5, for nine runs. Use only GCN validation MR, not the GAGA test set. Rank ideas by their mean validation gain over the corresponding pilot baseline across three seeds. If the gains differ by no more than 0.25 percentage points, choose A because its training traversal matches the baseline. If both ideas lose, retain the results and fully evaluate the better one; do not reopen the search to conceal negative results. Course approval remains unconfirmed, so these runs cannot be described as already satisfying the course's approved-pilots requirement.

The [descriptive training/validation diagnosis](TRAINING_DIAGNOSIS.md) provides motivation and limitations. Low direct-neighbor coverage only motivates a complementary-context hypothesis; it is not an established defect. High cosine similarity alone does not prove candidate redundancy.

Full evaluation: run the selected idea and its mechanism control three times each, using seeds 0/1/2 and the baseline's same full configuration, splits, detectors, budgets, and early stopping. Evaluate each condition once on the final test set. Report individual seeds, mean/sample standard deviation, and paired 95% intervals. Reuse the frozen baseline without retraining or rewriting it.

The additional compute limit is 12 GPU-hours; preserve failures and limit-exceeded records. Implementation tests must at least cover coverage and deterministic ties, candidate count/legality/uniqueness, learned-fill gradients, equivalence when the mechanism is disabled, sampling probabilities and total draws, and the existing budget/evaluation checks.
