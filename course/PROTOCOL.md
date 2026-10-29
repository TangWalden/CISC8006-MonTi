# CISC8006 MonTi Component reproduction protocol: historical feasibility draft

## Historical context and current status

This document preserves a feasibility-stage draft and its execution update. The statement below that the baseline was not yet frozen describes that historical stage, not the current experiment status. It is not a retroactively created preregistration. The authoritative frozen baseline specification is [frozen_protocol.json](frozen_protocol.json), recorded before the final runs at tag `reproduction-protocol-v1`. All six formal baseline runs have since completed; the prespecified conclusion is **inconclusive**. See [reproduction_summary.json](../evidence/reproduction_summary.json). Instructor approval of the Component level and course resource limits remains unconfirmed.

Historical draft status: environment and evaluation checks were in progress; the formal baseline was not yet frozen; instructor confirmation of the Component level and course compute limits was pending.

Historical execution update: the official full graph and split0 passed source, partition, and model checks. The learned and random conditions each completed a feasibility pilot with 16 training groups, 8 validation groups, and 2 epochs, without reading test predictions. Initial preprocessing plus the learned pilot took 189.18 seconds in total (4.20 seconds for training/validation); the random pilot took 13.70 seconds (4.94 seconds for training/validation). Peak reserved memory was 3.127/2.762 GiB respectively. Pilots were not used to decide the scientific claim or select hyperparameters. The formal experiment retained the official maximum of 100 epochs and patience=10. The full graph has 937 training groups, 592 validation groups, and 909 test groups. Resource estimates were for queueing and limit control, not a guarantee of exact runtime.

The hardware is a shared server. At inspection, other projects already used both GPUs; their processes were neither modified nor stopped. Each formal run records available GPU memory and utilization. Shared load affects runtime, so it is not used to claim reproduction of the paper's efficiency results.

## Single primary claim

On official GossipCop-S (code identifier `gossipcop_0.05`), split 0, with fixed official GCN surrogate and GAGA victim checkpoints, rho=0.05 and xi=0.1, learned candidate-node selection increases target-group-size-weighted target misclassification rate, MR%, relative to random selection.

Paper source: AAAI 2025, DOI 10.1609/aaai.v39i15.33760. Table 5, GAGA column: MonTi 43.70%, random candidates 39.13%, a difference of 4.57 percentage points. The corresponding MonTi entry in Table 4 is 41.34 +/- 1.90%. The discrepancy remains unexplained; retain both rather than choosing whichever table is closer to the measured result.

Upstream: https://github.com/bdi-lab/MonTi , source revision `10f233e953679ec3715125dc058ef6d28bee4691`. The remote repository's initial commit is a source-file snapshot of that revision, without the full upstream history. The original local clone retains the full upstream history.

## Code-review findings and treatment

1. Upstream passed `random_candidates` directly as `cand_selection`: False actually selected randomly, and True used the learned selector. The fix negates the flag and tests the actual branch and gradients. This does not establish that historical paper experiments or released checkpoints used the reversed condition. Both conditions are retrained using the same corrected code.
2. The upstream neighborhood-budget helper sums target-neighbor incidences without deduplicating shared neighbors, potentially differing from the paper's set-size definition of B. A synthetic graph yields 4 incidences versus 3 unique neighbors. This project retains the official budget calculation consistently across conditions and discloses the difference as a limitation; it is not an exact formula-level reproduction.
3. Upstream evaluation could skip a target group when no perturbed graph was produced. This project fails the run and retains the error instead of reporting success with a changed metric denominator.
4. The GCN representation-cache filename omits split/checkpoint identity. Formal runs use a separate project directory and fixed split/checkpoint, recording cache checksums after initial generation. A dependency change requires cache rebuilding.

## Operational definition of the control

The random condition uses the same k-hop exploration subgraph and non-target candidate pool. When the pool exceeds K, it samples K candidates uniformly without replacement; it resamples on every forward pass during training and evaluation. When the pool has at most K candidates, it retains all candidates. The learned condition uses the official CandidateSelector; the remaining modules are identical. Preserve the official random-number streams and fix each run's seed without claiming that all random events match between conditions. Model evaluation can remain stochastic because of injected-node noise and random candidates.

This is a "component comparison based on the official implementation with corrected flag mapping." Public evidence does not establish complete equivalence to Table 5's historical training procedure; that uncertainty remains unresolved.

## Staged execution

- One-command check: `bash course/run.sh -m course.test_protocol`. It covers known-answer weighted MR, actual branch selection, CPU/GPU forward/backward execution, candidate legality, feature/node/edge budgets, and the upstream budget-definition difference.
- Feasibility pilots: full original graph, target-group subsampling with fixed seed 1729, training/validation only, and few epochs. Test results must not inform decisions. Preserve failures and revisions.
- Formal reproduction: after pilots, freeze the official configuration or explicit changes, source code, dependencies, data/model checksums, all target groups, resource limit, and decision rule before running three paired seeds, 0/1/2, for each condition.
- Do not implement the three improvements in this reproduction stage. Establish a separate improvement protocol after reproduction, based on its evidence.

## Metric, randomness, and decision rule

MR = the number of misclassified node occurrences across all target groups / the number of node occurrences across all target groups. Count overlapping nodes once per group occurrence; do not replace this denominator with unique nodes. Report MR in percent and differences in percentage points. Also report clean MR, the increase after attack, coverage, individual run results, mean and sample standard deviation, runtime, and peak reserved memory.

Compute the mean paired MR difference (learned minus random) across three seeds and its 95% interval using t(df=2). A lower bound >0 means supported; an upper bound <=0 means not supported; otherwise the conclusion is inconclusive. This uncertainty covers run seeds under fixed splits and fixed pretrained models only; it does not generalize to other data/models.

After an audit, infrastructure failures may be retried with the same frozen settings, retaining every failure and its cause. Permit only one valid completed result for each condition/seed. If multiple completed results exist, the aggregation program stops; selecting a result by score is forbidden. Raw per-target predictions are retained on the server and checksummed in summaries; large prediction/model files are not placed in Git.

## Resources and evidence

Remote directory: `/home/tjy/Project/MonTi-course`; two RTX3090 24GB GPUs. Reuse the existing monti environment and set CUDA dynamic-library paths through the wrapper. Use Chinese mirrors if dependency downloads are required; obtain original data and checkpoints through official sources.

At this draft stage, the provisional resource limit was 48 GPU-hours, to be confirmed against pilot measurements and course rules before formal freezing. Every run uses a unique ID and preserves its registration, code commit, exact command, configuration, per-target predictions, full logs, checkpoint checksum, and GPU time. Retain all failures. Do not commit data, models, or downloaded archives to Git. Use the test set only for final evaluation after the formal freeze.

This document is not an instructor approval record. Do not record human checks, peer reviews, or course approval that have not occurred.
