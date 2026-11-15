# Target-aware two-step rewriter: positive but modest development gain

The registered neural rewriter passes its mean criterion on the reused
GossipCop development cohort. It improves on native, both existing single-step
methods, and both own dose-matched random controls. **It does not beat the
stronger two-step greedy/independent controls or achieve +1 pp over existing
neural retrieval.** No new TEST or YelpChi experiment was run in this phase.

| Method | GAGA MR, mean | Difference from existing neural |
|---|---:|---:|
| Native author upstream | 57.9101% | -1.2259 pp |
| Existing single neural | 59.1360% | reference |
| Existing single direct | 59.2528% | +0.1168 pp |
| Existing neural matched random | 58.0269% | -1.1092 pp |
| **New two-step neural rewriter** | **59.4863%** | **+0.3503 pp** |
| Its same-position random | 58.3187% | -0.8173 pp |
| Its joint region/endpoint random | 58.0852% | -1.0508 pp |
| Two-step greedy | 59.6030% | +0.4670 pp |
| Greedy matched random | 58.4939% | -0.6421 pp |
| Independent single-edit composition | 59.6030% | +0.4670 pp |
| Independent composition matched random | 58.6690% | -0.4670 pp |

Higher MR is a stronger attack. Each of three conditional repeats retains all
64 groups / 571 target occurrences / 240 unique targets. Repeats reuse one
author checkpoint and overlap in target identities; 1,713 occurrences are not
independent news samples. These rates are not comparable to the separate
[complete TEST cohort](../portal_full/).

## What changed and what the evidence supports

The frozen author outputs feed a downstream target-level attention encoder,
region head and conditional endpoint head. Exact two-edit TRAIN lookahead
labels train the model by imitation; this is not reinforcement learning.
It modifies up to two of three nominated candidate slots, preserves injected
features, target connections, all injected degrees and total physical budgets,
and reconstructs actual context between decisions. Original endpoint degrees
may change. See [implementation and controls](../../experiments/target_rewriter/).

Against existing neural, all repeats improve by +0.1751/+0.5254/+0.3503 pp.
The nominal paired t95 interval is [-0.0848, 0.7853] pp, conditional on three
fixed repetitions; this is modest exploratory evidence, not a confirmed
population improvement. There are six gained occurrences involving five
unique targets, zero lost occurrences, six improved group-repeat cases and
186 ties. Against native the gain is +1.5762 pp; against the two own random
controls it is +1.1675/+1.4011 pp.

Greedy and independent composition have **identical class predictions** on
all 1,713 occurrences, and exceed the rewriter by two successes on one repeated
target. This study does not establish learned interaction or sequential-policy
superiority. Joint-random region matching has only one feasible region subset
in 32 of 168 active cases (56/192 including inactive cases).

Post-hoc example: seed 1 / group 1356 has six targets. Native has zero misses,
existing neural one, and the rewriter two; its endpoint-random has zero. Seed
0 / group 1441 instead has three misses under both native and rewriter, while
the teacher and victim mean margin changes have opposite signs. These are
illustrations after outcomes, not selection rules or causal proofs.

Recorded selection context totals, including reused initial evidence, are 360
for rewriter, 1,552 for the old shared single-candidate cards, 2,168 for
independent composition, and 3,440 for greedy. **These are not comparable
minimum teacher-forward requirements or measured latency speedups.** The old
neural ranker uses physical sequence changes and features, not teacher logits;
it can rank without a teacher classifier forward. Its shared cards were also
scored for direct-search controls. The new rewriter does use teacher margins.
Physical reconstruction, classifier forwards, training and latency must be
distinguished. See the [source-based accounting clarification](cost_interpretation.json);
all original counters and results are retained. Training used 32 fixed TRAIN groups per repeat, versus the old
ranker's 128; search pools also differ (three disjoint pools of four versus
the old first pool of eight). Same graph budgets are not equal information,
edit dose, or computation across all methods.

## Reproducible record

- [All repeats, counters, contrasts and decision](summary.json)
- [Frozen protocol](protocol.json), scientific source `73e3487`
- [Saved tensor/JSON audit](audit.json): 31,123 checks, root-authored, not human
  or independent implementation review; no model forwards
- [Paired cases and post-hoc interpretation](paired.json)
- [Execution, source hashes, preflight and training curves](execution.json)

One launch completed: 12,864 training updates, 7,621 main physical builds,
10,440 own-teacher contexts in 1,099 batches, 360 policy forwards and 2,112
official GAGA VALID forwards. Preflight separately used eight physical builds,
four teacher contexts and two disposable functional optimizer steps on TRAIN
inputs; 127 synthetic checks passed. Whole driver wall time was 6,551.96 s;
main process CPU window was 15,864.76 s. GPU, API, new upstream generation and
TEST use were zero. Raw attempts remain private in the recorded QY stage.
