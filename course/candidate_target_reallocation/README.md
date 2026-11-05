# Candidate-to-target edge-budget reallocation

**Status: completed once, gains in all three seeds; the frozen development
criterion failed because its nominal confidence-interval lower endpoint remains
below zero. No candidate qualifies.** The owner-requested development work has
produced an observed gain, while the registered stable-effect objective remains
unachieved. This study is closed without added seeds, changed gates, promoted
controls or automatic further experiments. Automation monti remains deleted.
Full TEST and final submission/tag remain separate checkpoints.

## Completed results

| Fixed arm | Mean official-GAGA MR |
|---|---:|
| `baseline` | 46.94881890% |
| `balanced_target_fill` | 52.42782152% |
| `random_target_fill` | 52.52624672% |
| `candidate_rewire` | 46.88320210% |

Higher MR is a stronger attack. The primary gain is +5.47900262 percentage points,
with seed gains +7.77559055 / +5.90551181 / +2.75590551 pp. It passes the fixed
+1 pp threshold, all-three-positive condition and comparison against candidate
rewiring (+5.54461942 pp). The nominal paired-t95 interval is
[-0.82294779, 11.78095304] pp (`n=3`, `df=2`), so the positive-lower-bound condition
alone fails. No threshold is relaxed and no seeds are added to force that gate.

Across all 3,048 correlated target occurrences, the primary gains 184 wrong
predictions and loses 17, net +167. All 384 cases remain: 112 group-seed wins,
267 ties and five losses. A post-hoc maximum-gain example has 51 targets and
17 baseline wrong predictions versus 27 after reallocation. A negative example
has four targets and changes from four wrong to three. These illustrate saved
outcomes, not prospective case selection or a validated outcome-based gate.

Random target filling performs similarly and is slightly higher on average;
180 of 384 primary/random topologies are identical. This is consistent with a
role-allocation question, not proof that balanced target priority is superior.
The random control is not promoted. No first-novelty or LLM-benefit claim follows.
The [trace](../../evidence/candidate_target_reallocation_v1/execution_trace.json),
[complete summary](../../evidence/candidate_target_reallocation_v1/summary.json) and
[separate saved/tensor audit](../../evidence/candidate_target_reallocation_v1/independent_saved_tensor_audit_v1.json)
retain the results and limits.

## Evidence motivating the question

The completed sharpening comparison improved mean official GAGA MR from
47.973382% to 48.306110%, or +0.332728 percentage points, with eleven gained and
zero lost successes across 3,306 correlated target occurrences. All three seeds
were positive, but the fixed +1 pp criterion and positive nominal confidence
interval were not met. This is a small measured effect, not a qualified method.

A separately frozen, saved-geometry-only derivation recounted all 384 earlier
baseline scenarios without reading labels, predictions or tensors. Across the
128 separate scenarios in each seed, candidate edges accounted for 86.24%,
92.55% and 58.77% of summed non-self undirected injected-edge costs. These pooled
fractions are not means of per-scenario fractions. Candidate connections can
carry useful context; their count does not establish waste or a causal defect.
These geometry counts motivated the separately registered question: can some
candidate connections be reassigned to the original targets while retaining
every previous target connection and every injected-node degree? The completed
comparison above provides development evidence, not a general optimum or proof
that the original candidate edges were wasted.

## One fixed allocation and two controls

For each injected row, define
`k = min(candidate ones, target zeros, candidate zeros)` from the unchanged
baseline adjacency. The candidate-zero cap makes the strict candidate-rewiring
control feasible with exactly the same number of changed edges. There is no
outcome-selected fraction, validation threshold or learned/LLM gate.

| Arm | Removal | Addition |
|---|---|---|
| `baseline` | None | None |
| `balanced_target_fill` | Fixed SHA-sampled k original candidate edges | k missing targets, ordered by baseline injected-attachment count and SHA ties |
| `random_target_fill` | The exact same donors | k missing targets using fixed private random keys |
| `candidate_rewire` | The exact same donors | k previously absent original candidate connections |

Primary ranking uses target column sums across injected rows, computed **once
from the baseline**, not targets' full clean graph degrees. Per-row tie keys use
physical node IDs. The target list's known fraud-target assumption is inherited
from MonTi; no clean prediction, victim answer or new label lookup chooses an
endpoint. Both controls are comparators and cannot become the candidate later.

All original target edges, original node features, injected features x0, candidate
membership/order, target membership/order, injected-to-injected edges, row edge
costs, total node/edge counts and each injected node's physical non-self degree
remain fixed. Original candidate and target degrees intentionally change.
All nonbaseline arms have identical per-row final Hamming displacement and
identical removed candidate edges. Candidate-to-target and candidate-to-candidate
additions have different roles, as intended by the control.

The rule uses the maximum **common feasible** k, not a small-edit guarantee. A row
may lose every original candidate connection. Actual donor fractions, remaining
candidate incidence, target coverage and identity cases must therefore be
reported. No constraint is relaxed or scenario removed when k is zero.

## Registered cohort, preserved costs and actual execution

Source, meaningful synthetic checks and the separate auditor were frozen before
real cohort metadata and model calls. Selection drew 128 original VALID group
IDs without replacement with `random.Random(20261106)` from the sorted original
592 IDs after excluding both the original 128 and sharpening's 128 contexts.
The remaining eligible population is 336. Group memberships are unchanged; no
size, label, margin or outcome selected a group. The cohort has 1,016 occurrences
and 373 unique targets per seed. Of those identities, 306 overlap the union of
the two earlier cohorts and 67 are new relative to it. These are fresh contexts
for this intervention, not independent unseen news or an untouched validation
population in the adaptive project history. Earlier absolute scores are not a
paired baseline for this new cohort.

Execution used the same source-role `legacy` terminal-eight family as feature sharpening,
with the frozen original TRAIN groups, cached h_GCN, candidate policy, old score
role, tie policy and case RNG. Generate one common baseline for each of three
seeds and 128 groups: 384 generator calls. Three preparatory GCN and three
generator models load before proposals, but hooks prohibit proposal GCN forwards.
Seal all 1,536 complete adjacency/canonical-sequence inputs before loading either
dedicated evaluator. Topology changes require canonical reconstruction for every
arm; old known-relation-token invariance was not assumed. Only SELF token 0 was
required identical; changes in all seven canonical tokens were retained.

Every arm/seed's 128 separate scenarios sums 8,210 injected-node instances and
377,838 actual non-self undirected edges. These are not one combined graph.
All arms retain x0, every old target edge, physical injected degrees and budgets.
Across all three seeds, 122,420 candidate edges are moved, raising target
incidence from 59,103 to 181,523. There are 3,474 injected rows that lose every
candidate neighbor. This maximal-k risk is retained without dropping cases;
the observed gain does not make every reallocation beneficial.

The registered identity-only stop rule did not trigger. The completed original
run spent all 384 generator forwards and 1,664 original-GCN / 1,664 official-GAGA
forwards (128 clean and 1,536 attacked per classifier). Three preparatory GCN
loads made zero forwards, alongside three generator loads. No training, updates,
autograd, new teacher, API, TEST or GPU work was added.

Source was `c3238ee62272d39f0337828718a3560abe101a00`; protocol SHA256 was
`24dfa6cadd2b1c2a90a1b283bf659a5a0234e4d336e10d5142a813fe8ada0d14`.
The original run started at 10:10:55 UTC and completed at 11:29:05 UTC on
5 November: CPU / wall / outer seconds were 15,482.323859363 /
4,687.314927649 / 4,689.334797081, within the 5,400 / 5,460 wall-second limits.
It was neither stopped nor retried, and no scientific source, inputs or caps
changed during execution. Torch intra-operation threads were set to eight;
there was no total OS-thread cap. An operational snapshot showed 30 native/
background threads, without establishing that all 30 computed simultaneously.

Eleven synthetic tests, including 288 legal geometries, passed before calls
(0.287128046 CPU / 0.307375527 wall seconds). The implementation-author-independent
agent audit passed 231,878 checks / 1,186 byte-hashed files (240.269836838 CPU /
93.149031946 wall seconds), with zero model calls. Producer-author JSON reporting
passed 159,613 checks (0.78125 CPU / 1.187 wall seconds); its source was frozen
before this cohort and its model outcomes. It is not another independent method
audit. Agent checks do not replace human verification.

Only the primary qualifies with at least +1 pp mean official GAGA MR, positive
gains in all three seeds, nominal paired-t95 lower endpoint above zero and mean
above `candidate_rewire`. Random target addition need not be worse: when k fills
every missing target in a row, the two target additions are algebraically
identical. The hypothesis is target-role allocation, not superior target ranking.
An earlier source-preparation brief required beating both controls; this was
corrected during design review **before any new cohort metadata or model call**,
because of that identity, not because of observed outcomes. Both controls remain
fully reported and neither can be promoted as a winner. Keep every case and
denominator. The n=3 interval is
conditional on fixed checkpoints and an adaptively inspected development history,
not independent-news uncertainty or confirmatory generalization. No parameter,
cohort or seed expansion follows observed outcomes. Successful development would
still need its concrete owner-confirmed full-TEST protocol. Novelty, semantic
realizability, stealth and course completion are not established by this study.

## Closed study and remaining gates

Publish this complete matrix and incorporate genuine human feedback. No further
seeds, cohorts, model calls, API calls or TEST queries are launched automatically
to force the nominal interval above zero. Research GPU charges remain
5.065219865793663 hours plus 0.02 reserve, with 6.914780134206337 remaining.

A private conditional full-TEST plan is unselected, unapproved and nonexecutable:
the formal seeds-0/1/2 adapter, label-safe proposal handling and delayed victim
loading still need implementation and review. Its existence is not approval.
The private faster-canonical helper and seven test sources also remain unselected
and unrun; no profiling diagnostic was executed. The actual run completed within
its original limits despite slow execution and uncertain operator confirmation;
no single cause is assigned to that confirmation delay.

All older science, the original delivery snapshot `0c5e227` and dated PDF/PPTX
remain unchanged. Stable-effect qualification, full TEST, genuine human/course
checks, verified actual contributions, rehearsal and final tag remain incomplete.
