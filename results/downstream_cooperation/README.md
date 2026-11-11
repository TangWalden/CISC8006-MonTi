# Frozen upstream downstream training results

On the registered GossipCop-S split 0 development cohort, a trained downstream
candidate-edge scorer increased mean GAGA miss rate from **53.6486% to
58.1436%**, a **+4.4950 percentage-point** difference. Its nominal paired
three-seed 95% interval is **[0.5715, 8.4185] pp**. The primary met its registered
screening rule. Its additional gain over matched random candidate connections
was **+1.1092 pp**, with interval **[-3.9892, 6.2075] pp**; one seed declined.
The experiment therefore does not establish a reliable learning advantage over
random connections or a benefit specifically caused by observing upstream
decisions. **No final TEST evaluation was performed.**

Higher MR means more fraud targets are classified as benign and hence a stronger
attack. MR includes targets already misclassified on the clean graph.

## Cohort and training

The three seeds are 3, 4 and 5. Each uses 128 original TRAIN groups for four
downstream epochs and the same 64 original VALID groups for a single terminal
comparison. Each arm has **571 target occurrences per seed**, or 1,713 correlated
occurrences across the three seeds. There are **240 unique validation targets**;
**233** appeared in earlier development contexts. These are unused group
contexts, not an independent unseen-news population.

The upstream MonTi checkpoints and original GCN training-feedback model stay
frozen. Only the **151,872-parameter** downstream scorer is optimized, using
TRAIN feedback from the fixed GCN. It receives generated auxiliary features,
the feature mean and count of actually attached targets, candidate features and
cached GCN representations. This is an attachment summary, not a lossless
encoding of every upstream action. Four fixed epochs produce 500 updates per
seed: 1,500 updates in total, with 36 no-choice training occurrences skipped.
Checkpoints are the final epoch, without VALID checkpoint selection.

All three quota arms share the same upstream features, target attachments,
candidate lists, total structural budgets and projected role quotas. The
native MonTi arm retains its original output. See the
[experiment definition](../../experiments/downstream_cooperation/README.md) and
[frozen execution protocol](../../evidence/downstream_cooperation_v1/execution_protocol.json).

## GAGA results

| Arm | Seed 3 MR % | Seed 4 MR % | Seed 5 MR % | Mean MR % |
|---|---:|---:|---:|---:|
| Native MonTi | 52.5394 | 51.4886 | 56.9177 | 53.6486 |
| Projected quotas with original scores | 56.7426 | 56.9177 | 56.9177 | 56.8593 |
| Projected quotas with random connections | 56.7426 | 57.0928 | 57.2680 | 57.0344 |
| Projected quotas with trained downstream | **58.6690** | 55.8669 | **59.8949** | **58.1436** |

Each contrast below is trained downstream minus its named control; units are
percentage points. Intervals use the three seed differences, with the
Student-t critical value for two degrees of freedom.

| Control | Seed 3 difference | Seed 4 difference | Seed 5 difference | Mean difference | Nominal paired 95% interval |
|---|---:|---:|---:|---:|---|
| Native MonTi | +6.1296 | +4.3783 | +2.9772 | +4.4950 | [0.5715, 8.4185] |
| Quotas with original scores | +1.9264 | -1.0508 | +2.9772 | +1.2843 | [-3.9060, 6.4746] |
| Quotas with random connections | +1.9264 | -1.2259 | +2.6270 | +1.1092 | [-3.9892, 6.2075] |

The registered rule requires at least +1 pp over native MonTi, positive native
differences in all three seeds, a positive lower native-contrast interval, and
a mean above both quota controls. It does not require the intervals against
the quota controls to exclude zero. Passing that rule is therefore compatible
with unresolved evidence for a learning benefit over either matched control.

## Clean errors and attack transitions

These are pooled counts over the 1,713 repeated seed/group/node occurrences in
each arm, not counts of independent news articles. A new miss is a clean-correct
target becoming incorrect; a recovery is a clean-incorrect target becoming
correct. Every row satisfies `attacked wrong = clean wrong + new misses - recoveries`.

| GAGA arm | Clean wrong | Attacked wrong | New misses | Recoveries |
|---|---:|---:|---:|---:|
| Native MonTi | 666 | 919 | 253 | 0 |
| Quotas with original scores | 666 | 974 | 308 | 0 |
| Quotas with random connections | 666 | 977 | 311 | 0 |
| Quotas with trained downstream | 666 | 996 | 330 | 0 |

The trained arm has 77 more wrong occurrences than native, 22 more than the
original-score quota control and 19 more than the random quota control. These
are net count differences. They do not describe which individual targets were
gained or lost relative to each control. The complete
[case counts](case_counts.csv) retain case-level transitions.

## GCN results

| Arm | Seed 3 MR % | Seed 4 MR % | Seed 5 MR % | Mean MR % |
|---|---:|---:|---:|---:|
| Native MonTi | 96.8476 | 98.0736 | 96.4974 | 97.1395 |
| Quotas with original scores | 97.7233 | 97.8984 | 97.3730 | 97.6649 |
| Quotas with random connections | 97.7233 | 97.8984 | 97.3730 | 97.6649 |
| Quotas with trained downstream | 97.7233 | 97.8984 | 97.3730 | 97.6649 |

The three quota arms have equal GCN aggregate rates. Equal rates alone do not
prove identical node predictions. All per-seed clean errors, new misses and
recoveries for both detectors are in [seed metrics](seed_metrics.csv).

## Actual resource allocation

This is a **closest-feasible allocation study**, not an exact-half experiment.
For `b` auxiliary nodes, `t` targets, `c` candidates and total edge budget `e`,
the target quota is `ceil(e/2)` projected into
`[max(b, e-b*c), min(b*t, e)]`; the rest goes to candidates. Each auxiliary node
retains at least one target edge. Duplicate edges and self-loops do not satisfy
the budget. Quota arms have no auxiliary-to-auxiliary edges.

Of 192 validation seed/group cases, **150 (78.125%)** required projection. Across
all 192 cases, the actual target-edge fractions range from
**6.3830% to 100%**, with a case median of **25.5319%**. Only 24 cases are
exactly half; odd-budget rounding can also occur without projection. The
edge-weighted target fraction is **23.9922%**. Across all 576 TRAIN/VALID caches,
405 required projection.
The full [per-case quota table](quota_by_case.csv) records requested and realized
quotas, role counts and physical auxiliary-degree summaries for every arm.

Total auxiliary-node and edge budgets are preserved. Individual auxiliary-node
degrees can change. Compared with native MonTi, the quota arms also change
target allocation and remove native auxiliary-to-auxiliary connections.
Consequently, their gain over native does not isolate downstream learning.

## Execution cost and retained failures

The main run used 576 frozen-upstream generations, 1,500 GCN training forwards,
832 GCN validation forwards and 832 official GAGA validation forwards. The
separate TRAIN preflight used four upstream generations, 12 GCN forwards and
four downstream updates. No TEST or API queries were made; GPU use was **0 hours**.

Main-process wall time was **3,574.04 seconds**. Active guardian time across
recorded startups and completed phases was **3,619.56 seconds**. Elapsed time
including completed-phase review wait was **3,660.80 seconds**; recorded child
process CPU time totalled **9,223.58 seconds**. These shared-server measurements
are descriptive, not a dedicated hardware benchmark or a monetary cost estimate.

Two startup failures are retained: missing CUDA-linked DGL shared libraries,
and a Python 3.8 incompatibility in the test-log writer. Both stopped before
real-data model queries. Their guardian time totalled **17.84 seconds**, already
included above. Separate amendments corrected startup compatibility without
changing the scientific policy. [Detailed costs and failed starts](latest.json)
and the [execution receipts](../../evidence/downstream_cooperation_v1/) preserve
the records.

## Audit scope and interpretation

The [saved-logit audit](audit_saved.json) passed the prediction, transition and
aggregate recount. The separately authored [saved-tensor audit](audit_tensors.json)
passed file and tensor hashes, endpoint identities, quota projection and owned
adjacency geometry. They performed no new inference and are not human review.
The geometry audit checks stored feature bindings, not an executed DGL graph;
seal byte bindings do not prove wall-clock chronology.

The intervals are nominal three-seed development intervals after an adaptive
research history, not an independent population guarantee. No equal-capacity
training control or action-hidden ablation isolates the effect of receiving
upstream decisions. Training uses a biased selected-edge straight-through
surrogate rather than the exact derivative of discrete topology. Results do
not establish mutual upstream adaptation, a real fraud supply chain, first
novelty, course approval or final TEST improvement.

[Latest structured results](latest.json) and [publication provenance](provenance.json)
bind these compact exports to scientific source
`9e013cb182b5af71a9d0b719b5a939deda3c88e2` and protocol SHA256
`5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13`.
Original private assets are still required for a full ML rerun; these exports
are saved-record evidence, not an end-to-end public execution bundle.
