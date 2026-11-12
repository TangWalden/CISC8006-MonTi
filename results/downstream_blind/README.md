# Action summary ablation results

Providing the downstream policy with the upstream target-attachment summary
did **not** improve the registered primary comparison. Visible minus blind
GAGA miss rate averaged **-0.5838 percentage points**, with nominal paired
three-seed 95% interval **[-4.2838, 3.1163] pp**. The direct-summary criterion
failed. This does not prove that every form of upstream information is useless.

The blind policy exceeded fixed random candidate connections by **+1.6929 pp**
with nominal interval **[0.2944, 3.0914] pp**, a secondary development result.
The control is not promoted to a new winner. No final TEST evaluation ran,
and the [parent pilot](../downstream_cooperation/README.md) retains its original
results and qualification rule.

## Matched design and reused measurements

Only the new action-blind downstream was trained. Both trained policies have
**151,872 parameters**, identical initial state hashes per seed, the same
training order, optimizer settings, four terminal epochs, frozen upstream
captures and fixed GCN TRAIN feedback. There is no GAGA training feedback or
VALID checkpoint selection.

The visible policy receives each auxiliary node's actually attached-target
feature mean and `log1p(actual target count)`. The blind policy instead receives
the whole target-list feature mean and `log1p(registered target quota / number
of auxiliary nodes)`, repeated across rows. All parameter branches remain.
Auxiliary features, candidate features, cached representations and the physical
target connections remain unchanged. Those shared inputs and TRAIN feedback
can retain information correlated with upstream actions; this design removes
only the direct realized-attachment summary.

Seeds are 3, 4 and 5, with 128 TRAIN groups and the parent's same 64 VALID groups.
Each arm has **571 target occurrences per seed**, or **1,713 correlated** pooled
occurrences. Of 240 unique validation targets, **233** appeared in earlier
development contexts. The current cohort outcomes were already inspected in
the parent experiment, so this is a post-hoc mechanism ablation, not independent
confirmation on unseen data.

The four parent arms and 64 clean groups were reused without fresh inference:
768 parent arm/case records plus 192 newly evaluated blind cases give 960
combined records. The old random control is candidate-edge selection on the
same cached input, not the historical random target-fill policy.

## GAGA results

Higher MR means more fraud-target predictions are incorrect and hence a
stronger attack. MR includes errors already present on the clean graph.

| Arm | Seed 3 MR % | Seed 4 MR % | Seed 5 MR % | Mean MR % | Origin |
|---|---:|---:|---:|---:|---|
| Native MonTi | 52.5394 | 51.4886 | 56.9177 | 53.6486 | Reused parent |
| Quotas with original scores | 56.7426 | 56.9177 | 56.9177 | 56.8593 | Reused parent |
| Quotas with random connections | 56.7426 | 57.0928 | 57.2680 | 57.0344 | Reused parent |
| Visible trained downstream | 58.6690 | 55.8669 | 59.8949 | 58.1436 | Reused parent |
| Blind trained downstream | 58.6690 | 58.1436 | 59.3695 | 58.7274 | New blind fit |

| Contrast | Seed 3 pp | Seed 4 pp | Seed 5 pp | Mean pp | Nominal paired 95% interval pp |
|---|---:|---:|---:|---:|---|
| Visible minus blind, primary | 0.0000 | -2.2767 | +0.5254 | -0.5838 | [-4.2838, 3.1163] |
| Visible minus random, reused | +1.9264 | -1.2259 | +2.6270 | +1.1092 | [-3.9892, 6.2075] |
| Blind minus random, secondary | +1.9264 | +1.0508 | +2.1016 | +1.6929 | [0.2944, 3.0914] |

The registered primary requires positive visible-minus-blind differences in
all three seeds and a positive nominal interval lower bound. Neither condition
was met. These are three-seed intervals on a fixed, adaptively inspected
development cohort, not independent population guarantees or a correction for
the wider research history.

## Clean errors and new misses

The following counts pool repeated seed/group/node occurrences; they are not
counts of independent news articles. A new miss changes a clean-correct target
to incorrect. Each row satisfies
`attacked wrong = clean wrong + new misses - recoveries`.

| GAGA arm | Clean wrong | Attacked wrong | New misses | Recoveries |
|---|---:|---:|---:|---:|
| Native MonTi | 666 | 919 | 253 | 0 |
| Quotas with original scores | 666 | 974 | 308 | 0 |
| Quotas with random connections | 666 | 977 | 311 | 0 |
| Visible trained downstream | 666 | 996 | 330 | 0 |
| Blind trained downstream | 666 | 1006 | 340 | 0 |

Blind has 10 more wrong occurrences than visible and 29 more than random.
These are net aggregate differences, not a claim about the same individual
nodes changing in every seed. GCN MR for blind matches the three existing quota
arms in each seed: **97.7233%, 97.8984%, 97.3730%**, mean **97.6649%**. Equal
aggregate rates do not imply equal individual predictions. Complete detector
counts and per-case transitions are in [seed metrics](seed_metrics.csv) and
[case counts](case_counts.csv).

## How much action information was removed

Saved-input algebra was independently recounted for all 576 parent caches.
In every full-target block, each auxiliary node connects to every target, so
the visible mean/count are already the blind replacements.

| Context and seeds | Cases per seed | Full target block | Same target mean | Same log count | Same complete summary |
|---|---:|---:|---:|---:|---:|
| TRAIN, each seed | 128 | 84 | 84 | 98 | 84 |
| VALID, seeds 3 and 4 | 64 | 50 | 50 | 59 | 50 |
| VALID, seed 5 | 64 | 50 | 50 | 58 | 50 |

Thus the direct summary is unchanged for **50/64 validation contexts** per seed
and **84/128 training contexts** per seed. The other cases can still share
their count input without sharing the whole summary. Only 14/64 validation
contexts per seed have different complete summaries, limiting the contrast.

Nevertheless, blind and visible final adjacency plans match exactly in only
**20/192** seed/group cases; blind and random match in 6/192. This is not a
contradiction: training on the other contexts can change fitted weights and
therefore decisions even for cases whose direct input summaries are identical.
No prediction equivalence is inferred merely from summary equality.
[Conditioning records](conditioning_scope.csv) retain all algebra cases.

## Budgets execution and audits

All role quotas, upstream features, target blocks and total structural budgets
are inherited unchanged. These are projected, closest-feasible quotas rather
than exact halves in every case; the [per-case quota table](quota_by_case.csv)
retains actual counts for all five arms. This ablation changes conditioning,
not the parent's quota policy.

The registered preflight passed **31 functional checks** without skips and four
real TRAIN checks. Main training made **1,500 updates**, with 36 no-choice
training occurrences skipped. It reused all upstream captures and made no new
upstream generation. New main evaluation used **192 GCN and 192 GAGA forwards**;
preflight separately used four GCN forwards and four updates. TEST queries,
external API calls and GPU hours were all **zero**.

Main wall time was **1,926.89 seconds**, approximately 32 minutes. Active
guardian time was **1,952.70 seconds**; elapsed phase time including review wait
was **1,992.52 seconds**. Recorded main/preflight child CPU time totalled
**5,039.17 seconds**. The two saved-artifact audits took 3.58 and 6.69 wall seconds
separately. There were no failed startup attempts in this blind phase; parent
costs and its retained failures remain in the parent report. These shared-server
timings are descriptive, not a hardware benchmark or a monetary cost estimate.

Both the [saved-logit audit](audit_saved.json) and
[saved-tensor and summary audit](audit_tensors.json) passed. The latter hashes
checkpoint bytes without loading parameter states, deserializes own cached
inputs and plans, and checks geometry and summary algebra. It does not inspect
executed DGL graphs, independently prove seal chronology or constitute human
technical review. Real fraud supply-chain behavior, novelty, course approval
and final TEST effectiveness remain unestablished.

See the [design](../../experiments/downstream_blind/README.md),
[frozen protocol](../../evidence/downstream_blind_v1/execution_protocol.json),
[execution evidence](../../evidence/downstream_blind_v1/),
[structured results](latest.json) and [publication provenance](provenance.json).
Scientific source is `b7809f420fc1b5453e748a9fd19c52234541cfd8`; protocol SHA256
is `8bafbaf6c7590cd97f492f9d5edca90848193c76e5e3da0c4fdf5d8399500fd8`.
Compact exports preserve saved evidence; original private assets are required
for a complete model rerun.
