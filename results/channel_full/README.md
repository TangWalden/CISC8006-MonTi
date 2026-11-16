# Frozen two-dataset complete TEST comparison

The predeclared mean-MR criterion **passed**. Higher miss rate means a stronger
attack. Native means the author's original attack checkpoint, not a clean graph.
No hyperparameter, method, target cohort or repetition was changed using these
TEST results. Negative cases and every control remain included.

| Method | GossipCop MR | YelpChi MR |
|---|---:|---:|
| Author checkpoint | 44.2638% | 30.9098% |
| Frozen two-channel retrieval | 49.1932% | 37.6957% |
| Same-class equal-dose random | 44.1591% | 32.9605% |
| Joint-pool equal-dose random | 44.2194% | 31.1708% |

| Dataset | Gain over native (pp) | Over same-class random (pp) | Over joint-pool random (pp) |
|---|---:|---:|---:|
| gossipcop | +4.9295 | +5.0341 | +4.9738 |
| yelpchi | +6.7860 | +4.7353 | +6.5250 |

## Fixed method and fair comparison

The [unchanged development-selected method](../../experiments/channel_retrieval/)
uses an independent fixed TRAIN-fitted teacher to rank real endpoints in both
known TRAIN classes and verify shortlisted substitutions. At most two candidate
slots change; original target membership/order, target attachments, injected
features/degrees, injected-to-injected edges and actual node/edge budgets remain
fixed. All modified and random graph inputs are physically rebuilt and
parity-checked; original GossipCop inputs reuse the hash-pinned native captures.

Same-class random matches chosen slots, endpoint classes and exact exchanged
edge dose. Joint-pool random matches slots/dose without fixing class frequency.
No official victim prediction selects an action. Both datasets' complete action
sets were sealed before either official scoring phase. Equal graph budgets do
not imply equal compute: the proposed search incurs extra teacher computation.

GossipCop includes all 909 original TEST groups, 10,515 target occurrences and
1,221 unique targets per repeat. YelpChi includes all 548 original TEST groups,
2,682 occurrences and 2,452 unique targets. Repetitions 0/1/2 share fixed author
and teacher checkpoints. GossipCop reuses exact saved original captures;
YelpChi generates the original author attack once per case, shared by all arms.
There is no new upstream or teacher training, API call, GPU work or LifeIns run
in this full-cohort comparison. A separate owner-requested
[GPU timing/parity check](../channel_gpu/) used four fixed development cases
and 0.0188359131 GPU hours; it found no useful end-to-end speedup, so the
already-running CPU evaluation was preserved without interruption.

## Interpretation and limits

These are full-cohort measured outcomes, not proof that every case improves.
Per-repeat results, losses, gains and group-equal summaries are in
[summary.json](summary.json). Nominal paired three-repeat intervals condition
on the fixed checkpoints; overlapping targets are not independent samples.
Every repetition beats native and both random controls in both datasets.
YelpChi native/channel error counts happen to be identical across the three
repeats; their zero-width conditional interval is not zero population or
generalization uncertainty.
GossipCop TEST has prior historical exposure and is not an untouched holdout.
The [saved-only paired details](paired_details.json) include positive and negative
outcome-selected examples for explanation, never for retuning.

For illustration, GossipCop group 1785 has 35 targets and changes from 17
native misses to 24 in each repeat. YelpChi group 1287 has nine targets and
changes from two to eight. Negative examples are retained: GossipCop group
1531 changes from one miss to zero out of seven targets; YelpChi group 1403
changes from two to one out of two. These examples were selected after the
outcomes and do not define the evaluation cohort.

Two predetermined development cases per dataset reproduced the frozen decisions
before the full run. The root-authored saved tensor audit passed
192,372 checks; this is not independent or
human verification. A second saved-only arithmetic implementation by the same
root author passed 61,194 checks without model calls. Whole-driver elapsed
time was 15502.97 seconds (about 4 hours 18 minutes).
Concurrent worker walls and process CPU times are separate, not additive elapsed
time. All unsuccessful attempts, if any, stay in the provenance record.

[Compact frozen protocol](protocol_summary.json), [execution and hashes](execution.json),
[full runner source](../../experiments/channel_full/). Large raw graph cards,
models and predictions remain in the recorded private stage. Instructor
requirements, human verification/contributions, rehearsal and final submission
remain separate; this result does not declare course completion.
