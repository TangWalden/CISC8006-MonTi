# Two-channel relation retrieval: measured development comparison

The registered two-dataset primary screen **passed**. Higher GAGA MR means
a stronger attack; native below is the frozen author's attack, not clean data.
All alternatives and negative contrasts are retained.

| Method | GossipCop MR | YelpChi MR |
|---|---:|---:|
| Frozen author attack | 57.9101% | 29.1304% |
| Existing single neural | 59.1360% | 29.1304% |
| Existing two-step rewriter | 59.4863% | 29.1304% |
| Previous benign-only full-pool search | 60.4203% | 29.1304% |
| Its matched random | 58.4355% | 29.1304% |
| New two-channel gradient search | 64.1565% | 34.7826% |
| Same-class matched random | 57.6182% | 30.5797% |
| Joint-pool matched random | 57.5598% | 29.2754% |
| Two-channel uniform-shortlist search | 61.4127% | 32.8986% |
| Its same-class matched random | 57.3263% | 30.1449% |

New-method contrasts:

- gossipcop: native +6.2464 pp, benign_retrieval +3.7361 pp, random_channel +6.5382 pp, random_joint +6.5966 pp, uniform +2.7437 pp.
- yelpchi: native +5.6522 pp, benign_retrieval +5.6522 pp, random_channel +4.2029 pp, random_joint +5.5072 pp, uniform +1.8841 pp.

## What changed

Every primary contrast is positive in each of the three conditional repeats,
including against the uniform-shortlist search. Relative to native, GossipCop
gains 113 and loses 6 miss occurrences (net 107/1,713). YelpChi gains 39 and
loses zero across the three repeats: **13 additional unique targets per
230-target repeat**, not 39 independent targets. YelpChi's identical repeat
gains do not imply zero population uncertainty.

Selected illustrative repeat-0 cases (misses / targets):

| Dataset / group | Native | Previous benign-only search | New method | Same-class random |
|---|---:|---:|---:|---:|
| GossipCop / 1402 | 15/42 | 18/42 | 20/42 | 16/42 |
| GossipCop / 1473, negative example | 3/6 | 3/6 | 2/6 | 2/6 |
| YelpChi / 655 | 2/7 | 2/7 | 5/7 | 2/7 |

The previously non-flipping YelpChi target 41429 changes from fraud margin
+0.042818 (native) and +0.029866 (benign-only search) to -0.138584. Same-class
random also crosses for this particular target (-0.022149), so this example
alone does not establish superiority over random. The complete contrasts do.

The search chooses TRAIN-fraud endpoints for 297/332 accepted GossipCop
substitutions and 381/384 YelpChi substitutions. In YelpChi, all 39 gained
occurrences lie in cases choosing only TRAIN-fraud endpoints. These are
post-action descriptive strata, not randomized causal effects of class choice.
The sizable advantage over same-class random shows that merely choosing the
same class does not reproduce the full measured result.

The [downstream implementation](../../experiments/channel_retrieval/) searches
both known TRAIN-benign and TRAIN-fraud endpoint pools. It first considers
which group-summary channel a connection changes, ranks real endpoints by
the fixed independent teacher's input derivative, then verifies shortlisted
choices. At most two candidate slots change. All target memberships/order,
target edges, injected features/degrees and actual node/edge budgets remain
fixed. No validation/test label enters either endpoint pool. No LLM is used.

GAGA's existing label-group aggregation and HGT's typed-message principle
motivate the channel distinction; HotFlip motivates discrete gradient ranking.
These are prior ideas, not proof of first novelty or a newly trained HGT model.
This is a new downstream optimizer with broader endpoint eligibility than
the previous TRAIN-benign-only method.

Same-class random matches selected class, donor slots and exact edge dose;
joint-pool random matches slots/dose but not class frequencies. Uniform
shortlist uses both classes and the same per-step verification width without
gradient ranking. The new search doubles per-region shortlist opportunities
relative to the previous benign-only round. Same physical budgets therefore
do not imply equal computation, and the comparison does not by itself isolate
class expansion from extra search. STOP decisions also change total work.

## Evidence and limits

Both datasets' actions sealed before either official validation phase.
GossipCop retains 64 groups, 571 occurrences/240 unique targets per repeat;
YelpChi 64 groups, 230/230. Three capture repeats share one author checkpoint
and one teacher per dataset. Correlated occurrences and adaptive reuse of
development cohorts preclude an untouched-confirmation claim. Conditional
three-repeat intervals are reported in full, without treating targets as IID.

All fixed TRAIN mixed-channel checks and final physical canonical sequence
checks passed. The root-authored saved audit passed
31,224 checks; this is not independent
implementation or human verification. Whole-driver wall time was
6056.26 seconds; process CPU times and overlapping phase
walls are separate. No upstream/teacher training, optimizer updates, GPU,
API, TEST or LifeIns evaluation occurred.

[All metrics, contrasts and counters](summary.json), [frozen protocol](protocol.json),
[paired interpretation](paired_interpretation.json), [chosen-channel details](channel_details.json),
[illustrative cases](cases.json),
[execution, hashes and audit](execution.json). Outcome-selected illustrations
are explanatory and were not used to select the evaluated cohort. Large raw
models/graphs/logs remain private. Course approval, human verification and
final submission are separate outstanding requirements.
