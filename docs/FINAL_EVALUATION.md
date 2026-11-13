# Final evaluation: upstream/downstream role division

> This is the earlier staged evaluation. The later author-checkpoint context-
> retrieval benchmark is [complete and reported separately](../results/portal_full/).
> Its positive comparison does not change the negative result below.

Status: registered before model calls. Scientific source: `32d563f49e19cea3b74bc74f048ddd9a5a934074`.
Protocol SHA256: `2716a20b1b8c251466e35f7e93b750a3ab72b668c38d0199f9026b9c884f675f`.
Execution and a saved-record audit are complete. The registered criterion
failed: learned MR 44.2257%, native 43.8168%, random 44.0165%; differences
are uncertain and seed 2 is negative. [Full results and audit scope](../results/final_staged/).

## Question and fixed comparison

Does the complete staged scheme increase GAGA fraud-target miss rate relative
to the frozen reproduced MonTi generators, under the same total structural
budget? The owner authorized this final evaluation, Bento presentation and
demonstration on 12 November 2026. No new direction or ablation is included.

| Arm | Auxiliary features and target connections | Surrounding candidate connections |
|---|---|---|
| Native MonTi | Original frozen generator output | Original frozen generator output |
| Random completion | Frozen features; closest-feasible target quota | Private seeded random completion |
| Staged trained | Same frozen features and target block as random | Trained downstream scorer |

The main comparison is staged trained minus native MonTi. Random completion is
a secondary control, not another candidate from which to select a winner.
The staged scheme includes the quota policy and separate edge permissions.
The native comparison tests that complete bundle.

## Frozen scope

* Original processed GossipCop-S split 0, all 937 TRAIN groups and all 909 TEST
  groups. Every TEST group remains in the occurrence-weighted denominator.
* Original learned-candidate MonTi best checkpoints from the completed formal
  reproduction, seeds 0, 1 and 2. Their selected epochs are 4, 3 and 5.
  The upstream weights never update.
* Original official GCN surrogate and GAGA evaluation checkpoints. GCN supplies
  TRAIN feedback; GAGA is loaded only after all downstream weights and TEST
  plans are sealed. There is no VALID or TEST checkpoint selection.
* The existing 151,872-parameter downstream scorer, hidden dimension 64, four
  terminal epochs, Adam learning rate 0.001, weight decay 0.0001, gradient norm
  cap 5, and the existing selected-edge straight-through approximation.
* Fixed original node and edge budgets. Request half of the edge slots for
  targets, then project to the nearest feasible quota. At least one target
  connection per auxiliary node is required. Random and trained arms preserve
  identical target blocks, auxiliary features, candidate lists and quotas.
* No auxiliary-to-auxiliary edges in the two completion arms; native MonTi keeps
  its native output. Individual auxiliary degrees can differ.
* CPU execution, four threads, no GPU use. The main process has a 12-hour wall
  limit, the guarded phase a 16-hour limit, and automatic retries are disabled.

The [machine-readable registration](../evidence/final_staged_v1/protocol.json)
and [prior metadata boundary](../evidence/final_staged_v1/metadata_scope.json)
retain exact source and input identities. TEST contains 10,515 target occurrences
per seed; TRAIN contains 11,211. Raw predictions,
features and checkpoints remain in ignored server run storage.

## Ordering and limits

All 5,538 upstream TRAIN/TEST captures and their capacity checks must complete
before the first downstream update. Four training epochs allow at most 11,244
GCN/backward/update opportunities; cases with no candidate choice are logged
and do not cause an optimizer update. No TEST cases are removed.

All final downstream weights and 2,727 three-arm TEST plans are sealed before
evaluation. One shared clean evaluation plus the three arms allows exactly
9,090 GCN and 9,090 GAGA final forwards. The bounded preflight uses four TRAIN
contexts and at most 12 GCN forwards, with no GAGA or TEST calls.

The predeclared primary criterion requires a positive mean difference, all
three paired seed differences positive, and the nominal paired t 95% interval
entirely above zero. All outcomes are retained. A failed criterion does not
trigger extra seeds, new controls, larger budgets or a replacement winner.

## Metric and interpretation

MR is the percentage of incorrect fraud-target occurrences among all registered
target occurrences. Higher MR means a stronger attack and a weaker detector.
Pre-existing clean misses remain included; clean-to-attacked changes are also
reported separately. Auxiliary and candidate nodes never enlarge or filter the
target denominator. Groups can overlap, so occurrences are not independent
replications; the three training seeds are the replication unit.

The fresh paired CPU replay uses the exact frozen formal generator weights,
but its native absolute MR is not assumed to equal the historical 43.7280%
GPU result. That earlier reproduction result remains separately preserved.
The earlier development pilot used another checkpoint family and a validation
cohort; its 53.6486% to 58.1436% result is not the final TEST comparison.

## Formal results and delivery

The completed results are in the [compact result export](../results/final_staged/).
The report and presentation drafts predate this result and require an explicit
final update. Human review, rehearsal, presentation and course submission
are recorded only when performed. Subsequent development cannot treat this
observed TEST set as untouched confirmation.
