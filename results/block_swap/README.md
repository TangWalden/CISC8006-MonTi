# Whole-column recipient learning: null result

Same frozen author checkpoint and cached 128 TRAIN/64 VALID cases per repeat.
The learned and random arms share the exact donor column, removed rows,
activation and number of replaced edges. All injected features, target edges,
injected degrees and total budgets are preserved. Higher MR is stronger attack.

| Official GAGA MR | Repeat 0 | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|---:|
| Native | 58.1436% | 58.1436% | 57.4431% | 57.9101% |
| Learned block recipient | 58.1436% | 57.9685% | 57.6182% | 57.9101% |
| Matched random | 58.1436% | 58.3187% | 57.4431% | 57.9685% |

Learned minus native: 0 pp. Learned minus random: -0.0584 pp,
nominal paired t95 [-0.7229, 0.6062] pp. Both the original strict screen and
the owner's later overall-mean criterion fail. The three repeats share one
upstream checkpoint; 571 occurrences/240 unique targets per repeat are reused
development data, not independent confirmation.

The 79 activated learned cases contain 573 occurrences. Two GAGA decisions
change relative to native: repeat 1, group 1020 loses one miss; repeat 2,
group 1444 gains one. Net zero. On those active cases only 26 GCN loss gains
exceed 1e-6, while 15 changes worsen loss by more than 1e-6. Prediction error
is present. It does not prove that removing prediction error will improve
GAGA transfer.

The model-free membership check finds 307/573 active occurrences invariant.
Only 2/79 whole cases are entirely invariant, versus 27/79 in the prior
single-edge study, but their activated cohorts differ; this is not a paired
causal estimate. More visible graph changes did not produce better MR.

## Next measured question

A [direct-GCN block search](../../experiments/block_swap/hard_search.py)
measures the same small proposal pool rather than predicting its GCN gain.
Matched random retains the exact donor, activation and exchange count.
It adds surrogate inference cost. The study is running; no efficacy claim.
The owner clarified before GAGA evaluation that overall mean need only exceed
both native and random, without requiring every case/repeat to improve.
Original strict qualification and uncertainty will also be reported.

## Evidence

- [Complete seed metrics and counters](summary.json), [paired failure cases](diagnosis.json).
- [Membership diagnosis](reachability.json), [TRAIN group-held-out diagnostic](train_generalization.json).
- [Saved audit](audit.json): 13,538 checks passed; root-authored, not independent human review.
- [Original driver receipt](execution.json), [original audit failure](audit_failure_attempt1.json), [audit amendment](audit_amendment_receipt.json), [closure binding](audit_closure.json).
- [Implementation](../../experiments/block_swap/).

The original audit failed because an empty JSON feature list lost its `(0,32)`
shape. The separately recorded amendment restores that shape. Scientific
outputs and original failed receipts remain unchanged. Main wall 3,051.01 s,
process CPU 8,416.88 s; GPU/API/TEST use zero. Large raw outputs stay private.
