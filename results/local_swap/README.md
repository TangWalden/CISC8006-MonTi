# Author-checkpoint local swap: no demonstrated gain

One original author checkpoint is frozen. Three cached generation-noise
repetitions use 128 TRAIN and 64 VALID groups each. The downstream learns
one candidate-edge exchange from hard-graph GCN loss differences. Learned
and matched random activate on the same cases and exchange one edge each.

| Official GAGA MR | Repeat 0 | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|---:|
| Author native | 58.1436% | 58.1436% | 57.4431% | 57.9101% |
| Learned local swap | 58.1436% | 58.1436% | 57.4431% | 57.9101% |
| Matched random swap | 58.1436% | 58.1436% | 57.2680% | 57.8517% |
| Always random, auxiliary | 57.9685% | 58.1436% | 57.2680% | 57.7933% |

Learned minus native is zero in every repetition. Learned minus matched
random is +0.0584 pp, nominal paired t95 [-0.1928, 0.3096] pp. The registered
criterion **fails**. These are conditional repetitions of one author model,
not three independently trained upstreams. Each arm/repetition retains
571 correlated target occurrences and 240 unique targets, with prior
development exposure. No TEST calls were made.

## Evidence-guided revision

The 79 activated learned cases contain 669 target occurrences. Only two
GAGA decisions change relative to native: group 1432 in repeat 1 gains one
miss, while group 1492 loses one. Net gain is zero, not universal identical
predictions. A saved-graph, model-free diagnosis finds **455/669 occurrences
have exactly unchanged canonical added-neighbor memberships**. With fixed
features and group masks their mathematical aggregate input is unchanged;
saved GAGA logits match exactly on those occurrences. The remaining 214
occurrences do not establish a sufficient margin change or successful transfer.

This motivates a [whole-column block exchange](../../experiments/block_swap/):
remove one donor candidate's attachments together, and learn a replacement.
Its random control shares the exact donor rows and activation. This is a
new development hypothesis, not a positive result inferred from this diagnosis.

## Trace and audit

- [Summary, counters, source and protocol](summary.json), [paired cases](saved_diagnosis.json).
- [Saved fit/logit audit](audit.json): 9,633 checks, 772 hashes, 832 evaluation records.
- [Physical geometry audit](geometry.json): 576 native scenarios and 768 final plans; exact individual injected degrees and budgets preserved.
- [Model-free reachability diagnosis](reachability.json), [execution receipt](execution.json).
- [Registered source/settings summary](registration_summary.json), [diagnosis registration](reachability_registration.json), [adjacency directionality check](reachability_directionality_check.json).
- [Implementation](../../experiments/local_swap/).

Main wall time 4,283.80 s; guardian including preflight/audit 4,305.86 s.
Main process CPU 11,544.98 s. GPU/API/TEST use is zero. The root-authored
recounts are not independent implementation review or human verification.
Large raw tensors and prediction logs remain in private run storage.
