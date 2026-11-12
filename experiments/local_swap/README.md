# Frozen-author local exchange development study

This study keeps the authors' released MonTi checkpoint fixed and learns a
small downstream predictor of **the benefit of replacing one candidate edge**.
It does not refit MonTi, redistribute target/candidate quotas, or use an agent.

For each case, eight distinct legal row/remove/add triples are sampled before
outcomes. Each triple removes one existing injected-to-candidate edge and adds
one absent edge in the same row. All features, target connections, injected-to-
injected connections, candidate identities, total budgets and injected-node
degrees remain unchanged. At most one exchange is executed per case.

The downstream represents an exchange using differences in eight observable
relationship descriptors, crossed with four base-case context values. A
zero-intercept, 32-weight ridge model predicts actual hard-graph GCN loss
improvement measured on TRAIN. Its score changes sign when old/new descriptor
roles are reversed in the same context. This is an implementation choice, not a
claim of a new learning theorem or established novelty. No straight-through
topology gradient, victim output or validation label trains the predictor.

## Fixed comparison

- Same previous 128 TRAIN groups and 64 VALID groups: 571 target occurrences,
  240 unique targets per repeat. These development contexts have been used.
- Three noise/fitting repeats, seeds 0/1/2, conditional on **one** author model.
- Native author plan, learned local exchange, count-matched random exchange,
  and an auxiliary always-random exchange. Controls are not candidates.
- Matched random uses the same legal pool and exactly the learned per-case
  activation count. It does not isolate the activation rule; always-random is
  retained for that distinction. No-choice and abstention cases remain in all
  denominators.
- Gate: predicted absolute Softplus gain exceeds
  `max(1e-6, 0.1 * TRAIN gain RMS)`. This is a fixed tolerance, not calibrated
  confidence or a guarantee of GAGA benefit.
- One closed-form float64 fit per repeat, regularization 0.1, no hyperparameter
  search, extra seeds or checkpoint selection. All decisions are sealed before
  official-GAGA VALID evaluation. No TEST queries.

The primary contrast is learned minus count-matched random official-GAGA MR;
native comparison is also required. Reliable improvement is reported only if
both mean gains, all three paired differences, and both nominal paired t95
lower bounds are positive. This is conditional, reused development evidence,
not independent confirmation from 1,713 news articles.

## Execution and records

Source is committed and protocol/input hashes frozen before model calls. CPU
only, four threads, 300-second preflight and 7,200-second main ceilings, no
automatic retries. Maximum main counts are 576 upstream generations, 3,456
TRAIN GCN forwards, 832 evaluation forwards per classifier and three ridge
fits. No upstream updates, API calls or GPU use. A failed phase preserves its
attempt and stops. `audit.py` recounts saved logits, closed-form fits, cache
hashes and exact swap geometry; this is not independent human verification.

Results are pending execution. The previous full experiment is not relabeled
or replaced by this new development study.
