# Contextual feature residuals: negative development result

All author features begin unchanged, then a new downstream network predicts
bounded feature corrections on the fixed original topology. The independent
full-TRAIN GAGA teacher supplies feedback, not the official evaluation victim.
Three downstream seeds share one released author checkpoint and cached noise
repetitions. Each arm/seed retains 64 VALID groups, 571 occurrences and
240 unique targets. Development has been reused.

| Official GAGA MR | Repeat 0 | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|---:|
| Author native | 58.1436% | 58.1436% | 57.4431% | 57.9101% |
| Attention residual | 57.6182% | 57.4431% | 56.9177% | 57.3263% |
| Matched random signs, attention magnitudes | 58.3187% | 57.9685% | 57.6182% | 57.9685% |
| Mean-context residual | 57.6182% | 57.4431% | 57.0928% | 57.3847% |
| Matched random signs, mean magnitudes | 58.1436% | 58.1436% | 57.4431% | 57.9101% |

Higher MR is stronger attack. Both candidates fail the owner's overall-mean
criterion. Attention minus native is -0.5838 pp; attention minus its random
control is -0.6421 pp. Mean minus native/random is -0.5254 pp. Attention gains
zero new wrong occurrences and loses ten relative to native across the
1,713 correlated occurrences. Mean gains zero and loses nine.

## What the failure supports

The independent teacher's VALID MR improves by only one occurrence per repeat.
Its TRAIN group-mean attack loss changes little after the first epoch. This
is not evidence of a well-optimized transferable attack. Same-victim scale
checks find median absolute margin changes of about 0.0072-0.0115, while
median absolute native margins are about 1.86-1.88. Raw teacher/victim logits
are not assumed calibrated to the same scale. Many margin changes have
opposing directions in the two models.

The two unknown-group feature tokens can be edited with exact gradients on
fixed topology, but this tested architecture/budget did not yield a gain.
This is not a universal feature-attack capacity bound. It motivates testing
new known-benign relation context, not claiming attention generally fails.

## Design and audit

Feature magnitude is bounded by one original-TRAIN standard deviation and
the symmetric room inside MonTi's original full-graph feature box. Random
signs exactly match every coordinate's learned magnitude. All nodes, edges
and injected degrees remain fixed. The simpler mean network has fewer
effective parameters, so it is not a capacity-matched attention attribution.
Architecture-matched GAGA feedback changes the original GCN-only attack setting.

- [Complete metrics, counters and intervals](summary.json), [frozen protocol](protocol.json).
- [Saved audit](audit.json): 10,266 checks passed, including independent feature bounds and amplitude matching; root-authored, not human verification.
- [Independent teacher](teacher.json): fixed 38 TRAIN epochs, in-sample quality only; [canonical feature-gradient preflight](preflight.json).
- [Paired transfer diagnosis](transfer_diagnosis.json), [same-victim scale check](same_victim_scale.json).
- [Execution](execution.json), [implementation](../../experiments/residual_attention/).

Six downstream models completed 3,072 updates. Main wall time 1,276.79 s,
recorded execution-region CPU 2,854.83 s, guardian 1,282.98 s. Imports/version
checks precede that CPU timer; their CPU is unmetered, while guardian wall
covers child startup. Teacher/preflight costs are separate. GPU/API/TEST use
was zero. The current
[context-retrieval decoder](../../experiments/portal_decoder/) subsequently
completed a [positive matched full-cohort comparison](../portal_full/).
That separate method does not change this negative result.
