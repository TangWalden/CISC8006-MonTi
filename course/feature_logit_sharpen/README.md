# Fixed feature-logit sharpening on new VALID contexts

**Status: completed, small positive effect; the frozen development criterion
failed and no candidate qualifies.** This is a separate exploratory study after
the completed negative [prototype-mixing phase](../train_prototype_mix/README.md).
That earlier phase remains fixed at ratio 0.5 and failed; it is not expanded
with extra ratios, seeds or a favorable promoted control.

The owner's manual `continue` instruction authorizes bounded development work.
Automation monti remains deleted. Final TEST, final freeze/tag, human technical
review and course acceptance remain separate.

## Completed result

| Fixed arm | Mean official-GAGA MR |
|---|---:|
| `baseline` | 47.97338173% |
| `logit_sharp` | 48.30611010% |
| `linear_contrast` | 48.33635814% |
| `shuffled_delta` | 47.82214156% |

Higher MR is a stronger attack. Sharpening gains +0.33272837 pp versus the fresh
baseline, with seed changes +0.45372051 / +0.36297641 / +0.18148820 pp. The nominal
paired-t95 interval is [-0.01160766, 0.67706440] pp (`n=3`, `df=2`). All seeds are
positive, but the fixed +1 pp threshold and positive interval lower bound fail.
The linear comparator is slightly higher; no nonlinear superiority is claimed
and neither control becomes a candidate. Eleven wrong predictions are gained
and none lost across all 3,306 correlated occurrences: 11 group-seed wins, 373
ties, zero losses. This is a small measured effect, not a qualified improvement.

The [execution trace](../../evidence/feature_logit_sharpen_v1/execution_trace.json),
[complete saved summary](../../evidence/feature_logit_sharpen_v1/summary.json) and
[separate audit](../../evidence/feature_logit_sharpen_v1/independent_saved_tensor_audit_v1.json)
retain all arms, cases, controls and accounting. Fixed beta 2, seeds and cohort
remain closed; there is no multiplier search or outcome-driven expansion.

## Hypothesis grounded in the failed experiment

Half-mixtures with benign, all-TRAIN and fraud-TRAIN means all harmed official
GAGA performance. Benign mixing changed mean MR from 52.642835% to 47.479377%,
with one gained success and 170 lost baseline successes across the full 3,273
correlated occurrences. Benign and all-TRAIN mixtures produced identical GAGA
classes throughout. The independent audit reconstructed the reference means
with exact float32-bit agreement, so an incorrect TRAIN mean was not observed.

This supports retaining the original learned feature preferences as a testable
direction. It does **not** prove that strengthening them will improve attacks.
No validation-margin threshold or selected positive case determines this method.

## Four fixed transforms

The unchanged continuous feature head is
`x = min + (max-min)*sigmoid(linear_feat(z))`. It has no feature temperature.
For each nonconstant coordinate, recover `u` in the original raw-coordinate
box and apply one fixed contrast:

| Arm | Unit-coordinate operation | Role |
|---|---|---|
| `baseline` | Keep exact original feature bytes | Common fresh baseline |
| `logit_sharp` | `u*u / (u*u + (1-u)*(1-u))` | Primary, fixed beta 2 |
| `linear_contrast` | Clip `0.5 + 1.5*(u-0.5)` to the box | Simple contrast comparator |
| `shuffled_delta` | Permute the primary's unit delta within each injected row, then project | Direction control |

For interior real coordinates, the primary equals doubling sigmoid-head logits;
it is **not calibrated classifier confidence**. For example, 0.25 becomes 0.1
and 0.75 becomes 0.9. Endpoints remain bounded. Float32 can round an interior
value to an endpoint, so no universal no-collapse claim is made. The reference
box is the original all-node unlabeled transductive min/max box. Baseline bytes
and zero-width coordinates are retained exactly; no new L2/L0 constraint or
row normalization is imposed. Generated numeric features are not article text
and have no independently established semantic realizability or stealth.

The shuffled control permutes only movable coordinates using fixed SHA-derived
row/case keys. It matches the **pre-projection unit-space L2** displacement of
the actual stored primary delta. Clipping and raw-coordinate ranges can change
its final unit/raw displacement. Both are recorded; terminal/raw doses are not
called matched. The claim concerns practical feature contrast, while linear
equivalence or a stronger linear comparator must be reported without a new
curve-specific novelty claim or control promotion.

## A prospectively frozen new context cohort

Before any new model call, committed selection code loaded original VALID group
IDs, excluded the old 128 groups, and drew 128 without replacement using
`random.Random(20261105).sample(sorted(eligible),128)`. It does not select by
group size, model outputs or labels. Original target-list membership is never
regrouped. The ordered list, input hashes, occurrence count, unique targets and
overlap with old contexts were sealed in the protocol. The actual cohort has
1,102 target occurrences and 357 unique targets per seed: 246 overlap the prior
cohort and 111 are new target identities relative to it. It was sampled from
464 eligible original VALID contexts after excluding the earlier128. All target
lists remain unchanged. These are new contexts for this postprocessor, not
unseen independent news or an untouched validation population; old absolute
scores are not a paired comparison with this cohort.

Use the exact **source-role `legacy` terminal-eight checkpoint family** pinned
by the previous feature-polish base protocol, not the alignment-agent
`legacy_hinge_off` family and not the formal full-test checkpoints. Deep-copy
the old engine protocol and change only VALID IDs. All 256 TRAIN groups, weights,
candidate policy, original score role and case RNG remain unchanged. The helper
restores eight scheduler ticks; actual getters are recorded without inventing
historical equality evidence. Cached `h_GCN` is reused.

Generate one hard graph per seed/context, common to all four conditions. Three
GCN surrogates and three generators are loaded during proposal preparation;
hooks prohibit proposal GCN forwards. All 384 generator calls are counted.
Seal all 1,536 feature/canonical-sequence proposals before loading the official
victim or dedicated evaluation GCN. Tokens 0/1/2/4/5 must remain exactly unchanged;
all physical edges, node/edge budgets, candidate/target order and original node
features stay fixed within each case. Runtime labels verify/record original
fraud-target identities and metrics; they never enter the feature transform.

The registered rule was: if any seed has only identity primary changes, stop after sealing, before
dedicated classifier queries. The 384 generator calls and preparatory model
loads are still spent and retained. The completed comparison passed that gate;
each evaluation classifier received
128 shared clean plus 1,536 attacked forwards: 1,664 original-GCN and 1,664
official-GAGA calls. No fitting, optimizer/autograd, API or TEST call is added.
CPU execution used eight threads and a 5,400-second inner wall limit. Actual
process CPU / wall / outer seconds were 5,639.965564446 / 1,380.318315148 /
1,381.348569339. There were zero GPU hours, updates, autograd, API or TEST calls;
384 generator forwards and 1,664 forwards per evaluation classifier were spent.
Three preparatory GCN loads made zero forwards, alongside three generator loads.

Twelve synthetic checks passed before execution (2.470270328 CPU /
1.608656344 wall seconds). The implementation-author-independent agent audit
passed 711,082 checks / 1,182 byte-hashed files (184.896039025 CPU /
47.592551682 wall seconds), with no model calls. Producer-author saved-JSON
reporting passed 472,249 checks (0.9375 CPU / 1.594 wall seconds); its source was
frozen before current outcome reads, but it is not another independent method
audit. None of these agent checks is human technical verification.

## Fixed interpretation

Primary `logit_sharp` qualifies only with at least +1 pp mean GAGA MR versus its
fresh baseline, positive gains in all three seeds, nominal paired-t95 lower
endpoint above zero, and mean above `shuffled_delta`. The linear comparator need
not be worse: the hypothesis is contrast enhancement, not unique nonlinear-curve
superiority. Controls are not promoted after outcomes. All arms, losses,
recoveries, identity cases, effective feature doses and group-macro summaries
remain reported, with no multiplier/cohort/seed expansion after results.

The df=2 interval is conditional on fixed generators/classifiers and repeated
target identities in an adaptive development history. A positive result would
still need its concrete owner-confirmed full-TEST protocol; it would not establish
generalization, novelty, course approval or completed human work. Dated PDF/PPTX
drafts and the initial delivery snapshot remain unchanged.

## Separate next structural question

A separately frozen saved-geometry derivation passed 4,614 checks on 384 earlier
scenarios (0.109375 CPU / 0.125 wall seconds, zero GPU or model/tensor calls).
Candidate edges account for 86.24% / 92.55% / 58.77% of summed injected-edge
costs across the three seeds. These pooled counts do not show that candidate
edges are wasted or that moving them improves MR.

The [candidate-to-target package](../candidate_target_reallocation) is prepared
but unexecuted. It retains old target edges and injected-node degrees, with
fixed matched donor/count controls. Its primary question is role allocation,
not superiority of balanced priority over random target filling. A third
128-context ID-only sample will exclude both earlier lists, using seed 20261106;
its occurrence/unique-target counts are unknown. Eleven synthetic tests are
source-prepared but unrun. Finish that separate protocol/control freeze before
any calls. This completed feature study, old science, dated PDF/PPTX and initial
delivery snapshot stay unchanged, and automation stays deleted.
