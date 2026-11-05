# Fixed feature-logit sharpening on new VALID contexts

**Status: implementation and synthetic-test source prepared; no new cohort
metadata, generation or efficacy calls have run.** This is a separate exploratory
study after the completed negative [prototype-mixing phase](../train_prototype_mix/README.md).
That earlier phase remains fixed at ratio 0.5 and failed; it is not expanded
with extra ratios, seeds or a favorable promoted control.

The owner's manual `continue` instruction authorizes bounded development work.
Automation monti remains deleted. Final TEST, final freeze/tag, human technical
review and course acceptance remain separate.

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

Before any new model call, committed selection code loads original VALID group
IDs, excludes the old 128 groups, and draws 128 without replacement using
`random.Random(20261105).sample(sorted(eligible),128)`. It does not select by
group size, model outputs or labels. Original target-list membership is never
regrouped. The ordered list, input hashes, occurrence count, unique targets and
overlap with old contexts are sealed in the protocol. New counts are unknown
until this permitted metadata extraction; old 1,091/374/budget totals must not
be copied. These are new contexts for this postprocessor, not unseen independent
news or an untouched validation population.

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

If any seed has only identity primary changes, stop after sealing, before
dedicated classifier queries. The 384 generator calls and preparatory model
loads are still spent and retained. Otherwise, each evaluation classifier gets
128 shared clean plus 1,536 attacked forwards: 1,664 original-GCN and 1,664
official-GAGA calls. No fitting, optimizer/autograd, API or TEST call is added.
CPU eight threads, 5,400 inner seconds, zero GPU hours; retain all partial files,
actual counts, native failures and elapsed time. No automatic scientific retry.

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
