# Frozen author checkpoint + context retrieval: complete TEST benchmark

**The neural downstream exceeds the native author attack and its matched
random control on the complete GossipCop-S split-0 TEST cohort.** Higher GAGA
miss rate (MR) means a stronger attack, not a better defense.

| Arm | Repeat 0 | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|---:|
| Native author MonTi attack | 44.1845% | 44.2606% | 44.3462% | 44.2638% |
| Neural context retrieval | 45.2877% | 45.3447% | 45.4874% | **45.3733%** |
| Matched random, neural activation | 44.5459% | 44.5839% | 44.5649% | 44.5649% |
| Direct independent-teacher search | 45.3352% | 45.3543% | 45.4969% | **45.3955%** |
| Matched random, direct activation | 44.6125% | 44.6410% | 44.5934% | 44.6156% |

| Candidate contrast | Mean gain | Nominal paired 95% interval |
|---|---:|---:|
| Neural minus native | +1.1095 pp | [1.0374, 1.1817] pp |
| Neural minus matched random | +0.8084 pp | [0.5617, 1.0550] pp |
| Direct minus native | +1.1317 pp | [1.0499, 1.2136] pp |
| Direct minus matched random | +0.7798 pp | [0.5136, 1.0461] pp |

Both candidates pass the predeclared owner screen: pooled wrong count exceeds
native and the candidate's own random control. Every repeat is positive for
both contrasts, although the screen did not require that. Direct search is
slightly above the neural network; neural superiority over direct search is
not established. The gain is about one percentage point, not the earlier
4.495-point development gain from a different generator/cohort.

## What the downstream changes

MonTi's released checkpoint and generated fake features stay fixed. A new
dual encoder scores a target-context query against eight TRAIN-benign endpoint
keys and their actual counterfactual relation-feature changes. It can keep the
native action or replace one used candidate identity. An independently fitted
GAGA supplies TRAIN supervision; direct search uses that teacher's loss at
inference. The official victim never supplies selection feedback.

The replacement reaches new graph context outside the targets' original one-
and two-hop neighborhoods. This can change the grouped relation evidence,
where earlier local swaps often left aggregation membership unchanged. This
is an operationally distinct downstream design, not a claim that a larger
network alone helps or that the method is first novelty.

All fake features, target attachments, injected-to-injected edges, each
**injected** degree, candidate-slot count and total structural budget are
preserved. Original endpoint degrees can change. Each random control uses the
same bank, donor, activation and exact replaced-edge dose as its candidate.
The method has broader context access and architecture-matched surrogate
information than original local GCN-only MonTi. Equal node/edge cost does not
mean equal information or computation.

## Cases and weighting

Across 31,545 correlated occurrences, neural creates 357 wrong predictions and
loses seven versus native: **net +350**. Versus matched random it gains 258,
loses three: **net +255**. Group-seed comparisons versus native are 267 wins,
2,454 ties and six losses. This meets the owner's request without requiring
every individual case to improve. Direct is net +357 versus native and +246
versus its random control. These are scenario occurrences, not unique news.

For an explicitly outcome-selected example, repeat-0 group 1847 has 57 targets
and gains six wrong predictions without a loss; group 1907 has 18 targets and
gains four. Complete positive and negative case counts remain in the audit.
These examples were not used to choose a new gate or filter the population.

Post-hoc saved-only weighting checks also retain the direction: neural-minus-
random is +0.9856 pp with equal group weights and +1.1559 pp with equal target
weights after averaging each target's scenario outcomes. These are different
descriptive estimands, not a replacement primary metric or a deduplicated graph.
See [weighting definitions and all rates](weighting_sensitivity.json).

## Scope, execution and evidence

- **Population:** all 909 original TEST groups, 10,515 occurrences and 1,221
  unique fraud targets per repeat; no case filtering or replacement.
- **Training:** reuse the three existing 128-TRAIN-group, eight-epoch downstream
  fits. No full-benchmark retraining. The independent teacher used all 6,595
  original TRAIN nodes. Its reported training quality is in-sample.
- **Repetitions:** one released author checkpoint, three generation-noise and
  downstream-initialization repetitions, not three author model trainings.
  Group memberships overlap. Earlier TEST outcomes from other studies were
  already observed; this is exploratory, not untouched confirmation.
- **Controls:** two candidates were fixed and both reported. Nominal intervals
  describe these three conditional repeats; they are not IID-news uncertainty,
  multi-dataset generalization, course approval or proof of first novelty.
- **Cost:** each repeat's 909 separate scenarios sums 41,264 injected nodes and
  1,835,045 non-self undirected edge additions, equally for every arm. These
  are not one combined attack graph. The run used CPU four threads, nice 10,
  21,649.80 s measured main wall and 52,978.73 s execution-region CPU. Guardian
  wall was 21,685.63 s including preflight. GPU/API use was zero.

All 2,727 author captures and their choices were sealed before official GAGA
loaded. Physical canonical inputs were constructed for all options, then reused
for scoring. There were 20,975 physical inputs, 2,727 batched teacher calls
covering 20,975 hard contexts, 2,727 policy calls and 13,635 official GAGA TEST
calls. No optimizer update occurred in this complete-cohort phase.

- [Complete metrics and counters](summary.json), [frozen protocol](protocol.json),
  [raw trace hashes](raw_trace_hashes.json).
- [Saved audit](audit.json): **157,331 checks passed**, three downstream state
  loads and 2,727 policy replays; zero pretrained-classifier replays. Root-
  authored, not independent implementation or human verification.
- [TRAIN preflight](preflight.json), [scientific execution](execution.json),
  [automatic server audit](offline_audit_execution.json),
  [executed analysis sources](analysis_source_manifest.json).
- [Executable implementation](../../experiments/portal_full/),
  [development method and results](../portal_decoder/).

The server finished computation and its queued audit independently of the
local client. This was a detached QY pipeline, not a Dot cloud handoff.
Report/PPTX drafts predate these results; instructor approval, human review,
actual contributions, defense and final submission freeze remain unconfirmed.
