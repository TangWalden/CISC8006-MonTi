# Fixed TRAIN-prototype feature mixing

**Status: complete and negative. The fixed benign mixture fails its frozen
development criterion in all three seeds. No candidate qualifies; the full-TEST
gate remains closed and stable effect is not achieved.** The owner manually
continued after the initial delivery; automation remains deleted. Source
`edb703e80c7378aa0f425cf047a03c962774e3bf`, protocol SHA256
`0f7f6510552b7c1a51ea001dd000868712f682b74b46cb0e3054c72c4ae62c27`
completed at 2026-11-05 07:52:49 UTC. Existing delivery snapshot `0c5e227`, dated
artifacts and final TEST/tag/human-course boundaries remain separate.

## Evidence motivating this isolated contrast

The previous [target-attachment experiment](../target_degree_balance/README.md)
raised mean official-GAGA development MR from 52.642835% to 54.017721%, but one
seed declined and matched random was higher. Its original criterion failed.
A new saved-only endpoint diagnosis retains all 3,273 correlated occurrences:

- 61 of the 64 gained successes already had an injected edge. Newly covered
  occurrences contribute only three gains.
- Eight of the 19 lost successes have increased injected incidence, ten have
  reduced incidence, and one is unchanged. Degree alone is not a sufficient
  outcome rule.
- Balancing and random share 55 gains and 12 losses. The observed VALID margins
  are descriptive, uncalibrated outputs; they do not select or train a gate.

The [aggregate diagnosis](../../evidence/train_prototype_mix_v1/prior_endpoint_diagnosis.json)
was computed once from 13 retained JSON inputs, with 8,499 data checks and no
tensor/dataset/model/API/GPU operations. All data checks and files completed,
then final stdout formatting failed because `flush=True` was passed to
`json.dumps`. The original exit 1, source and completed output hashes remain
retained; the analysis was not rerun or relabeled as a successful process.
Its measured analysis CPU/wall time is 0.28125 / 0.407 seconds; whole-process
CPU time after finalization is unmeasured.

The new method keeps the original topology and tests a simple TRAIN data prior,
after learned-teacher gradients and global degree balancing failed to establish
stable transfer. It makes no novelty or efficacy guarantee.

## One fixed method and two noncandidate controls

Let `x0` be each saved baseline injected feature row. Compute three raw-coordinate
means once over **distinct original TRAIN nodes**: benign class 0, all TRAIN,
and fraud class 1. Accumulate in float64, then store float32. No VALID or TEST
label selects these pools; their partition IDs are used only for disjointness
checks. The unchanged standard loader still contains whole-graph labels and
class-count metadata.

| Arm | Injected raw features | Role |
|---|---|---|
| `baseline` | `x0`, exact original bytes | Common fresh baseline |
| `benign_mix` | `0.5*x0 + 0.5*mean(TRAIN class0)` | Primary method |
| `all_train_mix` | `0.5*x0 + 0.5*mean(all TRAIN)` | Generic shrinkage control |
| `fraud_mix` | `0.5*x0 + 0.5*mean(TRAIN class1)` | Label-direction control |

The mixture ratio is fixed before reference extraction and is not scanned.
All original edges, target/candidate membership and order, edge roles and actual
physical budgets remain identical. Only injected raw feature values change.
The legal coordinate box remains the original all-node **unlabeled transductive**
min/max box. Numerical projection and exact zero-width coordinates are retained;
there is no invented row-unit-norm, small-L2 or continuous-L0 restriction.

Injected nodes remain in GAGA's unknown group even when a reference came from a
benign TRAIN node. At fixed topology, SELF/known-group tokens 0/1/2/4/5 must be
bit-identical across arms; only unknown tokens 3/6 can vary. Numeric means are
not generated articles. No audited raw-article mapping or independent semantic
stealth/realizability claim is available. Prototype norms and generic shrinkage
remain possible explanations; the controls do not match every geometric aspect.

## Registered development scope (completed)

Reuse all 384 original CPU baseline captures from seeds 3/4/5, each with 128
fixed VALID groups, 1,091 target occurrences and 374 unique fraud targets. These
are the original unbalanced hard graphs, not the selected outputs of the later
balancing intervention. Identity changes remain in every denominator. Each arm
and seed's 128 separate scenarios sum 7,776 injected nodes and 358,491 actual
nonself undirected edges; they are not one combined graph.

Source/input hashes and a synthetic-test receipt must be frozen before means
are extracted. Seal the reference and all 1,536 feature/canonical-sequence
variants before loading either classifier. Construct 384 base graphs per stage
and 1,536 local feature graph views per stage, using new full feature tensors
to avoid in-place aliasing. Use canonical inputs directly, not the affine
feature shortcut.

Fresh evaluation is capped at 1,664 original-GCN and 1,664 official-GAGA forwards
(128 clean plus 1,536 attacked each), CPU eight threads, 5,400 inner seconds.
There are zero new generator/teacher fits or forwards, optimizer/autograd/API/
TEST calls and zero GPU hours. Failed partial records and outer elapsed time
must be retained; there is no automatic scientific retry or outcome-based
ratio/seed/cohort expansion.

## Fixed interpretation and gate

The primary `benign_mix` qualifies only if official-GAGA occurrence MR has mean
gain at least one percentage point over fresh baseline, all three seed gains
are positive, its mean exceeds **both** reference controls, and the nominal
paired t95 lower endpoint exceeds zero. Use the analytic df=2 critical value.
Controls are not promoted after outcomes. All arms, raw transitions, negative
cases, group-macro summaries and actual costs remain reported.

The three-seed interval is conditional on fixed classifiers and repeatedly
inspected development data; 3,273 repeated occurrences are not IID news. Even a
passing development result does not demonstrate generalization, first novelty,
course acceptance or authorize final TEST. Original formal reproduction and the
dated PDF/PPTX drafts remain unchanged.

## Actual results: no useful class-conditioned effect established

All 384 baseline group-seed cases remained in every arm; each arm/seed retains
128 groups / 1,091 occurrences / 374 unique targets. Mean official-GAGA MR and
all three seeds are:

| Arm | Seed 3 MR (%) | Seed 4 MR (%) | Seed 5 MR (%) | Mean MR (%) |
|---|---:|---:|---:|---:|
| Baseline | 51.420715 | 50.412466 | 56.095325 | 52.642835 |
| Benign mixture (primary) | 46.471127 | 45.187901 | 50.779102 | 47.479377 |
| All-TRAIN mixture (control) | 46.471127 | 45.187901 | 50.779102 | 47.479377 |
| Fraud mixture (control) | 46.471127 | 45.096242 | 50.779102 | 47.448824 |

Higher MR means a stronger attack. Primary seed changes are -4.949588 /
-5.224565 / -5.316224 pp; mean -5.163459 pp, nominal paired t95
[-5.637441, -4.689476] pp (`n=3`, `df=2`). The interval is conditional on these
fixed classifiers/saved attacks and repeatedly inspected development contexts;
overlapping occurrences are not IID news or fresh-teacher uncertainty.
The primary gains one new error and loses 170 existing successes: one group-
seed win, 271 ties, 112 losses. The criterion fails and no control is promoted.

The all-TRAIN control and benign primary have identical **GAGA predictions at
all 3,273 paired occurrences**, with zero pairwise gains/losses. This is stronger
than equal aggregate MR, but does not imply identical feature vectors or GCN
predictions: the GCN predictions do differ. Benign-label selection has no
observed GAGA advantage over generic all-TRAIN mixing here. This does not prove
a universal causal explanation for every possible mixing rule; the fixed
half-mixture remains a completed negative study and is not retuned.

## Reference, execution and audit evidence

The actual unique TRAIN population is 6,595 nodes: 5,036 benign and 1,559 fraud.
All three independently recomputed means match every stored float32 bit, with
maximum 0 ULP error under the predeclared at-most-1-ULP reduction-order bound.
This exact observed result is separate from the allowed bound. Whole-graph
labels/class counts are materialized by the unchanged producer loader; DGL
materialization also exposes global `y` to the audit before it indexes only
TRAIN truth and discards global labels. No held-out label selects a pool and
no TEST prediction is read. TRAIN-only selection does not mean the entire
loader never materializes held-out metadata.

Twelve committed CPU Torch synthetic checks passed before execution in
3.395333711 CPU / 2.437720392 wall seconds. Actual execution used 384 base graph
reconstructions and 1,536 local feature views in each of geometry/evaluation
stages, 1,536 canonical updates, and 1,664 fresh forwards each for original GCN
and official GAGA (128 clean plus 1,536 attacked). It used zero new generator/
teacher forwards, training updates, autograd, API, TEST or GPU work. All means
and proposed canonical inputs were sealed before classifier construction.
CPU/wall/outer seconds were 5,115.328740816 / 1,320.5656187059358 /
1,321.322383858962.

The separate method-author-independent agent audit passed 138,606 checks and
byte-hashed 1,185 actual files in 96.779026648 CPU / 30.062361425952986 wall
seconds. It inspected one original DGL graph, three partition-ID arrays and
769 reference/capture/geometry tensors, with no classifier weight load, model
construction or inference. It independently verifies actual TRAIN pools/means,
stored feature/token/edge invariants and raw predictions. Original-original
arcs, full canonical input construction and seal-before-model timing retain
producer/source boundaries. This is not human verification.

Producer-author saved-JSON reporting separately passed 50,593 checks / all
384 count-only cases, 0.21875 CPU / 0.547 wall seconds, using source SHA256
`1997fcd97368d72b197b497dfb29376c553a93f915bf323f0ce76f3e5d477b42`.
It loaded no tensors/datasets/models and is not another independent method
audit. The [public evidence directory](../../evidence/train_prototype_mix_v1)
retains raw result, reference checks, all-case summary, audits and execution
trace. Cumulative GPU research charge stays 5.065219865793663 hours plus the
separate 0.02 reserve. Human review, actual contributions, course approval,
rehearsal and full TEST/final tag remain incomplete; dated PDF/PPTX are unchanged.

## Next isolated hypothesis: prepared, not executed

Root prepared [unit-coordinate feature sharpening](../feature_logit_sharpen)
with fixed beta 2 and four arms: baseline, logit sharpening, linear contrast
1.5 and shuffled delta. Shuffling matches the pre-clip unit-coordinate delta
L2; terminal post-clipping L2 is not promised equal. It uses no prototype mean,
class-conditioned text or new learned model fit. This is a new prospective
128-context ID-only original-VALID sample outside the prior 128, selected
before calls, rather than another ratio for the failed v1 mixture. Target
occurrence/unique-node counts and new outcomes remain unknown, and different
group IDs do not imply IID news or a held-out TEST population.

The planned checkpoint family is the frozen legacy source-role baseline /
feature-refinement family, not the earlier alignment-policy models. Planned
limits are 384 generator forwards and 1,664 each GCN/GAGA forwards, CPU eight
threads / 5,400 seconds / zero GPU. Before sealing there are three generator
loads and three preparatory GCN loads with zero GCN forwards; the official
victim loads only after all inputs are sealed. These are expected counts, not
completed calls. No gain, new-cohort numbers, automation recreation, full TEST
or final tag is claimed by this preparation.
