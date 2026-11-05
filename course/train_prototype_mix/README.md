# Fixed TRAIN-prototype feature mixing

**Status: implementation prepared; no new real reference statistics or efficacy
results have been extracted.** The owner's new `continue` instruction on
5 November authorizes another bounded development comparison. The existing
automation was already deleted at the initial delivery boundary and is not
recreated. Final TEST, submission freeze/tag and actual human/course checks
remain separate.

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

## Frozen development scope

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
