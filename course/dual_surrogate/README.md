# Independent relationship-surrogate development study

**Status: teacher preparation and real-TRAIN preflight passed; the twelve-job
training matrix is running. Attack efficacy, sealed-victim evaluation and all
method comparisons remain pending.** This records an owner-authorized development
study, not a completed improvement, final test or course approval.

The question is whether feedback from a fresh GAGA-style relationship classifier
helps the same MonTi generator transfer to the sealed official GAGA victim.
Access to an additional architecture changes the original GCN-only adversary
knowledge. A gain from a stronger single teacher must be separated from a gain
from combining teachers.

## Frozen comparison

| Arm | Attack-loss teachers | Fixed weights |
|---|---|---:|
| `gcn_only` | Original GCN surrogate | 1 |
| `gaga_only` | Fresh independently trained GAGA | 1 |
| `two_gcn` | Original GCN and fresh independently trained GCN2 | 0.5 / 0.5 |
| `gcn_gaga` | Original GCN and fresh GAGA | 0.5 / 0.5 |

All arms use attack seeds 3, 4 and 5, eight epochs, one group per optimizer step
and the terminal epoch-eight checkpoint. Each completed job is required to retain
all 256 TRAIN groups and perform 2,048 optimizer updates. There is no validation
checkpoint search during training. The loss is the group mean of
`Softplus(logit_fraud - logit_benign, beta=1)`, with fixed equal mixtures of raw
teacher losses in the two-teacher arms. Teacher logit scales are not adjusted
using observed outcomes. Equal epoch counts do not imply equal compute costs.

The MonTi architecture, original `h_GCN` generator-input cache, candidate policy,
role encodings, existing edge-score source indexing, hard top-k convention,
incidence-based node/edge budgets, optimizer and weight decay are shared. Thus
`gaga_only` is not an exact reproduction of the paper's GAGA-surrogate setting:
its generator input still uses `h_GCN`. Per-case RNG keys are shared across arms;
bitwise device/kernel determinism is not claimed. There is no alignment penalty,
LLM controller or new API call in this phase.

See the [master protocol](../DUAL_SURROGATE_PROTOCOL_V1.json),
[training engine](experiment.py), [queue](queue.py) and
[provenance checks](verify.py). Actual runs require separately frozen execution
protocols with source/input hashes and fresh exclusive output directories.

## Teachers and label boundaries

The original TRAIN nodes are split by a fixed stratified hash rule into FIT80
and internal TUNE20. Only FIT labels enter the fresh GAGA's group masks. TUNE,
original VALID and test nodes remain unknown in every group input, including
during subsequent attacker training. Original graph features/topology are
transductive inputs; labels outside original TRAIN are not used to fit or select
either new teacher.

GAGA uses the original three-layer, 64-dimensional classifier architecture.
GCN2 uses the original two-layer 64-dimensional GCN with one full-graph FIT-loss
update per epoch and a separate internal-TUNE evaluation. That full-graph
normalization is also retained for GCN2 attack feedback; the original GCN keeps
its original sampled-block path. The teachers use fixed class-weighted cross
entropy and Adam settings, at most 80 epochs and patience ten. The earliest
maximum internal macro-F1 checkpoint is selected.

| Fresh teacher | Selected / completed epoch | Internal TUNE macro-F1 | Internal TUNE AUC |
|---|---:|---:|---:|
| GAGA | 38 / 48 | 0.884669 | 0.953011 |
| GCN2 | 22 / 32 | 0.691885 | 0.753863 |

Both passed the frozen quality gates on the 1,320-node internal TUNE cohort.
Preparation used 736 teacher forwards and 560 optimizer updates. These selected
internal scores establish readiness, not calibration, equal teacher strength or
attack-transfer improvement. The completed teachers are reused across the
engineering continuations and have not been retrained.

The [teacher preparer](prepare_teachers.py) writes separate private FIT-only
seven-token sequence, feature-sum, degree and group-mask caches. It never reuses
the original full-TRAIN-labelled sequence/sum/degree caches as FIT inputs.
Immutable original two-hop topology may be reused. Raw checkpoints, node IDs,
labels and caches remain in ignored storage; normalized initial/selected content
hashes, byte hashes, partition hashes and training receipts record provenance.

## Sequence and gradient validity

The [sequence adapter](sequence_adapter.py) takes authoritative masks from its
explicit data view. Fresh-teacher FIT masks and original full-TRAIN victim masks
are separate views. Injected nodes are always unknown. For binary generated
graphs, the hard forward follows the canonical updater for the same caches and
masks, retaining masked-self tokens and deduplicating two-hop reachability.

One-hop aggregation differentiates weighted sums and degrees. Two-hop topology
is discrete: its backward uses an explicit straight-through approximation with
`rho(P) = P / (1 + P)` on local path scores. It is not the exact derivative of
discrete reachability. Nonzero gradients show connectivity to an optimization
path, not a guarantee of useful updates or improved attacks.

Twenty-one CPU synthetic known-answer checks passed. Real-TRAIN
`preflight_attempt3`, from source
`14e2471a6791b90afcd3a395bab1ce72b9a27f19`, then passed four fixed TRAIN cases:

- Four attack generations, sixteen teacher forwards and eight autograd calls.
- Zero optimizer updates and zero official-victim loads/forwards.
- Maximum reported sequence discrepancy `6.5565e-7` and logit discrepancy
  `4.7684e-7`, within the frozen tolerances.
- Generated-feature/edge paths were checked; both actual MonTi output-head
  gradients were nonzero in all four cases. Frozen teacher states were retained.

This is bounded functional evidence. It does not establish equivalence on every
future graph, an exact topology gradient or attack efficacy.

## Preserved build and resource failures

The first preflight failed on integral floating-point TRAIN indices used for
PyTorch indexing, before attack generations, inference or updates. The
continuation validates integral values and supplies the required index dtype.

The second preflight exited with native status `-6` and a CUDA allocation failure
in DGL while another user's workload occupied approximately 22 GiB of GPU0.
There was no Python result or guardian receipt, so its exact model/generation
counters are unknown and are not reported as zero. Its measured outer lifetime,
23.151359 seconds, remains charged and its native log is preserved.

The explicitly registered continuation moved to GPU1, another RTX 3090, using
numeric `CUDA_VISIBLE_DEVICES=1`. Its visible logical device zero is checked
against the own-process physical UUID
`GPU-1bb984a3-e0f6-138a-cfb7-dc3365d8f9cf`, with both visibility counts at one
device and an 8 GiB free-memory gate.
No other user's job was changed. The existing teachers were retained; GPU
migration is not a claim of bitwise equivalence. These are infrastructure/build
continuations, not outcome-driven changes to the scientific comparison.

## Evaluation and results — pending

`matrix_attempt3` completed eleven terminal trainings; its last gcn_gaga seed5 hit the individual480-second timer after2012updates. A timer-only continuation will retain all eleven completed models and the passed preflight, then rerun only the incomplete job with a900-second individual ceiling. The aggregate two-GPU-hour ceiling, eight epochs, seeds, cohorts and all scientific source remain fixed. No original-victim validation was performed before this amendment. All twelve terminal trainings must
complete before the separate evaluation gate can deserialize the official GAGA
victim. Training cannot query that victim. The planned primary result is its
occurrence-weighted misclassification rate on the full registered 128-group
original-VALID cohort, with all 1,091 target occurrences retained. Matched clean
and attacked transitions will distinguish new errors from pre-existing errors
and recoveries. Overlapping occurrences are not independent news samples.

No attack MR, improvement, method ranking or superiority claim is available at
this checkpoint. The eventual record must include every arm/seed, paired
contrasts against GCN-only, GAGA-only and two-GCN, and the uncertainty from only
three attack seeds. A three-seed paired interval has two degrees of freedom.
Negative, mixed, incomplete or inconclusive outcomes will remain visible.

The complete phase is capped at two aggregate GPU-hours, including teacher
preparation, preflights, failures, all training and gated evaluation. Conservative
process/guardian charges and actual counters determine the final total, which
is still pending. No final test or final submission tag is authorized by this
development protocol. Existing report/slides remain dated drafts. Agent checks
and receipt consistency checks are not human verification, and genuine human
contributions, review, rehearsal and instructor approval remain separate.
