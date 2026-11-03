# Independent relationship-surrogate development study

**Status: all twelve terminal trainings, gated development evaluation and saved-
record agent audits completed. The proposed joint GCN/GAGA method produced a
negative result: its GAGA attack MR was lower than GCN-only in all three seeds.**
No improvement is selected. This is an owner-authorized development study;
final testing, human review and course completion remain separate.

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

`matrix_attempt3` completed eleven terminal trainings. Its final `gcn_gaga`
seed-5 attempt hit the 480-second individual timer after 2,012 optimizer updates
and 2,013 completed generations. Its GAGA dispatch count includes interrupted
sequence preparation and is not a completed-model-forward count. The preserved
timeout occurred before any official-victim evaluation. A registered timer-only
continuation raised that job's individual ceiling to 900 seconds, retained the
eleven completed models and passed preflight, and reran only the incomplete job.
Scientific settings and the aggregate two-GPU-hour ceiling were unchanged.

The primary twelve models comprise eleven trainings from
`14e2471a6791b90afcd3a395bab1ce72b9a27f19` and the completed retry from
`255b897714b3526eb56205dcdeac1101c7f1ce00`. All twelve completed before the
separate evaluation gate loaded the official GAGA victim. The final execution
protocol SHA256 is
`d816e1e1aaf409b414afe4a26beafba71d3c07e4f4eccc58e6258daf6cd52406`.

## Completed development results — negative

The primary outcome is occurrence-weighted GAGA misclassification rate (MR),
where higher means a stronger attack on these fraud targets. Each arm/seed
retains the same 128 original-VALID groups, 1,091 target occurrences and 374
unique target nodes. Overlapping occurrences are not independent news samples.

| Arm | Seed 3 MR (%) | Seed 4 MR (%) | Seed 5 MR (%) | Mean MR (%) | Sample SD (pp) |
|---|---:|---:|---:|---:|---:|
| GCN-only | 51.4207 | 48.5793 | 54.0788 | 51.3596 | 2.7503 |
| GAGA-only | 47.1127 | 45.0046 | 46.1962 | 46.1045 | 1.0571 |
| Two GCNs | 49.9542 | 50.1375 | 43.5380 | 47.8766 | 3.7584 |
| Joint GCN/GAGA | 35.5637 | 40.4216 | 40.7883 | 38.9245 | 2.9163 |

Joint minus GCN-only is **−12.4351 percentage points**, with all three paired
differences negative. Its paired 95% t interval is **[−22.1737, −2.6965] pp**
(`n=3`, `df=2`), conditional on the fixed teachers and development cohort. It
does not quantify uncertainty across new teacher fits, datasets or independent
news samples. The other comparisons and their intervals remain in the full
[saved aggregate](../../evidence/dual_surrogate_v1/saved_aggregate.json).
No architecture benefit, novelty or effective improvement is established.

Clean GAGA MR on this cohort is 398/1,091 = **36.4803%**. Attacked MR includes
pre-existing errors and recoveries: joint seed 3 creates 25 new errors but
recovers 35 previously wrong occurrences, leaving 388 wrong. MR is therefore
not a count of new attack-induced flips. Every scenario set retains summed
budgets of 7,776 injected nodes and 358,491 edges across the 128 independent
group attacks; these are not one jointly modified global graph.

The twelve successful trainings performed **24,576 optimizer updates and
36,864 teacher forwards**. Gated evaluation performed **1,536 generations,
1,664 original-GCN forwards and 1,664 official-GAGA forwards**, including clean
evaluation. These successful-matrix counts exclude the extra 2,012 updates in
the preserved timeout attempt. Teacher preparation remains 736 forwards / 560
updates. Unknown partial counters from the native abort remain unknown; an
exact all-attempt forward total is not claimed.

The complete conservative process/guardian charge is **1.333470031947719
GPU-hours**, within the unchanged two-hour ceiling. It includes teacher
preparation, preflights, failed/native-aborted and interrupted attempts, the
successful trainings and evaluation. No new API calls or test queries occurred.
The [execution trace](../../evidence/dual_surrogate_v1/execution_trace.json),
[results](../../evidence/dual_surrogate_v1/results.json),
[teacher audit](../../evidence/dual_surrogate_v1/independent_teacher_audit.json)
and [phase audit](../../evidence/dual_surrogate_v1/independent_phase_audit.json)
preserve sources, counters and limitations. Both audits passed saved-record
checks without new inference; they are agent checks, not human verification.

The [post-hoc saved-log diagnosis](../../evidence/dual_surrogate_v1/failure_diagnosis/summary.json)
reads no models and makes no new queries. In the joint arm, occurrence-weighted
online TRAIN GAGA wrong rate rises from 46.4077% in epoch one to 77.5974% in
epoch eight, while mean Softplus falls from 1.517893 to 0.758922. All three
seeds improve these own-teacher metrics. This argues against complete failure
to optimize that training teacher, but does not identify overfitting, model
mismatch, objective interaction or gradient-approximation bias as the cause.
These are changing-model online TRAIN observations over 3,141 occurrences per
epoch, not frozen-checkpoint predictions or a paired gap to VALID.

An outcome-selected illustration is seed-3 group 938: nine matched targets,
122 injected nodes and 5,734 edges under either method. Clean GAGA has four
wrong occurrences; GCN-only leaves five (one new error), whereas joint leaves
three (one recovery and no new errors). The rule chose the smallest group ID
with fewer joint errors after matching all 128 cases; it is not a representative
population estimate. The complete diagnosis preserves all 96 epoch cohorts and
its source-before-TRAIN-read receipt, explicitly after overall results were known.
It used 0.3445 seconds of diagnostic wall time, zero tensor loads/model calls/
API calls/GPU work. The executed summary mislabeled its monotonic elapsed timer
as `CPU_seconds`; the original source/output are preserved with a
[timing-label correction](../../evidence/dual_surrogate_v1/failure_diagnosis/timing_label_correction.json).
Process CPU time was not measured. The training teacher and validation victim
also differ in weights and FIT80 versus full-TRAIN group-label conventions;
these differences are not independently isolated as causes.

No further tuning, seeds, API or GPU work is planned for this phase. These
development results do not replace the earlier 43.728% formal test reproduction
or its inconclusive learned-versus-random paired interval. No final test or final
submission tag is authorized by this protocol. Existing report/slides remain
dated drafts; actual contributions, human review, rehearsal and instructor
approval remain separate.
