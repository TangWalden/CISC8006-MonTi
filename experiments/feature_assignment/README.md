# Fixed-graph feature assignment

**11 November 2026: implemented and evaluated; no measured benefit.** All four
arms have identical official-GAGA classifications across 3,213 correlated target
occurrences. The primary is not a qualified improvement.

| Condition | Seed 3 MR (%) | Seed 4 | Seed 5 | Mean |
|---|---:|---:|---:|---:|
| Random-fill identity baseline | 55.0887 | 54.7152 | 55.3688 | 55.0576 |
| Relation assignment (primary) | 55.0887 | 54.7152 | 55.3688 | 55.0576 |
| GCN proposal, common filter | 55.0887 | 54.7152 | 55.3688 | 55.0576 |
| Cycle-matched random, common filter | 55.0887 | 54.7152 | 55.3688 | 55.0576 |

Primary minus baseline and both controls is exactly zero in every seed. The
degenerate conditional t95 `[0,0]` describes these three fixed observations;
it is not population equivalence. The primary is accepted in 103/89/84 cases
(276/384), reducing its own-GAGA loss but changing no own-GAGA class or official
class. There are no favorable or unfavorable classification-change examples
to select. This does not identify a sole causal explanation.
[Same-input saved diagnosis and unchanged example](../../results/feature_assignment/diagnosis.md).

The complete comparison uses 128 groups, 1,071 occurrences and 323 unique
targets per seed; 175 identities repeat across groups, at most 32 memberships.
Each seed's 128 separate scenarios sum 7,867 injected nodes and 362,999 nonself
undirected edges in every arm. These are not one combined graph.
[Compact results and source provenance](../../results/feature_assignment/).

## Executed scope

Scientific source `bc31f324ba1e2b1ab541b719695cfa51b9e49155`, historical public
runtime `ca093d8b12ea153a076b02bbdbc4ede58508560c`, protocol SHA256
`b15527fa882fcdb808a303cf5e606239f97abffb95f0c9aea64981a1a9d2d679`.
The fixed preflight passed four TRAIN cases. Main execution made 384 generator
forwards, 768 feature-gradient calls, 1,324 physical forwards per own classifier
and 1,664 official-GAGA forwards. All 1,536 final inputs sealed before official
victim loading. Main wall time was 1,748.86 seconds / process CPU 5,624.25
seconds, CPU threading settings 8, zero GPU/API/TEST or parameter updates.

A separate author's saved-tensor audit passes 190,857 checks and 819 file hashes;
it does not replay neural models or constitute human verification. Its source
was frozen before primary VALID execution, after TRAIN preflight dispatch. An
original freeze status overstated that chronology; the original receipt and
explicit correction remain private. No scientific source/output was changed.

This development study tests a post-rewiring compatibility repair. MonTi first
generates features and edges; the existing random target-fill rule then changes
some edges while leaving those features in their original positions. Here the
filled graph stays fixed, and a one-to-one assignment changes which injected
position receives each already generated feature vector.

## Mechanism and controls

The primary uses input-feature gradients from a separately trained, frozen GAGA
surrogate to form a linear assignment cost. A Hungarian solver proposes a
permutation minimizing that first-order cost. This is a linearized objective,
not a globally optimal neural-network attack. It changes no feature value,
edge, injected degree, target list or physical attack budget.

The raw proposal falls back to identity unless its linear improvement exceeds
the recorded conservative FP64 roundoff bound. The random control matches this
post-certification raw permutation before the common nonlinear filter, not the
unfiltered solver return or the final accepted movement.

All three proposals use the same acceptance filter: strictly lower the own-GAGA
mean Softplus loss by more than 1e-7 and retain every baseline class-0 prediction
of the original GCN. Otherwise the entire case returns to the original features.
Neither official-GAGA predictions nor VALID truth select or accept a proposal.

| Arm | Feature placement on the identical random-fill graph |
|---|---|
| Identity | Keep original positions; strong random-fill baseline |
| Relation assignment (primary) | Own-GAGA gradient assignment, common filter |
| GCN proposal | GCN gradient assignment, common GAGA/GCN filter |
| Matched random | Randomly relabel the primary proposal's permutation cycles, common filter |

Cycle matching fixes moved-index counts and cycle lengths before acceptance;
it does not equalize actual feature displacement or accepted changes. The GCN
proposal is not a GCN-only attack: it shares the additional GAGA filter.

## Fair validation

Before model calls, freeze an ID-only draw of 128 original VALID groups from
the 208 contexts outside the preceding three disjoint lists. Keep all original
members and all three frozen generator seeds (3/4/5). These are overlapping,
adaptively studied development data, not unseen news or independent samples.

Seal every arm's final features and canonical inputs before loading the official
GAGA detector. Retain rejected, identical and one-injection cases in every
denominator. Report actual extra surrogate calls, gradients, CPU/GPU time and
the complete seed/case results. No training, external LLM API or final TEST is
part of this study.

The primary must gain at least one mean MR percentage point over its fresh
identity baseline, improve in all three seeds, have a positive nominal paired
t95 lower endpoint, and exceed both controls' mean MR. Controls cannot replace
a failed primary. The interval is conditional on fixed teachers and repeated
development news identities.

## Scope of the contribution

The project contribution under test is an explicit **fixed-topology,
fixed-feature-multiset assignment constraint** after rewiring. Transformer
models and joint structure/feature optimization already exist; no first-novelty
claim is made. See related [AGIA/HAO](https://arxiv.org/html/2202.08057v2),
[CANA](https://arxiv.org/html/2208.01819v4) and
[GraphTextack](https://arxiv.org/html/2511.12423v1).

Using an independent architecture-matched GAGA surrogate changes the original
GCN-only attacker knowledge. Previous own-GAGA feature optimization transferred
poorly; the filter protects only the own GCN. Preserving the feature multiset
does not prove local camouflage, article realizability or useful transfer.
This completed study finds no added classification benefit in its registered
cohort; broader effectiveness and originality are not established.

## Reproduction boundary

The compact result arithmetic is independently runnable with
`python -m tools.feature_assignment_results check --results results/feature_assignment`.
The numerical assignment tests require the recorded NumPy/SciPy environment.
Full ML execution requires the immutable scientific source above, the public
historical runtime, frozen data/caches and the three existing terminal generator
checkpoints. It also requires the source manifest and passed numeric-check
receipt consumed by `prepare.py`; these are not a claim of a one-click public
model rerun. The two `driver.py` phases are explicit, separately guarded
`preflight` and `run` invocations. Current curated main must not silently replace
the historical runtime. Raw assets, protocol inventories and attempts remain
private; their recorded hashes are retained.
