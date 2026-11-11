# Fixed-graph feature assignment

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
Measured effectiveness remains an experimental question.
