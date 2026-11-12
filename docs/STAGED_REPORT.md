# MonTi with Staged Role Division

> Draft status, updated 13 November: the text and dated PDF below describe
> the earlier development study. The [completed formal TEST](../results/final_staged/)
> failed the registered improvement criterion. The [author-checkpoint local
> correction](../results/local_swap/) is also null. These result addenda must
> be incorporated before final submission; this draft is not final evidence
> of a successful improvement.

## A reproduction and development study

CISC8006 team: Junyi Tang and Yifan Fan. Prepared 12 November 2026.

**Abstract.** We study a staged extension of MonTi for graph-based fraud
detection. A frozen upstream generator determines auxiliary-node features and
target connections. A separately trained downstream scorer completes the
connections to surrounding candidate nodes. Both stages share one structural
budget. A requested equal split between the two connection roles is projected
to the nearest feasible allocation, preserving every case and its original
total node and edge counts.

On GossipCop-S split 0, the complete staged scheme raises mean GAGA target miss
rate from **53.6486% to 58.1436%**, a **+4.4950 percentage-point** difference.
The three seed gains are positive, with a nominal paired 95% interval of
**[0.5715, 8.4185] pp**. Random candidate completion reaches 57.0344%; the smaller
trained-minus-random difference remains uncertain. The positive result supports
the complete staged scheme relative to its native MonTi baseline in this
validation study.

The report first establishes the attack task and reproduction context, then
explains the role division, budget constraints, protocol and controlled results.
All numerical tables derive from the immutable evidence snapshot
**70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980**. The scientific execution source,
registration, seed outputs and audits are separately identified for traceability.

<!-- pagebreak -->

## 1 Original MonTi and the reproduction context

A graph detector classifies nodes using their features and relationships.
MonTi adds auxiliary nodes, their features and their connections to make a
specified target group harder to detect. In GossipCop-S, nodes are news
articles; fraud targets are fake-news articles grouped using shared social
engagement metadata. A target group is a list of existing fraud nodes, not an
isolated graph or a single new node. Groups can overlap.

MonTi's central idea is to represent targets, surrounding candidates and new
auxiliary nodes together with a Transformer. A learned generator jointly plans
attributes and connections, while adaptive edge allocation lets auxiliary nodes
receive different numbers of edges. Candidate selection bounds the context
used for generation. This report retains those upstream capabilities and adds
a separate completion role. The original formulation is described in
[Choi, Kim and Whang, AAAI 2025](https://arxiv.org/html/2412.18370v3).

Our earlier formal component reproduction compared learned and random
candidate selection on a separate full-test cohort with seeds 0, 1 and 2.

| Component comparison | Learned MR % | Random MR % |
|---|---:|---:|
| Paper Table 5 | 43.7000 | 39.1300 |
| Our frozen reproduction | 43.7280 | 40.1902 |

The paired difference was +3.5378 pp, with nominal 95% interval
[-1.5415, 8.6171] pp: the frozen conclusion remains **inconclusive**. That result
and the later staged study use different cohorts and generator families; their
absolute rates are not paired before-and-after measurements.

The paper's main Table 4 reports 41.34 +/- 1.90% for GCN-to-GAGA, a different
value from its Table 5. The [paper and reproduction comparison](REPRODUCING.md#claim-and-paper-comparison)
records this unresolved difference and the implementation's budget differences.

For the staged experiment, the attack generators are **our three terminal
eight-epoch legacy MonTi checkpoints**, not the authors' supplied best MonTi
checkpoint. The GCN surrogate and GAGA evaluation weights are fixed official
detector checkpoints. Historical scoring semantics and the code's
incidence-based budget implementation are retained and documented.

<!-- pagebreak -->

## 2 The staged improvement

The research question is whether a fixed MonTi planner can support a specialized
completion policy under the same structural budget. The design assigns explicit
permissions to two stages instead of updating the whole attack generator.

| Stage | Decisions | Learning in this experiment |
|---|---|---|
| Upstream | Auxiliary features and fixed target connections | Frozen MonTi parameters |
| Downstream | Auxiliary-to-candidate connections | Train the new scorer only |
| Combined output | One graph with the original total budget | Evaluate the complete scheme |

The upstream first generates a case-specific representation and feature tensor.
Its frozen scores determine the target-connection block under the registered
quota. The downstream sees the auxiliary features and a summary of these actual
target connections before choosing surrounding connections. It cannot change
features, target edges, candidate identities or the total budget.

The downstream is a **151,872-parameter scorer**, not a second complete MonTi.
It projects auxiliary features, the mean features and count of attached targets,
candidate features and cached GCN candidate representations into 64-dimensional
hidden vectors. Their interactions score auxiliary-candidate pairs. A fixed
selection rule converts scores to an exact number of distinct edges.

Role division provides an operational analogy: a planner commits a partial
solution and a specialist completes it. Here the roles are algorithmic
permissions, rather than observed identities of real criminals. The experiment
tests the complete resulting scheme; its main comparison does not separately
identify every component's causal contribution.

Only downstream parameters enter the optimizer. The fixed GCN supplies TRAIN
feedback using the mean softplus of the fraud-minus-benign score. Hard graphs
are binary; a selected-edge straight-through surrogate supplies approximate
gradients during learning. Neither the upstream nor detector weights update.

<!-- pagebreak -->

## 3 Feasible allocation and matched controls

Equal role budgets are not always geometrically possible. With b auxiliary
nodes and t distinct targets, only b*t different auxiliary-to-target pairs
exist. Adding duplicate edges cannot legitimately satisfy a larger quota.

Let c be the actual candidate count and e the original total new undirected
edge budget. The requested target quota is ceil(e/2). Keeping at least one
target edge per auxiliary node gives the feasible interval:

**lower = max(b, e - b*c); upper = min(b*t, e).**

The target quota is the requested value projected into this interval; the
candidate quota is e minus the target quota. An empty interval stops preparation
before training. All cases in this run remained feasible and retained their
complete denominators.

Across 192 validation seed/group cases, **150 required projection**. Actual
target fractions range from **6.3830% to 100%**; the case median is 25.5319%.
Only 24 cases have an exact equal split. This is therefore a closest-feasible
allocation experiment, rather than an exact-half experiment.

| Arm | Target connections | Candidate connections |
|---|---|---|
| Native MonTi | Native output | Native output |
| Quota original | Frozen quota block | Original frozen scores |
| Quota random | Same frozen quota block | Fixed private random selection |
| Staged trained | Same frozen quota block | Trained downstream scores |

The three quota arms share byte-identical auxiliary features and target blocks,
the same candidate lists, quotas and total node/edge budgets. They contain no
auxiliary-to-auxiliary edges; native MonTi retains its native structure. Thus
native-to-staged changes include allocation and edge-role policy. Individual
auxiliary degrees may change even though global structural costs are constant.
The matched quota controls help distinguish these effects from learned scoring.

<!-- pagebreak -->

## 4 Protocol and measurement

The dataset is the official processed GossipCop-S split 0 graph, with news
features of dimension 768. We preserve the original group memberships and
TRAIN/VALID partitions. The 64 validation groups were selected by a fixed
ID-only draw excluding four earlier context lists. They contain 571 target
occurrences and 240 unique targets; 233 identities appeared in earlier contexts.

| Setting | Registered value |
|---|---|
| Generator and downstream seeds | 3, 4, 5 |
| TRAIN groups | 128 |
| VALID groups per seed | 64 |
| Downstream training | Four epochs; final checkpoint only |
| Optimizer | Adam; learning rate 0.001; weight decay 0.0001 |
| Gradient clipping | Norm limit 5 |
| Execution | CPU; four threads |
| Training feedback | Frozen official GCN |
| Terminal evaluation | Frozen official GAGA and GCN |

All 576 upstream TRAIN/VALID captures were saved before downstream fitting.
Final weights and validation topologies were sealed before loading GAGA for
evaluation. There was no GAGA checkpoint selection or performance-based early
stop. Each seed performed 500 updates; 36 no-choice training occurrences were
skipped across the nominal 1,536 opportunities. No cases were dropped from the
registered VALID cohort.

**MR = 100 * incorrect fraud-target occurrences / all target occurrences.**

All target labels are fraud, so an incorrect target prediction is a missed
fraud. Auxiliary and surrounding nodes are not added to this denominator. MR
weights groups by their number of target occurrences. A group of four targets
with one missed target has 25% MR; pre-existing clean misses are also counted.

We compute paired differences across three seeds. The nominal interval is
mean difference plus or minus 4.30265 times its sample standard deviation
divided by sqrt(3). The 1,713 pooled occurrences are correlated observations,
not 1,713 independent replications.

<!-- pagebreak -->

## 5 Primary validation result

The complete staged scheme improves GAGA MR over native MonTi in every seed.

| GAGA arm | Seed 3 MR % | Seed 4 MR % | Seed 5 MR % | Mean MR % |
|---|---:|---:|---:|---:|
| Native MonTi | 52.5394 | 51.4886 | 56.9177 | 53.6486 |
| Quota original | 56.7426 | 56.9177 | 56.9177 | 56.8593 |
| Quota random | 56.7426 | 57.0928 | 57.2680 | 57.0344 |
| Staged trained | 58.6690 | 55.8669 | 59.8949 | **58.1436** |

| Trained minus control | Mean difference pp | Nominal paired 95% interval pp |
|---|---:|---|
| Native MonTi | **+4.4950** | **[0.5715, 8.4185]** |
| Quota original | +1.2843 | [-3.9060, 6.4746] |
| Quota random | +1.1092 | [-3.9892, 6.2075] |

The native comparison has seed gains of **+6.1296, +4.3783 and +2.9772 pp**.
It satisfies the registered screen: at least +1 pp mean gain, all three seed
differences positive, a positive lower nominal interval endpoint, and a mean
above both quota controls. This is positive evidence for the full staged scheme
relative to its native baseline on the registered validation cohort.

The quota controls also improve on native MonTi, showing that allocation and
edge-role changes account for part of the observed gain. The trained arm has
the highest mean, but its extra advantage over random completion is not stable
across seeds: seed 4 is lower by 1.2259 pp. The wide secondary interval does
not establish a reliable win over random completion. The complete scheme's
positive primary result and this more limited attribution are both retained.

<!-- pagebreak -->

## 6 Error transitions and training feedback

To distinguish existing weaknesses from attack effects, we separately count
clean errors, newly missed targets and recoveries. A new miss changes a
clean-correct target to incorrect; a recovery reverses a clean error.

**Attacked wrong = clean wrong + new misses - recoveries.**

| GAGA arm | Clean wrong | Attacked wrong | New misses | Recoveries |
|---|---:|---:|---:|---:|
| Native MonTi | 666 | 919 | 253 | 0 |
| Quota original | 666 | 974 | 308 | 0 |
| Quota random | 666 | 977 | 311 | 0 |
| Staged trained | 666 | 996 | 330 | 0 |

Each row pools the same 1,713 repeated seed/group/node occurrences. The staged
scheme has **77 additional wrong occurrences over native MonTi** and 19 over
random completion. These are net totals, not counts of independent articles.
The unchanged clean denominator prevents a pre-existing miss from being
presented as newly caused by the attack.

| GCN arm | Seed 3 MR % | Seed 4 MR % | Seed 5 MR % | Mean MR % |
|---|---:|---:|---:|---:|
| Native MonTi | 96.8476 | 98.0736 | 96.4974 | 97.1395 |
| Quota original | 97.7233 | 97.8984 | 97.3730 | 97.6649 |
| Quota random | 97.7233 | 97.8984 | 97.3730 | 97.6649 |
| Staged trained | 97.7233 | 97.8984 | 97.3730 | 97.6649 |

GCN rates are near saturation and equal across the three quota arms. Equal
aggregate rates do not imply identical individual predictions. GAGA's separate
architecture supplies the more discriminating transfer evaluation here.
The result demonstrates additional exposure of a detector under the complete
staged intervention; it does not itself provide a defense or establish why
each individual connection changes a decision.

<!-- pagebreak -->

## 7 Verification cost and conclusion

Saved-logit and separately authored saved-tensor audits passed. They recount
predictions, transitions, complete denominators, hashes, node identities,
projected quotas and owned-edge geometry. The latter verifies stored bindings,
rather than independently inspecting every executed DGL graph. Registered
TRAIN checks also confirmed finite downstream gradients, parameter updates and
unchanged upstream/detector states.

Main execution used 576 upstream generations, 1,500 GCN training forwards,
832 GCN validation forwards and 832 GAGA validation forwards. Main wall time was
**3,574.04 seconds**, with **0 GPU hours** and no external API calls. The extra
151,872 parameters and training time are additional computational costs, despite
unchanged node/edge budgets. Two startup failures before any real-data model
query and their 17.84
guardian seconds remain recorded; all-attempt active guardian time was
3,619.56 seconds. Shared-server timing is descriptive.

**Conclusion.** Explicit role division between a frozen MonTi planner and a
trained candidate-edge completer produced a consistent three-seed gain over
native MonTi in this validation study. The +4.4950 pp primary improvement
supports the complete staged scheme at fixed total structural budgets. The
matched controls show that quota and edge-role changes are material, while the
extra learned advantage over random completion remains uncertain.

| Evidence | Immutable source at snapshot 70c48d1 |
|---|---|
| Structured result | [Summary](https://github.com/TangWalden/CISC8006-MonTi/blob/70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980/results/downstream_cooperation/latest.json) |
| Seed and transition counts | [Seed metrics](https://github.com/TangWalden/CISC8006-MonTi/blob/70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980/results/downstream_cooperation/seed_metrics.csv) |
| Per-case allocations | [Quota records](https://github.com/TangWalden/CISC8006-MonTi/blob/70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980/results/downstream_cooperation/quota_by_case.csv) |
| Exact protocol and inputs | [Registration](https://github.com/TangWalden/CISC8006-MonTi/blob/70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980/evidence/downstream_cooperation_v1/execution_protocol.json) |
| Audit attribution and byte bindings | [Provenance](https://github.com/TangWalden/CISC8006-MonTi/blob/70c48d1a74aa4e5b3e1b1c6ce15c068ba8931980/results/downstream_cooperation/provenance.json) |

Scientific execution source: 9e013cb182b5af71a9d0b719b5a939deda3c88e2.
Protocol SHA256: 5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13.
Additional studies and historical evidence are indexed in
[Supplementary studies](SUPPLEMENTARY_STUDIES.md).

This is a three-seed development result under an adaptively developed setup,
with substantial target-identity reuse and no final TEST evaluation of the
staged scheme. A full fair extension against the frozen reproduction baseline
within the instructor-approved level remains unconfirmed. Human technical
review, instructor approval and actual member contributions are not established
by automated evidence; planned duties remain in the [contribution record](../CONTRIBUTIONS.md).
Codex assisted implementation, experiment supervision, saved-record audits and
writing; the [AI-use record](../AI_USAGE.md) documents those roles.
