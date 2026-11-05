# Actual injected-target degree balancing

**Status: the complete three-condition, three-seed CPU study and separate agent audit passed execution/integrity checks, but the frozen development criterion failed. No candidate or control winner is selected; stable effect is not achieved.** This is a post-hoc practical structural study, not a novelty or validated-efficacy claim. The preceding [feature-refinement study](../feature_polish/README.md) was negative; improving a mathematical degree objective does not promise improved attacks.

The method keeps all baseline injected features `x0`, complete target lists, candidate selection and original-original graph edges fixed. It postprocesses actual injected-to-target edges after generation. It is not the earlier candidate-neighbor coverage pilot, group/list regrouping, a new agent, feature alignment or a new teacher fit. The private intra-injection and candidate-to-target drafts remain unselected and are not part of this package.

## Three matched conditions

The captured binary matrix has rows for injected nodes and columns `[targets, original candidates, injected nodes]`. Baseline is unchanged. `balanced_target` redistributes only the target-prefix entries. `matched_random` makes the same final per-row number of target-edge removals/additions as balancing, using fixed case/row/role random keys. Every column after the target prefix remains exactly unchanged in every condition.

Let `d_t` be target t's incoming degree from injected nodes, `S = sum(d_t)` the total injected-target incidence, and `Phi = sum(d_t^2)` the imbalance potential. While the largest degree exceeds the smallest by more than one:

1. Choose highest-degree target h and lowest-degree target l; ties use the lowest global target ID.
2. Choose the eligible injected row with smallest physical injected ID, where its h edge is present and l edge absent.
3. Swap h to l in that row.

Eligibility exists because `d_h > d_l`: otherwise the support of h would be a subset of l and could not have higher degree. Each step changes the potential by `-2 * (d_h - d_l - 1) < 0`. The high target retains at least one injected edge, so balancing never uncovers a previously covered target. The process terminates with degrees differing by at most one. Writing `q = S // T` and `r = S % T`, its final potential is `r*(q+1)^2 + (T-r)*q^2`, the minimum under the fixed total S. This proves a column-degree property, **not attack optimality, useful feature influence or classifier performance**.

For each row, compute final `k_i` from its difference to the original target prefix: k_i original ones removed and k_i original zeros added. `K = sum(k_i)` counts final rewired edges; iteration count is recorded separately and does not define the random control. `matched_random` samples k_i original target ones and k_i original target zeros without replacement. It matches final Hamming distance `2*k_i`, not the balanced degree distribution. It can uncover targets, which must be reported rather than filtered away.

## Cost and information invariants

All three conditions preserve:

- Every source-row target count and total edge count, including its mandatory target attachment.
- The entire original-candidate and inter-injected matrix block, endpoint-role counts and total nonself undirected edges.
- Every injected node's physical degree: its owned row count plus incoming edges from later injected rows are both unchanged.
- Candidate endpoint degrees, original-original edges, every feature vector, candidate pool and target list.

Only allocation among target endpoints redistributes; target physical degrees intentionally change. Balancing counts injected incidence, not total clean target degree or learned influence. No prediction, label, surrogate score or victim feedback selects graph changes. The method uses the original legal target endpoint space and adds no new attacker knowledge.

At fixed original features and group masks, GAGA self and known-TRAIN one-hop tokens 0, 1 and 2 should remain unchanged. Unknown one-hop token 3 and two-hop tokens 4, 5 and 6 may change. A real execution must check canonical tokens 0..2 before loading detector weights. GCN normalization can change with target degrees; there is no GCN class-preservation promise. Reassigned features can be unhelpful at another endpoint, and group averages can cancel or move in an unfavorable direction.

## Completed geometry gate and results

All 384 saved baseline group-seed cases were retained, with nonidentity balancing
in 95 / 85 / 99 cases for seeds 3 / 4 / 5. Identity cases stay in all denominators.
All 1,152 geometry variants/canonical sequences and token-invariant checks were
prepared and sealed before classifier loading. Each arm/seed retains 128 groups,
1,091 target occurrences and 374 unique targets. Summed budgets remain 7,776
injected nodes / 358,491 nonself undirected edges across separately attacked
scenarios; they are not one jointly modified graph.

The primary outcome is fresh CPU official-GAGA occurrence-weighted attack MR;
higher means a stronger attack. Baseline is newly reconstructed and freshly
inferred, rather than copied from old stored predictions.

| Condition | Seed 3 MR (%) | Seed 4 MR (%) | Seed 5 MR (%) | Mean MR (%) |
|---|---:|---:|---:|---:|
| Baseline | 51.420715 | 50.412466 | 56.095325 | 52.642835 |
| Balanced target | 52.612282 | 53.528873 | 55.912007 | 54.017721 |
| Matched random | 52.520623 | 53.895509 | 56.461962 | 54.292698 |

Balanced-minus-baseline differences are +1.191567 / +3.116407 / -0.183318 pp;
the mean is +1.374885 pp, nominal paired t95 [-2.742531, 5.492302] pp (`n=3`,
`df=2`). This is conditional on fixed saved attacks and a repeatedly inspected
development cohort; repeated target occurrences are correlated, not IID news
uncertainty. Across all 3,273 paired occurrences, balancing gains 64 new errors
and loses 19 baseline successes (39 group-seed wins / 331 ties / 14 losses).

The frozen rule required at least +1 pp mean versus baseline, positive differences
in all three seeds and a mean above matched random. It **fails**: seed 5 declines,
and balancing is -0.274977 pp below the random mean. Matched random is a control
and cannot be promoted to a qualified winner. No extra seeds, keys or tuning
were chosen to force a gain. Mathematical degree optimality and nonzero headroom
do not establish useful attack performance. The stable-effect objective remains
unachieved; the full-TEST candidate gate stays closed.

## Concrete cases and the remaining structural problem

The preselected follow-up, seed 4 / group 1071, has 14 targets and unchanged
122-node / 5,734-edge budgets. Its GAGA wrong count moves from 7 to 8 under
balancing, but matched random also gives 8. All 14 targets already had injected
incidence before rewiring. This recovers an illustrative failure case without
demonstrating a benefit specific to degree optimality.

For the other fixed example, seed 3 / group 1001, coverage rises from 22 to all
25 targets and GAGA wrong count rises from 6 to 7; random also gives 7. The same
group under seed 5 is a counterexample: coverage rises from 18 to 25, while
wrong count falls from 8 to 7. Better incidence coverage is therefore not a
sufficient performance rule. Across all retained occurrences, coverage rises
from 3,051 to 3,126 under balancing, versus 3,094 for random; the more uniform,
more covered condition still has lower mean GAGA MR than its random control.

All effects occur in the multi-injection stratum (3,060 occurrences, net +45
GAGA wrong predictions). The 48 one-node/one-edge and 165 one-node/multiple-edge
occurrences are unchanged. Of 384 cases, 279 change structurally and 105 remain
identity cases. These are repeated scenario occurrences, not distinct news.

The openly post-hoc largest gain is seed 4 / group 1130: 10 to 14 wrong out of
22, with random also 14. The largest loss is seed 3 / group 938: 5 to 4 wrong
out of 9, with random also 4. These examples are selected for explanation only;
all 384 count-only records remain in the summary. A gate selected from the
39 observed wins would reuse outcome information and is not an evaluated fix.
The evidence supports retaining the controlled structural probe, while leaving
the stable-improvement goal open; neither an agent nor another layer is shown
to solve the remaining feature/relationship transfer problem.

## Execution, independent audit and reporting amendment

Execution completed on 4 November and was checked/reported at the 5 November
delivery boundary; it was not relaunched. Scientific execution source is `5d0f5a67c3b98f427f2a458bd47e5f946b404e37` and
protocol SHA256 is
`422ae756a052a2db343937c90269e8339f78b421a74b89058f89848bd35bf009`.
The run made 1,152 geometry reconstructions/canonical updates and 1,152 separate
evaluation reconstructions. Fresh classifiers each completed 1,280 forwards:
1,152 attacked plus 128 clean. There were zero new generator/teacher forwards,
optimizer updates, autograd, API calls, TEST queries or GPU use. CPU/wall/outer
time was 7,213.101857664 / 1,737.2862673660275 / 1,737.804671398 seconds.

The separate method-author-independent
[saved/tensor audit](../../evidence/target_degree_balance_v1/independent_saved_tensor_audit_v1.json)
passed 223,007 checks and byte-hashed 1,183 actual files in 111.631158848 CPU /
36.909056378062814 wall seconds. It loaded 384 original captures and 384 new
geometry tensors; opaque input files were byte-hashed, but no original dataset,
graph/label array or classifier weight was deserialized and no model inference ran. Relative edge legality, replayed balancing and
random control, injected degrees, row costs, stored features and token 0..2 bits
were independently checked. Whole canonical construction, original-original
arcs and seal timing retain producer/source assertions. This is agent checking,
not human verification or course approval.

The first saved-JSON reporting attempt is preserved after a CI-precision failure
(61,419 checks, 0.1875 CPU / 0.375 wall seconds). An unused v2 amendment replaces
the rounded df=2 critical value with its analytic formula at the same 1e-10
tolerance. Executed v3 additionally corrects the inherited pre-outcome flag;
only the original v1 reporting plan preceded outcome inspection. V2/v3 changes
were made after aggregate outcomes were known. V3 passes 63,811 checks / all
384 count-only cases, using 0.28125 CPU / 0.313 wall seconds and script SHA256
`44d716637c86fd0b52eb35a0ddf4b57b6c581995e6cd29425f2b903de7fc8c69`.
This is producer-author arithmetic reconciliation, not another independent
implementation audit. Scientific inputs, method/case selection and tolerance
did not change; there was no extra model replay.
The execution trace clarifies inherited pre-outcome wording preserved in the raw
summary and reporting module: that wording refers only to the original v1 plan.

[Execution trace](../../evidence/target_degree_balance_v1/execution_trace.json),
[raw result](../../evidence/target_degree_balance_v1/result.json),
[complete summary](../../evidence/target_degree_balance_v1/summary.json),
[all-case counts](../../evidence/target_degree_balance_v1/all_case_counts.json),
[reporting receipt](../../evidence/target_degree_balance_v1/saved_summary_receipt.json),
[v2 amendment](../../evidence/target_degree_balance_v1/reporting_amendment_v2.json)
and [v3 metadata amendment](../../evidence/target_degree_balance_v1/reporting_amendment_v3.json)
retain the checks, failures and scope. Cumulative research GPU charge stays
5.065219865793663 hours plus 0.02 reserve. Dated PDF/PPTX remain unchanged,
outdated drafts; current results are this addendum. Full TEST/tag, actual human
checks/contributions, course approval and rehearsal remain incomplete. Delivery
on 5 November is not a claim that the stable-effect objective or course is complete.

## Implementation and synthetic scope

The [pure standard-library core](balance_target.py) is byte-identical to the private tested prototype. The [tests](test_balance_target.py) cover known answers, invariant cost/blocks, target/row ID ties, single-injection identity, candidate-capacity independence, mandatory edges, final-Hamming matching, stateless control replay and target-column permutation equivariance. An exhaustive small-case check covers 343 nonempty three-by-three target-prefix matrices and eight inter-injected patterns, 2,744 cases in total.

The public test source ran once after its own source/hash freeze: ten methods passed with zero errors/failures, including all 2,744 exhaustive cases, in 0.546875 process-CPU seconds / 0.688 wall seconds on local Python 3.12.14. No real capture, dataset, tensor, model, API or GPU work occurred. The core SHA256 is `e84a4b5291cff37ba42d2356064eb2c0dae5b7f25da965b80b9ab37e987a9e6b`. The public test changes only imports from the retained private test; its original source and private execution receipt remain unchanged.

Control sampling uses SHA256-derived seeds and `random.Random.sample` on sorted physical IDs. Synthetic checks use the bundled local Python 3.12; real control replay must use the frozen server Python 3.8.20 and record that runtime. Cross-version or native-execution bitwise identity is not claimed. Synthetic success verifies mathematical/functional properties only; it does not establish real geometric headroom, canonical behavior on every future graph, efficacy, novelty, human review or course completion.
