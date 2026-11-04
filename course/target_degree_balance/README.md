# Actual injected-target degree balancing

**Status: the isolated core and ten public synthetic checks are complete. Real capture headroom and detector effects have not been measured by this package.** This is a practical structural hypothesis, not a novelty, efficacy or cost-saving claim. The preceding [feature-refinement study](../feature_polish/README.md) was negative; improving a mathematical degree objective does not promise improved attacks.

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

## Headroom before model calls

The intended capture-only checkpoint covers all 384 saved baseline graphs from three seeds and 128 fixed VALID groups per seed. It must be registered with source/input hashes before capture reads. It measures d_t, S, Phi, final k_i/K, coverage, identity and balanced/random graph equality, unchanged role blocks/features/injected degrees, actual physical budgets and canonical token invariants. No detector, new generator, teacher training, API or TEST query is needed for that geometric stage.

A single injection already has zero/one target columns and is identity. Already balanced columns and other zero-action cases are retained in all denominators. Under the planned gate, a seed whose cases are all identity stops before detector loading; cases are not selected by prior detector outcomes or removed from evaluation. The full target lists and the 374 original target identities remain fixed. Repeated group/seed occurrences are correlated.

Only nontrivial registered geometry can justify the next bounded evaluation. The intended primary criterion is at least +1 pp mean official-GAGA MR versus baseline, positive differences in all three seeds, and a larger mean than matched random. Higher MR means a stronger attack. Baseline, balancing and random results, all negative cases, actual query counts and conditional three-seed uncertainty must be retained. No extra seeds, different random keys or tuning are chosen after outcomes to force a winner. Independent final TEST and final submission freeze/tag remain separate checkpoints.

## Implementation and synthetic scope

The [pure standard-library core](balance_target.py) is byte-identical to the private tested prototype. The [tests](test_balance_target.py) cover known answers, invariant cost/blocks, target/row ID ties, single-injection identity, candidate-capacity independence, mandatory edges, final-Hamming matching, stateless control replay and target-column permutation equivariance. An exhaustive small-case check covers 343 nonempty three-by-three target-prefix matrices and eight inter-injected patterns, 2,744 cases in total.

The public test source ran once after its own source/hash freeze: ten methods passed with zero errors/failures, including all 2,744 exhaustive cases, in 0.546875 process-CPU seconds / 0.688 wall seconds on local Python 3.12.14. No real capture, dataset, tensor, model, API or GPU work occurred. The core SHA256 is `e84a4b5291cff37ba42d2356064eb2c0dae5b7f25da965b80b9ab37e987a9e6b`. The public test changes only imports from the retained private test; its original source and private execution receipt remain unchanged.

Control sampling uses SHA256-derived seeds and `random.Random.sample` on sorted physical IDs. Synthetic checks use the bundled local Python 3.12; real control replay must use the frozen server Python 3.8.20 and record that runtime. Cross-version or native-execution bitwise identity is not claimed. Synthetic success verifies mathematical/functional properties only; it does not establish real geometric headroom, canonical behavior on every future graph, efficacy, novelty, human review or course completion.
