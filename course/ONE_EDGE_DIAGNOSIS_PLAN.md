# One-edge validation diagnosis: pre-execution plan

## Status and question

This plan must be committed before the measurements below begin. It registers a post hoc validation diagnostic, not a new reproduction claim or an improvement experiment. Improvement selection remains **ON HOLD**. The previous protocols, negative pilots, and inconclusive reproduction conclusion remain unchanged. This document records neither instructor approval nor a human evidence check.

The question is: within the existing one-node/one-edge budget, does changing only the sole target attachment endpoint improve a reconstructed saved learned attacker's GCN validation outcome, with its reconstructed generated feature held fixed? Historical prediction agreement does not establish identity of the originally generated feature or edge, which were not saved.

The results-first extraction found 16 of 592 full-validation groups with `node_budget=1` and `edge_budget=1`. Across three seed passes, they account for 243 of 14,889 evaluated occurrences (1.63%), but 147 of 438 learned residuals (33.56%). Their learned residual rate is 147/243 = 60.49%; the corresponding random rate is 151/243 = 62.14%. A residual is an occurrence still classified correctly after attack. These observations motivate measurement; they do not identify allocation as the cause. Degree, budget, and group structure are confounded.

## Fixed scope and inputs

- Include **all 16** full-validation groups satisfying both one-node and one-edge conditions in the learned rows of `evidence/results_first_diagnosis_v1/group_outcomes.csv`. Do not filter by observed success, residual count, or a favorable subgroup. Their group sizes sum to 81.
- Use the saved learned best attacker checkpoints from the full baseline seeds 0/1/2. Read their source `result.json` files only for checkpoint identity, best epoch, and related metadata, not victim test scores.
- Use only the fixed official GCN surrogate. Retain the full original transductive graph, original split, rho/xi, features, and registered model budgets. CPU execution is an explicit diagnostic difference from the original GPU runs.
- The saved selected-best GCN validation prediction file may be parsed to extract only these 16 groups, with membership, labels, denominator, epoch identity, and prediction hashes checked. Other groups are not used for this probe. No test prediction file may be opened.
- Before inference, record SHA-256 hashes of this plan, the clean committed adapter/source revision, fixed inputs, configurations, caches, GCN checkpoint, attacker checkpoints, and permitted source prediction files. Freeze the complete group list and its published metadata source in a separate protocol JSON. Do not regenerate caches silently.
- A new diagnostic adapter is permitted. Do not modify the original frozen scientific files, baseline settings, or evidence. Registration must identify the adapter commit and hashes without implying that the adapter was used in the original runs.

## Measurement procedure

1. Set `CUDA_VISIBLE_DEVICES` to the empty string and use CPU only. Freeze weights, call `model.eval()`, and disable gradient recording. Do not call training methods or take optimizer steps.
2. For each of the 16 groups, evaluate the clean GCN once. Retain per-target clean margins and correctness privately. Reuse this clean outcome across the three attacker seeds rather than counting it as three separate clean queries.
3. For each group and attacker seed, reconstruct and evaluate an attack at diagnostic RNG seed 1729. Record the generated feature, actual injected incidence, sole selected endpoint, clean and attacked margins, clean-correct transition counts, and agreement with the permitted saved selected-best validation predictions. Call all reconstructed features and edges reconstructed, even if predictions agree with the historical file. Prediction disagreement limits historical attribution but is retained rather than filtered out.
4. Independently regenerate and evaluate that same checkpoint/group at diagnostic RNG seed 2718. Compare feature hashes, selected endpoint positions, incidence counts, and predictions with the seed-1729 replay. This is a fixed-checkpoint stability check; it is not another training seed or an independent claim test.
5. For each seed-1729 reconstructed attack, hold its generated feature and all other graph inputs fixed. Enumerate **every target in that group** as the sole legal attachment endpoint. Preserve exactly one injected node and one non-self undirected injected edge, with the original self-loop handling and feature constraints. Evaluate every endpoint, including the reconstructed selected endpoint, without adaptive stopping or endpoint selection based on partial outcomes. Do not enumerate non-target endpoints: with this budget, the original mandatory target-attachment rule consumes the sole edge. Do not assume that targets without a direct injected edge are unaffected: GCN propagation and normalization can change their outcomes. Count absolute correct-class margin changes greater than 0.00001 as above the prespecified numerical tolerance; smaller changes are below this probe's resolution.
6. Preserve complete group-level counterexamples, including cases with no oracle headroom, worse alternative endpoints, disagreement with saved predictions, or replay instability. Diagnostic RNG seeds do not replace the original run seeds or reconstruct an unrecorded historical random state.

The adapter must not load GAGA or any victim model. Its logits entry point must reject a victim request. It must not read test predictions or perform victim/test inference.

## Prespecified summaries

For each group, checkpoint seed, and endpoint position, report total target occurrences, detector-correct and misclassified counts, the clean-correct denominator, clean-correct-to-incorrect counts, and clean-incorrect-to-correct counts. These conditional counts supplement the diagnosis; they do not replace the frozen primary MR definition.

Let `A(e)` be the number of misclassified target occurrences with endpoint `e`, using the fixed seed-1729 generated feature, and let `e0` be the originally selected endpoint. The primary restricted oracle headroom is `max_e A(e) - A(e0)`; the worst-endpoint change is `min_e A(e) - A(e0)`. Record ties and every endpoint result. Since `e0` is included, the maximum cannot be below the original outcome. Also report the corresponding range and headroom for the clean-correct-to-incorrect count separately; its maximizing endpoint need not match the total-MR maximizer.

Aggregate over all 48 group/checkpoint combinations and all 243 target-occurrence exposures, including zeros and negative endpoint changes. Report counts of positive/zero headroom, best/worst restricted-oracle totals, original totals, and source-prediction agreement counts. The oracle uses validation outcomes to characterize headroom only. It is not a trained method, a deployable label-based selection policy, a fair extra-query improvement comparison, or evidence of held-out gain.

Overlapping nodes, groups, and repeated seed passes are not independent sample sizes. No p-values, invented uncertainty intervals, causal allocation claim, or significance-based continuation will be added.

## Resource and failure rules

Expected GCN forward calls: 16 clean + 48 original + 48 repeat + `3 * 81` endpoint calls = **355**. Count every GCN forward call, including any incidental call; do not hide cache construction or retries outside this counter. Hard limits are **400 GCN forwards**, **600 wall-clock seconds**, and **0 GPU-hours**. Register the limits before inference and stop at whichever limit is reached first. Do not expand a limit after inspecting outcomes.

Use a new exclusive attempt directory. Preserve registration, progress, counters, failures, and partial outputs. A timeout or partial endpoint enumeration must be reported as incomplete; do not present partial maxima as complete oracle results. A later repair/retry requires a new attempt and explicit linkage to the failed attempt, preserving the original record.

Stop and record a failure for a hash or group-manifest mismatch, illegal incidence, changed generated feature during endpoint enumeration, nonfinite output, test/victim access attempt, or resource-limit violation. Differences from historical predictions or between the two diagnostic replays are recorded rather than silently corrected. If input integrity remains valid, complete the prespecified measurements within the limits and qualify their comparability; do not choose whichever replay better matches the source.

## Evidence release and interpretation

Keep raw per-target node IDs, labels, logits/margins, predictions, generated features, and detailed graph artifacts in an ignored remote raw directory. Publish only the already published target-group IDs with endpoint-position aggregate counts, integrity and stability summaries, hashes, limits, timings, and the adapter/protocol provenance. Do not publish raw node IDs, per-node labels/predictions or generated features.

- Positive restricted headroom would show that an alternative legal endpoint can help **for the same generated feature on these validation groups**. It would motivate examining learnable allocation signals, not establish a successful improvement.
- Zero headroom across most groups would weaken the sole-endpoint-allocation hypothesis for these checkpoints/features. It would leave feature quality, graph structure/reachability, the allowed budget, and clean baseline behavior as competing explanations. It would not prove a fundamental capacity bound or rule out all possible generated features.
- Clean GCN outcomes determine whether an attacked residual is a persistent clean-correct case or a clean-incorrect case changed back to correct. Without this comparison, residuals cannot be interpreted as proven attack failures or flips.
- Replay instability or mismatch with historical selected validation predictions would limit attribution of a small endpoint advantage. Report the discrepancy before interpreting it; temperature/epsilon are not assumed to change deterministic evaluation decisions.
- Even a stable positive validation oracle result cannot establish victim transfer or generalization. Any subsequent method proposal requires separate evidence review, owner confirmation of direction, a frozen protocol, fair controls, and a separate final evaluation. This diagnostic itself selects no improvement.
