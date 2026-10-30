# Score-Role Instrumentation and Cross-Device Replay: Pre-Execution Plan

## Status, motivation, and limits of the question

This registers a bounded validation diagnostic, not a source correction, training experiment, or selected improvement. Commit this plan, the diagnostic adapter, and an exact protocol/input manifest before execution. Preserve the earlier protocols, failed attempts, negative pilots, and inconclusive reproduction conclusion. Coverage full evaluation remains **ON HOLD**; this plan records neither course approval nor completed human verification.

The [CPU one-edge reconstruction](../evidence/one_edge_diagnosis_v2/summary.json) matched only 17/48 historical prediction vectors. A subsequent [offline endpoint join](../evidence/one_edge_endpoint_attribution_v1.json) found that all seven positive restricted-oracle cases had selected an endpoint already incorrect on the clean GCN graph. These observations motivate checking the actual score path and reconstruction concordance, rather than assuming an RNG explanation or implementing a clean-priority policy.

The [official paper](https://ojs.aaai.org/index.php/AAAI/article/view/33760/35915), printed page 16031, Eq. 6 and the following unnumbered score definition, projects target/candidate/injected representations and uses the normalized injected representation as the edge-score source. The [implementation](../models/attack_model.py) concatenates z=[targets, candidates, injected nodes], but indexes the score source as e[src], where src is a row of the injection-adjacency mask. With one injected node, src=0 addresses the first target representation; the injected representation starts at T+C. This establishes a formulation/source discrepancy. It does not establish the behavior of historical unpublished implementations, explain earlier prediction differences, or demonstrate a performance effect.

## Fixed cohort, checkpoints, and information boundary

- Include all 16 groups in the [v2 one-edge protocol](one_edge_diagnosis_protocol_v2.json): 972, 987, 1034, 1139, 1185, 1188, 1196, 1267, 1285, 1348, 1357, 1367, 1386, 1408, 1415, and 1463. Do not filter by residuals, replay agreement, ties, or oracle headroom.
- Use all three saved learned attacker checkpoints, seeds 0/1/2, and the same official GCN checkpoint, complete graph, split, feature bounds, cached representations, target assignment, and node/edge budgets of exactly 1/1.
- Use CPU and physical GPU 0 as the two generator and detector devices. Pin CUDA visibility to GPU 0's preregistered UUID so upstream all-device seed resets see only this GPU. Record the actual GPU identity and original registered device for each historical run; GPU 0 is not assumed to be the same physical device as every original run.
- Use diagnostic RNG seed 1729 for each generator replay. These are saved checkpoints, not newly trained seeds. Disable gradients and set attacker/GCN to eval mode. A surrogate-only adapter must reject training, optimizer steps, victim loading, victim requests, and test evaluation.
- Permitted comparison data are the same groups' saved selected-best GCN validation predictions and the retained private v2 reconstruction records. Validate membership, labels, target ordering, selected-epoch identity, and file hashes. Do not inspect GAGA test scores to choose behavior or open test prediction files.
- Load frozen caches; do not regenerate them through hidden GCN calls. A missing or mismatched cache fails the attempt.

## Faithful initialization and recorded runtime

Preserve upstream initialization, including ExperimentHandler's call to [set_seed](../utils/utils.py). That helper sets cudnn.deterministic=True and cudnn.benchmark=False and updates the seed environment value. Record flags **before initialization** and the actual **post-initialization and measurement-time** flags. This source-defined initialization is disclosed rather than bypassed by a custom constructor or undone by restoration.

Reset seed 1729 through the upstream helper immediately before each generator forward. Do not manually tune or restore backend settings to seek agreement. Matmul TF32, cuDNN TF32, deterministic-algorithm policy, and other backend options are otherwise read-only. Record their available values, environment settings, CPU thread count, dtype, PyTorch/DGL/NumPy/CUDA/cuDNN versions, driver, device identity, and runtime configuration; mark unavailable historical settings unknown.

Hash source files, configurations, fixed inputs/caches, GCN weights, all three attacker checkpoints, permitted historical predictions, v2 raw comparison data, this plan, and the protocol before the first model forward. Verify loaded weight identities and retain a clean committed source snapshot. The original frozen scientific files must remain unchanged.

## Capture the actual execution path without changing outputs

Instrument the actual edge-generation call. Do not reconstruct a substitute score pipeline and present it as an observed score.

1. Record T, C, sequence length, startAtk=T+C, actual mask src/dst indices, and the roles and positions of row 0 and the injected row. Retain ordered target/candidate global IDs and their mapping to sequence positions privately.
2. Use a forward hook on linear_edge to make detached cloned observations of its input/output; the hook must return None. Record row norms for the first target and injected row. Distinguish directly observed values from any normalized norms derived later from detached copies.
3. Use temporary observational wrappers around the original edge-path gumbel_softmax and st_top_k calls. Capture the actual raw score vector **before** gumbel_softmax and the actual 1xT target-probability slice received by mandatory-target st_top_k. The helper can modify an aliased input in place, so capture detached clones before invoking it.
4. Each wrapper calls its original function exactly once and returns that original result unchanged, including the actual selected index. It must not substitute copied outputs, recomputed probabilities, a different tie rule, alternative scores, or an altered graph. Separate edge-path captures from candidate-selection calls; fail on missing or ambiguous captures.
5. Retain native dtype/value snapshots, call arguments, relevant shapes, and the actual selected endpoint privately. Restore temporary wrappers/hooks in a finally block. Verify original scientific-file hashes after execution.

For both the captured raw target scores and captured target probabilities, report exact maximum multiplicity, membership/count in absolute near-maximum bands of 1e-6 and 1e-5, and the top-1 minus top-2 gap. Preserve native captured values; do not round before computing ties. Report the score space and units separately. These bands are descriptive score-resolution bands, not widened logit/margin tolerances or a new selection policy.

A paper-role reference may be derived mathematically from detached captured representations by using the normalized injected row as source. Label it a reference calculation. Never feed it into actual edge selection, construct a replacement attack from it, or evaluate a reference-selected graph with GCN.

### Identical-vector top-k algebra check

For each of the 96 captured 1xT probability vectors, copy the **same captured values** to CPU and GPU 0, check bitwise vector/dtype/shape identity after transfer, and apply top-k with k=1 on each device. Record both reference indices, gaps, and tie membership alongside the actual native index. This gives exactly **192 reference top-k calls**.

These are algebra-only checks. They do not replace actual attack decisions or add MonTi/GCN forwards. A device-dependent index on an identical vector would characterize that selection operation; it would not alone prove that a historical disagreement was caused by a tie.

## Complete generator-by-detector matrix

For every group/checkpoint case, capture one attack generated on CPU and one generated on GPU 0. Reuse each **same captured attacked graph** for GCN evaluation on both detector devices:

| Captured generator device | Detector CPU | Detector GPU 0 |
|---|---|---|
| CPU | Evaluate captured CPU attack | Transfer and evaluate that same attack |
| GPU 0 | Transfer and evaluate captured GPU attack | Evaluate captured GPU attack |

Transfer using graph/tensor .to operations, not rebuilding or sorting edges, reattaching an endpoint, or regenerating features. Before each cross-device evaluation, require equal feature values, node counts, target order, and ordered edge sequences, recorded as hashes; ordered-edge identity also implies multiset identity. CPU/GPU clean graph construction may itself produce different edge orders: record that observation, but require equal features and edge multisets before proceeding. Retain self-loop handling and every relevant field. A transfer-integrity failure stops the attempt. Verify sampled output-node order and final-block destination IDs against the requested target order before interpreting detector output.

The matrix contains 48 cases x 2 generator devices x 2 detector devices = **192 attacked GCN forwards**. Evaluate each of the 16 clean groups once per detector device and reuse its result across checkpoints: **32 clean forwards**. Total: **224 GCN forwards** and **96 MonTi generation forwards**. No additional model call is allowed for reference calculations, capture checks, cache construction, endpoint search, or an implicit retry.

Keep a fixed, recorded case/device traversal and flush progress before and after every generation and GCN forward, with separate started/completed counters. Audit the actual model entry points so helper calls cannot bypass the counters.

## Prespecified comparisons and public summaries

- Fresh CPU generation/detection versus retained v2: compare generated-feature hashes, actual endpoint, ordered graph/target/candidate identity, and prediction vectors; retain disagreements.
- GPU generation/GPU detection versus historical selected-best validation records: report vector agreement, count agreement, target-level disagreement totals, and ordering checks. Agreement does not establish original perturbation identity because historical generated features/graphs were not saved.
- Within each captured attack, compare CPU/GPU GCN predictions, margins, and numerical differences. Across generators, compare features, actual score captures, chosen endpoints, ordering, and detector outcomes on each detector device.
- Join selected endpoints to the corresponding detector device's clean correctness, without using clean labels/correctness to rerank an actual endpoint.
- Retain every case, including exact/near ties, different endpoints with equal counts, no disagreement, and differences that do not fit the proposed explanation. Do not convert the 192 attacked evaluations into independent experimental replications or add p-values.

The crossed matrix can localize an observed difference to generation, detector evaluation, or their interaction under this diagnostic. It cannot establish original perturbation identity, a universal tie mechanism, historical causality, a successful source fix, or held-out improvement.

Publish English group/checkpoint/device aggregates, role indices, norms/gap/tie summaries, agreement counts, input/order hashes, counters, costs, and provenance. Keep raw IDs, embeddings, features, logits, margins, prediction vectors, score vectors, and detailed graph artifacts in an ignored private raw directory.

## Hard resource limits, failures, and next decision

The main diagnostic has a hard **300-second elapsed limit**, starting before preparation, hashing, loading, transfers, reference math, and matrix work. Conservatively charge one GPU for the entire main elapsed time, including CPU/setup work. The aggregate forensic reservation, including any failed attempt and separately registered repair, is **0.1 GPU-hour (360 seconds)**, with 60 seconds outside the main budget for overhead/cleanup. An external guardian sends termination at 330 seconds and a hard kill 10 seconds later; its lifetime and actual cost must also be recorded within the reservation. Charge the greater of main elapsed and whole guarded-process elapsed. Stop only this attempt's own process. Any repair must fit the remaining aggregate reservation; no automatic retry is authorized.

The model-call caps are exactly 96 generation and 224 GCN forwards; the complete algebra check has 192 reference top-k calls. If time expires or any cell is unavailable, mark incomplete. Do not drop difficult groups, enlarge limits, add automatic retries, or present partial comparisons as a completed matrix.

Use an exclusive attempt directory. Preserve registration, immutable input hashes, private progress/raw outputs, completed counters, partial results, and failure reasons. Fail on source/input/weight mismatch, illegal incidence, nonfinite values, changed output behavior, transfer-integrity failure, forbidden victim/test access, or a resource violation. Recorded CPU/GPU or historical disagreements are diagnostic outcomes, not grounds for silently changing inputs or excluding cases.

An independent audit after execution must reconcile captured outputs, counters, hashes, comparisons, and incomplete attempts without further inference. Only then interpret the evidence and propose a next controlled measurement. This probe selects no method and does not authorize full improvement evaluation.
