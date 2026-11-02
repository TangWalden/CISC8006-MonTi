# Isolated self-model and numeric evidence runner

## Actual validation pilot on 3 November 2026

The owner explicitly requested implementation and execution. Three self-model
seeds were trained, and the external DeepSeek agent was called on numeric
self/graph evidence. The primary 32 cases are eight original news nodes under
four clean/attack conditions, not 32 independent articles. GAGA remains the
fixed graph branch. No article body/title mapping is available in this pilot.

| Method on the identical 32 cases | Correct | Benign false positive rate | Fraud recall |
|---|---:|---:|---:|
| Self-only MLP, seed 3 | 16/32 | 25% | 25% |
| Fixed GAGA graph branch | 20/32 | 25% | 50% |
| Fixed half fusion | 20/32 | 25% | 50% |
| Numeric neural gate, registered action grid | 20/32 | 25% | 50% |
| External LLM with template evidence | 20/32 | 25% | 50% |
| External LLM with generated descriptions | 20/32 | 25% | 50% |

The last five methods make exactly the same 32 class predictions. A pre-API
amendment adds four supervised TRAIN examples in context and assesses 20 fixed
validation cases: 11/20 correct, again identical to zero-shot LLM, numeric gate,
graph branch and fixed fusion on those same 20 cases. This is in-context
adaptation, not LLM fine-tuning or a trained text student. No LLM advantage was
observed. Both branches are uncalibrated; original graph TRAIN-meta predictions
are in-sample, not out-of-fold. The numeric gate learns from 662 meta examples,
whereas the LLM extension receives four demonstrations, so adaptation budgets
are not matched even though query facts and decision grids are matched.

The API completed 120 physical calls, no retries/errors/fallbacks, with 101473
prompt tokens and 14198 completion tokens. The registered peak/cache-miss cost
upper bound is USD 0.0474795, not an invoice. Returned model alias was
`deepseek-flash`; recorded system fingerprint does not independently certify
backend weights. Labels and raw node/group/attack identities stayed local.
Eight joint branch errors cannot be corrected by fixed-threshold convex fusion.
The GCN-trained attacks provide a transfer probe, not an adaptive attack against
the complete LLM pipeline. A post-hoc join of retained predictions finds zero
clean-to-attacked GAGA class changes for each arm's eight probe nodes. This cohort
therefore does not demonstrate prevention of attack-induced errors, and remains
a functional/selected-utility pilot rather than positive defense evidence. No
new cases or calls were added after this finding. See the [saved-output recount](../../evidence/new_methods_20261103/final_saved_recount.json)
and [frozen protocol](../NEW_METHODS_EXECUTION_PROTOCOL_V1.json).

This module implements preparation/training/evidence extraction for the new
validation-only pilot. It makes no API calls, initializes no attacker, does not
predict test nodes, and does not alter the frozen baseline. No execution is
implied by these source files.

Run from the repository root in the locked server environment, only after the
root agent freezes the source and concrete configuration. Raw output must be
inside the ignored `runs/` directory. All commands require exclusive new output
directories and externally supplied frozen digests.

```sh
python -m course.fusion_pilot.self_evidence prepare-manifest --config runs/fusion-config.json --expected-config-sha256 CONFIG_SHA --output runs/fusion-manifest-attempt1
python -m course.fusion_pilot.self_evidence run --config runs/fusion-config.json --expected-config-sha256 CONFIG_SHA --manifest runs/fusion-manifest-attempt1/cohort_manifest_private.json --expected-manifest-sha256 MANIFEST_SHA --output runs/fusion-self-attempt1
```

The required JSON configuration has `phase: "validation_pilot"`,
`graph_model: "GAGA"` (or the explicit GCN control), `device: "cpu"` or an
explicit CUDA device, `source_commit`, `max_wall_seconds` (at most 3600),
`max_graph_forwards` (at most 1000), optional `cpu_threads` (default 2), and
`expected_sha256` mapping repository-relative files to their actual SHA256.
Registration checks require this new source, frozen loader/model sources, all
official split inputs, graph, selected graph configuration/checkpoint and the
GAGA sequence cache when applicable. Extra frozen file hashes are checked too.
No missing cache is generated. The source commit must equal HEAD.

Set `probe_group_ids: [1185, 1267, 1285, 1415]` for the root's specified export
cohort. If absent, preparation instead selects four eligible validation groups
by its documented hash order; these are different explicitly recorded rules.
The frozen protocol must pick the intended rule before preparation/execution.

The manifest is written before training. It splits original training nodes
within each class by hash seed 20261103: floor(80%) base fit, floor(10%) self tune,
remainder meta. Original-validation clean evidence uses 32 nodes from each class
chosen by an independent fixed hash namespace. Each selected one-node/one-edge
group has one hash-chosen fraud target and one unique validation benign node;
probe benign nodes are also disjoint from the 64-node clean cohort. Original
target overlap is not removed. This preparation does not call a model.

The self-only MLP uses the existing MLP layers, hidden sizes [64,32], seeds 3/4/5,
Adam lr .001, weight decay 0, batch size 256, at most 30 epochs and patience 5 on
self-tune cross entropy. It is not calibrated. The graph checkpoint is fixed.
**Graph-branch meta scores are in-sample for original training labels, NOT
out-of-fold.** The new self branch does not fit meta labels; this does not make
the graph branch out-of-fold. The original validation nodes remain the later
held-out assessment cohort. No test results select epochs or parameters.

`run` produces label-free `records.jsonl`, separate private
`labels_private.jsonl`, best self checkpoints, per-epoch records and a summary.
Clean graph predictions are computed once and reused across self seeds under
the same context. Batch size is fixed at 64; this clean node-cohort inference
does not claim bitwise identity with historical target-group inference.

Root can import `load_self_checkpoint`, `load_graph_branch`,
`graph_branch_logits` and `extract_current_context` to evaluate retained attack
graphs. For GAGA, pass current-context sequences **already aligned to the query
node order**. The executed bridge in `extract_attacked.py` reconstructs canonical
current-graph one-hop and Boolean two-hop class-group sequences, including queried
benign centers, and checks the clean result against the frozen cache before model
calls. It passed combined `atol=1e-5, rtol=1e-5`; the maximum absolute difference
was 1.8537044525e-5, so this is not an absolute-only 1e-5 bound or bitwise equality.
Do not reuse clean sequences for an attacked context. A matching context string
is an assertion, not proof of correct sequence construction. The wrapper checks
test exclusion, numerical validity, exactly one self-loop and current degree.

The module's `run` stage executes clean/meta extraction only. Attack generation,
probe extraction, API calls, numeric-gate training, fusion evaluation and their
complete cost/protocol remain root-owned separate stages.
