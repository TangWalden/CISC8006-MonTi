# Isolated self-model and numeric evidence runner

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
node order**; use the original update helper with the full relevant membership,
including queried benign nodes. Do not reuse clean sequences for an attacked
context. A matching context string is an assertion, not proof of correct sequence
construction. The wrapper checks test exclusion, numerical validity, exactly one
self-loop and records current degree rather than cached original degree.

The module's `run` stage executes clean/meta extraction only. Attack generation,
probe extraction, API calls, numeric-gate training, fusion evaluation and their
complete cost/protocol remain root-owned separate stages.
