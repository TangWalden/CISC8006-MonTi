# Conditional relationship alignment: isolated pilot implementation

## Completed validation pilot

The explicitly authorized first matrix completed all nine fresh runs, seeds
3/4/5, with the frozen 256 TRAIN and 128 validation groups. Every selected
validation checkpoint covers 1091 target occurrences. Selection uses the
earliest maximum occurrence MR. The [aggregate recount](../../evidence/new_methods_20261103/final_saved_recount.json)
and [separate-agent raw-output audit](../../evidence/new_methods_20261103/alignment_full_audit.json)
preserve every seed and all 89 validation epochs.

| Arm | Seed 3 MR | Seed 4 MR | Seed 5 MR | Mean MR | Mean group distance | Outside tolerance |
|---|---:|---:|---:|---:|---:|---:|
| OFF | 97.5252% | 97.4335% | 97.5252% | 97.4947% | 41.19860 | 91.2509% |
| ALWAYS | 95.1421% | 94.5005% | 94.2255% | 94.6227% | 0.26192 | 5.8342% |
| GATED | 96.0587% | 94.8671% | 95.0504% | 95.3254% | 0.52096 | 25.4201% |

Higher MR means a stronger attack against the GCN surrogate. Mean paired MR
differences versus fresh OFF are -2.87198 percentage points for ALWAYS and
-2.16926 for GATED. GATED is 0.70272 points above ALWAYS but below OFF in all
three seeds. These are selected-checkpoint validation comparisons, not final
victim results or a selected positive improvement. Historical negative pilots
are not substituted as the control.

Distance averages group mean distances, then seeds. Outside tolerance is the
fraction of injected nodes exceeding the frozen TRAIN threshold, then averaged
across seeds. These are different denominators. Policy activation counts are
not outlier counts: OFF activates zero, ALWAYS all, GATED only those outside.
Each selected arm/seed physically adds 7776 nodes and 358491 non-self undirected
edges across the same 128 scenarios; these are sums across independently
perturbed graphs, not one combined graph. Counts match the exact implementation
budgets. Optimizer updates total 22784; GCN and generator forwards total 34176
each. Whole alignment GPU charge is recorded separately from self training.

The loss lowers its optimized two-moment distance, with a reduction in attack
effectiveness. An independent camouflage detector, matched-activation random
control and matched-penalty-strength control were not evaluated. Consequently
these measurements do not isolate intelligent gate selection, establish full
distribution matching, or prove practical stealth. The later selected GAGA/LLM
probe is a transfer test and induces zero graph-branch class changes in its eight
queried news nodes. No adaptive attack against the complete API defense or final
test was performed. No seeds/cases/calls were added to improve the outcome.

This package adds a training-time attack regularizer, not an LLM defense gate.
It changes no frozen scientific file, candidate selector, edge-score source,
tie policy, feature range or physical budget. The owner authorized actual method
work; training still requires a committed source, frozen protocol and guardian.
There is no efficacy claim from implementation or synthetic checks.

For each injected node, the relationship signature is the first and second raw
moments of cosine similarity between its generated feature and the features of
its physical non-self neighbors. The original benign TRAIN nodes provide the
reference centers. Reference neighborhoods use the original transductive graph
and raw features; only center labels from the registered TRAIN node partition
are consulted. No validation/test labels or predictions fit this reference.

Degrees use predefined floor-log2 bins. A supported bin needs eight reference
nodes. An unsupported bin uses the closest supported bin, breaking ties toward
the lower bin; every fallback is counted. Per-bin coordinate scales are the
population standard deviations with a floor of 0.05. Distance is the mean of
the two squared standardized signature coordinates. Tolerance is the TRAIN
90th percentile of that distance, frozen before optimization.

All arms compute and record the same distance. OFF adds zero; ALWAYS adds
0.1 times mean distance; GATED adds 0.1 times mean of distance times the detached
indicator `distance > tolerance`. The mean includes every injected node, not
only active nodes. No gate parameter is optimized, so it cannot turn itself off
to reduce its own objective. An implementation that uses an outside-band hinge
in every arm would make ALWAYS/GATED identical and is deliberately not used.

This is two-moment, degree-bin-conditional alignment, not equality of entire
distributions. A one-edge injected node has one cosine value, not an independent
sample distribution. Its degree and some neighborhood statistics are fixed by
the budget/endpoint. Relationships to genuine fraud/benign targets are not
globally flattened. Homophily and camouflage have prior art; no first-novelty
claim is made. Reduced distance alone is not evidence of an effective attack.

## Interfaces and sequence

1. Freeze a preparation protocol with all new package Python source hashes,
   relevant frozen scientific source/input hashes, and the choices below; commit
   source before any execution.
2. Fit the single reference with the registered runtime and CPU visibility:

   `CUDA_VISIBLE_DEVICES= bash course/run.sh -m course.conditional_alignment.prepare_reference --protocol <protocol.json> --output <reference.json>`

   This loads original graph/split metadata but no model/checkpoint, performs no
   model forwards and writes no original cache. Freeze the output SHA256 as
   `reference_sha256` before training. All nine jobs must use that same file.
3. Run one fresh job under the root-owned shared-stage guardian:

   `CUDA_VISIBLE_DEVICES=0 bash course/run.sh -m course.conditional_alignment.run --protocol <protocol.json> --reference <reference.json> --arm gated --seed 3 --gpu 0 --output runs/alignment_gated_s3_attempt1`

   The runner requires one visible GPU and checks its own process UUID. It has
   a 480-second maximum alarm and saves partial/failure records. The guardian
   remains responsible for the aggregate two-GPU-hour stage cap, startup time,
   outer hard termination and no automatic retries.

Protocol fields: `status="frozen"`, `arms=["off","always","gated"]`,
`seeds=[3,4,5]`, the unchanged 256/128 group lists, `epochs=10`, `patience=5`,
`alignment_lambda=0.1`, `job_wall_seconds=480`, `base_config_sha256`, `code`,
`inputs`, `reference_sha256` for training, and
`reference={"degree_rule":"floor_log2_nonself","min_count":8,"gate_quantile":0.9,"scale_floor":0.05,"chunk_size":4096}`.
Set `physical_gpu_uuid` to pin the guardian's physical GPU. Export groups are
exactly `[1185,1267,1285,1415]`, retained from the old 128-group validation cohort.

## Evidence and integration

Only the fixed GCN surrogate checkpoint and original `h_GCN.npy` cache are used.
Victim/GAGA loading, test evaluation and silent cache regeneration are rejected.
The GCN sampler must preserve requested target order. Selection uses the original
occurrence-weighted GCN validation MR and original patience behavior. Validation
reuses each GCN logits calculation for prediction and hinge summaries, rather
than adding an otherwise duplicate forward. Training still uses the original
MonTi loss plus the explicitly recorded penalty.

Each output directory is exclusive and must live under ignored `runs/`.
Per-epoch group-count and alignment JSON files, progress, registration, source and
reference hashes, best checkpoint hash and forward counters support review.
Raw validation IDs/labels/logits/predictions stay in `raw/`, never public logs.
Original model/optimizer checkpoints are retained. Public release should copy
only reviewed aggregate records, not this ignored directory wholesale.

Every normal validation pass directly exports the four registered groups'
`x_atk`, `adj_attack`, global `target_ids` and global `candidate_ids` to private
CPU `.pt` files. Adjacency columns are ordered target, candidate, injected, as
upstream; attack rows are injection-relative. `original_num_nodes` is recorded.
The final `selected_attack_exports.json` selects the files from the best epoch,
with hashes. This costs no additional generator/GCN query and lets the root's
separately registered fusion evaluator reconstruct the exact perturbation.
No GAGA context is constructed by this package.

`python -m course.conditional_alignment.synthetic_checks` contains small CPU
known-answer/gradient/source-boundary checks. These are preparation until an
execution receipt confirms a committed-source run; they load no real data or
model weights and measure no method efficacy.
