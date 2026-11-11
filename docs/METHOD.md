# Candidate-to-target edge reallocation

The current method is a deterministic postprocessing rule applied to a MonTi
attack graph. It changes selected injected edges after generation, without
retraining MonTi, calling an LLM agent or querying a detector to choose edges.
The standalone implementation accepts explicit node IDs and a binary adjacency.

## Rule and controls

Each row represents an injected node. Columns contain original targets,
original candidates and injected nodes, in that order. The injected block uses
strict lower-triangular ownership, counting each undirected edge once.

From each unchanged baseline row, form three sets:

- `C1`: candidate nodes already connected to this injected node.
- `T0`: targets not connected to it.
- `C0`: candidate nodes not connected to it.

Set `k = min(|C1|, |T0|, |C0|)`. The candidate-zero limit permits a matched
candidate-rewiring control. All nonbaseline arms remove the same `k` candidate
edges, selected with a fixed SHA-derived random key.

| Arm | Additions replacing those candidate edges |
|---|---|
| `baseline` | No changes |
| `balanced_target_fill` | `k` missing targets with the lowest baseline injected-attachment counts, then SHA ties |
| `random_target_fill` | `k` missing targets sampled with a separate fixed key |
| `candidate_rewire` | `k` previously absent candidate connections |

Primary target counts are column sums across injected rows, computed **once
from the baseline**. They are not full clean-graph target degrees and are not
updated after each addition. Keys bind the seed, original group key, physical
injected-node ID, purpose and, for ties, target ID. Do not reindex IDs and expect
the same result. No labels, margins or model outputs enter this rule.

## What stays fixed

All arms retain original target edges, target/candidate identities and order,
the injected-to-injected block, every row edge count, each injected node's
physical degree and total injected node/edge costs. The original graph and
injected features `x0` remain unchanged in the experiment. Each nonbaseline arm
changes exactly `2k` binary entries per row. Target and candidate attachment
counts intentionally change in the target-fill arms.

The inherited target lists assume known fraud targets; this rule does not
discover fraud labels. A zero `k` keeps the case unchanged. Maximal feasible
reallocation can remove every candidate neighbor from a row, so it is not a
small-edit guarantee. All such cases remain in evaluation.

## Measured development result

On 128 original VALID groups, three generator seeds and 1,016 target occurrences
per seed, mean official-GAGA MR rises from **46.9488% to 52.4278%** (+5.4790 pp).
All three seed gains are positive, but the nominal paired 95% interval is
**[-0.8229, 11.7810] pp**. Its lower bound fails the frozen qualification rule;
no candidate qualified and no final TEST evaluation was performed.

MR counts incorrect predictions over all fraud-target occurrences, including
clean errors. Higher MR means a stronger attack. The 3,048 occurrences include
repeated news identities and are not independent observations.

Random target filling averages **52.5262%**, slightly above the primary;
candidate rewiring averages **46.8832%**. This supports investigating edge roles,
not superior balanced priority, an optimal allocation or first novelty.
See [complete compact results](../results/README.md) for controls, seed values,
retained failures, costs and provenance. Formal reproduction results use a
different cohort and are reported separately.
