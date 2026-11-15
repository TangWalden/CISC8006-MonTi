# YelpChi development transfer: no measured improvement

The owner authorized a fair second-dataset comparison after asset preparation.
One registered pipeline completed on 15–16 November 2026. **All eleven arms
have 29.1304% GAGA miss rate and identical paired class predictions.** The
registered improvement screen failed. LifeIns was excluded; no TEST evaluation
or post-outcome tuning was performed.

Higher MR means a stronger attack on the original fraud targets. Native below
means the author's attack, not the clean graph.

| Method | Mean MR | Difference from native |
|---|---:|---:|
| Frozen author upstream | 29.1304% | reference |
| Existing single neural | 29.1304% | 0.0000 pp |
| Existing single direct search | 29.1304% | 0.0000 pp |
| Existing neural matched random | 29.1304% | 0.0000 pp |
| New two-step neural rewriter | 29.1304% | 0.0000 pp |
| Its same-position endpoint random | 29.1304% | 0.0000 pp |
| Its joint region/endpoint random | 29.1304% | 0.0000 pp |
| Two-step greedy / its matched random | 29.1304% / 29.1304% | 0.0000 pp |
| Independent composition / its matched random | 29.1304% / 29.1304% | 0.0000 pp |

Each repeat retains 64 VALID groups, 230 occurrences and 230 unique targets,
with 67 misses. Three capture-noise/downstream repeats share one author
checkpoint and one independent teacher; 690 is not 690 independent targets.
All comparisons have zero gained/lost class predictions. Group-equal MR is
also identical across arms (30.8319%). The retained conditional t-intervals
of [0,0] describe these fixed repeats, not zero population uncertainty.

## What happened in the actual graphs

The rewriter changed connections in 187/192 cases, totaling 4,949 exchanged
edges across separate scenarios. Both own random controls match those cases,
edit counts and edge dose. Logits changed on 676/690 paired occurrences, so
this is not simply an unexecuted or identity-only intervention. None crossed
the decision boundary. Joint-region random is degenerate in 72/192 cases.

Saved-logit examples, selected after outcomes for illustration:

| Repeat / group / target | Native fraud-minus-benign margin | Rewriter margin | Class changed? |
|---|---:|---:|---|
| 0 / 619 / 41429, closest native margin | +0.042818 | +0.037866 | No |
| 0 / 822 / 6067, largest absolute margin change | +8.223686 | +7.669014 | No |

Mean absolute margin change is 0.006692. Among movements toward the boundary,
the largest closes 21.31% of that same occurrence's original margin. This
establishes non-crossing for the saved cases, not a sole causal explanation,
a capacity bound, or proof that another downstream cannot work.

GossipCop's corresponding rewriter development gain was +1.5762 pp over native
and +0.3503 pp over existing neural. **It did not carry into a positive YelpChi
comparison under the same downstream training rule.** Raw MR across datasets
is not paired: their cohorts, graph geometry and original budgets differ.

## Fairness, scope and evidence

- Original author YelpChi checkpoint/configuration, features and physical
  node/edge budgets are fixed within every comparison. Target membership and
  order, injected degrees and all denominators remain fixed.
- Same downstream architectures, optimizers, epochs and respective TRAIN-group
  counts as GossipCop; feature width follows YelpChi. Models are fitted on
  YelpChi TRAIN: this is method transfer, not zero-shot weight transfer.
- 64 of 331 author-provided VALID groups were sampled with a frozen RNG before
  model calls. This does not prove the author's original list construction is
  unbiased. All selected cases remain. Original transductive features/graph
  are used; known-label groups and benign endpoints use only original TRAIN.
- All methods' actions sealed before original GAGA scoring. The independent
  teacher's TRAIN quality check is in-sample, not calibration or held-out proof.
- Shared candidate-card counts are **not** minimum teacher-forward costs for
  the old neural model; see the [accounting clarification](../target_rewriter/cost_interpretation.json).
  Same physical budget is not equal training, classifier or reconstruction cost.

[All numerical results](summary.json), [frozen protocol](protocol.json),
[paired saved analysis](paired.json), [saved-logit details](margin_details.json),
[audits and preflights](audits.json), [execution and costs](execution.json),
[executable adapter/source](../../experiments/yelpchi_validation/).

The root-authored baseline and saved tensor/JSON audits passed 3,850 and 31,782
checks; these are not independent implementation or human verification.
There were 580 author generations including preflight, 1,368 teacher updates,
3,072 single-ranker updates, 14,976 rewriter updates, two disposable functional
preflight updates and 2,112 official VALID forwards. The main rewriter phase
used 9,041 physical builds and 11,891 teacher contexts. Whole driver wall time
was 13,191.01 seconds, with 0 GPU/API/TEST use. Preparation and saved-only
interpretation timings are separate. Raw models, cards and full logs remain
private; source hashes, exact compact outputs and failures remain traceable.
