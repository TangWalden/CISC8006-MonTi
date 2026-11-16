# Full-pool gradient retrieval: GossipCop improves, YelpChi remains null

Both fixed development cohorts completed. The registered two-dataset criterion
**failed**: GossipCop improves, while every YelpChi arm has identical classes.
Native means the frozen author's attack, not the clean graph; higher MR is
stronger attack performance.

| Method | GossipCop MR | YelpChi MR |
|---|---:|---:|
| Frozen author attack | 57.9101% | 29.1304% |
| Existing single neural | 59.1360% | 29.1304% |
| Existing two-step rewriter | 59.4863% | 29.1304% |
| Full-pool Softplus search | 60.4203% | 29.1304% |
| Its matched random | 58.2604% | 29.1304% |
| Primary boundary-focused search | 60.4203% | 29.1304% |
| Its matched random | 58.4355% | 29.1304% |
| Uniform-shortlist teacher search | 59.6030% | 29.1304% |
| Its matched random | 58.0269% | 29.1304% |

The primary GossipCop gain is +2.5102 pp over native, +1.9848 pp over its
random control, +0.9340 pp over the existing rewriter and +0.8173 pp over
uniform-shortlist search. Each contrast is positive in all three repeats.
Against native there are 43 gained and zero lost occurrences; against the
rewriter 16 gained and zero lost. Both loss choices reach the same MR;
this does not establish an advantage for the boundary-focused objective.
Nominal three-repeat intervals are conditional; those for rewriter/uniform
contrasts cross zero. Repeated targets are not independent observations.

YelpChi has changed logits on 687/690 occurrences but zero class changes in
both victim and own teacher for either gradient-search objective. This is
not evidence that teacher success failed to transfer in these cases: the
teacher did not gain class flips either. It does not identify a sole cause.

Outcome-selected examples, not rules for selecting evaluation cases:

| Example | Native | Existing rewriter | Primary search |
|---|---:|---:|---:|
| GossipCop repeat 0, group 1402: misses / targets | 15/42 | 16/42 | 18/42 |
| YelpChi repeat 0, group 619, node 41429: fraud margin | +0.042818 | +0.037866 | +0.029866 |
| YelpChi repeat 0, group 822, node 6067: fraud margin | +8.223686 | +7.669014 | +7.227038 |

Both YelpChi margins remain positive. Own-teacher margins and victim margins
are different uncalibrated scales, not comparable geometric distances.

## Method, fairness and limits

The [implementation](../../experiments/gradient_retrieval/) adapts HotFlip-style
input gradients to rank all legal TRAIN-benign remote endpoints, then checks
eight candidates per region with an independent fixed teacher. It preserves
injected features, target edges, physical degrees and node/edge budgets, with
at most two candidate-slot substitutions. Same-donor random matches exact
edge dose. Uniform shortlist shares the pool, horizon and per-step verifier
width; gradient/pool scoring is additional compute and STOP changes totals.

Both datasets' choices were sealed before either official VALID evaluation.
All 64 groups and three capture repeats remain: GossipCop has 571 occurrences
and 240 unique targets per repeat; YelpChi 230/230. These are reused exploratory
development sets, not independent confirmation or three author model fits.
No upstream generation, optimizer update, new teacher, API, GPU or TEST call.

TRAIN preflights and every final new/random graph passed physical sequence
equivalence. The root-authored saved tensor/JSON audit passed 27,372 checks;
this is not independent implementation or human verification. Whole-driver
wall time was 4,035.01 seconds; process CPU timings are separate and must not
be added to overlapping wall times. All alternative outcomes remain available.

[All numbers and counters](summary.json), [frozen protocol](protocol.json),
[paired interpretation](paired_interpretation.json), [execution/hashes](execution.json).
Large raw graphs, predictions and logs remain private. The following
channel-aware iteration is a separately frozen hypothesis, not a retroactive
change to this study.
