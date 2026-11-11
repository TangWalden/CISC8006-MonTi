# Loss improvement without classification improvement

A post-hoc, saved-JSON-only recount joins the own teacher and official detector
on the same sealed native inputs. All 384 cases remain; no new neural inference,
tensor/dataset load, training, API or TEST work occurred.

| Seed | Accepted primary cases /128 | Own-GAGA wrong, before/after assignment | Official-GAGA wrong, before/after |
|---|---:|---:|---:|
| 3 | 103 | 330 /330 | 590 /590 |
| 4 | 89 | 324 /324 | 586 /586 |
| 5 | 84 | 337 /337 | 593 /593 |

Each row has 1,071 target occurrences. Every proposal arm has zero own-teacher
and zero official class changes. The primary's accepted loss reductions therefore
do not produce even own-class boundary crossings. Baseline own/official
disagreements are 276/276/266 occurrences; this does not identify a sole cause.
Original-clean own-teacher predictions were not saved, so these before/after
comparisons are **attacked random-fill baseline to feature assignment**, not
own-clean-to-attack measurements. Official clean-to-attack counts are separate.

One deterministic post-hoc tied example has 9 targets, 11 injected nodes and
264 edges: own-GAGA wrong stays 2, official wrong stays 6, while own mean
Softplus decreases by 8.172258e-6. Both maximum-gain and maximum-loss selectors
tie at zero. There is no positive classification-change example to promote.

The [recount source](../../tools/analyze_feature_assignment_transfer.py) passes
18,745 saved-record checks. It is by the method-core author, **not independent
of implementation authorship**, and is not human verification. The separate
saved-tensor audit is documented in [provenance.json](provenance.json).

Recount source SHA256: `bcf69834206e056d8da4247c0cc93034e875df9adda4d433d3b4d862954f7c7d`.
Original private result SHA256: `4bd73489b4b28e65da636e8c6cd1f698d59965112d843712197d905ac89da3fc`.
This page is a derived explanation, not a relabeled original executed record.
