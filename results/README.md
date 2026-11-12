# Compact result exports

## 11 November: feature assignment

[Latest study](feature_assignment/): all four conditions have identical
official-GAGA classifications, mean MR **55.05757859%**; primary gain **0 pp**.
The own-model filter accepts 276/384 primary proposals without changing even
the own-GAGA classes. No qualified candidate or final TEST result.

The new directory contains four compact data exports: summary, all seed metrics,
all 1,536 arm-case count rows and provenance, plus a [brief diagnosis](feature_assignment/diagnosis.md). Check with
`python -m tools.feature_assignment_results check --results results/feature_assignment`.

## Earlier edge-reallocation study

[Preserved deterministic method](../docs/LEGACY_REALLOCATION.md).

These files are derived, compact exports of committed historical records, not
the original executed JSON. Full source locations and hashes are retained.

| File | Contents |
|---|---|
| [latest.json](latest.json) | Earlier edge-reallocation snapshot; historical filename retained for exact checks |
| [seed_metrics.csv](seed_metrics.csv) | All 24 edge-reallocation seed/arm/detector metrics and denominators |
| [case_counts.csv](case_counts.csv) | All 384 cases across four arms: 1,536 rows with GCN/GAGA counts and topology costs |
| [previous_studies.csv](previous_studies.csv) | Earlier attempts, including negative and inconclusive outcomes |
| [audit.json](audit.json) | Original audit attribution, scope and saved verification results |
| [provenance.json](provenance.json) | Source commits, original record hashes and export provenance |

`python -m tools.check_results` checks the saved counts and summary arithmetic.
It performs no neural-model inference and is not another independent raw-logit
or tensor audit. The original separately authored audit is identified in
`audit.json` and the [historical archive](../docs/ARCHIVE.md).

Higher MR means a stronger attack. Every target occurrence remains in the
denominator, including clean errors, unchanged cases and losses. Edge-reallocation results
cover 1,016 occurrences / 373 unique targets per seed, with repeated identities
across groups; the 3,048 combined occurrences are correlated.

The earlier +5.4790 pp mean gain has a nominal paired 95% interval spanning zero,
so the frozen qualification failed. Random target filling is slightly higher;
no control is promoted. The formal reproduction's 43.7280% versus 40.1902%
inconclusive result uses a different cohort and is not this comparison's baseline.
