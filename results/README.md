# Compact result exports

These files are derived, compact exports of committed historical records, not
the original executed JSON. Full source locations and hashes are retained.

| File | Contents |
|---|---|
| [latest.json](latest.json) | Latest method, cohort, aggregate comparisons, qualification and limits |
| [seed_metrics.csv](seed_metrics.csv) | All 24 latest seed/arm/detector metrics and their denominators |
| [case_counts.csv](case_counts.csv) | All 384 cases across four arms: 1,536 rows with GCN/GAGA counts and topology costs |
| [previous_studies.csv](previous_studies.csv) | Earlier attempts, including negative and inconclusive outcomes |
| [audit.json](audit.json) | Original audit attribution, scope and saved verification results |
| [provenance.json](provenance.json) | Source commits, original record hashes and export provenance |

`python -m tools.check_results` checks the saved counts and summary arithmetic.
It performs no neural-model inference and is not another independent raw-logit
or tensor audit. The original separately authored audit is identified in
`audit.json` and the [historical archive](../docs/ARCHIVE.md).

Higher MR means a stronger attack. Every target occurrence remains in the
denominator, including clean errors, unchanged cases and losses. Latest results
cover 1,016 occurrences / 373 unique targets per seed, with repeated identities
across groups; the 3,048 combined occurrences are correlated.

The latest +5.4790 pp mean gain has a nominal paired 95% interval spanning zero,
so the frozen qualification failed. Random target filling is slightly higher;
no control is promoted. The formal reproduction's 43.7280% versus 40.1902%
inconclusive result uses a different cohort and is not this comparison's baseline.
