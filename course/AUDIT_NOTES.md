# Additional pilot audit requirements

Recorded after the pilot freeze. These checks append evidence without changing the registered v1 trials, their selection rule, or the frozen baseline.

A read-only cross-module agent audit found no scientific error requiring the nine-run queue to stop. It checked the six registered code hashes, control settings, selection rule, and train/validation information boundary. This is an agent audit; it is not human course approval.

Before freezing full improvement evaluation:

1. Independently recount each pilot's saved predictions at its selected validation epoch. Verify group count, target-occurrence denominator, error count, prediction checksum, stored MR, selected epoch, and saved checkpoint hash. Do not make new test predictions.
2. Verify each saved configuration checksum, variant and seed, target-group subset against the protocol, and registered source commit. The pilot selection analyzer alone is not this independent audit.
3. Reconcile queue events, registration files, result files, and full terminal logs. A process terminated before registration or by SIGTERM may lack `result.json`; append an interrupted/failed incident record and elapsed cost without rewriting the original evidence.
4. Maintain an aggregate GPU-hour ledger covering diagnosis, checks, failed attempts, pilots, full evaluation, and controls. The pilot queue enforces its 2 GPU-hour cap; the 12 GPU-hour total is not enforced by that queue alone. Any subsequent queue must use the remaining budget.

Fairness limitations: size-proportional sampling preserves group updates per epoch but may process more target occurrences and may change early stopping. Report actual updates, occurrences, epochs and cost. Pilot GCN validation MR is a method-selection signal, not evidence of improved GAGA test performance.

Cross-platform note: Windows checkout line endings can differ from Linux bytes. Scientific source normalized to LF matched the original hashes in the local audit; server checks use the actual frozen Linux files. Do not infer a server source change from a Windows CRLF difference.
