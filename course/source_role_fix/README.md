# Injected-source scoring development control

Status: implementation and source freeze in preparation; no new outcome claimed.

The source stacks Transformer representations as targets, candidates, then injected nodes. Feature generation reads the injected tail, while the legacy edge scorer uses an injected-relative index directly in the complete projected representation. This study changes only the score source to `e[T+C+src]` and retrains consistently. It is a paper-conformance control, not a novelty claim or a proven cause of the earlier joint-teacher failure.

The owner requested an improvement grounded in concrete failure cases and actual execution. Both fresh arms use the original GCN feedback, identical 256 TRAIN / 128 VALID groups, seeds 3/4/5 and eight terminal epochs. All six models must finish before official GAGA development evaluation. Existing target overlap means the 1,091 occurrences per run are not independent news samples. No final TEST or API calls are permitted.

The original candidate selection, feature constraints, architecture, optimizer, schedules, h_GCN input, native top-k/ties and physical node/edge budgets stay fixed. The original lower-triangular edge mask permits connections to earlier injected nodes; the adapter preserves that behavior. Frozen original scientific files remain unchanged.

The [protocol](../SOURCE_ROLE_FIX_PROTOCOL_V1.json) fixes the comparison and resource ceiling before calls. An observed positive development delta is not guaranteed. Mean gain at least one percentage point and positive in all three seeds is a predeclared consistency criterion; uncertainty, negative cases, new errors/recoveries and group-equal summaries are reported regardless. No additional resources are added after outcomes to obtain a gain.

A separately bounded companion checks fresh-GAGA teacher decisions on the 96 previously saved attack graphs and joins the existing victim predictions. It does not generate new attacks or query the victim, and its eight exported groups do not represent all 128 groups. It does not select the new method's settings.

Final testing, human review, actual contributions, course approval and final submission freeze remain separate.
