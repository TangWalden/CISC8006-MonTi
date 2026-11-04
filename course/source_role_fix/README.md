# Injected-source scoring development control

**Status: all six trainings, complete development evaluation, saved-graph teacher measurement and saved-record audits completed. No consistent development gain was found; no improved method is selected.**

The source stacks Transformer representations as targets, candidates, then injected nodes. Feature generation reads the injected tail, while the legacy edge scorer uses an injected-relative index directly in the complete projected representation. This study changes only the score source to `e[T+C+src]` and retrains consistently. It is a paper-conformance control, not a novelty claim or a proven cause of the earlier joint-teacher failure.

The owner requested an improvement grounded in concrete failure cases and actual execution. Both fresh arms use the original GCN feedback, identical 256 TRAIN / 128 VALID groups, seeds 3/4/5 and eight terminal epochs. All six models must finish before official GAGA development evaluation. Existing target overlap means the 1,091 occurrences per run are not independent news samples. No final TEST or API calls are permitted.

The original candidate selection, feature constraints, architecture, optimizer, schedules, h_GCN input, native top-k/ties and physical node/edge budgets stay fixed. The original lower-triangular edge mask permits connections to earlier injected nodes; the adapter preserves that behavior. Frozen original scientific files remain unchanged.

The [protocol](../SOURCE_ROLE_FIX_PROTOCOL_V1.json) fixes the comparison and resource ceiling before calls. An observed positive development delta is not guaranteed. Mean gain at least one percentage point and positive in all three seeds is a predeclared consistency criterion; uncertainty, negative cases, new errors/recoveries and group-equal summaries are reported regardless. No additional resources are added after outcomes to obtain a gain.

A separately bounded companion checks fresh-GAGA teacher decisions on the 96 previously saved attack graphs and joins the existing victim predictions. It does not generate new attacks or query the victim, and its eight exported groups do not represent all 128 groups. It does not select the new method's settings.

Final testing, human review, actual contributions, course approval and final submission freeze remain separate.

## Completed development comparison: no consistent gain

Both arms completed seeds 3/4/5, eight fixed terminal epochs and 2,048 updates per run before the original official GAGA was loaded for evaluation. Each run retains the same 128 groups, 1,091 fraud-target occurrences and 374 unique targets. Higher attack MR is a stronger attack. These repeatedly inspected development data are not the earlier formal test cohort.

| Scoring source | Seed 3 GAGA MR (%) | Seed 4 | Seed 5 | Mean | Sample SD (pp) |
|---|---:|---:|---:|---:|---:|
| Legacy `e[src]` | 51.8790 | 50.9624 | 55.2704 | 52.7039 | 2.2694 |
| Injected `e[T+C+src]` | 48.5793 | 55.1787 | 54.2621 | 52.6734 | 3.5751 |
| Paired difference (pp) | -3.2997 | +4.2163 | -1.0082 | -0.0306 | |

The correction fails the frozen criterion of at least +1 pp mean gain with positive differences in every seed. Its paired 95% t interval is [-9.6000,+9.5389] pp (n=3, df=2), conditional on fixed models and this cohort. The wide interval establishes neither improvement nor equivalence or harm. The positive seed is retained alongside both negative seeds and is not selected as a winner.

Across 3,273 correlated occurrences, legacy leaves 1,725 wrong and the corrected arm leaves 1,724 wrong. Of the original attack successes, 441 are shared, 90 are lost and 89 new ones are gained. Neither arm recovers already-clean-wrong predictions in this comparison. The one-count deficit is therefore 89-90=-1. There are 59 group-seed wins, 268 ties and 57 losses. Group-equal GAGA MR changes 50.2637% to 50.5357%; this small descriptive direction change does not replace the registered primary metric or establish a winner.

| Disjoint budget regime | Occurrences | Net wrong-count change | Difference (pp) |
|---|---:|---:|---:|
| One injected node, one edge | 48 | -2 | -4.1667 |
| One injected node, multiple edges | 165 | +1 | +0.6061 |
| Multiple injected nodes | 3,060 | 0 | 0.0000 |

Zero net change in the multi-node regime is not graph or prediction identity: 87 successes are gained and 87 lost. Every run's 128 separate scenarios sum 7,776 injected nodes and 358,491 actual nonself undirected edges, matching its registered budgets. These are not one combined attack graph.

There are concrete successes and failures. The post-hoc largest-gain example (tie-break by seed, then group ID), seed 4/group 1071, has 14 targets and the same 122-node/5,734-edge caps under both arms: wrong predictions increase 7 to 11, gaining four successes. The largest-loss example, seed 3/group 1001, has 25 targets and 14-node/140-edge caps: wrong predictions fall 6 to 2, losing four successes. These are outcome-selected illustrations, not representative samples, causal proof or a gate-selection rule. The [case receipt](../../evidence/source_role_fix_v1/case_examples.json) discloses selection, and all 384 group-seed pairs remain privately retained.

## What the new failure says

The correction is not enough to raise aggregate development performance under the fixed training budget. Its original-GCN validation MR also varies: the mean changes 95.9059% to 91.2313%, with seed 3 dropping 96.9753% to 84.5096%. Thus the result cannot be explained only by transfer to another victim. These observations do not prove optimization instability, inadequate epochs, a sole scoring cause or a required tuning fix. The isolated source correction is retained as a conformance control; it does not replace the unchanged baseline or explain the earlier dual-surrogate loss.

The separately frozen companion reconstructed all 96 old exported attack graphs (four old arms, three seeds, eight predetermined groups) and made 104 fresh-GAGA teacher forwards: eight clean and 96 attacked. The official-victim channel uses only already-saved logits, with zero new generator or victim calls. In the old joint arm's 261 correlated target occurrences, the fresh teacher has 47 new errors while the saved official victim has only 2. Across all old arms there are 1,044 correlated occurrences: teacher new errors 111 versus saved-victim 24. This is selected-subset evidence of differing teacher/victim responses, not full 128-group coverage or proof that weights, masks, overfit or approximate gradients individually caused it. No new setting was selected from the companion results.

## Execution, checks and cost

Scientific execution source is `dad856c9d2697d887bc6eeea5dd26fab432fe682`; the preparation and experiment protocol SHA256 values are respectively `b8246361e67cc20cc5d889988271bd03c3ad6e5408d84d45b7d2be1e08a8f8ff` and `e9028d0370714089e6139c932b0fc84f5aa6b72ca6d45a15b4f7495cae9b1d4e`. The [execution trace](../../evidence/source_role_fix_v1/execution_trace.json), [results](../../evidence/source_role_fix_v1/results.json) and [saved summary](../../evidence/source_role_fix_v1/saved_summary.json) retain the comparison.

Twelve CPU synthetic methods passed, including native legacy forward/backward parity, the T=2/C=1/B=2 known answer, gradients, masks and budgets. Four real TRAIN cases then passed with four GCN forwards/four autograd calls and zero updates or official-victim calls. All six primary trainings completed on their first attempt: 48 epochs/12,288 updates/12,288 generations/12,288 GCN forwards. Development evaluation adds 768 generations and 896 GCN+896 official-GAGA forwards (including 128 clean forwards per classifier). The companion adds 104 fresh-GAGA forwards and no generation/update/victim calls. No new dependencies or experimental LLM API calls occurred; no final TEST was queried.

A separate agent recounted all 48 TRAIN logs, all clean/attacked raw logits and all counters/budgets/checkpoints: 117,125 checks and 405 actual-byte hashes passed with no warnings. This auditor did not author the source-role model/training/evaluation/queue, but did author the companion diagnostic. A different agent independently recounted the companion's saved records: 6,533 checks/118 files; that auditor authored the source-role model. The [phase audit](../../evidence/source_role_fix_v1/independent_phase_audit.json) and [companion audit](../../evidence/source_role_fix_v1/independent_companion_audit.json) disclose these boundaries. These are agent saved-record checks, not human verification or fresh model replay. Tensor graph topology for the main matrix is not independently reconstructed by the saved-only audit.

The nine serial model processes charge the maximum outer/guardian lifetime each: 0.8056387150377526 GPU-hours, below the frozen 2-hour ceiling. Cumulative additional research is 5.065219865793663 GPU-hours plus 0.02 reserve, leaving 6.914780134206337 of 12 hours. CPU-only checks/audits are recorded separately. No retry, extra seed, extension or outcome-driven tuning was performed. Raw checkpoints, exports, full predictions and immutable metadata remain in private ignored storage.

Dated PDF/PPTX drafts are unchanged and predate these results. Genuine human/course checks, complete owner-confirmed final TEST and final submission freeze/tag remain outstanding. No new study starts merely because a heartbeat fires.
