# Saved-graph canonical own-teacher diagnostic

**Status: complete. All 768 registered canonical-versus-stored-affine
comparisons passed; own-teacher classes agree in every comparison.** This is a
post-hoc implementation diagnostic after the negative
[feature-refinement study](../feature_polish/README.md), not a selected improvement
or final TEST result.

The check retained all 384 saved graphs, both baseline and GAGA-PGD endpoints,
and all 1,091 target occurrences per seed. The fixed own teacher received 768
new canonical forward evaluations on CPU. These logits were compared with its
immutable stored affine-path logits; the affine model path was not rerun.
Maximum absolute sequence error was `2.9802322387695312e-6`; maximum absolute
teacher-logit error was `1.5497207641601562e-6`. There were zero class differences,
zero parity violations and all 384 guard-alias checks passed.

The original topology/cache checks also passed: binary symmetric zero-diagonal
adjacency, binary zero-diagonal two-hop topology, selected two-hop support,
masked-self counts and unknown-token denominators on all 374 unique VALID
targets. Teacher weights, original features, feature sums and degrees have
identical before/after content hashes. These producer checks are distinct from
the saved-only audit; that audit did not rebuild sequences or independently
replay cache properties.

## Observed teacher/victim mismatch

The following are wrong target-occurrence counts out of 1,091, **baseline
attack to GAGA-PGD**, not clean-to-attacked transitions. Official-victim values
are joined from already saved predictions, with no new official-victim query.

| Seed | Canonical own teacher: baseline to GAGA-PGD | Saved official victim: baseline to GAGA-PGD |
|---|---:|---:|
| 3 | 336 to 553 | 561 to 500 |
| 4 | 328 to 526 | 550 to 492 |
| 5 | 364 to 694 | 612 to 537 |

The refinement increases own-teacher errors while reducing official-victim
errors. Measured canonical/affine agreement gives no evidence of a material
shortcut discrepancy at these endpoints. The teacher/victim response mismatch
is observed; these records do not prove it is the sole cause of failed transfer,
prove useful topology changes, or establish calibration or novelty. Repeated
target memberships, seeds and contexts are correlated.

## Execution, cost and audit

Execution used eight CPU threads, with a registered 1,800-second inner limit.
Measured wall time was 671.4419598969398 seconds, process CPU time
1,707.287493063 seconds, and outer process time 671.9515510279452 seconds.
Multithreaded process CPU time is not elapsed time. GPU charge is zero.

The diagnostic made 768 completed own-teacher forwards, 768 canonical updates,
384 hard-graph reconstructions and 384 affine sequence builds. It loaded the
registered data/cache inputs, saved perturbation tensors and frozen own-teacher
checkpoint. There were zero new generator or GCN forwards, official-victim
loads/forwards, optimizer updates, autograd calls, API calls or TEST queries.
This is not a JSON-only analysis.

Source `784b246f5c43fc4f12c3da81fb2a13cc40e1e027` and protocol SHA256
`c0b168a88f54543570f2de9cfa69d2bcd5db89dea592441b900403858a88e9d9`
were registered before execution. The [result](../../evidence/feature_parity_check_v1/result.json)
and [execution trace](../../evidence/feature_parity_check_v1/execution_trace.json)
record all endpoints, input hashes and counters. The separate
[saved-record audit](../../evidence/feature_parity_check_v1/independent_saved_audit_v1.json)
passed 53,967 checks in 8.986756472964771 wall seconds / 8.535665459 process-CPU
seconds, with zero model calls. It independently recounted saved logits,
endpoint joins and artifact hashes; this is an agent check, not human review.
Cumulative research GPU charge remains 5.065219865793663 hours plus the separate
0.02-hour setup reserve.

## Next bounded structural question

The [target-attachment study](../target_degree_balance/README.md), described as
ready/synthetic-only at this earlier checkpoint, has subsequently completed its
three-condition, three-seed CPU evaluation. Its mean gain fails the frozen
criterion: one seed is negative and matched random has a higher mean. No
candidate or control winner is selected; the stable-effect objective remains
unachieved. See that package's completed result/audit addendum. The present
parity result still establishes bounded implementation agreement, not attack
efficacy. Full TEST/tag and genuine human/course requirements remain separate;
dated report/slides are unchanged and outdated for the newer results.
