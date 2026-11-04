# Fixed-graph injected-feature refinement

**Status: all five arms, three seeds, complete development evaluation and both saved-record audits are complete. The primary guarded method reduces GAGA attack MR in every seed. No improvement is selected.**

The preceding source-role correction did not improve aggregate development MR. Its saved online TRAIN objective was not globally weaker in the poorly transferring seed, so merely extending epochs is unsupported. The earlier joint-teacher companion also showed differing teacher/victim responses on the same saved graphs. This study preserves existing strong GCN-generator hard graphs and asks whether an independently fitted relationship model can improve only the injected features.

The [protocol](../FEATURE_POLISH_PROTOCOL_V1.json) fixes five conditions: unchanged features, GCN feature-PGD, full-TRAIN GAGA feature-PGD, terminal GAGA-PGD with a GCN prediction-preservation guard, and seeded random-direction modification. Each seed/group is generated once and all conditions use identical topology and initial injected features. Feature updates remain within the original box; extra optimization calls and actual movement are explicitly charged/reported. No victim feedback selects a proposal or step.

PGD uses 20 fixed sign-gradient steps of size 0.02 in normalized coordinates, with box projection and frozen zero-range dimensions; only the terminal candidate is used. The guard accepts the entire GAGA-PGD candidate only if every target predicted class 0 by the baseline original GCN remains class 0, otherwise the whole group returns to baseline features. No evaluation label or official-victim response enters that decision. The random route uses one fixed Rademacher direction for the same 20 steps; terminal displacement and query cost are not claimed equal to PGD.

A fresh GAGA is fitted from its independently seeded initialization for 38 fixed epochs on original TRAIN labels and corresponding full-TRAIN group inputs. It never loads the victim weights, and its post-fit TRAIN performance is in-sample. Under the stated fixed-support/cache assumptions, seven-token means are affine in injected features, allowing analytical feature derivatives without a straight-through edge/reachability derivative. Hard-update equivalence and finite differences are required before refinement.

All three successful refinement jobs finished before official GAGA development evaluation. Three attack seeds and all 128 frozen VALID groups remain. The teacher and complete paired study actually ran on CPU because shared GPUs did not meet the memory gate. Device selection occurred before graph generation; no other workload was interrupted. Historical CPU/GPU graph identity is not assumed, and actual CPU/wall time is reported separately from GPU hours.

Positive effects are a research goal, not a promised result. Prior attempts stay immutable. No final TEST, LLM API, novelty, human review, course acceptance or submission completion is implied.

The [saved TRAIN diagnosis](../../evidence/feature_polish_v1/saved_train_diagnosis.json) is post-hoc online evidence, not a frozen-checkpoint replay or a paired TRAIN/VALID gap. Source-fixed seed 3 ends with 94.0146% online TRAIN wrong versus 93.7918% legacy and lower group-mean Softplus (0.239847 versus 0.249276). This supports testing a more direct hard-graph feature optimization path; it does not prove its cause or benefit.

A development candidate must lower its own GAGA mean group loss in every seed, improve official GAGA MR by at least one percentage point on average with all three seed differences positive, and exceed GCN-PGD/random means. The guarded arm is primary; the unguarded GAGA arm is a predeclared secondary candidate and never replaces the primary result. All intervals and negative controls remain reported; independent final TEST confirmation is separate.

## Completed development results: negative

Each arm/seed retains 128 groups, 1,091 fraud-target occurrences and 374 unique targets. The same targets repeat across seeds and overlapping groups, so the occurrences are correlated. Higher MR is a stronger attack. Baselines are newly generated on this phase's selected CPU device; historical MR values are not a paired baseline.

| Condition | Seed 3 MR (%) | Seed 4 | Seed 5 | Mean | Sample SD (pp) | Mean change vs baseline (pp) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 51.4207 | 50.4125 | 56.0953 | 52.6428 | 3.0321 | 0 |
| GCN-PGD | 50.2291 | 49.8625 | 55.6370 | 51.9096 | 3.2333 | -0.7333 |
| GAGA-PGD | 45.8295 | 45.0962 | 49.2209 | 46.7156 | 2.2005 | -5.9273 |
| GAGA-PGD + GCN guard (primary) | 46.9294 | 45.3712 | 49.8625 | 47.3877 | 2.2805 | -5.2551 |
| Random direction | 48.0293 | 47.3877 | 52.6123 | 49.3431 | 2.8493 | -3.2997 |

Primary paired differences are -4.4913, -5.0412 and -6.2328 pp. The nominal paired 95% t interval is **[-7.4666, -3.0436] pp**, over three attack seeds (`df=2`), conditional on the fixed teacher, victim, generators and repeatedly inspected development cohort. It does not quantify uncertainty across fresh teachers, independent news samples or datasets. The primary fails its frozen rule and is below both controls. The secondary also fails; it is not promoted into a replacement primary. See the [complete numeric summary](../../evidence/feature_polish_v1/results_summary.json).

Clean GAGA is wrong on 398/1,091 occurrences per seed. Across 3,273 correlated attacked occurrences, baseline leaves 1,723 wrong and the guarded method leaves 1,551 wrong. It loses **175** existing new-error successes, gains **5**, and additionally corrects **2** already-clean-wrong occurrences: `5 - 175 - 2 = -172`. Its group-seed comparisons are 5 wins, 279 ties and 100 losses. Group-equal MR also falls, 50.1667% to 45.6521%, so the direction is not solely an occurrence-weighting artifact.

| Disjoint budget regime | Occurrences across seeds | Guarded net wrong-count change |
|---|---:|---:|
| One injected node, one edge | 48 | 0 |
| One injected node, multiple edges | 165 | 0 |
| Multiple injected nodes | 3,060 | -172 |

Every arm/seed uses the same summed 7,776 injected nodes and 358,491 actual nonself undirected edges across its 128 separate scenarios. These are not one combined attack graph. Equal physical budgets do not imply equal optimization cost or terminal feature movement.

## Cases and the remaining diagnostic question

The guard accepts 362 of 384 graph cases and falls back in 22. It preserves baseline original-GCN class-0 predictions, with stored GCN MR unchanged at 95.3559%, yet official-GAGA attack MR decreases. GCN prediction preservation therefore does not guarantee transfer preservation.

For continuity with earlier named illustrations, seed 4 / group 1071 has 14 targets and a 122-node / 5,734-edge budget. Its baseline gives seven GAGA-wrong targets; GCN-PGD still gives seven, random gives six, and both GAGA-PGD and guarded GAGA-PGD give four. They lose three previous successes with no new success. Seed 3 / group 1001 has 25 targets and a 14-node / 140-edge budget; all five conditions give six wrong. This is unchanged, not a repaired failure. These are illustrative cases, not representative or causal evidence; [case records](../../evidence/feature_polish_v1/paired_group_cases.json) disclose all matched group-count comparisons; underlying raw logits and capture tensors remain private.

The optimized own-GAGA mean group loss decreases from 2.3175 to 1.5911 for unguarded PGD and to 1.6292 with the guard; each seed's mean decreases. This confirms optimization of the recorded surrogate objective, not useful transfer. Random feature movement also hurts MR. These facts do not isolate teacher/victim mismatch, a feature sensitivity mechanism, a canonical/affine discrepancy or an optimal step size as the cause.

Before another method change, the next proposed measurement is own-teacher affine-versus-canonical hard-sequence parity on **all 384 saved graphs**. The four TRAIN checks do not establish that full-cohort property, and the tensor audit does not replay classifier predictions. The [isolated diagnostic package](../feature_parity_check/README.md) is prepared for a separate source/input freeze. This is a diagnostic question, not a positive result, selected new method or final-TEST authorization.

## Preserved CPU build continuation

The twelve affine CPU checks passed (0.299 seconds of test loop). The independent full-TRAIN teacher completed its fixed 38 epochs / 494 updates and 507 forwards on CPU, retaining the registered initial-state hash. Its final in-sample TRAIN accuracy is91.0538%, not held-out quality.

The first real-data preflight then failed before any generation, classifier forward, autograd or update: the inherited original-GCN loader attempted to restore CUDA storages while CUDA was deliberately hidden for CPU execution. An isolated loader subclass maps the same hash-verified original GCN checkpoint to CPU. No original scientific file or method setting changed. The [amendment](../../evidence/feature_polish_v1/build_amendment_v1.json) preserves the failure and exact receipts. The completed teacher and unchanged passed tests were reused. That continuation retained the selected CPU device, original cumulative cost records and original phase ceilings; no official-victim outcome existed at that construction checkpoint.

A second TRAIN-only preflight stopped during group238: its float32 GCN central finite difference at h=.001 differed from automatic differentiation by.00445332, exceeding the frozen tolerance.00429145. It retained two generations,30fresh-GAGA/20original-GCN forwards,8autograd calls and6canonical sequence updates, with zero updates or official-victim calls. This is a partial failed check, not two passed cases. The [second amendment](../../evidence/feature_polish_v1/build_amendment_v2.json) fixes sampled block order and uses a separately copied float64 GCN for a bounded convergence/activation-branch check. The actual float32 PGD, original tolerances and scientific comparison stay unchanged; no cause or positive performance is inferred from the failed number.

The amended four-case real preflight passed, retaining60GAGA/184GCN forwards (20actual float32 plus164isolated float64 functional forwards),24autograd calls and12canonical updates. It validates the stated fixed-graph feature path; it does not identify a unique cause for the earlier float32 difference. The first CPU refinement process then reached its original1800-second deadline after118whole-case records and119generations. Its2837GCN/2833GAGA dispatches,4724autograd dispatches,4723applied feature steps and8142.0932CPU seconds remain extra failed work. No official-victim evaluation started.

The [timer-only amendment](../../evidence/feature_polish_v1/build_amendment_v3.json) retained CPU/eight threads and original phase origin, while changing per-refinement/evaluation maxima to 5,400 seconds and total elapsed phase ceiling to six hours. The incomplete seed-3 job was retried as a whole; its partial cases are excluded from the primary matrix. Completed teacher/tests/preflight were reused. Scientific settings and the effect criterion remained unchanged. A job restart does not promise DGL native bitwise trajectory identity; every new five-arm case still shares one actual captured graph.

## Execution, audits and cost

The successful primary matrix performs 384 baseline generations, 9,216 original-GCN and 9,216 fresh-GAGA forwards, and 15,360 feature-gradient calls/steps, with no generator or classifier parameter updates. Final evaluation makes 2,048 official-GAGA forwards (128 clean plus 1,920 attacked), 128 clean-GCN forwards and 1,920 canonical sequence updates. Its attacked GCN channel reuses the exact stored refinement logits with capture/feature bindings; it is not an independent prediction replay. Official GAGA uses canonical hard-graph sequences, not the affine optimization shortcut.

Successful refinement/evaluation source is `8222dfa7bea7ba8115f31420e2a2420a93e5b5b1`, distinct from the eventual publication HEAD. The teacher and passed preflight retain their original execution sources; they were not rerun under that later source. The [execution trace](../../evidence/feature_polish_v1/execution_trace.json) records the exact source and protocol roles.

An independent agent [saved-record V3 audit](../../evidence/feature_polish_v1/independent_saved_audit_v3.json) passes 218,298 checks and 1,280 actual-byte file hashes. A separate [tensor audit](../../evidence/feature_polish_v1/independent_tensor_audit_v1.json) passes 30,400 checks and 395 file hashes, inspecting 384 captures / 1,920 feature branches for topology, bounds, aliases and random-route facts. It loads no classifier checkpoint and makes no model forward. The same separate agent authored both audit scripts and did not author the method implementation; their file sets overlap. Both report no warnings. These are agent checks, not human verification, gradient replay, novelty or generalization proof.

All model processes ran on CPU. Total phase elapsed time is 9,645.5745 seconds, including retained failures and repair gaps, below the amended six-hour ceiling; per-process CPU/wall times are separately recorded. GPU charge is zero. Cumulative additional research remains 5.065219865793663 GPU-hours plus 0.02 reserve, leaving 6.914780134206337 of 12 hours. No final TEST, experimental external LLM API call or new dependency was used. Dated PDF/PPTX drafts are unchanged and predate these results. Human/course approval, actual contributions, application review, rehearsal/defense and final submission freeze/tag remain incomplete.
