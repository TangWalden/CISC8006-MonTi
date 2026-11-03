# Failure-driven alignment and external-agent follow-up

**Executed on 3 November 2026.** All 18 fresh trainings completed. This is an
exploratory GCN-validation study, not final GAGA/test evaluation, proof of novelty,
or an overall successful improvement. The owner explicitly requested concrete
failure analysis, changes, and execution. Historical attempts remain unchanged.

## Concrete motivation

The [saved failure analysis](../../evidence/alignment_agent_v2/failure_analysis.json)
compares the previous OFF and q90-GATED trainings at their separately selected
checkpoints. Entries below count fraud targets still classified correctly after
attack; more remaining correct targets means a weaker attack.

| Group | Targets | OFF, seeds 3/4/5 | Old GATED, seeds 3/4/5 |
|---|---:|---|---|
| 957 | 42 | 0 / 0 / 1 | 5 / 3 / 5 |
| 1198 | 12 | 1 / 3 / 0 | 5 / 4 / 4 |
| 1328 | 14 | 3 / 3 / 1 | 4 / 5 / 4 |
| 1249, mixed counterexample | 3 | 1 / 1 / 1 | 0 / 2 / 0 |

Distance reductions did not ensure stronger attacks. A zero validation gate
does not undo earlier training. These are separately trained comparisons, not
same-weight interventions or clean-to-attacked transitions. Margin extrema in
the saved analysis are ranges over changed subsets, not paired endpoint values.

The separately frozen [TRAIN gradient diagnostic](../../evidence/alignment_agent_v2/gradient_summary.json)
made 32 generations and 32 GCN forwards on 16 predetermined TRAIN groups and two
old seed-3 checkpoints, without optimizer updates. Hinge gradients were zero in
32/32 cases; Softplus gradients were nonzero in 32/32. At the OFF checkpoint,
11/16 complete-gradient comparisons conflicted; the remaining five still had a
feature-head conflict. The median weighted alignment/Softplus gradient-norm
ratio was 9,141.36. This supports a local saturation/conflict hypothesis, not a
causal reconstruction of all historical training failures.

## Frozen changes and controls

[Scientific plan](../ALIGNMENT_AGENT_V2_PROTOCOL.json) and
[exact execution protocol](../../evidence/alignment_agent_v2/execution_protocol_v2.json)
were frozen before new training/API calls. Execution source is
`5a1d9e57a99541006c6d749adda9602217575421`; protocol SHA256 is
`1f7098358dd6f625642100de0242ccc25ae62389f75cef94ec9e0d777efa5ee3`.

Six arms each used seeds 3/4/5, eight fixed epochs, batch size one, the original
256 TRAIN/128 validation groups, and the terminal epoch only. There was no
interim validation, early stopping, or best-checkpoint search. First-epoch
alignment was OFF for every arm. Architecture, candidates, original score role,
tie policy, physical budgets and original TRAIN benign moment reference stayed
fixed. The cohort has 1,091 target occurrences but only 374 unique targets;
717 occurrences repeat nodes in different group contexts.

1. `legacy_hinge_off`: original hinge objective, no alignment.
2. `smooth_off`: Softplus attack objective, no alignment.
3. `smooth_threshold`: Softplus plus the original nodewise q90 penalty after warmup.
4. `smooth_rule`: same penalty with per-parameter-block conflict projection and
   norm capping, always enabled after warmup.
5. `smooth_agent`: an external DeepSeek agent chooses epoch-level ON/OFF from
   lagged TRAIN statistics; ON uses the identical protected update.
6. `smooth_random`: the same-seed agent's actually applied ON flags are shuffled
   among epochs 2-8. This matches enabled epochs, not node activations or dose.

Protection removes the opposing alignment-gradient component and caps the
remaining norm at the attack-gradient norm in each fixed parameter block.
Near-zero attack gradients suppress that block's alignment contribution. Adam
and weight decay remain unchanged. This is not an exact orthogonality claim or
a guarantee of a beneficial Adam step; projection and capping are one combined
control, not separately identified inventions.

The agent receives numeric TRAIN margins, losses, distance/outlier summaries,
eight fixed TRAIN gradient probes and prior actions. It receives no validation
outcomes, node IDs, article text, victim feedback or test labels. This is online
API scheduling, not LLM fine-tuning or a learned local neural action policy.

## Complete results

Higher MR means more fraud targets misclassified as benign: a stronger attack.
Each entry averages the same three seeds and all 128 groups/1,091 occurrences.
The primary endpoint is MR after the fixed isolation filter. Distance is the
directly optimized moment statistic, not independent camouflage evidence.

| Arm | Raw MR, % | Filtered MR, % | Group-equal distance | Outside fraction |
|---|---:|---:|---:|---:|
| Original hinge, OFF | 96.6697 | 93.2783 | 42.8173 | 0.9534 |
| Smooth, OFF | 96.7919 | 93.6144 | 39.5811 | 0.9143 |
| Smooth, threshold | 94.7449 | 91.0480 | 0.4571 | 0.1176 |
| Smooth, protected rule | 96.5170 | 93.2478 | 34.1499 | 0.9312 |
| Smooth, hybrid agent | 96.8225 | 93.6450 | 32.2863 | 0.9344 |
| Smooth, matched random | 96.4253 | 93.0339 | 26.5927 | 0.9322 |

[All jobs](../../evidence/alignment_agent_v2/results/jobs.csv),
[all paired seed differences](../../evidence/alignment_agent_v2/results/paired_seed_differences.csv),
[prediction-vector comparisons](../../evidence/alignment_agent_v2/results/prediction_vector_identity.csv),
[all eight preselected cases](../../evidence/alignment_agent_v2/results/preselected_cases.csv),
[actual actions](../../evidence/alignment_agent_v2/results/applied_epoch_actions.csv),
and the [full saved summary](../../evidence/alignment_agent_v2/results/summary.json)
retain positive and negative observations.

- Unprotected alignment lowers filtered MR relative to smooth-OFF by
  2.5665 percentage points; every seed is negative.
- Protection recovers 2.1998 points relative to unprotected alignment; every
  seed is positive, but the optimized distance rises from 0.4571 to 34.1499.
  It mitigates the tradeoff rather than achieving both goals.
- Hybrid-agent minus smooth-OFF is only +0.030553 points in both mean raw and
  filtered MR: a net one additional wrong prediction over 3,273 occurrences.
  Many individual predictions differ; this is not a one-prediction identity.
- Agent minus protected-rule filtered differences are +0.2750 / -0.8249 /
  +1.7415 points. There is no consistent three-seed advantage.
- Agent minus matched-random filtered mean is +0.6111 points. Seeds 3 and 4
  have identical all-ON schedules in agent, rule and random, yet different
  trained outputs despite matching initial hashes/noise keys. Those differences
  cannot be attributed to agent-selected timing. Only seed 5 has a nontrivial
  timing contrast: agent OFF at epoch 3, random OFF at epoch 6.

Group 957 illustrates mixed effects within the new common harness: raw/unfiltered
residual correct counts are 2/0/4 for legacy-OFF, 2/0/2 for smooth-OFF, and 0/0/5 for
hybrid-agent. The third seed gets worse. Historical best-checkpoint scores above
are motivation, not paired controls for this new terminal-eight experiment.

## Filter validity and clean harm

The unoptimized filter uses an unlabeled original-TRAIN 80/20 fit/calibration
split, robust raw-feature/degree descriptors and a fixed 99th-percentile cutoff.
It scores all current nodes without origin or injection IDs, then isolates
flagged nonself connections. Its descriptors differ from the optimized moments
but are not statistically independent. All targets remain in evaluation.

**Every run flags zero of its 7,776 injected-node instances.** Filtered MR is
therefore not evidence that it detected the injected nodes. On the original
clean graph it flags 113 original nodes. Across the fixed target occurrences,
clean MR changes from 498/1,091 to 461/1,091: 68 recoveries and 31 new errors.
Each attacked condition is compared with its matching raw/filtered clean
baseline; existing filter errors are not counted as new attack gains.

On the separately fixed 64-benign/64-fraud clean cohort, benign false positives
increase 18 to 19 while correctly detected fraud increases 46 to 47. Accuracy
remains 92/128. This weak proxy cannot establish a camouflage benefit or adaptive
robustness. The recorded moment distance is not full distribution equality.

## Agent fidelity and failures

The [saved API audit](../../evidence/alignment_agent_v2/api_audit_attempt2.json)
checks 861 conditions. There are 21 decision slots but 15 physical calls, all
HTTP 200/stop with zero HTTP retries: 9,255 prompt and 704 completion tokens.
The cache-miss peak-price upper bound is USD 0.0036213, not a provider invoice.
The old 120-call fusion phase remains separate and closed.

Only 11 decisions were actually applied from the LLM; ten used the registered
fallback. Per seed the LLM/fallback counts are 1/6, 4/3 and 6/1. Applied ON counts
are 7/7/6. Three contract-valid answers were recorded but not applied after the
server's timeout/invalid-response handling. Six slots made no physical API call.

The original metadata bridge exited on an SSH publication failure. Recovery 1
also exited after metadata retries; recovery 2 completed. All failures, cached
answers, late/nonapplied responses and actual decisions remain immutable. The
recovery reuses the same ledger, cache, prompt and caps, retries metadata only,
and never replaces an already applied action. No training was repeated.

One additional interface defect is explicit: an otherwise valid Boolean OFF
answer had a 314-character rationale, exceeding the frozen validator's 240
limit. The prompt requested a short rationale without stating that numerical
limit. Fallback ON was applied; this request also had nonapplication/latency
conditions, so length alone is not established as the sole cause. Its proposed
OFF is counterfactual, not an accepted action or measured training outcome.
This is a contract-design limitation, not evidence that the LLM's action was bad.
The frozen acceptance rule was not relaxed after seeing the answer.

## Audit, costs and decision

The separate alignment auditor verifies all 18 saved outputs and 232 artifact
hashes, including 18 models, 18 optimizers and 144 perturbation exports, with no
missing bytes. The [original failed audit](../../evidence/alignment_agent_v2/independent_audit_attempt1.json)
and [passed audit with warnings](../../evidence/alignment_agent_v2/independent_audit_attempt2.json)
are both retained. An [explicit numerical amendment](../../evidence/alignment_agent_v2/numerical_materiality_amendment_v2.json)
classifies ten scalar float64 cancellation residuals; worst negative dot is
7.52e-18, not material opposition, and no cap violation was found. Science and
raw outputs did not change. The API auditor separately preserved its own first
temporal-predicate failure and [audit-only amendment](../../evidence/alignment_agent_v2/api_audit_logic_amendment_v2.json).
The amendment's preserved private CRLF bytes and equivalent public LF bytes
have an explicit [publication hash map](../../evidence/alignment_agent_v2/publication_line_ending_receipt_v1.json).
One extra manifest mismatch was caught before push and retained; scientific
outputs did not change. API audit references still identify the private original.
The alignment auditor is separate from the experiment author; the API auditor
also authored the bridge. Neither is human verification.

Real-data training totals are 144 epochs, 36,864 optimizer updates, 39,168
generations and 46,116 GCN forwards. The separate diagnostic adds 32 generations,
32 GCN forwards and 128 autograd calls, with no updates. Each run's 128 separate
attack scenarios sum to 7,776 injected nodes and 358,491 actual nonself undirected
edges; these are not one combined graph. Forty CPU functional tests passed;
the flow test includes 256 SGD updates of one toy scalar, separately from the
real-data totals. Earlier CPU setup/test failures remain recorded.

Whole-process charged GPU time including diagnostic and API waits is
**1.3617947545 hours**, below the frozen 2-hour phase ceiling. The cumulative
[research ledger](../../evidence/research_cost_ledger.json) is 2.9261111188 hours
plus 0.02 reserve, leaving 9.0538888812 of 12 hours. Equal update/physical budgets
do not imply equal computation or API cost. Shared-GPU times are descriptive.

**Recommendation:** do not select an overall winner or market agent superiority.
The strongest observed contrast is recovery from an alignment penalty's harm,
while most of its distance reduction is lost. A future protocol should first
separate action acceptance from optional rationale formatting, make transport
reliable, and validate an informative defense with acceptable clean utility on
TRAIN/development data. Any clean agent study needs meaningful action variation,
matched controls and a learned-policy comparator before claiming LLM superiority.
Do not tune a filter or expand seeds after these outcomes to obtain a win.

No final GAGA/test study, adaptive complete-LLM-defense study, instructor/course
approval, human review/contribution/rehearsal or final submission/tag is implied.
Dated report/PPTX artifacts and frozen baseline/previous scientific attempts stay
unchanged. Further final testing and final freeze retain separate checkpoints.
