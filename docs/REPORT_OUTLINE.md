# Final Report Outline and Evidence Map

This is a report structure and completion checklist. A factual [ten-page PDF draft](MonTi_Report_Draft.pdf) and [editable source](REPORT_DRAFT.md) are prepared and visually checked; the final report and course submission remain incomplete. Full improvement/control results, human review, course approval, presentation materials, and the final delivery freeze remain to be completed or confirmed. The latest UMMoodle instructions govern the actual course submission format and deadline.

## Front matter and abstract

- Current draft title: **MonTi Component Reproduction and Validation Diagnostics**. A final title must reflect the actually completed scope.
- Team: Junyi Tang and Yifan Fan. Distinguish planned responsibilities from verified completed contributions.
- Reproduction level: Component; instructor approval has not been confirmed.
- Technical delivery target: 2026-11-05, as requested by the user. This is separate from the unprovided common Week 12 course deadline.
- Abstract: state the single claim, the fixed setting, the baseline result and uncertainty, the selected improvement and control once available, and the principal limitations. Do not claim an improvement before its full GAGA test evaluation.
- Final report version, delivery tag, commit, course approval evidence, and member confirmation: **[pending]**.

## 1. Claim, paper understanding, and scope — 5 points

State the claim: on the official GossipCop-S split 0, with fixed official GCN surrogate and GAGA victim checkpoints and identical per-group injection budgets, learned candidate selection increases target-occurrence-weighted misclassification rate relative to the defined random selector.

Explain why this is an important component and describe the roles of target groups, exploration candidates, the learned selector, injected nodes, and the surrogate-to-victim transfer. Higher MR indicates a stronger attack in this study.

Map the claim to the paper:

| Reference | Learned/MonTi MR | Random-candidate MR | Difference |
|---|---:|---:|---:|
| Paper Table 5, GAGA column | 43.70% | 39.13% | 4.57 percentage points |
| Paper Table 4, corresponding MonTi entry | 41.34 +/- 1.90% | Not the same reported comparison | Unresolved Table 4/5 discrepancy |

Do not imply identical historical training conditions or select only the paper result closest to this experiment.

Evidence: [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33760), [upstream code](https://github.com/bdi-lab/MonTi), [reproduction protocol](../course/PROTOCOL.md), and [frozen baseline protocol](../course/frozen_protocol.json).

## 2. Reproduction protocol and evaluation design — 10 points

Describe the settings frozen before the six final baseline runs:

- Complete official graph, split 0, fixed pretrained detectors; rho=0.05, xi=0.1, K=128.
- Separately retrained learned/random attackers; paired seeds 0/1/2; official maximum 100 epochs and patience=10.
- Random candidates sampled uniformly without replacement at every training and evaluation forward; small pools retained in full.
- MR = total incorrect target occurrences across groups / total target occurrences. Overlaps count once per occurrence within each group, rather than once per unique graph node.
- Percent for MR; percentage points for differences. Report mean and sample standard deviation separately from confidence intervals.
- Independent random replication unit: a run seed, not an individual target occurrence.
- Registered decision rule: paired t interval with df=2; lower bound above zero supports the claim, upper bound at or below zero does not support it, otherwise the conclusion is inconclusive.
- Fixed data/model scope, test information boundary, compute limit, failure handling, and source/input/cache checksums.

Include the known-answer evaluation test: groups of sizes 1 and 3, with one incorrect occurrence in the first group and none in the second, yield MR=25%, rather than the unweighted mean of group error rates, 50%.

Evidence: [baseline configuration](../course/configs/baseline.json), [protocol freeze](../course/frozen_protocol.json), [known-answer and branch tests](../course/test_protocol.py), [audited evaluator](../course/runner.py), [analysis code](../course/analyze.py), [input manifest](../evidence/data/input_manifest.json), and [partition audit](../evidence/data/partition_audit.json).

## 3. Pilot implementation and audit readiness — 10 points

Explain the distinction between the original feasibility pilots and the later improvement pilots.

- The two baseline feasibility pilots used 16 training groups, 8 validation groups, and 2 epochs; their purpose was execution/resource checking, not scientific confirmation.
- The baseline synthetic tests check the actual candidate branch, gradients, known-answer MR, budget behavior, CPU/GPU behavior, and refusal to silently drop failed targets.
- Explain the candidate-flag correction through the actual call chain, not just a configuration label.
- Disclose the incidence-count versus unique-neighbor budget difference and retained official budget implementation.
- Explain fixed-cache provenance and the risk of reusing a cache across different splits/checkpoints.
- Describe environment locking, the one-command smoke check, failed checks, corrective changes, and their retained logs.
- Actual live human code/configuration review and course approval evidence: **[pending]**.

Evidence: [remote execution](../course/README.md), [test logs](../evidence/bootstrap/), [candidate caller](../modules/experiment_handler.py), [attack model](../models/attack_model.py), [budget implementation](../modules/data_handler.py), [budget helper](../utils/utils.py), and [locked environment](../environment/runtime.json).

## 4. Final reproduction and difference analysis — 20 points

Use the completed frozen baseline, with all three seeds shown:

| Seed | Learned MR (%) | Random MR (%) | Learned minus random (pp) |
|---|---:|---:|---:|
| 0 | 44.2891 | 39.0870 | 5.2021 |
| 1 | 42.8816 | 41.6262 | 1.2553 |
| 2 | 44.0133 | 39.8573 | 4.1560 |
| Mean +/- sample SD | 43.7280 +/- 0.7459 | 40.1902 +/- 1.3019 | 3.5378 +/- 2.0447 |

The paired mean difference is 3.5378 pp, with a paired 95% interval of [-1.5415, 8.6171] pp. Under the rule frozen before evaluation, the conclusion is **inconclusive**. A positive average and three positive observed differences do not override that rule. Do not add seeds until significance is reached.

Report the clean MR from the registered outputs, attacked MR, costs, and peak memory with evidence. Explain the independent recount of all six saved prediction outputs: each contains 909 test groups and 10,515 target occurrences.

Discuss each material difference:

1. Corrected flag mapping; no inference that historical paper experiments or released checkpoints used reversed conditions.
2. Explicitly defined random resampling semantics; historical Table 5 training/inference details remain unconfirmed.
3. Retained incidence-based budget algorithm and its difference from the paper's neighbor-set definition.
4. Unresolved Table 4/5 discrepancy.
5. Three-seed uncertainty conditional on a fixed split and pretrained detectors.
6. Shared-GPU runtime is descriptive and does not reproduce an efficiency claim.

Evidence: [complete baseline summary](../evidence/reproduction_summary.json), [registered runs](../evidence/run_registry.json), [independent recount](../evidence/baseline_audit.json), [published per-group counts](../evidence/published_baseline/), and [training logs](../evidence/training_logs/).

## 5. Improvements, controls, and ablations — 20 points

Current selection update: the owner required actual-outcome analysis before choosing a new intervention. The historical coverage follow-up is ON HOLD. Report the negative pilots and original selection rule truthfully, then explain the subsequent research decision separately. [Saved-outcome diagnosis](RESULTS_FIRST_DIAGNOSIS.md) and [reconstructed one-edge diagnosis](ONE_EDGE_DIAGNOSIS.md) are exploratory evidence, not final improvement results. The latter preserves the 79-call failure, reviewed numerical-check amendment, complete 355-call retry, independent raw-logit recount, historical prediction disagreement and zero-query clean-endpoint attribution. All seven restricted endpoint opportunities used a clean-GCN-incorrect target, but this establishes neither an improved trained method nor victim transfer. The role finding and replay concordance are addressed in the completed crossed-device study below; no new improvement direction has been confirmed. Do not present the restricted oracle's 2.8807pp selected-subset headroom as a full-validation or test gain.

Further evidence is now available in the [crossed-device study](SCORE_ROLE_REPLAY.md): actual runtime score roles differ from the published injected-source role in all 96 captures; GPU generation restores 48/48 historical prediction concordance, whereas fixed-graph detector-device changes produce no class differences. Preserve the distinction between observed tied probabilities, identical-vector top-k controls and naturally different generator probability vectors. Candidate sets also differ, and the old-weight paper-role reference remains often tied, so no correction performance is established. The [v2 method proposal](../course/IMPROVEMENT_PROPOSAL_V2.md) and [pilot draft](../course/outcome_v2/PILOT_PROTOCOL_DRAFT.md) await owner confirmation. The pilot remains DRAFT / NOT APPROVED / NOT FROZEN / NOT EXECUTED; no A/B direction has been selected. Original source and frozen baseline results remain intact.

### Historical motivation and three proposals

Explain structural-coverage candidates, sampling by group size, and budget-aware candidate count. Use the train/validation diagnosis as descriptive motivation. Low direct-neighbor coverage does not prove a defect; high cosine similarity does not by itself establish redundancy. Explain possible negative effects as well as hoped-for benefits.

Evidence: [three proposals](../course/IMPROVEMENT_PROPOSAL.md), [training diagnosis](../course/TRAINING_DIAGNOSIS.md), [diagnostic provenance](../evidence/training_diagnostic_v1/source_provenance.json), and [implementation tests](../course/improvements/test_policies.py).

### Two technical improvement pilots

Describe the nine registered runs: common pilot baseline, coverage, and size sampling; seeds 3/4/5; fixed 256 training groups and 128 validation groups; maximum 10 epochs and patience=5. Report every run and preserve failures or negative gains.

Use only GCN validation MR for the registered selection rule. **GCN pilot-validation MR is not GAGA test performance and is not evidence of a final improvement.** These small-subset pilots must not be placed in a final GAGA-test result column.

- All nine technical pilot runs and the independent selected-epoch audit are complete. Mean paired GCN validation gains were **-0.1833 pp for coverage** and **-0.5500 pp for size sampling**. The frozen rule selected coverage; its full follow-up is **ON HOLD**, rather than an active approved direction.
- Course approval of the two ideas: **[not confirmed; attach genuine evidence if obtained]**.
- Do not describe technical research pilots as approved course pilots without confirmation.

Evidence: [frozen pilot protocol](../course/improvements/pilot_protocol_v1.json), [completed pilot summary](../evidence/improvement_pilot_summary.json), [independent pilot audit](../evidence/improvement_pilot_audit.json), [pilot runner](../course/improvements/run.py), [selection analyzer](../course/improvements/analyze_pilots.py), and [additional evidence checks](../course/AUDIT_NOTES.md).

### Historical full-evaluation branches — ON HOLD

The branches below retain the earlier plan for historical context; neither full evaluation was executed. The active v2 direction and pilot draft remain pending owner confirmation. Any future full evaluation needs a separately confirmed and frozen protocol before new final test evaluation, with a fair comparison to the frozen baseline.

- Historical coverage branch, ON HOLD: compare against random anchors matching each group's actual greedy-anchor count, plus the learned complement. The frozen learned baseline represents removal of the added mechanism.
- Historical size-sampling branch, ON HOLD: compare against uniform sampling with replacement, separating size weighting from replacement. Record actual group updates, processed target occurrences, epochs, and cost; equal group updates do not imply identical computation.

The following full-evaluation table remains an unexecuted placeholder; no new improvement or mechanism control has been selected.

| Full GAGA-test condition | Seed 0 MR | Seed 1 MR | Seed 2 MR | Mean +/- sample SD | Paired gain vs frozen learned baseline and 95% CI | Evidence |
|---|---|---|---|---|---|---|
| Frozen learned baseline | See Section 4 | See Section 4 | See Section 4 | 43.7280 +/- 0.7459% | Reference | Existing baseline registry |
| Selected improvement | [pending] | [pending] | [pending] | [pending] | [pending] | [pending registered runs] |
| Mechanism control | [pending] | [pending] | [pending] | [pending] | [pending] | [pending registered runs] |

Interpret the full results and control contrast together. Report uncertainty and limitations even if the improvement is null or negative. Do not substitute pilot scores for missing final results.

## 6. Repository, report, and agent-use record — 15 points

Provide an evidence map from every reported result to its commit, resolved configuration, exact command, seed, runtime, GPU-hour cost, saved output, and checksum. Reconcile registrations, queue events, failures/interrupted attempts, and logs rather than treating a missing result file as success.

Document sources, versions, licenses, preparation, environment lock, one-command check, raw-output retention, and public English publication policy. English translations of earlier agent records must be explicitly labeled and linked to original Git/hash provenance.

Discuss important agent interactions in understanding/planning, implementation/testing, and improvement/analysis. Include a genuine team accept/change/reject decision after independent checking; an AI-to-AI critique is not that human decision.

- Human review records: **[pending completion of the review template]**.
- Actual contribution evidence: **[pending member entries]**.
- Final report, slides, demonstration instructions, delivery tag, and final commit: **[pending]**.
- Do not call a technical delivery tag the common Week 12 freeze tag without course confirmation.

Evidence: [AI usage](../AI_USAGE.md), [contributions](../CONTRIBUTIONS.md), [English DeepSeek record](../evidence/agent_reviews/deepseek_protocol_review.en.json), [grading gaps](../course/GRADING_AUDIT.md), and [human-review template](HUMAN_REVIEW.md).

## 7. Team presentation and demonstration — 5 points

Prepare a short presentation covering the claim, audited implementation, baseline uncertainty, improvement/control findings, failures, and limitations. Assign substantive speaking sections to both members and rehearse the transition between them.

Demonstration preparation:

1. Run the baseline one-command smoke check in the documented environment.
2. Run the improvement policy tests.
3. Open a registered baseline run, its configuration, published group counts, and summary; show how the score is traced to evidence.
4. On the authoritative environment with retained raw files, rehearse the independent baseline recount.
5. Add the selected full improvement/control evidence only after completion and audit.
6. Prepare an offline evidence copy and an honest failure fallback.

Slides, rehearsal, demonstration materials, and actual presentation: **[pending]**.

## 8. Individual defense and contribution — 10 + 5 individual points

Each member prepares to explain the main method, their actual work, at least one cross-checked module, metric denominator, random-control semantics, seed uncertainty, implementation differences, and negative results. Use the review checklist for practice, not as agent-generated answers during an assessed defense.

Each member records actual work and evidence independently. Balanced planned responsibilities are not proof of equal completed contribution. Human checks, signatures, actual defense, and contribution confirmation remain **[pending]**.

## Appendix checklist

- [ ] Claim-to-paper mapping and current course approval evidence.
- [ ] Baseline protocol/configuration/environment/input hashes and freeze chronology.
- [ ] Known-answer, branch, gradient, budget, and failure-handling checks.
- [ ] Complete baseline run table and independent raw-output audit.
- [ ] Three improvement ideas, two-pilot approval status, all pilot results, and independent pilot audit.
- [ ] Full improvement/control protocol, registered outputs, uncertainty, and negative/failure analysis.
- [ ] Aggregate additional GPU-hour ledger and shared-hardware limitation.
- [ ] Three-phase AI record, labeled translations, genuine human decision, and actual contributions.
- [ ] Final report, slides, demo/fallback instructions, and final tag/commit.
- [ ] Team confirmation that every submitted claim and result is supported by linked evidence.
