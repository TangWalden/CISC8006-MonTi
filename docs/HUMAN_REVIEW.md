# Team Human Review and Rehearsal Template

This document assigns balanced review work to Junyi Tang and Yifan Fan. It records no completed human checks. Fill the response and evidence fields only after the named member independently performs the check. Agent-generated text, an assigned task, or a successful automated test is not evidence that a team member reviewed it.

Repository commit reviewed: ____________________

Review date and environment: ____________________

Any course approval or instruction consulted, with date and evidence: ____________________

Preparation update, 31 October 2026: the historical negative pilots are complete;
their coverage follow-up is on hold. The [outcome-based v2 proposal](../course/IMPROVEMENT_PROPOSAL_V2.md)
and [bounded related-work check](RELATED_WORK_CHECK.md) remain review material,
not an approved direction. Report, slide and demo drafts do not fill any human
response/signature field below.

## Junyi Tang's review pack

| Item | Required independent check | Response / finding | Evidence / permalink |
|---|---|---|---|
| J1: Claim and paper comparison | Read the paper and map the single component claim to Table 5. Check the separate Table 4 value and explain the unresolved difference without choosing only the closest number. Confirm the scope and approval status in the protocol. | | |
| J2: Data and environment | Check source/model versions, input checksums, split separation, edge units, environment lock, and cache scope. Identify which artifacts are inherited and which are derived. | | |
| J3: Metric known answer | Independently calculate MR for two groups of sizes 1 and 3, with one wrong occurrence in the first and none in the second. Compare with the known-answer and audited-evaluator tests. Explain why overlaps are counted as target occurrences and why averaging group rates changes the metric. | | |
| J4: Baseline result trace | Pick a baseline run and trace its registration to commit, configuration, command, seed, saved result, group counts, and prediction checksum. Recompute its score from published counts, then check the six-run summary and independent audit. | | |
| J5: Improvement peer review | Inspect the completed negative coverage/size pilots, the unselected v2 proposal and limited related-work check. Check A's source-role meaning, B's fixed features/budgets and informative versus random-mask controls. Review a full protocol only when available; keep unexecuted results and direction approval pending. | | |
| J6: Baseline/demo rehearsal | Rehearse the one-command baseline check and evidence navigation. On the authoritative environment, inspect raw-output recount requirements. Record what actually ran, what failed, and the fallback that was tested. | | |

References: [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33760), [protocol](../course/PROTOCOL.md), [data audit](../evidence/data/partition_audit.json), [input manifest](../evidence/data/input_manifest.json), [runtime lock](../environment/runtime.json), [evaluation tests](../course/test_protocol.py), [run registry](../evidence/run_registry.json), [public counts](../evidence/published_baseline/), [baseline recount](../course/audit_baseline.py), and [improvement proposal](../course/IMPROVEMENT_PROPOSAL.md).

Junyi's completed items and unresolved questions: ____________________

Junyi's actual work evidence and time spent: ____________________

Junyi's confirmation/date: ____________________

## Yifan Fan's review pack

| Item | Required independent check | Response / finding | Evidence / permalink |
|---|---|---|---|
| Y1: Actual candidate-flag call chain | Trace random_candidates through ExperimentHandler.get_perturbed_graph, the fifth MonTi.forward argument, and MonTi.select_candidates. Write the actual branch for both flag values and inspect the branch/gradient tests. Explain what this establishes and what it does not establish about historical paper runs. | | |
| Y2: Independent metric cross-check | Repeat the known-answer MR calculation independently of Junyi. Inspect AuditedExperiment.check_performance and the failed-generation test. Check that no failed group can be silently omitted from the denominator. | | |
| Y3: Genuine AI accept/change/reject decision | Inspect DeepSeek item 2 and the main-agent correction. Independently compare generated content with budget quantities using the source and run evidence. Complete the dedicated decision record below; do not copy the agent's decision as your own. | | |
| Y4: Paired interval cross-check | Recompute learned-minus-random differences for seeds 0/1/2 and the paired t interval with df=2. Explain the registered inconclusive result when the interval crosses zero, despite a positive mean. Explain why 909 groups or 10,515 occurrences are not the number of independent seed replications. | | |
| Y5: Pilot and final-evaluation boundary | Check the nine completed pilot runs/audit, subsets, seeds, negative effects and historical selection rule. Distinguish GCN validation from GAGA final-test MR, CPU seven from GPU six opportunities, and endpoint eligibility from the MR denominator. Check draft v2 checkpoint/RNG/query fairness; leave new full results and approvals pending. | | |
| Y6: Improvement/demo rehearsal | Rehearse the improvement tests and explanation of the mechanism control. Review actual sampling traces, target occurrences, epochs, costs, and failures. Record demonstrated behavior and unresolved issues without claiming an unrun final experiment. | | |

References: [candidate caller](../modules/experiment_handler.py), [MonTi implementation](../models/attack_model.py), [protocol tests](../course/test_protocol.py), [audited evaluator](../course/runner.py), [budget calculation](../modules/data_handler.py), [neighbor-count helper](../utils/utils.py), [English DeepSeek record](../evidence/agent_reviews/deepseek_protocol_review.en.json), [paired analysis](../course/analyze.py), [baseline summary](../evidence/reproduction_summary.json), [pilot protocol](../course/improvements/pilot_protocol_v1.json), [pilot runner](../course/improvements/run.py), and [additional audit requirements](../course/AUDIT_NOTES.md).

Yifan's completed items and unresolved questions: ____________________

Yifan's actual work evidence and time spent: ____________________

Yifan's confirmation/date: ____________________

## Required human decision: the DeepSeek budget inference

Case under review: DeepSeek item 2 inferred that different injected content or node sets could invalidate the equal-budget comparison. The main agent subsequently challenged that inference. The team must check the distinction between the generated attributes/connections and the budget quantities for the same target group, rather than accepting either model's wording without evidence.

The [English review record](../evidence/agent_reviews/deepseek_protocol_review.en.json) is a labeled translation. Its original-source metadata identifies the original Git record and checksum. The [supplemental usage record](../evidence/agent_reviews/deepseek_usage_record.md) documents the interaction. Review the code and actual run evidence as well as these records.

Human reviewer and date: ____________________

Exact suggestion(s) independently examined: ____________________

Source functions and specific run records inspected: ____________________

Checks actually performed and observed results: ____________________

Decision about DeepSeek item 2 (accept / change / reject): ____________________

Reason for that decision: ____________________

Decision about the main-agent correction (accept / change / reject): ____________________

Reason, including any qualification or remaining uncertainty: ____________________

Evidence links, commands, and reviewed commit: ____________________

Cross-reviewer, check performed, and date: ____________________

Entry added to AI_USAGE.md after genuine completion, with evidence link: ____________________

## Joint submission and contribution check

| Check | Junyi response / evidence | Yifan response / evidence |
|---|---|---|
| Planned versus actual contribution | | |
| A substantive cross-check of the other member's work | | |
| Open issues, failed runs, and negative results are visible | | |
| Course approval and actual deadline are evidenced or explicitly pending | | |
| Every reported result links to code/configuration/seed/command/cost/output | | |
| English translations are labeled and original evidence is traceable | | |
| Final report, slides, demo/fallback, and final tag/commit actually exist | | |

Do not fill another member's confirmation or convert the approximately 50/50 work plan into an unsupported actual contribution percentage. Update [CONTRIBUTIONS.md](../CONTRIBUTIONS.md) only with work that occurred and evidence the member can explain.

## Presentation and practice preparation

This section supports rehearsal. It is not a bank of live answers for an assessed defense.

- Junyi prepares the claim, baseline protocol, evidence trace, and reproduction uncertainty.
- Yifan prepares improvement motivation, pilot selection, mechanism controls, and agent criticism.
- Both rehearse the full argument, explain at least one cross-reviewed module, and practice responding to uncertainty or a negative result without overstating the evidence.
- Both practice opening an exact source/configuration/output record and explaining how it supports a claim.
- Rehearse baseline and improvement smoke checks, evidence-only fallback navigation, and speaking transitions.
- Record actual rehearsal results and unresolved questions below. Slides and successful demonstrations are not presumed complete.

~~~bash
bash course/run.sh -m course.test_protocol
bash course/run.sh -m course.improvements.test_policies
# On the authoritative environment with retained raw inputs and predictions:
bash course/run.sh -m course.audit_baseline
~~~

Junyi's rehearsed segment, date, and evidence: ____________________

Yifan's rehearsed segment, date, and evidence: ____________________

Offline fallback tested and location: ____________________

Unresolved practice questions and responsible member: ____________________

Final team review of the report and presentation, date/commit: ____________________

For the actual assessed defense, each member must explain their work personally. Do not use a live agent to answer assessed questions unless the instructor explicitly permits a demonstration.
