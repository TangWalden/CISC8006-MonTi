# One-week execution and acceptance plan

User request: complete one week of technical work by 2026-11-05 (Beijing time). The public repository is https://github.com/TangWalden/CISC8006-MonTi . This plan is not evidence of course approval or of the actual UMMoodle deadline.

| Date | Work and acceptance criteria |
|---|---|
| 10-29 | Create the public repository and preserve the existing history; independently recompute the six baseline outputs and paired interval; publish training logs, per-group counts, and grading gaps. |
| 10-30 | Diagnose weaknesses using training/validation data; propose three improvements; implement and test two; freeze the pilot protocol and compute budget. |
| 10-31 | Run three seeds each for the two ideas and their common pilot baseline, using validation data only; retain negative results and select one idea using the prespecified rule. |
| 11-01 | Freeze the selected idea; evaluate seeds 0/1/2 with the same full data, detectors, attack budgets, and training limits. |
| 11-02 | Run the required controls/ablations with three seeds; independently check outputs, paired statistics, costs, and failure records. |
| 11-03 | Prepare the English report, slides, demonstration, and offline backup; complete all three phases of AI_USAGE. |
| 11-04 | Junyi Tang and Yifan Fan conduct human cross-checks, confirm actual contributions, and prepare their individual defenses; freeze the technical deliverables. |
| 11-05 | Reserve time for recovery and final delivery; disclose any genuinely unfinished items and stop scheduled follow-up. |

## Scientific constraints

- Do not modify the frozen baseline or add seeds in pursuit of significance. Its current conclusion is inconclusive.
- Select improvement directions and parameters using training/validation data only. Use the test set only for preregistered final evaluations.
- Create separate improvement configurations, registrations, and freeze files; do not rewrite baseline evidence.
- The provisional additional compute limit is 12 GPU-hours: 2 for pilots, 8 for full evaluation and ablations, and 2 for diagnosis/debugging. Preserve evidence and report when the limit is reached.
- Do not stop or modify processes belonging to other projects. Runtime on the shared GPUs is descriptive only.
- Record instructor approval, the latest course rules, and genuine human checks separately. Agent checks cannot replace human checks. If the actual Week 12 freeze has already occurred, label new results as extracurricular research rather than frozen course evidence.

## Balanced division of work (planned; record actual completion separately)

| Member | Primary responsibility, approximately half of the work | Required cross-check |
|---|---|---|
| Junyi Tang | Claim, paper/configuration differences, reproduction evaluation and statistics, data/environment/log audits, reproduction report section, and baseline demonstration. | Check Yifan's improvements, ablations, and interpretation of negative results. |
| Yifan Fan | Improvement motivation and design, pilot/full-improvement/ablation checks, critical review of agent suggestions, improvement report section, and improvement demonstration. | Check Junyi's MR denominator, candidate flag, seeds, and interval. |

Both members must approve and understand the final text, explain their own responsibilities and at least one module owned by the other member, and truthfully record contribution evidence. Codex-generated work does not automatically count as either person's completed contribution.
