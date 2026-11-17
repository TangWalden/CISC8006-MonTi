# Contributions

On 2026-10-29, the user named Junyi Tang and Yifan Fan as team members and requested a balanced division of work. The table below is a plan. Each member must confirm actual work and supporting evidence. Automated work must not be recorded as personal work that has not occurred.

| Member | Planned primary responsibilities, approximately 50% | Actual completion and human verification |
|---|---|---|
| Junyi Tang | Paper claim and implementation differences; baseline evaluation and statistics; data/environment audit; reproduction section of the report; baseline demonstration | To be completed by the member, with links to commits or review evidence |
| Yifan Fan | Improvement motivation; review of pilots, full improvement evaluation, and ablations; critical assessment of AI suggestions; improvement section of the report; improvement demonstration | To be completed by the member, with links to commits or review evidence |

Cross-checks: Junyi reviews the improvement mechanism and ablations. Yifan reviews the candidate flag, MR denominator, and paired interval. Both members must explain the main method and their own work and speak during the presentation. A balanced assignment does not establish an actual 50% contribution by each member.

Codex assistance includes paper/code review, remote-environment inspection, protocol drafting, the candidate-flag correction, synthetic tests, the runner, and evidence registration. Team members remain responsible for reviewing and explaining all submitted code and conclusions.

Human verification and peer evidence review are incomplete. Instructor approval of the reproduction level and resources has not been recorded. Current results are summarized in [results/](results/); historical execution records are indexed in the [archive](docs/ARCHIVE.md).

## Responsibilities in grading order

Updated 17 November 2026, following the [assessment mapping](docs/ASSESSMENT.md).
These are planned responsibilities, not awarded marks or verified actual work.

| Order | Assessment component | Points | Lead | Concrete responsibility |
|---|---|---:|---|---|
| 1 | Scope and understanding | 5 | Junyi Tang | Explain original claim, graph/target groups and reproduction scope; verify course approval. Yifan cross-checks. |
| 2 | Reproduction protocol and evaluation | 10 | Junyi Tang | Explain splits, checkpoint/seed settings, MR denominator and protocol. Yifan checks fairness. |
| 3 | Pilot implementation and audit readiness | 10 | Yifan Fan | Walk through downstream code, tests, configurations and retained failures. Junyi performs a peer walkthrough. |
| 4 | Final reproduction and differences | 20 | Junyi Tang | Explain six-run results, uncertainty, paper/setup differences and retained inconclusive conclusion. Yifan reviews interpretation. |
| 5 | Improvement experiments and ablations | 20 | Yifan Fan | Explain two-channel retrieval, both matched random controls, two-dataset results, ablations and negative cases. Junyi checks sources and counts. |
| 6 | Repository, report and agent record | 15 | Both | Junyi leads reproduction/setup sections and source traceability; Yifan leads method/results sections and AI-use review. Both verify the final report and package. |
| 7 | Presentation and demonstration | 5 | Both | Junyi presents slides 1-6; Yifan presents 7-12. Rehearse and demonstrate the recorded workflow. |
| 8 | Individual technical defense | 10 | Each member | Personally explain the full pipeline, own reviewed code and limitations; prepare questions for the other member. |
| 9 | Individual contribution | 5 | Each member | Record actual actions, dates and evidence; distinguish AI-generated work from personal review and implementation. |

Team components total 85 points; individual components total 15. Shared
components are not split into invented official per-person marks. Junyi prepares
backup slide 15, Yifan backup slide 14, and both prepare slide 13. Actual
completion/sign-off remains pending until each person provides evidence.
