# Grading gap audit

Based on the user-provided CISC8006 Student Guide dated 2026-09-02. The latest UMMoodle rules take precedence and have not yet been provided. This table does not predict a grade or claim course approval.

A factual [PDF report draft](../docs/MonTi_Report_Draft.pdf), editable source, [slide draft](../docs/MonTi_Slides_Draft.pptx) and offline demo script are prepared. The final report and course submission remain incomplete.

| Grading component | Points | Existing evidence | Remaining work |
|---|---:|---|---|
| Scope and understanding of the claim | 5 | Learned candidate selection versus an explicitly defined random control; paper Tables 4/5 and implementation differences. | Instructor approval of the Component level and specific claim. |
| Reproduction protocol and evaluation design | 10 | Protocol frozen before final runs; known-answer MR test; three paired seeds and a prespecified CI rule. | Team explanation and verification of the metric, independent replication unit, and information boundary. |
| Pilot implementation and audit readiness | 10 | Feasibility pilots for both conditions; six CPU/GPU tests; locked existing environment. | Actual live code/configuration review and feedback records. |
| Final reproduction and difference analysis | 20 | Six completed runs; successful recomputation from raw predictions; CI crosses zero, so the conclusion is inconclusive. The report draft includes direct paper comparison and limitations. | Team verification and final integration of the report and interpretation. |
| Improvement experiments and ablations | 20 | Three historical proposals; nine technical pilots and independent audit completed, with negative mean gains for coverage and size sampling. Historical full-evaluation branches are ON HOLD; the v2 direction and pilot draft await owner confirmation. | Pilots of two approved ideas; one full three-seed improvement evaluation; controls/ablations, uncertainty, and negative results. |
| Repository, report, and agent-use record | 15 | Public GitHub repository, meaningful history, configurations/commands/sources/hashes, failure records, and agent reviews across three phases. | Final report, genuine critical human checks, contribution records, and final technical tag. |
| Team presentation and demonstration | 5 | Reusable one-command check/logs, an offline demo script and an editable English slide draft with planned balanced roles. | Application check, rehearsal of demonstration/fallback, genuine role confirmation and actual presentation. |
| Individual technical defense | 10 | Question bank and explanation materials can be prepared. | Each member must complete the actual defense; an agent cannot answer on their behalf. |
| Individual contribution | 5 | A balanced work plan has been assigned to both members. | Actual completed work, evidence, and signed confirmation from both members. Planned work cannot be presented as completed contribution. |

## Issues that must remain visible

1. A positive mean difference is not statistical support. Do not add seeds until a result becomes significant.
2. The official `random_candidates` call-semantics fix has branch/gradient tests, but it does not establish that historical paper experiments or models used the reversed condition.
3. The official budget uses incidence counts, which differ from the paper's definition using the size of a neighbor set. Both conditions consistently retain and disclose the original implementation.
4. The discrepancy between Table 4 and Table 5 remains unexplained. Do not select only the reference result closest to this experiment.
5. Feasibility pilots of the learned and random conditions are not pilots of two **improvement ideas**.
6. Agent review in `AI_USAGE.md` is not the genuine human check required by the course. The team must accept, modify, or reject at least one important suggestion and explain its decision.
7. A technical delivery tag must not be represented as the common Week 12 course freeze tag. Actual course approval, freezing, and presentation follow the course schedule.
