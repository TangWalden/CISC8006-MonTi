# Grading points and evidence

Mapped to the **user-provided CISC8006 Student Guide, 2 September 2026**:
85 team points and 15 individual points. The retained guide text and
[archived grading audit](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/GRADING_AUDIT.md)
support this mapping. Final UMMoodle rules and instructor approvals remain
unconfirmed. This table describes evidence availability, not an estimated grade.

| Grading component | Points | Where to find the evidence | Current gap |
|---|---:|---|---|
| Scope and understanding of the claim | 5 | [Reproduced claim and paper comparison](REPRODUCING.md#claim-and-paper-comparison), [current method](METHOD.md) | Instructor-approved claim/level and team explanation remain unverified. |
| Reproduction protocol and evaluation design | 10 | [Frozen baseline protocol](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json), [seed results](../evidence/reproduction_summary.json), [current provenance](../results/provenance.json) | Recorded protocol and uncertainty are available; human verification is pending. |
| Pilot implementation and audit readiness | 10 | [Runnable synthetic checks](../tests/), [archived pilot/run registry](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/evidence/run_registry.json), [saved audit](../results/audit.json) | Functional checks and technical pilots exist; live code/configuration review is unverified. |
| Final reproduction and difference analysis | 20 | [Six-run reproduction summary](../evidence/reproduction_summary.json), [paper/setup differences](REPRODUCING.md#claim-and-paper-comparison), [baseline audit](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/evidence/baseline_audit.json) | Reproduction is complete and inconclusive; human review and final report integration remain pending. |
| Improvement experiments and ablations | 20 | [Method and controls](METHOD.md), [all compact results](../results/README.md), [earlier ideas](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/IMPROVEMENT_PROPOSAL.md) | Controlled VALID studies exist. Two course-approved idea pilots and a full improvement evaluation against the frozen reproduction baseline are not established; the latest method has no final TEST result. |
| Repository, report and agent-use record | 15 | [Setup/check commands](REPRODUCING.md), [environment](../environment/), [traceability](../results/provenance.json), [three-phase AI record](../AI_USAGE.md), [archived report draft](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/docs/MonTi_Report_Draft.pdf) | Standalone code/count checks work. Full ML reruns require private frozen assets; final report, genuine human checks and the course freeze tag remain pending. |
| Team presentation and demonstration | 5 | [Archived slides, guide and demo](https://github.com/TangWalden/CISC8006-MonTi/tree/ca093d8b12ea153a076b02bbdbc4ede58508560c/docs) | Outdated drafts only; updated materials, application checks, rehearsal and actual presentation remain pending. |
| Individual technical defense | 10 | [Method explanation](METHOD.md), [reproduction explanation](REPRODUCING.md) | Each member must personally explain and defend the work; prepared documents do not establish completion. |
| Individual contribution | 5 | [Contribution record](../CONTRIBUTIONS.md) | Duties are planned; actual completed work and member confirmation remain unverified. |
| **Total** | **100** | **85 team + 15 individual** | **No marks are self-awarded.** |

## How to interpret the gaps

- The guide requires **three improvement ideas, pilots of two approved ideas,
  and one full fair evaluation**, with controls, uncertainty and retained failures.
  Technical execution and instructor approval are different evidence.
- A careful null or negative improvement can receive full credit under the
  guide. Our +1 pp / positive-confidence-interval rule is an **experiment-specific
  qualification rule**, not a course requirement to obtain a positive result.
- Three-phase AI records need task, tool, decision, human checks and evidence.
  [AI_USAGE.md](../AI_USAGE.md) distinguishes recorded owner direction from
  genuine team technical review; [CONTRIBUTIONS.md](../CONTRIBUTIONS.md) does
  not convert automated work into unverified member contributions.
- Final report/slides/demo, genuine human checks, personal defense and the common
  Week 12 tag/hash still need real completion. Exact deadlines and submission
  rules must follow the final UMMoodle announcement.

The concise front page and compact tables are entry points. Complete source,
configurations, exact commands, failed runs and older artifacts remain in the
[immutable archive](ARCHIVE.md); this mapping adds no experiment or approval.
