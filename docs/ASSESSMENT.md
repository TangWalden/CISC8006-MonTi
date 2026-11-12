# Grading points and evidence

This mapping follows the user-provided **CISC8006 Student Guide, 2 September
2026**: 85 team points and 15 individual points. The main improvement is
upstream/downstream role specialization, with a positive registered validation
comparison against the reproduced native MonTi baseline. The
[current report](STAGED_REPORT.md) explains that result and its controls.

| Component | Points | Available evidence | Remaining completion item |
|---|---:|---|---|
| Scope and understanding | 5 | [Claim and paper comparison](REPRODUCING.md#claim-and-paper-comparison), [staged method](METHOD.md) | Confirm the instructor-approved reproduction level and each member's understanding. |
| Reproduction protocol and evaluation design | 10 | [Frozen reproduction protocol](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json), [seed results](../evidence/reproduction_summary.json) | Genuine team verification of the protocol and evaluation code. |
| Pilot implementation and audit readiness | 10 | [Functional tests](../tests/), [staged preflight](../evidence/downstream_cooperation_v1/attempt3_preflight.json), [saved prediction and geometry audits](../results/downstream_cooperation/) | Complete the human live code/configuration review. |
| Final reproduction and difference analysis | 20 | [Six-run reproduction](../evidence/reproduction_summary.json), [paper/setup comparison](REPRODUCING.md#claim-and-paper-comparison) | The frozen reproduction conclusion is inconclusive; incorporate the team's verified explanation in the final submission. |
| Improvement experiments and ablations | 20 | [Staged scheme and controls](METHOD.md), [three-seed results](../results/downstream_cooperation/), [supplementary studies](SUPPLEMENTARY_STUDIES.md) | Complete or confirm the full fair evaluation of the chosen idea against the frozen reproduction baseline within the approved scope. Instructor approval of the required idea pilots is unrecorded. |
| Repository, report and agent record | 15 | [Current report](STAGED_REPORT.md), [PDF](MonTi_Staged_Report.pdf), [setup and source records](REPRODUCING.md), [AI use](../AI_USAGE.md), [traceability](../results/downstream_cooperation/provenance.json) | Final human review, submission packaging and owner-confirmed course freeze tag. |
| Presentation and demonstration | 5 | [Historical slides and demo preparation](ARCHIVE.md) | Align presentation materials with the staged method, check the application, rehearse and present. |
| Individual technical defense | 10 | [Method explanation](METHOD.md), [report](STAGED_REPORT.md) | Each member must personally explain and defend the work. |
| Individual contribution | 5 | [Contribution record](../CONTRIBUTIONS.md) | Members confirm actual work and supporting evidence. |
| **Total** | **100** | **85 team + 15 individual** | **Evidence availability is not a self-awarded mark.** |

## What experiment work remains

The guide requires at least three improvement ideas, pilots of two approved
ideas, and a full evaluation of one idea against the frozen reproduction
baseline. Multiple technical ideas, pilots and controls already exist. The
current staged pilot uses seeds 3/4/5, its own terminal-eight upstream family
and 64 validation groups. The earlier formal reproduction uses seeds 0/1/2
and a different evaluation cohort. The current pilot is therefore not the
registered final matched evaluation of that frozen reproduction baseline.

The remaining experimental item is to lock that matched evaluation's scope,
baseline, controls, resources and decision rule, then complete it. It does
not require inventing another method or adding more mechanism studies. The
guide permits instructor-approved Exact, Scaled-faithful and Component levels,
all eligible for full marks. It does not independently require the full
original dataset, the author's released checkpoint or a particular TEST phase.

The staged pilot's primary mean gain is **+4.4950 pp**, with all three seed
gains positive and nominal paired interval **[0.5715, 8.4185] pp**. The
information-channel analysis is supplementary and does not replace this primary
question. Complete controls, uncertainty, negative and failed outcomes remain
indexed in [supplementary research](SUPPLEMENTARY_STUDIES.md).

## Human and submission requirements

Instructor approval, current UMMoodle rules, genuine technical reviews, actual
member contributions, rehearsal, presentation, defense and final freeze need
their own real evidence. Owner research direction and AI audits do not stand
in for them. The guide explicitly requires retaining failed and negative runs
and allows careful null or negative improvements to receive full credit.

The [retained guide and grading audit](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/GRADING_AUDIT.md)
provide the original mapping. Final UMMoodle announcements govern exact dates,
approved scope and submission details.
