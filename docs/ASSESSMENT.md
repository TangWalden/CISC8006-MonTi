# Grading points and evidence

This mapping follows the user-provided **CISC8006 Student Guide, 2 September
2026**: 85 team points and 15 individual points. Upstream/downstream role
specialization produced a positive native-baseline development comparison,
but the [completed full TEST](../results/final_staged/) failed its criterion.
The [author-checkpoint local-swap study](../results/local_swap/) is also null.
The report/presentation drafts predate these results and are not final submissions.

| Component | Points | Available evidence | Remaining completion item |
|---|---:|---|---|
| Scope and understanding | 5 | [Claim and paper comparison](REPRODUCING.md#claim-and-paper-comparison), [staged method](METHOD.md) | Confirm the instructor-approved reproduction level and each member's understanding. |
| Reproduction protocol and evaluation design | 10 | [Frozen reproduction protocol](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json), [seed results](../evidence/reproduction_summary.json) | Genuine team verification of the protocol and evaluation code. |
| Pilot implementation and audit readiness | 10 | [Functional tests](../tests/), [staged preflight](../evidence/downstream_cooperation_v1/attempt3_preflight.json), [saved prediction and geometry audits](../results/downstream_cooperation/) | Complete the human live code/configuration review. |
| Final reproduction and difference analysis | 20 | [Six-run reproduction](../evidence/reproduction_summary.json), [paper/setup comparison](REPRODUCING.md#claim-and-paper-comparison) | The frozen reproduction conclusion is inconclusive; incorporate the team's verified explanation in the final submission. |
| Improvement experiments and ablations | 20 | [Completed full evaluation](../results/final_staged/), [author-checkpoint correction](../results/local_swap/), [supplementary studies](SUPPLEMENTARY_STUDIES.md) | Full evaluation exists but does not establish a reliable improvement. Instructor approval of the required idea pilots is unrecorded; current downstream iteration is development. |
| Repository, report and agent record | 15 | [Current report](STAGED_REPORT.md), [PDF](MonTi_Staged_Report.pdf), [setup and source records](REPRODUCING.md), [AI use](../AI_USAGE.md), [traceability](../results/downstream_cooperation/provenance.json) | Final human review, submission packaging and owner-confirmed course freeze tag. |
| Presentation and demonstration | 5 | [Bento and PowerPoint draft with speaking notes](SLIDES_GUIDE.md), [historical demo preparation](ARCHIVE.md) | Incorporate the now-completed negative full-test outcome; complete the final demonstration, human application checks, rehearsal and presentation. |
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

The matched evaluation was [registered and frozen](FINAL_EVALUATION.md),
executed, and recounted from saved records. It failed its declared improvement
criterion. Subsequent author-checkpoint development is explicitly exploratory;
the observed TEST set cannot become untouched confirmation again. The
guide permits instructor-approved Exact, Scaled-faithful and Component levels,
all eligible for full marks. It does not independently require the full
original dataset, the author's released checkpoint or a particular TEST phase.

The staged pilot's **+4.4950 pp** native gain did not carry into a qualified
full-test result: the latter is **+0.4089 pp**, nominal interval
**[-8.3199, 9.1378] pp**, with seed 2 negative. Full-test learning versus
random is **+0.2092 pp**, also uncertain. Complete controls, uncertainty,
negative and failed outcomes remain indexed in
[supplementary research](SUPPLEMENTARY_STUDIES.md).

## Human and submission requirements

Instructor approval, current UMMoodle rules, genuine technical reviews, actual
member contributions, rehearsal, presentation, defense and final freeze need
their own real evidence. Owner research direction and AI audits do not stand
in for them. The guide explicitly requires retaining failed and negative runs
and allows careful null or negative improvements to receive full credit.

The [retained guide and grading audit](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/GRADING_AUDIT.md)
provide the original mapping. Final UMMoodle announcements govern exact dates,
approved scope and submission details.
