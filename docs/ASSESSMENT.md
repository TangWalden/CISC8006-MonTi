# Grading points and evidence

This mapping follows the user-provided **CISC8006 Student Guide, 2 September
2026**: 85 team points and 15 individual points. Upstream/downstream role
specialization produced a positive native-baseline development comparison,
but the [completed full TEST](../results/final_staged/) failed its criterion.
The author-checkpoint local-swap study was null. The latest
[context-retrieval complete benchmark](../results/portal_full/) exceeds native
and matched random by +1.1095/+0.8084 pp, respectively, on the fixed original
TEST cohort. Earlier TEST exposure and conditional repeats limit inference.
Report/presentation drafts predate these latest results and are not final submissions.

| Component | Points | Available evidence | Remaining completion item |
|---|---:|---|---|
| Scope and understanding | 5 | [Claim and paper comparison](REPRODUCING.md#claim-and-paper-comparison), [staged method](METHOD.md) | Confirm the instructor-approved reproduction level and each member's understanding. |
| Reproduction protocol and evaluation design | 10 | [Frozen reproduction protocol](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json), [seed results](../evidence/reproduction_summary.json) | Genuine team verification of the protocol and evaluation code. |
| Pilot implementation and audit readiness | 10 | [Functional tests](../tests/), [retrieval preflight and 19,593-check development audit](../results/portal_decoder/), [full saved audit](../results/portal_full/audit.json) | Complete the human live code/configuration review. Automated audits are root-authored, not independent implementation/human review. |
| Final reproduction and difference analysis | 20 | [Six-run reproduction](../evidence/reproduction_summary.json), [paper/setup comparison](REPRODUCING.md#claim-and-paper-comparison) | The frozen reproduction conclusion is inconclusive; incorporate the team's verified explanation in the final submission. |
| Improvement experiments and ablations | 20 | [Complete retrieval benchmark and matched controls](../results/portal_full/), [earlier negative full study](../results/final_staged/), [all attempts](SUPPLEMENTARY_STUDIES.md) | Positive controlled means are measured; instructor approval of required pilots, scope and interpretation is unrecorded. No untouched-confirmation claim. |
| Repository, report and agent record | 15 | [Latest result addendum](../results/portal_full/), [dated report draft](STAGED_REPORT.md), [PDF](MonTi_Staged_Report.pdf), [setup/source](REPRODUCING.md), [AI use](../AI_USAGE.md) | Incorporate the latest addendum into final report/slides; human review, packaging and owner-confirmed freeze tag. |
| Presentation and demonstration | 5 | [Bento and PowerPoint draft with speaking notes](SLIDES_GUIDE.md), [historical demo preparation](ARCHIVE.md) | Incorporate positive and negative complete studies; final demonstration, human application checks, rehearsal and presentation. |
| Individual technical defense | 10 | [Method explanation](METHOD.md), [report](STAGED_REPORT.md) | Each member must personally explain and defend the work. |
| Individual contribution | 5 | [Contribution record](../CONTRIBUTIONS.md) | Members confirm actual work and supporting evidence. |
| **Total** | **100** | **85 team + 15 individual** | **Evidence availability is not a self-awarded mark.** |

## What experiment work remains

The guide requires at least three improvement ideas, pilots of two approved
ideas, and a full evaluation of one idea against the frozen reproduction
baseline. Multiple technical ideas, pilots and controls already exist. The
historical staged pilot uses seeds 3/4/5, its own terminal-eight upstream family
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

The later context-retrieval study keeps one released author checkpoint and
reuses three 128-TRAIN-group downstream fits on **all 909 TEST groups**. Neural
MR is 45.3733%, native 44.2638%, matched random 44.5649%. The nominal paired
random-contrast interval is [0.5617, 1.0550] pp across three conditional repeats.
Direct search is slightly higher than neural. This is one measured complete-
cohort improvement with matched controls, not completion of every paper dataset
or every course requirement. It does not replace the earlier reproduction or
erase negative results. Real instructor approval and individual work remain
separate requirements.

## Human and submission requirements

Instructor approval, current UMMoodle rules, genuine technical reviews, actual
member contributions, rehearsal, presentation, defense and final freeze need
their own real evidence. Owner research direction and AI audits do not stand
in for them. The guide explicitly requires retaining failed and negative runs
and allows careful null or negative improvements to receive full credit.

The [retained guide and grading audit](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/GRADING_AUDIT.md)
provide the original mapping. Final UMMoodle announcements govern exact dates,
approved scope and submission details.
