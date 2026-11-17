# Grading points and evidence

This mapping follows the user-provided **CISC8006 Student Guide, 2 September
2026**: 85 team points and 15 individual points. The latest frozen
[complete two-dataset comparison](../results/channel_full/) passed its
predeclared mean criterion. Current-minus-native gains are
+4.9295 pp on GossipCop and +6.7860 pp on YelpChi;
current-minus-same-class-random gains are +5.0341/+4.7353 pp.
All original TEST groups and all three conditional repeats were retained.
The earlier reproduction, nulls and failures are not replaced. Earlier GossipCop
TEST exposure limits confirmation. The [current slides](SLIDES_GUIDE.md) include
the latest complete results. The dated report still needs the result addendum;
both artifacts require genuine human review before submission.

| Component | Points | Available evidence | Remaining completion item |
|---|---:|---|---|
| Scope and understanding | 5 | [Claim and paper comparison](REPRODUCING.md#claim-and-paper-comparison), [current downstream design](../experiments/channel_retrieval/) | Confirm the instructor-approved reproduction level and each member's understanding. |
| Reproduction protocol and evaluation design | 10 | [Frozen reproduction protocol](https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json), [seed results](../evidence/reproduction_summary.json) | Genuine team verification of the protocol and evaluation code. |
| Pilot implementation and audit readiness | 10 | [Historical functional tests](../tests/), [current preflight and 192,372-check saved audit](../results/channel_full/), [development ablations](../results/channel_retrieval/) | Complete the human live code/configuration review. Automated audits are root-authored, not independent implementation/human review. |
| Final reproduction and difference analysis | 20 | [Six-run reproduction](../evidence/reproduction_summary.json), [paper/setup comparison](REPRODUCING.md#claim-and-paper-comparison) | The frozen reproduction conclusion is inconclusive; incorporate the team's verified explanation in the final submission. |
| Improvement experiments and ablations | 20 | [Frozen complete two-dataset study](../results/channel_full/), [development ablations](../results/channel_retrieval/), [earlier complete benchmark](../results/portal_full/), [all attempts](SUPPLEMENTARY_STUDIES.md) | The new method is fixed and completely evaluated; instructor approval of required pilots/scope remains unrecorded. No untouched-confirmation claim. |
| Repository, report and agent record | 15 | [Latest complete result addendum](../results/channel_full/), [dated report draft](STAGED_REPORT.md), [PDF](MonTi_Staged_Report.pdf), [setup/source](REPRODUCING.md), [AI use](../AI_USAGE.md) | Incorporate the latest addendum into the final report; human review, packaging and owner-confirmed freeze tag. |
| Presentation and demonstration | 5 | [Current Bento and PowerPoint with speaking notes](SLIDES_GUIDE.md), [historical demo preparation](ARCHIVE.md) | Final demonstration, human application checks, rehearsal and presentation. |
| Individual technical defense | 10 | [Method explanation](METHOD.md), [report](STAGED_REPORT.md) | Each member must personally explain and defend the work. |
| Individual contribution | 5 | [Contribution record](../CONTRIBUTIONS.md) | Members confirm actual work and supporting evidence. |
| **Total** | **100** | **85 team + 15 individual** | **Evidence availability is not a self-awarded mark.** |

## What experiment work remains

The guide requires at least three improvement ideas, pilots of two approved
ideas, and a full evaluation of one idea against the frozen reproduction
baseline. Multiple technical ideas, pilots and controls already exist. The
historical staged pilot uses seeds 3/4/5, its own terminal-eight upstream family
and 64 validation groups. The earlier formal reproduction uses seeds 0/1/2
and a different evaluation cohort. That historical staged pilot is therefore not the
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
