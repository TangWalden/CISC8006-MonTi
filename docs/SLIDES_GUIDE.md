# English presentation draft

The [editable PowerPoint draft](MonTi_Slides_Draft.pptx) contains seven main
slides and five optional evidence appendices. It presents the completed frozen
reproduction, negative technical pilots and exploratory GCN validation evidence.
It contains no new improvement/control result. Direction approval, full
evaluation, genuine human review and the final course freeze remain pending.

The source evidence is public commit
`9d5b62173249829063219beac8c3aaf4cf3aa45b`. The [report](REPORT_DRAFT.md)
and each slide's speaker notes supply versioned references. Codex prepared this
draft with agent reviews. The proposed speaking roles below do not establish
actual student authorship, contribution or rehearsal.

## Proposed main presentation

| Slide | Topic | Planned speaker | Proposed seconds |
|---|---|---|---:|
| 1 | Component question and draft status | Junyi Tang | 20 |
| 2 | Frozen comparison and occurrence-weighted MR | Junyi Tang | 55 |
| 3 | Baseline and paired interval | Junyi Tang | 75 |
| 4 | Earlier negative validation pilots | Junyi Tang | 30 |
| 5 | Score roles and crossed-device replay | Yifan Fan | 65 |
| 6 | Unapproved follow-up options | Yifan Fan | 65 |
| 7 | Completed evidence and remaining requirements | Yifan Fan | 50 |

This totals six minutes, with three planned minutes per member. The course
guide does not specify this duration. Final UMMoodle timing takes precedence.
The separate [offline evidence demo](DEMO_SCRIPT.md) is another preparation
option, not proof that either presentation or demo occurred. Fit any combined
talk/demo to the actual course slot rather than assuming twelve minutes.

Appendices 8-12 provide complete seed tables, residual denominators, the
restricted CPU endpoint oracle, tie/device limits and source/cost navigation.
They support evidence inspection. They are not live-agent answers to assessed
defense questions.

## Interpretation that must remain intact

- The original paired GAGA comparison remains **inconclusive**, with three
  independent run seeds and a fixed split/checkpoint scope.
- Earlier coverage and size pilots have negative mean GCN validation effects.
  Historical coverage follow-up is **ON HOLD**.
- Residual counts, occurrence exposures and endpoint configurations have
  different meanings. Budget and degree remain confounded.
- CPU's seven restricted-oracle opportunities and GPU's six eligibility
  opportunities are separate observations. Neither is a trained-method gain.
- GPU predictions agree with historical vectors in this cohort. Original
  generated features/graphs are unavailable, so this is not perturbation identity.
- Source-role alignment, endpoint priority and exact-tie controls have no
  demonstrated efficacy or novelty. Twelve synthetic checks establish functional
  preparation only. A new pilot and complete final testing need their gates.

## Editing and portability

All eight tables and three charts are native editable objects. Each chart has
an embedded workbook snapshot. Chart values use a declared maximum of eight
decimal places to satisfy Excel precision limits; original experimental
JSON/CSV values remain unchanged and are cited in the notes. Visible labels
use fewer decimal places appropriate to the slide. The workbook snapshots are
presentation data, not original spreadsheet formulas or training outputs.

The deck uses Arial and a 16:9 canvas. The bundled renderer, package checks,
chart/workbook consistency checks and first-party import are distinct from
manual testing in PowerPoint. An actual PowerPoint editing/presentation check
and human rehearsal have not been recorded. Keep the original draft and save a
copy before editing. Recheck signs, units, uncertainty and source references
after changing a chart or table.

The [builder](../course/build_slides_draft.mjs) uses the bundled Node.js
artifact-tool and isolated private build directories. It does not import ML
code, load datasets/checkpoints or launch training. Runtime paths must come
from the workspace dependency loader. Set its returned `RUNTIME_NODE_MODULES`
before launching child validators. Finalization must retain package, geometry,
font, chart/workbook and import checks, and must write a new output revision.
Validate and visually inspect the final exported deck before copying it into
this public repository. No new dependencies were downloaded for this draft.

The [build record](../evidence/slides_draft_build.json) and
[review summary](../evidence/slides_draft_qa.json) record sources, precision,
output hashes and the limits of these checks. Detailed tool receipts and
rendered previews remain in the private workspace. They are agent artifact
checks, not human technical verification.

The Windows chart-preview worker completed all twelve exact-file PNG writes,
then failed at native shutdown with `0xC0000409`. This incident is retained in
the review summary. File export/validation runs separately and exited zero;
the source does not clear failed exit codes or disable any validator. Every
completed preview was hash-checked and visually inspected. Preview generation
is opt-in rather than a required part of the default export command on this
runtime. This does not replace a real PowerPoint test or human rehearsal.

## Human rehearsal record

Leave these entries incomplete until genuine work occurs:

- Junyi's rehearsal date, actual duration, checked sources and unresolved issues:
  ____________________
- Yifan's rehearsal date, actual duration, checked sources and unresolved issues:
  ____________________
- Actual presentation application/version and chart-editing check:
  ____________________
- Offline evidence copy and failure fallback tested by the team:
  ____________________
- Actual contributions and independent technical cross-checks:
  ____________________

Use [HUMAN_REVIEW.md](HUMAN_REVIEW.md) and
[CONTRIBUTIONS.md](../CONTRIBUTIONS.md) for substantive evidence. Planned
speaking time does not replace either record.
