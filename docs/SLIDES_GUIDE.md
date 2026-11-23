# Current presentation: complete two-dataset results

Updated 17 November 2026. Fifteen slides: twelve main and three backup slides.

- [Download editable PowerPoint](slides/MonTi_Full_Results.pptx).
- [Download standalone Bento HTML](slides/MonTi_Full_Results.bento.html)
  (use GitHub's **Download raw file**, then open in Edge or Chrome).
- [Results and limitations](../results/channel_full/),
  [file checksums](slides/full_results_manifest.json).
- [Responsibilities in grading order](../CONTRIBUTIONS.md#responsibilities-in-grading-order).

The current deck covers the fixed author upstream, two-channel local correction,
both fair random controls, complete results, negative cases and course requirements.
Current-minus-same-class-random gains are **5.0341 pp on GossipCop** and
**4.7353 pp on YelpChi**. Higher MR means a stronger attack. Earlier studies
used the same GossipCop benchmark. The [evaluation timeline](../results/channel_full/#evaluation-order-and-case-selection)
explains when choices were fixed and when cases were reviewed. Repeats share
weights, targets recur and the teacher adds computation. The original reproduction remains inconclusive.

| Planned order | Speaker | Slides and purpose |
|---|---|---|
| 1 | Junyi Tang | 1-6: problem, motivation, method and fair controls. Prepare backup 15 on reproduction and execution. |
| 2 | Yifan Fan | 7-12: results, case gains/losses, contribution, limitations and conclusion. Prepare backup 14 on development ablations. |
| 3 | Both | Backup 13: grading evidence and missing requirements; individual technical questions. |

This is a plan, not verified contribution, rehearsal or a course timing rule.
The Bento file includes editable objects, English notes, sources and embedded
fonts, without collaboration credentials. Offline playback, text editing and
reopening a saved copy were tested. Use Save/Save a copy to keep edits.
The PowerPoint retains editable objects and notes; native PowerPoint application
checks and human review remain outstanding. Local Chinese-note copies remain
separate. The design follows the owner-supplied CaPGuard reference and Bento shell.

## Historical staged-method deck (not current results)

The remaining text describes the earlier development artifact only. Its pending
TEST wording is preserved as historical context and does not describe current status.

> The downloadable deck is a dated development draft. The now-completed
> [formal TEST](../results/final_staged/) failed its improvement criterion;
> [author-checkpoint local correction](../results/local_swap/) also has no
> native gain. The existing slides do not yet incorporate these outcomes.
>
> The latest [complete context-retrieval benchmark](../results/portal_full/)
> is positive versus native and matched random. It is also absent from this
> dated deck; use its result addendum for current evidence.

The revised English presentation follows one story: MonTi's joint construction,
a fixed upstream planner, a trained downstream candidate-edge policy, a matched
budget comparison, and the measured development result.

- [Open the editable Bento deck](https://tangwalden.github.io/CISC8006-MonTi/slides/MonTi_Staged_Role_Division.bento.html).
- [Download PowerPoint](slides/MonTi_Staged_Role_Division.pptx).
- [Download the self-contained Bento HTML](slides/MonTi_Staged_Role_Division.bento.html) or [document JSON](slides/MonTi_Staged_Role_Division.bento.json).
- [Conversion differences](slides/conversion_report.json) and [build/review record](../evidence/bento_story_v1/qa.json).

## Talk structure

There are 12 main slides and two optional linked appendices. Native graph
diagrams explain groups, original MonTi, the two construction roles and the
closest-feasible quota. Two charts show measured means and all three seed gains.
One comparison table distinguishes native MonTi, random completion and trained
completion. Every slide has an English speaking script and versioned sources.

The first six main slides are planned for Junyi Tang and the next six for
Yifan Fan. The proposed six-minute split is a rehearsal plan, not a course
timing rule or evidence of actual contribution or rehearsal.

## Results in this version

The result charts show the completed three-seed, 64-group **development VALID
pilot**. Its means are 53.6486% native MonTi, 57.0344% random completion and
58.1436% trained completion. The primary gain is +4.4950 percentage points;
the secondary advantage over random remains uncertain.

The [registered full TEST evaluation](FINAL_EVALUATION.md) is explicitly pending
on slide 11. Pilot scores are not substituted for its outcomes. The full run
uses the already selected method and controls; no new direction or ablation is
introduced by this presentation revision.

## Editing and presenting

The HTML opens as a native Bento presentation in a browser. Text, shapes, graph
connections, charts, the table and speaker notes remain separate editable
elements. Save a copy after editing. Open the slideshow control to present;
the MR example and quota appendix are reached through the labeled links.

The web link distributes an editable file copy. It is not a configured live
coediting room: changes made in separate copies do not synchronize automatically.
No collaboration credentials are included in the published files.

The PowerPoint contains 14 slides, with the two interactive appendices exported
as hidden slides. Its editable objects and notes are retained. The official
Bento converter reports that morphs become cuts, animations use their final
frame, and graph connectors become independent lines. Use the Bento version
for the original transitions and interaction behavior.

The two PowerPoint charts are native chart objects with cached data, but this
converter does not embed Excel workbooks. PowerPoint's Edit Data behavior is
unverified. The Bento chart data and the companion JSON remain editable sources.

All Bento slide bodies were visually checked, followed by the final two appendix
title revisions. Text overflow and a feedback arrow were corrected. The PPTX
package was checked separately; a PowerPoint application rendering/edit test
and human rehearsal have not been performed.

The deck uses the official [Bento Slides](https://bento.page/) app shell and
[official converter](https://bento.page/convert/). Numerical sources are pinned
to the saved pilot snapshot and frozen full-test protocol in the review record.
