# AI use

AI assistants supported paper/code review, experiment design, implementation,
execution supervision, saved-record audits and documentation. AI-written code
and agent audits are not human technical review or verified student contributions.

## Three recorded phases

| Phase | Task | Tool | Decision | Human check | Evidence |
|---|---|---|---|---|---|
| Understanding and planning | Review the paper, candidate flag and budget semantics | Codex | Define learned versus random; correct flag mapping; retain and disclose the incidence-based budget | Team technical verification pending | [Protocol][protocol], [frozen specification][frozen] |
| Implementation and reproduction | Implement checks, run the frozen baseline and recount saved outputs | Codex | Retain failed runs and use the raw-output recount; do not select results by score | Actual team live code/configuration review pending | [Run registry][runs], [baseline audit][baseline] |
| Improvement and critical analysis | Compare controls, examine failures and interpret development results | Codex and recorded DeepSeek experiments | Preserve negative controls; report the VALID gain and failed frozen CI gate without adding seeds | Team technical verification pending | [Current results](results/), [previous studies](results/previous_studies.csv), [full agent record][history] |

## Recorded experimental use

- An external DeepSeek fusion experiment made 120 physical calls. Its selected
  cases did not establish superiority over graph-only, static or neural gates.
- A later scheduling experiment made 15 physical calls; 11 decisions were applied,
  with registered fallbacks. No overall LLM scheduling advantage was established.
- The earlier edge-reallocation method uses no LLM/API call or extra training.
  Its development gain and failed stability criterion are both retained in
  [the results](results/).

## 11 November: feature assignment

At the owner's request, Codex used saved failures to propose a fixed-graph,
fixed-feature-multiset assignment and explained it before execution. The method,
four conditions, ID-only context draw and resources were frozen before new model
calls. Existing independently trained classifiers were reused; there was no
training update or experimental LLM call. Numerical checks, four TRAIN
construction cases and the complete three-seed VALID study actually ran.

A separate AI implementation of the saved-tensor auditor passes 190,857 checks;
the main result and all controls are retained. Loss improvements without any
classification improvement are reported as a failed method, not efficacy.
[Source and complete compact results](experiments/feature_assignment/).
Owner authorization and AI review do not establish student technical review,
verified contributions, instructor approval or a final course submission.

## 11 November: frozen upstream and trained downstream

The owner proposed sequential cooperation, fixed the upstream and authorized
training only a downstream candidate-edge policy. After a saved-geometry
capacity check showed literal equal quotas were often impossible, the owner
selected the nearest feasible ratio while retaining each original total budget.
Codex implemented and registered a four-arm, three-seed TRAIN/VALID pilot before
new model calls. The downstream observes actual frozen target attachments
through a mean/count summary. It is a small conditional scorer, not a second
complete MonTi. The frozen historical score-source semantics were retained.

Two zero-real-data-query startup failures were preserved and charged to the
same CPU phase. A separately frozen compatibility amendment passed 34 synthetic
tests and four TRAIN gradient cases. The completed main run made 1500 downstream
updates; upstream and classifier parameters remained fixed. All final weights
and graphs were sealed before official-GAGA development evaluation. There were
no experimental API calls, GPU work or final TEST queries.

Separately authored AI audits recounted saved logits and serialized geometry;
they added no model inference and do not establish human verification or seal
chronology. The pilot passed its registered rule and improved over native MonTi,
but the small advantage over random candidate selection remains uncertain.
Neither cooperation-information benefit nor novelty is established.
[Complete counts, controls and limits](results/downstream_cooperation/) retain
the negative seed and all controls. Human technical checks, actual student
contributions and final course authorization remain unverified.

## 12 November: matched action-summary ablation

The owner explicitly requested a downstream control without the upstream
connection decisions, holding other conditions fixed and comparing with random
connections. Codex preserved the 151872-parameter architecture, initial states,
training order, four epochs, optimizer and frozen structural budgets. The new
policy receives the whole target-list mean and the known quota average rather
than each auxiliary node's realized attachment mean/count. Shared features can
still carry indirect information. All 576 upstream captures and the completed
visible/random comparator predictions were reused through recorded byte hashes.

The scientific source and a distinct protocol were frozen before new model
work. Thirty-one synthetic checks and four real TRAIN gradient cases passed.
The new control then completed 1500 updates and 36 no-choice skips. All three
terminal weights and 192 final plans were sealed before 192 official-GAGA
development calls. No upstream regeneration, API call, GPU work, comparator
re-evaluation or final TEST query was performed.

Separate AI implementations audited saved predictions/statistics and own tensor
geometry/direct-summary algebra, adding no inference. The primary visible-minus-
blind screen failed. The blind control's positive secondary comparison with
random is retained without promoting it to a selected winner. This known-cohort
post-hoc ablation supplies no general proof that communication is useless,
first novelty, genuine human technical review or course approval.
[Complete results and limits](results/downstream_blind/).

## Critically reviewed direction decision

The owner rejected proposal-first justification and required analysis of saved
experimental outcomes before choosing another improvement. The workflow changed
to results-first diagnosis, and the historical coverage follow-up stayed on hold
([diagnosis][diagnosis], [full record][history]). This is actual research-direction
feedback, not member source-code verification or course approval. A genuine team
technical accept/change/reject check, with evidence and confirmation, remains pending.

## Publication cleanup

The 11 November 2026 cleanup extracted the existing algorithm without changing
its bytes, exported compact saved results, checked all case denominators, and
ran synthetic tests and the JSON example. It added no model inference, training,
dataset/checkpoint load, external experimental API call or GPU research.

Earlier positive, negative and failed experiments remain in the
[immutable archive](docs/ARCHIVE.md). Full raw-output review and genuine human
verification remain distinct from these automated checks.

[protocol]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/PROTOCOL.md
[frozen]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/course/frozen_protocol.json
[runs]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/evidence/run_registry.json
[baseline]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/evidence/baseline_audit.json
[history]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/AI_USAGE.md
[diagnosis]: https://github.com/TangWalden/CISC8006-MonTi/blob/ca093d8b12ea153a076b02bbdbc4ede58508560c/docs/RESULTS_FIRST_DIAGNOSIS.md
