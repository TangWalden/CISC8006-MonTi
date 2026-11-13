# Context-retrieval downstream: positive development mean

The released author MonTi checkpoint stays frozen. A contextual dual encoder
ranks eight original TRAIN-benign nodes as replacement endpoints; an independent
TRAIN-only GAGA supplies hard counterfactual supervision. Direct teacher search
is a separate candidate. Neither uses the official victim to choose an action.

Higher GAGA miss rate (MR) means a stronger attack.

| Arm | Repeat 0 | Repeat 1 | Repeat 2 | Mean |
|---|---:|---:|---:|---:|
| Native author model | 58.1436% | 58.1436% | 57.4431% | 57.9101% |
| Neural retrieval | 59.5447% | 59.1944% | 58.6690% | 59.1360% |
| Matched random, neural activation | 57.7933% | 58.6690% | 57.6182% | 58.0269% |
| Direct teacher search | 59.5447% | 59.3695% | 58.8441% | 59.2528% |
| Matched random, direct activation | 57.7933% | 58.8441% | 57.7933% | 58.1436% |

Both candidates exceed native and their own random control in mean and in
all three repeats. Neural gains **1.2259 pp** over native and **1.1092 pp**
over random; direct gains **1.3427 pp** and **1.1092 pp**, respectively.
The owner's overall-mean criterion passes. The nominal paired 95% interval
versus random is **[-0.4187, 2.6370] pp** for both: a reliable population
advantage is not established. Direct is slightly above neural; these results
do not establish neural-model superiority over direct search.

Each repeat has 64 VALID groups, 571 occurrences and 240 unique targets.
These are three noise repetitions of **one author checkpoint**, not three
independently trained author models. Development and target identities were
reused during adaptive design. The 1,713 occurrences are correlated.

## What changed in actual cases

Against native, neural creates 23 additional wrong occurrences and loses two,
net +21; direct creates 23 and loses zero, net +23. Against each matched
random control, both gain 19 and lose zero. These are occurrence counts,
not independent people or unique news items. For example, outcome-selected
repeat-0 group 1356 has six targets and gains two wrong predictions with no
loss; group 1402 has 42 targets and gains one versus native. Counterexamples
remain: neural loses one in repeat-1 group 1172 and repeat-2 group 1441.

The physical node/edge budgets, individual injected degrees, fake features,
target attachments and owned adjacency masks stay fixed. One candidate
identity changes, rerouting that column's attached edges. Context access is
broader than MonTi's original local candidates, and architecture-matched
teacher feedback changes the original GCN-only information setting.
Equal structural cost does not mean equal computation or information.

## Evidence and limits

- [Metrics, contrasts and counters](summary.json), [frozen protocol](protocol.json).
- [Saved audit](audit.json): 19,593 checks passed. It replays 192 downstream
  policy calls with three new parameter loads; zero pretrained-classifier
  calls. Root-authored, not independent implementation or human review.
- [Physical preflight](preflight.json), [890-node topology/cache check](cache_topology_check.json).
- [Execution](execution.json), [audit execution](audit_execution.json),
  [implementation](../../experiments/portal_decoder/).
- [Saved dose and transfer diagnosis](transfer_diagnosis.json): neural changes
  the physical aggregation input on 1,570 activated occurrences; it is not an
  identity-only gain. Typical logit shifts remain small. This post-hoc analysis
  makes no classifier calls.

The complete main run used 4,032 physical counterfactual builds, 576 batched
own-teacher calls covering 4,608 native/counterfactual graph contexts, 3,072
training visits and 960 official GAGA VALID calls. GPU/API/TEST use was zero.
The invalid first attempt was stopped before official scoring; its CSC-index
bug, failed preflights and costs remain recorded. No original cache changed.

The [complete TEST benchmark](../../experiments/portal_full/) keeps these models, pools, thresholds
and controls fixed. Earlier TEST outcomes have already been observed in
other studies; this cannot be called untouched statistical confirmation.
