# Offline Evidence Demonstration — Six-Minute Rehearsal Script

This is preparation material, not a record that either member has rehearsed or demonstrated the project. The suggested speaking roles are planned and approximately balanced. Actual contributions and human checks must be recorded separately.

**Current checkpoint:** the new A/B direction is awaiting owner confirmation. No new-direction training, full improvement evaluation, or final submission freeze is presented as complete. Recheck [project status](../course/PROJECT_STATUS.json) before using this script.

## Before the demonstration

Use an offline copy of the public repository and open the linked files in advance. The timed demonstration reads existing records and images only. It launches no training, model inference, GPU job, download, or SSH session. Raw data, model weights, keys, node-level predictions, and private captures remain outside the public demonstration.

Use a local Markdown/JSON viewer. Any Python command labeled READ ONLY below uses the standard library to display a saved file; it does not import the ML project. The separate future smoke command is not part of the offline run.

The quoted lines are a rehearsal talk track. File lists, setup instructions, and appendices are navigation notes rather than text to read aloud.

## Planned running order: 6:00 total

| Time | Planned speaker | Artifact and action |
|---|---|---|
| 0:00–0:40 | Junyi Tang | Scope, recorded environment, and recorded smoke output |
| 0:40–1:30 | Junyi Tang | Frozen six-run baseline and paired interval |
| 1:30–2:10 | Junyi Tang | Trace one result to its registered evidence |
| 2:10–2:50 | Yifan Fan | Earlier negative improvement pilots |
| 2:50–3:50 | Yifan Fan | Results-first and one-edge diagnostic evidence |
| 3:50–4:55 | Yifan Fan | Paper/source roles and crossed-device replay |
| 4:55–5:35 | Junyi Tang | Preserved failures, cost, and agent-use provenance |
| 5:35–6:00 | Both | Pending direction and genuine human responsibilities |

### 0:00–0:40 — Scope and system check

Open the [README](../README.md), [runtime record](../environment/runtime.json), and [recorded baseline smoke log](../evidence/publication_checks/baseline_smoke.log).

Suggested explanation: “We study one MonTi component under fixed settings. The recorded ML environment is Python 3.8.20, PyTorch 1.13.1+cu117, and DGL 1.1.2+cu117. This saved smoke log shows six checks passed, including the known-answer weighted MR and actual selector dispatch. We are displaying recorded output; no live check is running.”

The environment was reused. A fresh installation has not been independently verified.

### 0:40–1:30 — The frozen baseline

Show [stored results](../evidence/reproduction_summary.json), the [paired figure](figures/baseline_paired_effect.png), and [frozen protocol](../course/frozen_protocol.json).

Suggested explanation: “Six registered runs compare learned and random candidate selection, with three paired seeds. Mean attacked GAGA MR is 43.7280% versus 40.1902%. The paired difference is +3.5378 percentage points, with a paired Student t(df=2) 95% interval of [-1.5415, 8.6171]. Higher MR means a stronger attack. The interval crosses zero, so the rule registered before evaluation gives inconclusive.”

Point out that the uncertainty concerns three run seeds under fixed split/checkpoints, not independent target occurrences. Mention the corrected flag mapping and unresolved historical comparability; numerical proximity to the paper is insufficient.

### 1:30–2:10 — Trace a result

Use run 20261028T122617Z_reproduction_learned_s0_772a5d:

1. Open [registered.json](../evidence/runs/20261028T122617Z_reproduction_learned_s0_772a5d/registered.json): original execution commit, exact command, seed, mode, environment, and configuration checksum.
2. Open [config.json](../evidence/runs/20261028T122617Z_reproduction_learned_s0_772a5d/config.json) and [result.json](../evidence/runs/20261028T122617Z_reproduction_learned_s0_772a5d/result.json): resolved inputs, completed status, selected epoch, elapsed time, GPU-hours, and checkpoint checksum.
3. Open [final prediction summary](../evidence/runs/20261028T122617Z_reproduction_learned_s0_772a5d/016_victim_test_attack.json): 909 groups, 10,515 target occurrences, 4,657 incorrect, and the retained raw-prediction hash.
4. Show [terminal log](../evidence/training_logs/formal_learned_seed0_gpu0.log) and [independent count audit](../evidence/baseline_audit.json).

The public group-count hash and private raw-prediction hash identify different artifacts. [Publication mapping](../evidence/publication_history_map.json) relates original execution IDs to equivalent public source snapshots. Display the registered training command as provenance only.

### 2:10–2:50 — Negative pilots remain visible

Show [pilot results](../evidence/improvement_pilot_summary.json), [independent audit](../evidence/improvement_pilot_audit.json), and the [pilot figure](figures/pilot_validation_effects.png).

Suggested explanation: “The nine earlier pilots tested coverage and size-proportional sampling against a common pilot baseline, using seeds 3/4/5. Mean paired GCN validation gains were -0.1833 and -0.5500 percentage points. Both were negative. The old rule preferred coverage, but its follow-up is now on hold. These are surrogate-validation results, not GAGA test improvements.”

### 2:50–3:50 — Let outcomes change the question

Open [results-first diagnosis](RESULTS_FIRST_DIAGNOSIS.md) and [one-edge diagnosis](ONE_EDGE_DIAGNOSIS.md).

Suggested explanation: “Saved outcomes concentrated residual detections in low-allocation strata, but budget, degree, and structure are confounded. A bounded CPU-only one-edge probe then enumerated legal endpoints with each reconstructed feature fixed. Its optimistic oracle gained seven incorrect occurrences on that selected cohort. This is neither a learned policy nor a whole-validation improvement.”

Show [endpoint aggregates](../evidence/one_edge_diagnosis_v2/endpoints.csv) and [one-edge audit](../evidence/one_edge_diagnosis_audit_v2.json). Explain that clean-graph errors must be separated from newly induced flips. The CPU reconstruction agreed with only 17/48 historical prediction vectors, which limited historical attribution and motivated the next diagnostic.

### 3:50–4:55 — Roles and crossed replay

Open [paper/source check](../course/PAPER_EDGE_SCORE_CHECK.md), [replay report](SCORE_ROLE_REPLAY.md), and [matrix summary](../evidence/score_role_replay_v2/summary.json).

Suggested explanation: “All 96 observed calls scored edges using the first target embedding; the published formulation uses the injected embedding. The physical edge still comes from the injected node. This is a formulation/source discrepancy, not evidence that correcting it improves performance.”

Show the four matrix rows: CPU-generated attacks give 148 residuals and 17/48 historical vector matches on either detector; GPU-generated attacks give 147 and 48/48 on either detector. Prediction agreement does not establish original graph/feature identity.

Explain the limit: identical-vector top-k choices differed across devices in 32/96 controls, but natural CPU/GPU probability vectors differed in all 48 pairs; candidate sets and features also changed. A sole tie-based cause has not been established. Link the [raw-capture audit](../evidence/score_role_replay_audit_v2.json).

### 4:55–5:35 — Failures and accountability

Show the preserved [one-edge failure](../evidence/one_edge_diagnosis_v1/failure.json), [numerical amendment](../course/ONE_EDGE_DIAGNOSIS_AMENDMENT_1.md), [startup failure](../evidence/score_role_replay_v1/failure.json), and [cost ledger](../evidence/research_cost_ledger.json).

Suggested explanation: “Failed and incomplete attempts remain traceable. Repairs were separately recorded; partial results were not presented as completed experiments. The ledger includes failed-attempt and conservative diagnostic cost.”

Open [AI_USAGE.md](../AI_USAGE.md). Agent checks and translated records are disclosed; they do not establish completed human review.

### 5:35–6:00 — State the pending decision

Show [proposal v2](../course/IMPROVEMENT_PROPOSAL_V2.md), [human-review template](HUMAN_REVIEW.md), and [contributions](../CONTRIBUTIONS.md).

Suggested explanation: “Direction A concerns role alignment with controlled retraining; B concerns scarce-budget surrogate-guided endpoint priority. Both await owner confirmation and neither is a demonstrated improvement. Full evaluation and the final freeze have separate checkpoints. Our balanced speaking and work assignments are plans; actual human checks, contributions, and course approval remain to be confirmed.”

## Read-only display commands

Run these from the repository root only if a file viewer is unavailable:

~~~bash
# READ ONLY: display stored records; no ML imports or model execution
python -m json.tool environment/runtime.json
python -m json.tool evidence/reproduction_summary.json
python -m json.tool evidence/improvement_pilot_summary.json
python -m json.tool evidence/score_role_replay_v2/summary.json
~~~

## Future optional smoke check — human-run, outside the offline script

Only in the prepared matching Linux environment, a human may later run:

~~~bash
# FUTURE CPU-ONLY SMOKE; not executed by this offline demonstration
CUDA_VISIBLE_DEVICES="" bash course/run.sh -m course.test_protocol
~~~

Set MONTI_PYTHON as documented in the README if the default environment is unavailable. Record that later command's actual date, outcome, and failures. A CPU-only check does not revalidate GPU execution. Do not substitute a training or diagnostic-run command during the six-minute evidence demonstration.

## Rehearsal record — leave blank until completed

Junyi: actual rehearsal date, segment, evidence, and unresolved issues: ____________________

Yifan: actual rehearsal date, segment, evidence, and unresolved issues: ____________________

Offline copy/version used and fallback tested: ____________________

Team confirmation of factual wording and source links: ____________________

Each member must explain their work personally. This script supports a planned team demonstration; it is not live-agent assistance for assessed defense questions.
