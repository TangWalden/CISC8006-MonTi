# AI Usage Record

## 2026-10-30: Public-history cleanup and independent public-clone verification

- Task and tools: The owner requested that historical Chinese records no longer be publicly retrievable while retaining the grading evidence. Codex prepared a separate English-only publication history, made the original repository a private audit archive, and published the English edition at the original public URL.
- Decision: Preserve original execution records, registered commit IDs, numerical results, failures, and the full private Git history. Omit non-English text artifacts from historical public snapshots, retain current English translations at their actual publication stage, and record equivalent public source commits. Agent use remains disclosed in English.
- Checks: The public history scan found zero Chinese text objects;17 frozen scientific file hashes were unchanged. Anonymous checks returned404 for the private archive and old raw Chinese README, while the old execution commit was not resolvable in the fresh public repository. A fresh public-bundle clone without the original execution commit passed six baseline checks, five improvement checks and the nine-pilot saved-output audit using the public commit mapping. No full improvement run was started.
- Evidence: evidence/publication_history_map.json, evidence/publication_verification.json, and evidence/publication_checks/. Original evidence remains available privately for instructor verification; deleting copies from external clones or independent caches is not guaranteed. These are agent checks, not completed team human review or instructor approval.
- Owner checkpoints: The owner requested confirmation of the reproduction conclusion, improvement direction, complete experimental protocol and final freeze. The candidate-coverage evaluation design is prepared for review; full training remains gated on the appropriate confirmation.

## 2026-10-29: English public documentation and log publication

- Task and tools: The user requested English GitHub documentation, README, and uploaded logs. The main agent and two prose-review agents translated public documents and prepared a labeled English edition of the earlier Chinese DeepSeek interaction.
- Decision: Generate subsequent experiment logs in English. Preserve original evidence in Git history and a separate server archive, recording its hash and source revision. The English API record is a translation, not a new API call or verbatim English provider output; original usage and timestamps retain their original meaning.
- Checks: The main agent reviewed numerical results, pending human/course status, and the model's erroneous budget inference, which remains visible in translation alongside the recorded correction. Public document and log text was checked for remaining Chinese characters. No scientific source or frozen configuration was changed for translation. Human translation review remains pending.
- Evidence: README.md, course/AUDIT_NOTES.md, and evidence/agent_reviews/deepseek_protocol_review.en.json. The experiment server remains at the pilot freeze while its queue is active; documentation changes are published separately before later safe synchronization.

## 2026-10-29: Improvement implementation, review, and technical pilot registration

- Task and tools: Codex implemented an isolated coverage-candidate adapter, group-size sampling, a random control with matched anchor counts, sampling-trace records, and a finite pilot queue. A scientific-design agent independently challenged the interpretation of the diagnostic observations.
- Decision: Replace the initial two-seed pilot suggestion with seeds 3/4/5. Match the random-anchor control to each group's actual greedy-anchor count. Accept the warning that low candidate adjacency coverage does not establish a defect, and revise the motivation accordingly. Preserve all negative results and predefine the rule that the better of the two ideas will still receive a full evaluation if both pilot gains are negative.
- Checks: Five tests covering CPU/GPU candidates, gradients, coverage, sampling, and the actual loader passed; all new modules compiled. The train/validation diagnosis is post-hoc and descriptive. Its script was uncommitted when executed and was subsequently archived with identical bytes and an explicit provenance note; it is not presented as a previously committed diagnostic protocol. The technical pilot separately freezes code hashes, configuration, subsets, selection rules, and GPU-hour limits before execution.
- Evidence: [Improvement implementation](course/improvements/), evidence/improvement_smoke_attempt*.log, [training diagnosis](course/TRAINING_DIAGNOSIS.md), and [diagnostic source provenance](evidence/training_diagnostic_v1/source_provenance.json).
- Human and course status: A main-agent review is not team human verification. Course approval remains unconfirmed; research pilots are not represented as course-approved pilots.

## 2026-10-29: Full baseline analysis, public evidence, and improvement design

- Task and tools: The user authorized Codex to complete reproduction, improvement work, and a public GitHub project within one week under the grading requirements. The main agent independently reread the six saved prediction outputs and recomputed MR and the paired interval. A course-review agent audited grading gaps, and a scientific-design agent reviewed candidate coverage and training weights.
- Decision: Retain the frozen claim's conclusion as inconclusive; do not add seeds in pursuit of significance. Use train/validation diagnosis to propose coverage candidates, group-size sampling, and budget-aware K. Assign balanced planned responsibilities to the two members; actual contributions require their confirmation.
- Checks: The main agent recounted 909 groups and 10,515 target occurrences per run and checked the raw-prediction SHA-256 values. All six reported results matched. The paired t interval was independently recomputed with a separate statistics implementation. Scientific source files and all frozen inputs matched their hashes. These remain agent checks; the team's human review is incomplete.
- Evidence: [Baseline audit implementation](course/audit_baseline.py), [audit results](evidence/baseline_audit.json), [published group counts](evidence/published_baseline/), [grading audit](course/GRADING_AUDIT.md), and [improvement proposal](course/IMPROVEMENT_PROPOSAL.md).

## 2026-10-28: Independent DeepSeek protocol review (analysis and criticism)

- Task and tools: The user explicitly authorized the DeepSeek API and entered the key personally. The key was stored locally with encryption and was not placed in Git, logs, or the server repository. One deepseek-flash call used 1,035 tokens. Only public method information and a protocol summary were sent; personal and SSH information were excluded.
- Decision: Accept reminders about small-sample interval limitations, unknown historical semantics, and the distinction between agent and human checks. Correct the inference that different injected content implies unequal budgets: the two arms may generate different content while obeying the same per-target-group budget quantities. The frozen experiment was unchanged.
- Checks: This was the main agent's critical review of another model, not the course-required human acceptance, modification, or rejection case. Junyi or Yifan must independently inspect the inference and record their own decision, reasons, and evidence.
- Evidence: [English review record](evidence/agent_reviews/deepseek_protocol_review.en.json) and [supplemental usage record](evidence/agent_reviews/deepseek_usage_record.md). The English record is a labeled translation, not a verbatim English API response. The original Chinese record remains traceable through its original Git commit and SHA-256, identified in the English record.

## 2026-10-28: Paper understanding and reproduction planning

- Task and tools: The user authorized Codex to read the CISC8006 Student Guide, paper, and official repository. The main agent designed a component reproduction, and a review subagent independently checked the grading requirements.
- Decision: Test one candidate-selection claim with learned and random arms and at least three seeds. Checkpoint inference cannot replace retraining. Reproduction is the first stage.
- Checks: The main agent compared Tables 4 and 5 with the source code. Instructor approval of the claim and reproduction level has not been obtained or recorded; agent checks are not described as human checks.
- Evidence: [Protocol](course/PROTOCOL.md), the paper DOI, upstream revision 10f233e, and the Chinese proposal and execution request in this chat.

## 2026-10-28: Environment, implementation, and evaluation tests

- Task and tools: Codex inspected the server through SSH, reused the existing Python 3.8 / PyTorch 1.13.1 / DGL 1.1.2 environment, and wrote an auditable runner and synthetic tests.
- Decision: Correct the reversed candidate-flag mapping. Retain the official budget algorithm while disclosing its difference from the paper's set definition. Make a failed target fail the run rather than silently removing it.
- Checks: Branch regression tests, a known-answer MR example, CPU/GPU checks, and gradient tests were used. The first GPU test failed because its test input was on the CPU; the input device was corrected and the logs were retained. Team human review remains pending.
- Evidence: The minimal change in [experiment_handler.py](modules/experiment_handler.py), [protocol tests](course/test_protocol.py), and evidence/bootstrap/smoke_attempt*.log.

## 2026-10-28: Critical review

- Task and tools: The main agent and course-review subagent examined configuration semantics and comparability with the paper.
- Decision: Replace the initial suggestion of a direct Table 5 match with a component comparison based on the official implementation with corrected flag mapping. Do not infer that the paper's historical experiments or released checkpoints used reversed arms. Retrain the two modes separately.
- Checks: This is an inter-agent review, not the required human verification. The team must independently explain the call chain and record why it accepts, changes, or rejects an agent suggestion, with evidence, before claiming that human-review requirement is complete.
- Evidence: Control definitions and limitations in [the protocol](course/PROTOCOL.md). The formal improvement-analysis stage had not yet been completed at the time of this entry.

## 2026-10-28: Real data and two feasibility pilots

- Task and tools: Codex checked official ZIP-member CRC32 values and local SHA-256 values, node and target-group partitions, and agreement between the official pretrained models and Git LFS pointers. The existing network proxy made downloads unstable; the successful downloader used a direct connection only within that script. Interrupted and failed download logs were retained.
- Decision: Retain the complete official graph and best configuration. Run each mode on 16 training groups and 8 validation groups for 2 epochs, using validation only. Preserve old runs and prohibit test evaluation programmatically.
- Checks: Both arms completed real forward and backward passes, checkpoint saving, and validation. Six CPU/GPU synthetic tests passed. The first data audit encountered a SciPy int32-index conversion issue; explicitly using torch.long resolved it, and the failed audit log was retained. Pilot MR was not used to select the configuration.
- Evidence: evidence/data, evidence/runs/*_pilot_*, and runs/pilot_*_attempt1.log. Human verification remains pending.
