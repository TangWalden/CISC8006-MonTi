# Score Roles and Cross-Device Replay

**Status:** completed diagnostic measurements and independent retained-tensor recount, not a corrected model or demonstrated improvement. No training, corrected-edge evaluation, GAGA/test query, or new method selection occurred. Coverage full evaluation remains on hold.

## Setup and preserved startup failure

The [plan](../course/SCORE_ROLE_REPLAY_PLAN.md) covers all 16 one-node/one-edge validation groups and three saved learned checkpoints: 48 group/checkpoint cases. Each attacker runs on CPU and GPU 0; each captured attack is evaluated by GCN on both devices. Transfers preserve captured features and ordered edges rather than rebuilding attacks.

V1 failed before any model forward: UUID-based CUDA visibility yielded PyTorch device count zero but native-runtime count one. The [separate amendment](../course/SCORE_ROLE_REPLAY_AMENDMENT_1.md) documents metadata-only checks and v2's ordinal-zero visibility, with the process's own GPU UUID verified against physical GPU 0. Limits and scientific measurements were unchanged; [v1 records](../evidence/score_role_replay_v1/) remain preserved.

V2 ran from source 726de64 under its [frozen protocol](../course/score_role_replay_protocol_v2.json). Upstream initialization was preserved and backend settings recorded without manual tuning.

## Observed formulation/source discrepancy

All 96 captured score-source embedding indices were zero, corresponding to the first target. The physical edge still originated at the injected node: the discrepancy concerns the representation used to score that edge.

The [official paper](https://ojs.aaai.org/index.php/AAAI/article/view/33760/35915), printed page 16031, Eq. 6 and the following normalized score, uses the injected representation as source. In the [implementation](../models/attack_model.py), targets precede candidates and injected nodes, so the injected representation is at startAtk=T+C. Actual pre-Gumbel scores and target probabilities entering mandatory-target top-k were captured through unchanged-output wrappers.

This establishes a formulation/source difference, not the authors' historical intent or a performance consequence of correcting it.

## Crossed replay results

Each row below contains 48 cases and 243 target-occurrence exposures:

| Generator | GCN detector | Residuals | Historical vector agreement | Historical target disagreements |
|---|---|---:|---:|---:|
| CPU | CPU | 148 | 17/48 | 55 |
| CPU | GPU | 148 | 17/48 | 55 |
| GPU | CPU | 147 | 48/48 | 0 |
| GPU | GPU | 147 | 48/48 | 0 |

Within each captured attack, changing detector device caused no predicted-class disagreements, although maximum logit differences reached 5.5313e-5 for CPU-generated attacks and 4.0770e-5 for GPU-generated attacks. Under these conditions, the observed class difference localizes to the generator path, without establishing identical detector numerics.

Fresh CPU features, endpoints, and prediction vectors matched the earlier CPU reconstruction in all 48 cases. GPU-generated attacks recovered all historical prediction vectors. Historical generated graphs/features were not saved, so prediction concordance does not establish original perturbation identity. See [detectors.csv](../evidence/score_role_replay_v2/detectors.csv) and [paired device comparisons](../evidence/score_role_replay_v2/pairs.csv).

## Ties are observed, but no sole cause is established

[Generator captures](../evidence/score_role_replay_v2/generators.csv) contain multiple exact raw-score maxima in 58/96 cases and multiple exact target-probability maxima in 63/96. Applying CPU and GPU top-k to the same bitwise-identical captured probability vector produced different choices in 32/96 cases, all tied. The reference on the native device matched the executed choice in all 96 cases.

Naturally generated CPU/GPU probability vectors were bitwise identical in **0/48 pairs**. Thus the 32 identical-vector reference disagreements do not by themselves explain any or all 35 natural endpoint changes. Candidate sets/order also differed in 36/48 pairs, and generated-feature differences reached 0.00785291. Both inputs and execution device changed; final top-k handling is not an established sole cause.

A detached, paper-role raw-score reference on the **old weights** had exact maximum ties in 36/96 cases; its largest top-1/top-2 gap was only 6.20e-6. This calculation never selected an actual edge or received GCN evaluation. Raw-to-raw comparison is appropriate; comparing 63 soft-probability ties with 36 reference raw-score ties would mix score spaces. Fewer ties alone do not establish a successful fix.

The raw clean-endpoint join records nine GPU-generated choices already incorrect before attack: six had clean-correct alternatives, while three were all-clean-wrong group-1285 cases. All 39 new flips occurred at the selected endpoint. These observations suggest a question about scarce-budget priority; they do not establish the outcome of switching those six endpoints.

## Cost, audit, and next interpretation

The completed matrix used 224 GCN and 96 MonTi forwards, plus 192 same-vector top-k reference calls. Main elapsed time was 210.9085 seconds; guardian elapsed time was 213.5916. Conservatively charging the larger interval gives 0.059331 GPU-hours. Including 5.2767 seconds of prior startup/metadata work gives 0.0607968 GPU-hours, below the aggregate 0.1-hour reservation.

[Summary](../evidence/score_role_replay_v2/summary.json), [guardian](../evidence/score_role_replay_v2/guardian.json), [registration](../evidence/score_role_replay_v2/registered.json), [verified inputs](../evidence/score_role_replay_v2/verified_inputs.json), and [output checksums](../evidence/score_role_replay_v2/output_checksums.json) support reconciliation. Raw captures remain remote and ignored. The [independent raw-audit record](../evidence/score_role_replay_audit_v2.json), source 2048f54, passed checks of captured scores/logits and aggregate counts without inference. Backend, weight, and graph-integrity checks remain adapter/source-reviewed rather than independently regenerated. The [current audit implementation](../course/audit_score_role_replay.py) supports a fresh output path and verifies the necessarily updated cost ledger against its registered Git snapshot; model/data/source mismatches retain strict checks.

Role alignment would require controlled retraining and fair comparison; scarce-budget endpoint prioritization remains unproven. Neither was tested here. These selected-cohort observations do not establish whole-validation or victim-test gains. The [v2 proposal](../course/IMPROVEMENT_PROPOSAL_V2.md) presents those directions and their controls for owner confirmation before new training advances.
