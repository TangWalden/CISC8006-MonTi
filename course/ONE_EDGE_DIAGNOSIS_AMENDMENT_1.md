# One-edge diagnosis amendment 1: numerical equivalence check

Recorded after failed v1 and before v2 inference. The v1 plan, protocol, source and failed attempt remain intact. Source commit `8ecf14d6131bc90a191be1dfe076f1d3fdaac0b8` identifies the original adapter. This amendment changes an instrument check, not the cohort, primary metric, attack budget or method-selection policy.

## Observed failure

Attempt `evidence/one_edge_diagnosis_v1` stopped after 79 GCN forward calls and 54.42 seconds of CPU execution, at group 1285 / attacker seed 0. The same selected endpoint was rebuilt with the fixed generated feature. The maximum observed logit difference was 0.000035643577575683594, exceeding the original absolute logit check of 0.00001, while all predicted classes agreed. The attempt remains **failed and incomplete**. Completed forward measurements were flushed to ignored raw progress records. No partial oracle is presented as a completed result.

The discrepancy does not by itself establish a cause such as aggregation ordering. It shows that an absolute logit-equality guard was stricter than the observed numerical agreement. No alternative endpoints, cases, checkpoints or results are removed based on this observation.

## Revised guard and unchanged question

Use the separate `course/one_edge_diagnosis_v2.py` and `course/one_edge_diagnosis_protocol_v2.json`. Keep all 16 groups, three saved learned checkpoints, diagnostic seeds 1729/2718, exhaustive target endpoints, fixed reconstructed features, 355 expected forwards, 400-call / 600-second hard caps and CPU-only execution.

For the selected-endpoint rebuild, require exactly equal node features and equal directed edge multisets, record whether edge order also agrees, and retain **strict predicted-class equality**. Record maximum logit and correct-class-margin discrepancies instead of requiring an arbitrary absolute logit tolerance. A class disagreement or graph/feature mismatch still fails the attempt. Historical-prediction and diagnostic-repeat disagreements retain the v1 recorded-and-qualified handling.

The primary endpoint question remains the change in total misclassified occurrences with a fixed reconstructed feature. The selected endpoint is included, so the restricted best is optimistic by construction. Inspect the signed margins of predictions gained by the oracle and retain gains and losses separately. Preserve the nominal 0.00001 margin-change flags, but treat changes comparable to the measured reattachment/repeat discrepancy as numerically sensitive. Observed discrepancy maxima are empirical references, not guaranteed error bounds.

This correction was independently reviewed before v2 execution. It authorizes no new training, victim/test query, budget increase, selected improvement or full evaluation. Count both the failed attempt's 79 forwards and the complete retry in total diagnostic cost/query accounting. Raw node-level predictions/features remain remote and ignored; publish English aggregate evidence, provenance and the failure record.
