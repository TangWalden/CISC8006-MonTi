# Training/validation mechanism diagnosis

This is a post hoc descriptive diagnosis, not a preregistered claim test. It uses the first registered learned seed0 checkpoint and 64 randomly sampled groups from each of the training and validation partitions, with fixed seed 1729. There is no new test inference or test stratification.

| Partition | Groups | Groups where the selector is active | Mean direct coverage of selected candidates | Mean direct coverage of full pool | Mean cosine similarity of candidate h |
|---|---:|---:|---:|---:|---:|
| train | 64 | 64 | 11.11% | 97.13% | 0.9564 |
| valid | 64 | 62 | 18.47% | 98.79% | 0.9458 |

The minimum/25th percentile/median/75th percentile/maximum group sizes are 3/4/7/14/97 for training and 3/4/6/10/51 for validation. The total variation distance between uniform-group and target-size-weighted distributions is 0.3606/0.3118, supporting an examination of training versus evaluation weights.

The coverage observation only motivates the hypothesis that a few structural anchors might provide complementary context and potential non-target connection options; it does not prove that low direct-neighbor coverage is a defect in the original method. The selected set and full pool differ greatly in size, so their coverage gap alone does not establish abnormal redundancy or poor selection. A matched-size comparison is needed for that stronger inference. MonTi already encodes every target and keeps targets as eligible endpoints independently of selected candidates. Selected candidates additionally provide Transformer context and eligible non-target endpoints for injected edges. Its original budget mechanism ensures that every injected node connects to at least one target; this must not be described as ensuring an injected edge for every target. Anchors might displace useful learned candidates and have a negative effect. High cosine similarity may arise from GCN smoothing and does not by itself prove redundancy.

The diagnostic reconstructs the model from configuration and loads its parameter state. Scheduler state is not serialized in that parameter checkpoint, so these candidate observations describe that fixed reconstruction, not a bit-for-bit replay of the original training-time selector. The diagnosis remains a limited hypothesis-generating observation.

Source code, raw per-group descriptions, the checkpoint hash, and the post hoc source-attribution explanation are in [evidence/training_diagnostic_v1](../evidence/training_diagnostic_v1). The next step is comparison under the frozen pilot rules, without changing parameters based on test results.
