# Counterfactual context-retrieval downstream

Feature residual attention failed: mean GAGA MR 57.3263% versus native
57.9101% and matched random 57.9685%. Own-teacher margin shifts averaged
about -0.04 while final-victim shifts were often opposite; typical victim
boundary distance was about 1.86. Larger network size did not solve transfer.

This new decoder keeps the author checkpoint, all injected features and the
owned adjacency matrix. It replaces one used candidate **identity** by a
retrieved original TRAIN-benign node outside every target's original one/two-hop
neighborhood. Thus new context can reach a known-benign aggregation token,
rather than only changing unknown injected-feature tokens. Actual injected
degrees, target connections, candidate-slot count and node/edge budgets stay
fixed. Context access is broader than the original local candidate policy;
same structural budget is not equal information access or computation.

The donor is chosen geometrically, before outcomes, with preference for targets
with little known-benign two-hop context. Eight remote nodes form a fixed pool.
Random and learned controls share that pool and donor. A contextual dual encoder
learns listwise rankings from hard counterfactual losses of the independent
TRAIN-only GAGA teacher. A direct teacher-search decoder is a separate candidate.
Each candidate has its own activation-matched random control. Identity actions
and infeasible cases remain in the denominator.

The first proposed affine token shortcut failed its canonical preflight
(sequence discrepancy 0.0009402, logit discrepancy 0.0002037). That failure
is retained. A TRAIN-only diagnosis traced the failed premise to CSC indices
being read as CSR in the new retrieval code; the checked cached entries
agree with actual graph paths. The corrected filter converts row slices
to CSR and has an explicit CSR/CSC regression test. The new implementation constructs and scores each full physical
counterfactual graph, without using the shortcut or relaxing its tolerance.
Training labels come from the
attacker's known original TRAIN partition; no VALID/test truth or official
victim answers select a portal. The corrected development study completed:
neural 59.1360%, native 57.9101%, matched random 58.0269%; direct search
59.2528% versus its random 58.1436%. Both overall-mean criteria pass, while
the nominal random-contrast interval includes zero. See the
[complete results and limitations](../../results/portal_decoder/).
This is not established first novelty or complete-cohort efficacy.
Earlier negative attempts remain intact.

## Active files

| File | Purpose |
|---|---|
| `model.py` | Contextual query/key/change scorer |
| `geometry_hard.py` | Corrected CSR/CSC-safe eligibility and endpoint replacement |
| `run.py` | Physical counterfactuals, listwise training, seals and evaluation |
| `sparse_checks.py`, `preflight_hard_v3.py` | Format regression and real-graph eligibility/budget checks |

`geometry.py`, `preflight.py` and `preflight_hard_v2.py` are retained failed or
incomplete construction prototypes. The active run uses the corrected files
above, not the affine shortcut. All original data and caches remain unchanged.
