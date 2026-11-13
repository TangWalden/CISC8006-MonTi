# Fixed-upstream staged attack construction

> Historical staged-method description. The current author-checkpoint
> extension is [context retrieval](../experiments/portal_decoder/), with a
> [completed full-cohort comparison](../results/portal_full/). The older
> staged full evaluation was negative; the body below retains that method's
> original scope and development results.

The selected extension separates a MonTi attack into target-facing upstream
planning and downstream selection of candidate connections. The upstream is
frozen while a small specialized downstream is trained. The measured claim is
that this complete staged pipeline improves official-GAGA development miss
rate over our reproduced native MonTi under matched total structural budgets.

## Construction in two roles

1. Reuse our three terminal-eight legacy MonTi checkpoints for seeds 3, 4 and 5.
   Cache each case's generated auxiliary features, original target list,
   candidate list and physical node order. Preserve the documented historical
   scoring-source semantics; no silent source-role correction is introduced.
2. Give the fixed upstream the target-connection quota. It selects those target
   edges with its frozen scoring function, retaining at least one target edge
   per auxiliary node. This chosen target block remains fixed thereafter.
3. Train a separate downstream to score auxiliary-to-candidate pairs using the
   frozen upstream output and permitted original-graph candidate context. It
   cannot change features, target attachments, candidate identities or the
   registered role quotas. Hard selection fills exactly the remaining edge
   quota without duplicates or artificial self-loops.

The downstream has 151,872 parameters and hidden width 64. It is a specialized
candidate-edge scorer, rather than a second complete MonTi. Candidate inputs
include original features and cached representations from the fixed GCN.
Detailed interfaces and input validation are recorded in the
[implementation and execution specification](../experiments/downstream_cooperation/README.md).

This division provides a concrete experimental model of sequential specialized
roles. It is not evidence that real fraud groups have this organization or that
the frozen upstream and downstream jointly adapt during training.

## Closest-feasible allocation and structural cost

Let `B` be the number of auxiliary nodes, `T` the number of original targets,
`C` the number of available original candidates and `E` the original total new
undirected-edge budget. The requested target quota is `ceil(E/2)`. Project it
into the feasible integer interval

```text
lower = max(B, E - B*C)
upper = min(B*T, E)
q_target = min(upper, max(lower, ceil(E/2)))
q_candidate = E - q_target
```

The `B` lower bound preserves a target connection for each auxiliary node.
The other bounds prevent exceeding the distinct legal target/candidate pairs.
An empty interval stops construction before fitting; it cannot remove a case.
Each arm keeps the original node budget and total edge budget. Actual ratios
are reported individually, rather than relabelled as strict equal allocation.

In this cohort, 50 of 64 VALID cases per seed require projection; only eight
are exactly half. The case median target-edge fraction is 25.5319%, the
edge-weighted target fraction is 23.9922%, and fractions range from 6.3830% to
100%. [Every actual ratio](../results/downstream_cooperation/quota_by_case.csv)
is retained.

The quota arms have no auxiliary-to-auxiliary edges, whereas native MonTi keeps
its native structure. Individual auxiliary-node degrees need not stay fixed.
Target allocations also differ from native. Thus the native-to-staged change
bundles role quotas, target choices, edge-role restrictions and downstream
training. Equal node/edge budgets do not mean equal per-node geometry or equal
training/computational cost.

## Matched controls

| Condition | Target block | Candidate connections | Mean official-GAGA MR |
|---|---|---|---:|
| Native MonTi | Native output | Native output | 53.6486% |
| Projected quotas, original scores | Fixed upstream quota block | Frozen original scores | 56.8593% |
| Projected quotas, random | Same fixed block | Fixed private random stream | 57.0344% |
| Staged pipeline, trained downstream | Same fixed block | Learned downstream scores | 58.1436% |

All three quota arms share byte-identical upstream features, target blocks,
candidate lists and order, total node/edge budgets and role quotas. The
frozen-score control already improves over native, so the full staged gain
cannot all be attributed to downstream learning. The trained pipeline exceeds
this control's mean by 1.2843 percentage points. Its advantage over matched
random candidate connections is also not established with the present
three-seed uncertainty. The full seed tables and intervals remain in the
[main results](../results/downstream_cooperation/README.md).

The random candidate control changes only downstream candidate connections.
Earlier random target-fill studies use another policy and cohort and are
indexed separately in [supplementary research](SUPPLEMENTARY_STUDIES.md).

## Training and evaluation boundary

For each seed, fit only downstream parameters for four terminal epochs on the
same 128 original TRAIN groups. Use Adam with learning rate 0.001 and weight
decay 0.0001, gradient-norm clipping at 5 and frozen GCN TRAIN feedback. The hard
forward graph is legal and binary. The selected-edge straight-through gradient
is a biased surrogate, without an exact derivative through discrete topology
or hard graph-degree changes. The upstream and classifier parameters remain
fixed.

The main fit performs 500 downstream updates per seed, or 1,500 total. Its 36
no-choice training occurrences are skipped under the same rule and retained
in the accounting. No VALID checkpoint selection is allowed. All three
terminal checkpoints and 192 final four-arm plans are sealed before the
single planned official-GAGA development evaluation. There are no final TEST
queries or outcome-driven extra seeds, epochs or fits.

The 64 original VALID groups per seed contain 571 target occurrences and 240
unique targets. There are 233 identities seen in earlier development contexts;
unused group contexts do not imply an independent unseen-news population.
Original target memberships are retained, with an ID-only cohort draw rather
than selection by size, label difficulty or detector outcome. Every target
occurrence remains in the denominator, including clean errors and no-change
cases. Higher MR means a stronger attack. Clean errors, newly incorrect
predictions and recoveries are reported separately.

## Result and claim boundary

The staged pipeline improves over our native MonTi by 4.4950 percentage points.
Seed gains are 6.1296, 4.3783 and 2.9772 pp; the nominal paired three-seed 95%
interval is [0.5715, 8.4185] pp. It passes the pre-specified pilot rule. Seed is
the replication unit; the 1,713 pooled target occurrences are correlated and
must not be treated as 1,713 independent news articles. The interval is a
nominal development estimate after an adaptive research history, not a
population-wide guarantee.

The native value is our own evaluation of our earlier trained checkpoints on
this same cohort. It is not a paper-table value or a verified authors' best
checkpoint. The separate formal learned/random TEST reproduction remains
43.7280% versus 40.1902%, inconclusive, and uses a different comparison.
[Paper and reproduction limits](REPRODUCING.md#claim-and-paper-comparison)
remain unchanged. The staged result is a validation-stage comparison; the
complete matched evaluation within the approved course scope is tracked in
[the assessment mapping](ASSESSMENT.md).

## Evidence and further reading

- [Main report](STAGED_REPORT.md) and [PDF](MonTi_Staged_Report.pdf).
- [Frozen execution protocol](../evidence/downstream_cooperation_v1/execution_protocol.json),
  [source manifest](../evidence/downstream_cooperation_v1/execution_source_manifest.json)
  and [completed execution receipt](../evidence/downstream_cooperation_v1/completed_summary.json).
- [All seed metrics](../results/downstream_cooperation/seed_metrics.csv),
  [all case counts](../results/downstream_cooperation/case_counts.csv),
  [actual quotas](../results/downstream_cooperation/quota_by_case.csv) and
  [audit and publication provenance](../results/downstream_cooperation/).
- [Supplementary studies](SUPPLEMENTARY_STUDIES.md) preserve every other study,
  including negative results, diagnostic limits and failed attempts. The old
  deterministic reallocation method is preserved byte-for-byte in
  [its historical description](LEGACY_REALLOCATION.md).

The main CPU run took 3,574.04 seconds of process wall time and used zero GPU
hours or experimental API calls. Failed startup attempts and guardian/CPU
accounting are retained. Separate saved-logit and saved-tensor audits add no
new model inference and do not establish human verification or seal chronology.
Private frozen assets are still required for a full ML rerun; the compact
exports are saved-record evidence, not an end-to-end public execution bundle.
