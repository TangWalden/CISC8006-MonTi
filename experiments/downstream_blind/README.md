# Action-blind downstream control

This is a separately registered development ablation of the
[completed frozen-upstream pilot](../downstream_cooperation/README.md).
The owner authorized adding a downstream policy that cannot directly observe
the upstream target-connection decisions, while preserving the other conditions
and comparing it with the existing visible policy and random candidate control.
The implementation and actual execution JSON must be frozen before model work;
this document is the design specification, not an execution receipt.

The earlier visible-policy outcomes are already known. This is a post-hoc
mechanistic development ablation, not fresh confirmation on unseen data. It
does not change the earlier pilot's registered qualification rule or results.

## Question and information boundary

The visible downstream receives, for each auxiliary node, the mean features of
the targets actually connected to it and the logarithm of its actual target
connection count. The new control replaces those two action-dependent inputs:

| Input | Existing visible downstream | New action-blind downstream |
|---|---|---|
| Auxiliary features | Frozen upstream output | Identical frozen output |
| Target feature summary | Mean of this auxiliary node's attached targets | Mean of the complete fixed target list, repeated for every auxiliary row |
| Target-count summary | `log1p(actual row target count)` | `log1p(registered target quota / auxiliary-node count)`, repeated for every row |
| Candidate features and cached GCN context | Frozen cache | Identical frozen cache |
| Original targets, candidate identities and order, quotas | Fixed | Identical |

The blind summary must not read the actual target adjacency, its row counts,
its column totals or any statistic calculated from its selected connections.
The count input comes from the registered role quota. In particular, taking a
mean of logarithmic actual row counts is not the registered replacement.
The physical target connections still belong in the evaluated graph and the
TRAIN optimization feedback. They are hidden from the policy's direct
conditioning input, rather than deleted from the attack.

Both policies retain the same 151,872-parameter architecture and input sizes.
The target and count projection branches remain present. Their inputs use
permitted target-list context and the known budget, so this is not a comparison
against a smaller model with those branches removed or zeroed.

The term *action-blind* is limited to the direct target-attachment summary.
Shared auxiliary features and candidate representations may implicitly encode
information correlated with upstream actions. The TRAIN feedback also depends
on the actual graph. This design does not claim zero information about the
upstream or complete isolation of every possible information channel.

## Reused scientific inputs

Use the exact saved inputs from the 11 November pilot, executed at scientific
source `9e013cb182b5af71a9d0b719b5a939deda3c88e2`, with protocol SHA256
`5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13`.
The historical runtime stays at
`ca093d8b12ea153a076b02bbdbc4ede58508560c`.

- Reuse all 576 sealed TRAIN/VALID upstream captures with their original byte
  hashes. No upstream generation, candidate redraw or target-block change is
  allowed.
- Reuse the previous visible, frozen-score, native and fixed-random plans and
  saved predictions. Existing clean predictions are also reused. Their declared
  hashes and complete case coverage must be checked before new training.
- Retain the original 128 TRAIN groups, 64 VALID groups and seeds 3, 4 and 5.
  Each arm has 571 VALID target occurrences per seed and 240 unique targets;
  233 target identities had appeared in earlier development contexts. The
  current 64 group contexts have now also been evaluated in the visible pilot.
- Preserve every original node/edge budget, projected role quota, target block,
  candidate list and node feature tensor. No case may be dropped or replaced
  because its budget leaves no downstream choice.

The frozen-score and native results remain descriptive context. The primary
new comparison is visible versus blind; random candidate selection is the
matched fixed-random control. The older random target-fill study uses a
different policy and cohort and is not this control.

## Matched training and construction

Train only the new blind downstream. Preserve four terminal epochs, Adam
learning rate 0.001, weight decay 0.0001, gradient-norm clipping at 5, the fixed
GCN TRAIN feedback model, and the original selected-edge straight-through
surrogate. This surrogate is biased and does not differentiate the discrete
topology exactly. No GAGA feedback, VALID checkpoint selection or performance
based early stop is allowed.

Use `torch.manual_seed(20261111 + seed)` for each model initialization and the
same epoch order produced by `random.Random(seed * 1000 + epoch)`. Each blind
initial state hash must equal the saved visible initial state hash for that
seed. Preserve the original hard-selection tie purpose `learned_candidates`;
an arm-specific tie convention would introduce a second policy change.
Use the same exact quota, physical-ID tie rule, mandatory target attachments
and prohibition on auxiliary-to-auxiliary edges as the quota arms.

The upstream, classifier parameters and all reused caches stay frozen. The
optimizer contains only blind-downstream parameters. Cases with zero or full
candidate quota remain in evaluation; they are skipped for fitting under the
same no-choice rule as before. The expected main training workload is 1,500
updates and 36 skips, subject to the frozen geometry checks, with an absolute
cap of 1,536 training GCN forwards and optimizer updates.

A separate TRAIN-only preflight must check functional invariance to changes
in hidden target attachments, the initial-weight match, legal budgets, finite
nonzero gradients and unchanged frozen parameters. It may use at most four
TRAIN GCN forwards and four optimizer updates, with zero GAGA calls. Preflight
updates do not initialize or modify the main-run models.
These caps count real-data work; separate synthetic tests can also perform
optimizer steps on constructed tensors. The neutral target mean uses an
all-ones matrix multiplication and row-count division with the same geometry
as the visible model's fully connected target block, avoiding a separate
floating-point reduction convention in those boundary cases.

After all three blind fits finish, seal their terminal checkpoints and all
192 blind VALID plans before loading or querying official GAGA. Thereafter
only scoring and saved-record auditing are allowed. No additional optimization
or policy construction may use the resulting scores.

## Evaluation and fixed interpretation

Evaluate only the 192 new blind seed/group graphs: at most 192 GCN forwards and
192 official-GAGA forwards. Reuse old clean and control predictions without
new control queries. Recount saved logits, transitions and geometry separately
after completion. The complete fraud-target occurrence denominator is retained:
MR includes clean errors, and newly incorrect predictions and recoveries are
reported separately. Higher MR denotes a stronger attack.

The primary statistic is the three-seed paired difference in official-GAGA MR,
`visible minus blind`, in percentage points. Report every seed difference,
their mean, and the nominal paired 95% Student-t interval with two degrees of
freedom. The separately registered mechanistic screen requires all three
differences to be positive and the interval's lower bound to exceed zero.
There is no new practical-magnitude threshold. These requirements must be
fixed before blind outcomes are read.

Report visible-minus-random and blind-minus-random by the same method as
secondary descriptive contrasts. A visible-versus-blind advantage would support
the usefulness of the supplied action summary in this fixed setup; it would not
by itself establish superiority over random candidate connections. Conversely,
failure to pass this screen does not prove that upstream information has no
value in all architectures or settings. No control is promoted to a new winner,
and there are no extra seeds, epochs or repeated fits selected after outcomes.

Seed is the replication unit. Repeated target occurrences and overlapping news
identities are correlated; the nominal three-seed interval is not an independent
population guarantee. The study does not establish mutual upstream adaptation,
a realistic fraud supply chain, first novelty, human technical verification,
instructor approval or final TEST effectiveness.

## Bounded execution and retained evidence

Use the locked CPU environment with four threads and explicit empty CUDA
visibility. Main execution is bounded by 10,800 seconds internally and a
10,860-second outer guardian. The preflight limit is 900 seconds. The whole
registered phase is bounded by four elapsed hours including reviewed failures
and review waits. No automatic retry, GPU fallback, dependency installation,
external experimental API call or final TEST query is permitted.

Retain the exact new source and protocol, prior source/input hashes, reused
control bindings, initialization and final checkpoint hashes, training order,
per-epoch logs, all planned and actual budgets, raw predictions, query counters,
CPU process and guardian times, any failure or interruption, and the separate
post-run audit. Zero GPU/API/TEST use and the cost ledger remain explicit.
Earlier scientific files and raw evidence are preserved unchanged.
