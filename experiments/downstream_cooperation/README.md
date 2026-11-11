# Frozen upstream and trained downstream pilot

The owner selected a fixed MonTi upstream, a trainable downstream that sees the
upstream decisions, and equal target and candidate edge quotas on 11 November
2026. This new development study is after the earlier 5 November technical
delivery target. Its course submission status is unconfirmed. The earlier
reallocation results do not establish its effectiveness.

## Capacity decision

The current study cannot apply a literal half-target, half-candidate allocation
to every original case without changing another constraint. In the published
128-group reallocation cohort, 97 groups in each seed fail even the necessary
target-capacity test. With `b` injected nodes, `t` original targets and `e` total
new undirected edges, at most `b*t` distinct injected-to-target edges exist.
The requested quota is `ceil(e/2)`.

For one retained case, `b=122`, `t=3` and `e=5734`: the quota is 2867 target
edges, but only 366 distinct pairs exist. This result uses saved geometry only;
it is not a new model evaluation. Passing this necessary test does not establish
feasibility until the actual candidate list is known.

The [saved metadata audit](../../evidence/downstream_cooperation_v1/budget_audit.json)
records every checked row. Fourteen synthetic checks and a separate root
geometry recount are retained in the
[execution receipt](../../evidence/downstream_cooperation_v1/review_and_receipt.json).
The published opaque case keys are seed-scoped; this audit compares geometry
multisets and does not infer cross-seed group identity from those keys.

The owner selected preservation of the original total budget and projection
of the requested target quota to its feasible interval. Let the actual number
of original candidates be `c`. The interval is
`[max(b, e-b*c), min(b*t, e)]`. Project `ceil(e/2)` into this integer interval,
and give the remaining edges to the candidate branch. Report every actual
ratio; projected cases are not exact equal allocation. An empty interval stops
the preflight or upstream-cache phase before training; it cannot silently
remove a case. Training and real-data method evaluation await the source-bound
registration and passed preflight.

## Fixed upstream decisions

Reuse the three recorded terminal-eight legacy MonTi checkpoints for seeds
3, 4 and 5. Their parameters remain frozen, retaining the documented historical
score-source semantics. This study does not silently introduce a source-role
correction. Each case uses one cached upstream
feature tensor, one original target list and one candidate list with preserved
physical node IDs and order.

The upstream chooses the fixed target block under the selected quota from its
frozen scoring function. Every injected node retains at least one target edge.
The downstream may select candidate connections only. It cannot modify node
features, target connections, the candidate list or the total resource budget.
To allocate the entire edge budget between these two roles, the proposal arms
contain no injected-to-injected edges. The unmodified native baseline retains
its native structure, including such edges where present.

Odd edge budgets give the target branch the extra edge. Insufficient target
capacity, insufficient actual candidate capacity and conflict with mandatory
target attachments are separate recorded feasibility reasons. Duplicate edges
and artificial self-loops cannot satisfy a quota.

## Comparisons

| Arm | Target connections | Candidate connections |
|---|---|---|
| Native MonTi | Native output | Native output |
| Equal quotas with original scores | Fixed quota block | Frozen original scores |
| Equal quotas with random choices | Same fixed quota block | Fixed private random stream |
| Equal quotas with trained downstream | Same fixed quota block | Learned conditional scores |

The three quota arms share byte-identical upstream features and target blocks.
The extra frozen-score arm distinguishes a quota change from downstream
learning. Comparing against random candidate selection that changes the
upstream candidate context would not be a matched downstream control.

## Learning and evaluation boundaries

Only downstream parameters enter the optimizer. The downstream receives the
upstream generated features and actual target attachments, along with permitted
original graph context. Target labels, classifier predictions and final victim
outcomes are not policy inputs. A synthetic change to upstream attachments must
change the corresponding downstream conditioning input.

A fixed GCN provides TRAIN-only optimization feedback without parameter
updates. Official GAGA is reserved for a single terminal development evaluation
after the downstream weights and all graphs have been sealed. No TEST inference
is permitted. Use the final fixed training epoch, not a checkpoint chosen by
GAGA outcomes.

The execution settings are three seeds, 128 original TRAIN groups
and four downstream epochs per seed. Select 64 original VALID groups by a fixed
ID-only draw from contexts not used in the four earlier 128-group lists,
including the separate feature-assignment study. Target identities can still
overlap earlier groups, so these are fresh contexts, not independent unseen
news. All models and inputs must be hashed before execution. A separate
TRAIN-only gradient and geometry preflight must pass before the main run.

The runtime is CPU-only with four threads and a four-hour outer phase
limit. No GPU or external API call is required by the proposed pilot. Any
failure or interruption retains its partial output and actual CPU time. There
is no automatic retry, seed expansion or additional training after seeing
development outcomes. The source-bound execution registration must be fixed
before any model call.

The first preflight stopped before any model query because its launcher omitted
the recorded environment library directory. CUDA-linked DGL needs those shared
libraries even when execution is CPU-only. A separately committed launcher
amendment pins `LD_LIBRARY_PATH`, `PYTHONHASHSEED` and the DGL backend. The first
source, registration, logs and zero-query receipt remain in their original
attempt directory; the amendment charges its guardian time against the same
four-hour CPU ceiling. A second startup exposed a Python 3.8 incompatibility in
the synthetic-test log writer, also before any real-data model call. Its
separate source, failure and CPU time are retained. The compatibility amendment
uses an explicit text stream and does not change the experiment settings.

## Interpretation

Report occurrence-weighted MR, clean errors, newly incorrect predictions and
recoveries for all arms. Seed is the replication unit; overlapping group-node
occurrences are correlated. Report paired differences and the nominal
three-seed interval, together with actual resource counts and fallback rates.
Neither a larger model nor a quota restriction alone establishes cooperation.
The matched arms test a trained conditional downstream policy. Without a
separate action-hidden training ablation, any gain does not isolate the
performance contribution of receiving the upstream decisions themselves.
This pilot does not establish a real fraud supply chain, novelty, course
approval or final TEST effectiveness.
