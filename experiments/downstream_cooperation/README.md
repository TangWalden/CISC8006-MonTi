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

The owner has been asked to choose one of two explicit policies:

- Apply exact quotas to feasible cases, with the original attack retained in
  every arm for infeasible cases. Report the complete cohort and the feasible
  subset separately. No target is removed from the complete-cohort denominator.
- Preserve the original total budget and project the requested target quota to
  its feasible interval. Report actual ratios; do not label projected cases as
  exact equal allocation.

The choice is pending. Training and real-data method evaluation have not begun.

## Fixed upstream decisions

Reuse the three recorded terminal-eight legacy MonTi checkpoints for seeds
3, 4 and 5. Their parameters remain frozen. Each case uses one cached upstream
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

The draft execution settings are three seeds, at most 256 original TRAIN groups
and four downstream epochs per seed. Select 64 original VALID groups by a fixed
ID-only draw from contexts not used in the four earlier 128-group lists,
including the separate feature-assignment study. Target identities can still
overlap earlier groups, so these are fresh contexts, not independent unseen
news. All models and inputs must be hashed before execution. A separate
TRAIN-only gradient and geometry preflight must pass before the main run.

The planned runtime is CPU-only with four threads and a four-hour outer phase
limit. No GPU or external API call is required by the proposed pilot. Any
failure or interruption retains its partial output and actual CPU time. There
is no automatic retry, seed expansion or additional training after seeing
development outcomes. These draft settings become executable only after the
capacity policy and the source-bound execution registration are fixed.

## Interpretation

Report occurrence-weighted MR, clean errors, newly incorrect predictions and
recoveries for all arms. Seed is the replication unit; overlapping group-node
occurrences are correlated. Report paired differences and the nominal
three-seed interval, together with actual resource counts and fallback rates.
Neither a larger model nor a quota restriction alone establishes cooperation.
This pilot does not establish a real fraud supply chain, novelty, course
approval or final TEST effectiveness.
