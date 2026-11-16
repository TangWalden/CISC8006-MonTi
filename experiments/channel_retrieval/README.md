# Two-channel relation retrieval: completed development iteration

[The completed comparison](../../results/channel_retrieval/) passes the
registered two-dataset screen: GossipCop 64.1565% versus native 57.9101% and
same-class random 57.6182%; YelpChi 34.7826% versus 29.1304% and 30.5797%.
All three conditional repeats are positive against native, both random
controls, the previous method and uniform-shortlist search. This remains
exploratory reused-development evidence, not final TEST confirmation.

The preceding full-pool search improved GossipCop from 57.9101% to 60.4203%
MR, but YelpChi remained 29.1304%. Neither YelpChi victim nor own-teacher
classes changed, despite changed logits on 687/690 correlated occurrences.
This motivates testing the restricted relation channel, not declaring a cause.

GAGA separates known TRAIN-benign and TRAIN-fraud neighborhood summaries.
Previous remote endpoint retrieval could directly change only the two-hop
TRAIN-benign token. This iteration lets the frozen-teacher downstream search
both known TRAIN classes, then retrieve a real node within the chosen class.
These are label-grouped input channels, not new physical edge-relation types.
It changes endpoint eligibility, not labels or the author generator. No
validation/test label is used to form either pool. Unknown-label endpoints
remain excluded. The same three donor slots and maximum two substitutions
preserve all target edges, injected features/degrees and node/edge budgets.

The general motivation is type-specific relation processing in
[HGT (WWW 2020)](https://arxiv.org/abs/2003.01332), combined with discrete
gradient ranking from [HotFlip (ACL 2018)](https://aclanthology.org/P18-2006/).
GAGA's own [label-group aggregation](https://arxiv.org/abs/2302.10407) is direct
prior art. We do not implement HGT or claim that typed routing, gradients or
this combination establishes first novelty. Whether it transfers is measured.

For each region/class, eight real endpoints are shortlisted from the complete
legal pool, then scored with the fixed independent teacher. The fixed
sigmoid-margin objective and STOP rule are unchanged. This doubles the prior
per-region shortlist opportunities; it is not an equal-compute comparison.

Controls include the exact saved author attack, old single neural model,
two-step rewriter and benign-only full-pool search. Same-class random matches
the selected class, donor slots and edge dose, isolating within-class choice.
Joint-pool random matches slots/dose but can choose another class. A uniform
shortlist searches both classes with the same verification width per step;
STOP and derivative work can still change total cost. All outcomes are kept.

Both datasets retain their same 64 development groups/three capture repeats.
The registered primary requires the channel method to beat native, both old
models, benign-only search and both random controls in **each** dataset.
This is exploratory development on reused data, not untouched confirmation.

Fixed TRAIN checks cover both classes and all four mixed two-substitution
combinations. Every final graph is physically rebuilt and compared with the
canonical seven-token input. All actions on both datasets seal before either
official validation phase. No new upstream/teacher fitting, GPU, API or TEST.
Finite CPU-four-thread phases; no automatic scientific retry. Root-authored
checks are not independent or human verification. Results were unknown at
registration; execution source is pinned in the completed result record.
