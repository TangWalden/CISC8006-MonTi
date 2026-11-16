# Frozen two-dataset full evaluation

The owner authorized full GossipCop and YelpChi TEST evaluation of the fixed two-channel endpoint retrieval method. This runner changes orchestration and cohorts, not the selected method.

- Frozen method: channel retrieval source `6e2982ccf3634a1b924d561f92ff14b7163be075`; two substitutions, three donor slots, eight gradient-ranked candidates per class/slot, temperature 1, STOP improvement above 1e-5.
- Comparisons: original author checkpoint, channel retrieval, same-class equal-dose random, and joint-pool equal-dose random. All share the original captured attack features, physical budgets, target list/order and direct target edges.
- Full original cohorts: GossipCop 909 groups / 10,515 target occurrences / 1,221 unique targets; YelpChi 548 groups / 2,682 occurrences / 2,452 unique targets per repetition. Repetitions 0/1/2 condition on fixed checkpoints.
- First replay two predetermined development cases per dataset to verify integration against saved decisions. Then seal every action in both datasets before either official TEST evaluation.
- Four low-priority selection workers, four CPU threads each; no GPU, training, new teacher, API or automatic retry. Existing server workloads are untouched. Runtime checkout stays at its archived commit.
- No test-driven tuning, extra seeds, exclusions or winner switching. Report all contrasts even if negative. Old GossipCop TEST exposure prevents an untouched-confirmation claim.

`prepare.py` freezes source/input hashes and cohorts; `phase.py` executes replay, search and scoring; `driver.py` enforces exclusivity, deadlines and the joint sealing barrier; `audit.py` checks saved tensors, predictions, exact matching and denominators without model calls. Audit authorship is the same root agent, not independent or human verification.

The run and saved audit are now complete: [full outcomes and controls](../../results/channel_full/). The frozen execution source remains recorded at its original commit; this documentation update does not change that execution. Course completion and human requirements remain separate.
