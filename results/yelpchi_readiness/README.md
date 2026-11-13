# YelpChi assets prepared; experiments deferred

On 14 November 2026 the owner requested data/cache preparation, followed by
holding YelpChi experiments while reconsidering the GossipCop downstream.
LifeIns is excluded. No YelpChi efficacy result is claimed.

- Official graph, split-0 node lists and target membership are present:
  45,900 nodes and 32 features; 556 TRAIN, 331 VALID and 548 TEST target groups.
- The author's `GCN-GAGA-yelpchi_rtr` checkpoint and config, original surrogate
  GCN and victim GAGA weights pass strict state-dictionary shape/key checks.
- Five input caches are ready: GCN representations, GAGA sequences, feature
  sums, group counts and two-hop adjacency. Original TRAIN labels alone define
  known-label groups; graph/features use the original transductive convention.
- Cache work made 45 GCN representation forwards, **zero MonTi generations,
  GAGA forwards, optimizer updates, TEST predictions or GPU/API research**.
  Main preparation took 49.11 s wall / 89.46 s process CPU; server retrieval
  plus preparation took 97.36 s wall. Transfer time is separate.
- Eighteen staged files were checked against saved hashes. Large data, model
  binaries and caches remain private on QY under
  `runs/yelpchi_assets_20261114_attempt1`; the clean historical runtime was
  not modified. No active preparation job remains.

Sources are the [official upstream repository](https://github.com/bdi-lab/MonTi)
and its linked public asset archive. Range-retrieved ZIP members pass
header/size/CRC and local/server SHA256 agreement; classifier bytes match Git
LFS object IDs at upstream commit `10f233e953679ec3715125dc058ef6d28bee4691`.
This is not a signed author release or human review.

[Metadata, hashes and counters](preparation.json),
[executed source hashes](source_hashes.json),
[preparation scripts](../../experiments/yelpchi_assets/).
The [current positive GossipCop result](../portal_full/) is unchanged.
