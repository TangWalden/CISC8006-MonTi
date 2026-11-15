# Registered-method transfer to YelpChi development

The owner's latest instruction authorizes YelpChi validation, superseding the
earlier preparation-only hold. LifeIns is excluded. The registered pipeline
has now [completed with a null result](../../results/yelpchi_validation/): all
eleven arms 29.1304% MR, with identical paired classes. No model code was changed
after outcomes; executed source is commit 61eeb4a, with later documentation separate.

The study retains the author's YelpChi checkpoint/configuration (seed 2,
node ratio .05, edge factor .5) and freezes its generated features and mask for
all downstream arms. It uses original split 0 and an outcome-blind fixed sample
of 128 TRAIN and 64 VALID groups. All selected cases, including identity actions,
remain. No TEST evaluation is authorized. Three generation-noise repetitions
reuse one author model, not three independently trained author checkpoints.

The independent GAGA teacher follows the GossipCop full-TRAIN fit: fixed 38
epochs, balanced cross entropy, Adam .001/.0001, batch512, seed20261113. Its
quality gate is an in-sample sanity check, not held-out calibration. The old
single neural method retains128 TRAIN groups/eight epochs and its own random;
the two-step rewriter retains the first32 groups/twelve epochs. Architectures,
learning rates, candidate pools, lookahead labels and action rules remain as in
GossipCop. Only feature width, data paths, original budget config and cohort
denominators change. No victim parameters or VALID/test labels drive training.

Rather than copying or silently editing the completed method, tiny wrappers
execute hash-pinned shared source with four explicit bindings: cohort counts,
the YelpChi classifier loader, absence of a prior YelpChi victim-output file,
and the corresponding disabled historical-output parity comparison. The
original author/native graph is instead checked against physical canonical
reconstruction on every new capture. Dataset-specific adapters cannot load
the GossipCop victim or group caches.

All new/old neural and search decisions seal before any YelpChi original-GAGA
evaluation. The eleven arms and primary rewriter screen match GossipCop.
Same-position and joint random match actual exchanged edges and edit count;
degenerate matching cases remain reported. Structural budgets are equal within
YelpChi, not between datasets; actual compute/query use also differs across
methods and is recorded. CPU4/nice10, finite stage deadlines, no GPU/API, no
automatic scientific retry. Raw attempts and failures remain isolated on QY.

The inherited `old_single_selection_contexts` field counts shared candidate
cards, not the old neural ranker's minimum teacher forwards. That ranker uses
physical sequence changes and features without teacher logits at deployment;
the new rewriter uses teacher margins. Preserve the original counters, but do
not infer a teacher-query or latency speedup from their 2-versus-9 ceilings.
