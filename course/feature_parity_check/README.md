# Saved-graph canonical own-teacher diagnostic

Status: implementation and source freeze prepared after the complete negative
[feature-refinement study](../feature_polish/README.md). No positive candidate
or final TEST authorization follows from this diagnostic.

The independent full-TRAIN GAGA's optimization loss decreased, while official
GAGA development MR fell. Four TRAIN functional probes do not establish that
the affine shortcut and canonical updater agree on every saved VALID graph.
This bounded post-hoc check measures that missing consistency directly.

All 384 saved graphs and both baseline/GAGA-PGD endpoints are retained. The
fixed own teacher receives 768 newly built canonical sequences, compared with
its immutable saved affine-path logits. Guarded features are verified as the
appropriate endpoint aliases. The official victim is never loaded or queried;
its completed prediction JSONLs provide only saved endpoint comparisons.

The check also records full original adjacency symmetry, selected unknown-group
cache denominators and two-hop support on the 374 unique VALID targets. It
retains numerical, class, alias and cache discrepancies separately. A near-zero
argmax flip within the existing numerical tolerance is not automatically a
large adapter-error finding. Agreement cannot prove the cause of transfer
failure, efficacy, novelty, calibration or human verification.

Execution is CPU-only with eight threads and an 1800-second maximum. There are
zero new generations, GCN/victim calls, weight updates, feature gradients,
external API calls or final TEST queries. Existing original data/caches, the
frozen own-teacher checkpoint and saved perturbation tensors are explicitly
read; this is not a saved-JSON-only analysis. All scientific and diagnostic
source/input hashes must be committed and registered before those reads.

Both prior development gain criteria failed, so the complete full-TEST candidate
gate remains closed. The owner's request for further measured improvement
continues; this implementation check precedes another proposed method.
