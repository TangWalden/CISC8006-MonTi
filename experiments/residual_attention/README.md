# Contextual feature residual downstream

The owner requested a change of downstream architecture after candidate-only
corrections produced no mean gain. This proposal freezes the author's MonTi
weights and native topology, then learns a bounded residual of injected-node
features. Two local context streams attend to actually connected targets and
candidates; a learned gate fuses them. A zero-initialized decoder begins at
exactly the author output.

Per-feature residual magnitude is bounded by 1.0 original-TRAIN standard
deviation and the symmetric distance to the original MonTi feature box.
Thus both residual signs stay within the original per-feature full-graph
minimum/maximum range. No baseline clipping or extra nodes/edges is introduced.
The comparison will include magnitude-matched random signs and a simpler
mean-context network; the latter has fewer effective parameters, so it is
not a capacity-matched proof of attention's causal contribution.

Training feedback will come from a fresh, independent GAGA architecture,
with 38 fixed full-TRAIN epochs and the original full-TRAIN input convention.
The earlier internal TRAIN-only study selected epoch 38; this refit does not
access VALID/TEST labels or official victim weights. Its quality check is
in-sample and is not held-out calibration. Architecture-matched feedback
changes the original GCN-only threat setting and must be reported explicitly.

For fixed topology, only unknown-group one-hop/two-hop tokens depend on the
injected features. Precomputed linear coefficients permit exact feature
gradients, subject to a canonical-input/gradient preflight. No straight-through
topology derivative is used. The full downstream protocol must be frozen
after that construction check and before any new development scoring.

The [completed development result](../../results/residual_attention/) is
negative: attention57.3263%, native57.9101%, matchedrandom57.9685%; the mean
network also declines. Exact gradients and feature constraints passed,
but neither candidate met the owner mean criterion. No efficacy or
first-novelty claim. Original failed/null studies stay preserved.
