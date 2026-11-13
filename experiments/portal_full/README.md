# Frozen context-retrieval complete-cohort benchmark

This applies the already fitted three downstream models to **all 909 original
TEST groups**, 10,515 fraud-target occurrences and 1,221 unique targets per
noise repetition. It does not retrain or select models on TEST. Both the neural
decoder and direct-search candidate, plus their own activation/dose-matched
random controls and native author model, remain fixed from development.

The author checkpoint, bank size eight, geometric donor, TRAIN-benign pool,
native features/owned mask/degrees/budgets, independent GAGA teacher and gain
threshold 0.0001 are unchanged. All physical option inputs and decisions are
hash-sealed before the official GAGA victim loads. Evaluation reuses these
physical canonical inputs rather than reconstructing every duplicate arm.
Singleton GAGA outputs are restored to shape `[targets, 2]`.

The primary owner screen remains exact pooled wrong-count superiority to
native and the corresponding random arm. Both candidates, all repeats and
nominal paired intervals are reported; no all-repeat-positive requirement is
added. Correlated memberships and one upstream checkpoint limit inference.
Earlier TEST results in other studies were already observed. This is a full
exploratory benchmark, not untouched statistical confirmation.

CPU four threads, nice 10, no GPU/API/training. Preflight uses two fixed TRAIN
groups only. The main deadline is 43,200 s, with no automatic retry or outcome-
driven expansion. Caps: 2,727 author generations, at most 24,543 physical
canonical builds, 2,727 batched own-teacher calls (up to 24,543 hard contexts),
2,727 downstream policy calls and 13,635 official GAGA TEST calls. No case
filtering, replacement or early efficacy inspection is allowed.
