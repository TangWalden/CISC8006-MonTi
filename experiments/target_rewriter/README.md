# Target-aware two-step local rewriter

Owner-authorized **GossipCop development** experiment, executed 14 November 2026.
The [completed result](../../results/target_rewriter/) passes its mean screen,
but the gain over existing neural is modest and does not beat two-step greedy.
This phase made no YelpChi or new TEST calls. Author upstream/cached attacks
remain unchanged. The exact executed README is retained at source `73e3487`.

The new downstream retains individual target tokens. Cross attention links
legal candidate replacements to affected targets; separate heads select a
region and a conditional replacement, with STOP. It can modify at most two
distinct candidate slots from three geometrically nominated regions. Each
region has four disjoint original TRAIN-benign candidates outside original
target one/two-hop neighborhoods. Injected features, target edges, owned mask,
each injected degree and physical node/edge budgets remain fixed. Original
endpoint degrees can change, and context access exceeds the original policy.

Training uses exact physical one/two-edit outcomes of an independent TRAIN-only
GAGA teacher on the first 32 previously registered TRAIN groups per repeat.
Root labels use best attainable two-edit loss; second-state labels use final
loss. Twelve fixed epochs of listwise imitation train the hierarchical policy.
This is lookahead imitation, **not** completed reinforcement learning. Inference
recomputes actual context after the first edit. The current implementation does
not add a perturbed-teacher ensemble; that would be a separate ablation.

All 64 original development groups / 571 target occurrences / 240 unique targets
per repeat remain. Three cached generation-noise repeats of one author model
are not three independently trained author models. Original single neural and
direct methods retain their eight-candidate pool and saved fitted models.
Controls include native, existing methods/random, two-step greedy, independent
single-edit composition, and matched random for each new candidate. New-policy
endpoint random keeps its chosen positions; joint random also randomizes the
region subset when exact total edge-dose/count matching permits it. Degenerate
joint random subsets are reported. A common 21-context selection ceiling is
not equal actual compute: neural <=2, original <=9, independent <=13, greedy
<=21. TRAIN-label construction and all evaluation costs are counted separately.

All choices seal before official GAGA is loaded. Primary success requires the
neural rewriter's pooled MR to exceed native, both existing methods and both
own random controls. A one-percentage-point gain over existing neural is
reported separately. All outcomes, paired gains/losses and conditional-repeat
intervals remain visible. This reused development study cannot establish
untouched confirmation or first novelty.

Design references: [NeuRewriter](https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf)
motivates region/rule decomposition; [LPGIA](https://arxiv.org/html/2405.18824v1)
motivates considering connection combinations. Their improvements do not imply
improvement here. The exact protocol, source hashes and finite caps are frozen
by `prepare.py` before `driver.py` launches TRAIN-only preflight, the study, and
the root-authored saved-record auditor. Failures are preserved; no automatic
scientific retry or outcome-dependent resource extension is allowed.
