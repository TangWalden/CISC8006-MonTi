# Bounded related work check

Checked on 31 October 2026. This is a targeted primary-source inspection of four
papers, three official repositories and versioned PyTorch documentation. It is
not an exhaustive literature review and establishes neither exact prior absence
nor first-in-literature novelty. It selects no method and authorizes no training.
Existing report/slides, frozen protocols and experimental results are unchanged.

## What the sources establish

**MonTi already uses a surrogate.** The published candidate score and target
encoding incorporate pretrained surrogate representations in Eqs. 2-3. Eq. 6
and the following edge-score definition use the injected representation as
source. Thus A is a proposed paper-conformance study. B would not introduce
surrogate guidance into MonTi for the first time. [MonTi paper, printed pages
16030-16031](https://ojs.aaai.org/index.php/AAAI/article/download/33760/35915)
and [the existing paper/source comparison](../course/PAPER_EDGE_SCORE_CHECK.md).

**G-NIA already learns from classification margins.** Its CIKM 2021 paper,
Section 5.5 Eq. 8, minimizes a true-class versus competing-class score gap. The
official single- and multiple-target runners implement ReLU margin losses. This
is relevant prior work for prediction-aware injection, including tightly limited
injection settings. It is operationally different from B's proposed hard endpoint
eligibility rule. [G-NIA paper v2](https://arxiv.org/pdf/2108.13049v2),
[single-target loss](https://github.com/TaoShuchang/G-NIA/blob/19340eb5b8e21911a58b31189ec57e74714dd2cd/run_gnia.py#L232-L240),
[multiple-target loss](https://github.com/TaoShuchang/G-NIA/blob/19340eb5b8e21911a58b31189ec57e74714dd2cd/run_multar_gnia.py#L236-L250).

**TDGIA directly uses confidence in finite-budget edge selection.** Its KDD 2021
paper, Section 5.3 Algorithm 1, combines surrogate confidence with a
topological vulnerability factor before ranking attachment targets. The official
implementation likewise combines prediction scores with degree terms. Its
sequential allocation and feature optimization differ from fixed-feature B.
Prediction-informed use of an edge budget is therefore established broad prior
work, rather than a defensible general novelty claim here.
[TDGIA paper v2, Algorithm 1](https://arxiv.org/pdf/2106.06663v2#page=6),
[pinned attachment score](https://github.com/THUDM/tdgia/blob/29462748ae020b43015338de0f00df5b7b9748ca/tdgia.py#L158-L176),
[prediction refresh](https://github.com/THUDM/tdgia/blob/29462748ae020b43015338de0f00df5b7b9748ca/tdgia.py#L480-L486).

**AGIA/HAO provides another model-informed edge-optimization context.** The ICLR
2022 work studies homophily and unnoticeability. In its official repository,
AGIA can obtain clean-model pseudo-labels, alternate feature/adjacency procedures
and optimize edge weights with prediction loss and a sparsity term. This review
does not equate the paper's HAO objective with every detail of that AGIA file.
Neither mechanism establishes that B's exact conditional filter is the same
policy, or that the filter is absent elsewhere.
[Chen et al. paper v2](https://arxiv.org/abs/2202.08057v2),
[pinned clean-prediction and alternating steps](https://github.com/LFhase/GIA-HAO/blob/780852c78e138070d0a2c2b581109d43bd4e752e/attacks/agia.py#L68-L92),
[pinned edge objective](https://github.com/LFhase/GIA-HAO/blob/780852c78e138070d0a2c2b581109d43bd4e752e/attacks/agia.py#L155-L169).

## What the proposed contribution could be

| Option | Defensible description now | Evidence still missing |
|---|---|---|
| A | Unselected alignment of the executed score source with the published role | Matched retraining and controlled evaluation of its effect |
| B | Unselected MonTi extension restricted to node budget = edge budget = 1 | Efficacy, transfer, fair cost accounting and any stronger originality claim |
| C | Explicit exact-tie convention as an engineering control | Actual pipeline/device comparisons if approved, with changing scores/context distinguished |

B's specific question remains whether a hard clean-surrogate predicted-fraud
eligibility set uses the sole edge more effectively under this fixed MonTi
setting. Features, injected-node/edge counts and evaluation groups would remain
fixed. An empty eligible set falls back to all targets. The six observed GPU
opportunities motivate this question; they are not measured gains from B.
The already proposed native, all-target procedural and random count-matched
controls remain necessary. This review adds no new experimental arm or winner.
[Outcome-based proposal](../course/IMPROVEMENT_PROPOSAL_V2.md) and
[completed replay evidence](SCORE_ROLE_REPLAY.md).

Endpoint eligibility must remain distinct from removing evaluation targets.
G-NIA's inspected test loops iterate loaded split masks/groups; this check does
not support a universal initially-correct-only evaluation claim. External cohort
preprocessing was not audited. B must preserve the full registered MR denominator
and must not select endpoints using GAGA outcomes or evaluation-label oracles.
[Single-target test loop](https://github.com/TaoShuchang/G-NIA/blob/19340eb5b8e21911a58b31189ec57e74714dd2cd/run_gnia.py#L307-L355),
[multiple-target test loop](https://github.com/TaoShuchang/G-NIA/blob/19340eb5b8e21911a58b31189ec57e74714dd2cd/run_multar_gnia.py#L320-L378).

## Version limits for tie handling

The official v1.13.1 `topk` docstring describes sorting returned values but does
not promise stable tied indices. Its reproducibility notes separately decline a
general CPU/GPU reproducibility guarantee. The newer v2.15.0 docstring explicitly
warns about tied-index instability. That newer wording must not be retroactively
attributed to the locked v1.13.1 runtime. The project's 32/96 identical-vector
reference differences are empirical evidence under its own conditions, not proof
that all generator differences arise from ties.
[v1.13.1 topk](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/_torch_docs.py#L11464-L11506),
[v1.13.1 reproducibility notes](https://github.com/pytorch/pytorch/blob/v1.13.1/docs/source/notes/randomness.rst#L1-L24),
[v2.15.0 topk](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/_torch_docs.py#L11879-L11923),
[local replay report](SCORE_ROLE_REPLAY.md).

## Review boundaries and use

Code references above use the exact commits resolved during this check, or named
official release tags. Paper versions and inspected locations are recorded in
the [source log](../evidence/related_work_review_v1.json). No datasets or new
dependencies were downloaded, and no model inference, training, test queries or
GPU research occurred.
Follow-up access to the AAAI PDF encountered an access-denied page after indexed
primary text had been read; no challenge was bypassed. Other primary paper/code
sources remained available. Source locations were checked against pinned code,
correcting preliminary agent line-number errors before publication.

Human technical review, instructor approval and owner direction remain pending.
The response fields in the existing [human review template](HUMAN_REVIEW.md)
and actual-completion fields in the [contribution record](../CONTRIBUTIONS.md)
remain pending. A careful approved
null/negative study can satisfy a scientific question, but this source check
does not replace the course's required complete experiments and genuine human
work. No grade or novelty is guaranteed.
