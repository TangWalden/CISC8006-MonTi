# MonTi: Frozen Upstream, Relation-Aware Retrieval

CISC8006 reproduction and extension of [MonTi](https://github.com/bdi-lab/MonTi).
We freeze the author's attack checkpoint and optimize downstream connections
under unchanged physical node/edge budgets. **The latest two-channel search
improves development GAGA miss rate by 6.2464 pp on GossipCop and 5.6522 pp
on YelpChi over native, and by 6.5382/4.2029 pp over same-class matched random.**
Both datasets pass the registered screen in all three conditional repeats.
These are exploratory VALID results, not final TEST confirmation or first novelty.

## Results

Higher GAGA miss rate (MR) means a stronger attack. Each row is a separate
matched study; rates across rows use different cohorts/checkpoints.

| Study | Native MR | Random control MR | Modified MR | Conclusion |
|---|---:|---:|---:|---|
| [Two-channel retrieval, both VALID sets](results/channel_retrieval/) | GossipCop 57.9101%; YelpChi 29.1304% | Same-class 57.6182%; 30.5797% | 64.1565%; 34.7826% | Registered two-dataset screen passed |
| [Full-pool gradient search, both VALID sets](results/gradient_retrieval/) | GossipCop 57.9101%; YelpChi 29.1304% | 58.4355%; 29.1304% | 60.4203%; 29.1304% | GossipCop +2.5102 pp; YelpChi no gain |
| [YelpChi method transfer, VALID](results/yelpchi_validation/) | 29.1304% | 29.1304% | 29.1304% | No gain; all eleven arms have identical paired class predictions |
| [Two-step target-aware rewriter, VALID](results/target_rewriter/) | 57.9101% | 58.3187% / 58.0852% | **59.4863%** | +0.3503 pp over existing neural; below two-step greedy 59.6030% |
| [Author-checkpoint context retrieval, all TEST groups](results/portal_full/) | 44.2638% | 44.5649% | **45.3733%** | Positive against both controls in all three conditional repeats |
| [Context retrieval, VALID](results/portal_decoder/) | 57.9101% | 58.0269% | 59.1360% | Positive development mean; later complete-cohort result above |
| [Earlier staged TEST, different upstreams](results/final_staged/) | 43.8168% | 44.0165% | 44.2257% | Criterion failed; negative history retained |

The separate original six-run reproduction remains **inconclusive**:
43.7280% learned-candidate versus 40.1902% random-candidate MR,
paired 95% interval [-1.5415, 8.6171] pp. This random-candidate baseline is
different from the downstream random controls above.

## Current method and evidence

Freeze the **author's released checkpoint** and generated features. The latest
downstream compares known TRAIN-benign and TRAIN-fraud group-summary channels,
ranks legal real endpoints with the fixed teacher's input gradient, and checks
shortlisted choices. It preserves target edges, injected-to-injected edges,
every injected degree and total budgets. Same-class random matches class,
donor and exact edge dose; joint-pool random and uniform-shortlist search are
also included. Context/GAGA surrogate access is broader than original local
GCN-only MonTi, and extra search computation is disclosed.

The earlier learned decoder's [complete benchmark](results/portal_full/) covers all 909 TEST groups,
10,515 occurrences and 1,221 unique targets per repeat. Existing downstream
fits are reused; three repeats share one author checkpoint and overlapping
targets. Neural-minus-random nominal 95% interval is [0.5617, 1.0550] pp,
conditional on these repeats. Earlier TEST exposure limits confirmation.
Direct teacher search reaches 45.3955% against its 44.6156% random control,
slightly above neural. Earlier null/negative attempts remain recorded.

## Read and reproduce

| Purpose | Entry |
|---|---|
| Course scoring points and genuine remaining work | [Assessment mapping](docs/ASSESSMENT.md) |
| Current method, controls and executable source | [Two-channel retrieval](experiments/channel_retrieval/), [earlier learned decoder](experiments/portal_decoder/) |
| Latest results and audits | [Both-dataset result](results/channel_retrieval/), [preceding partial success](results/gradient_retrieval/), [earlier complete benchmark](results/portal_full/) |
| Setup and historical reproduction | [Reproducing](docs/REPRODUCING.md) |
| Other attempts, including failures | [Supplementary index](docs/SUPPLEMENTARY_STUDIES.md) |
| Report and presentation drafts | [Report](docs/STAGED_REPORT.md), [slides guide](docs/SLIDES_GUIDE.md) - predate the latest results |
| AI assistance and human contributions | [AI use](AI_USAGE.md), [contributions](CONTRIBUTIONS.md) |

Compact methods/results are public; large raw runs and frozen assets remain
in the recorded archive. Full ML reruns require that runtime and its assets.
Instructor approval, human review, actual contributions, defense and final
submission remain unconfirmed. [License](LICENSE.txt).
