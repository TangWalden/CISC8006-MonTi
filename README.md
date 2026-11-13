# MonTi: Frozen Upstream, Context Retrieval

CISC8006 reproduction and extension of [MonTi](https://github.com/bdi-lab/MonTi).
We freeze the upstream attack generator and learn downstream connections under
controlled node/edge budgets. **The context-retrieval downstream improves
complete-cohort GAGA miss rate by 1.1095 pp over native and 0.8084 pp over
matched random.** This is one exploratory benchmark, not a first-novelty claim.

## Results

Higher GAGA miss rate (MR) means a stronger attack. Each row is a separate
matched study; rates across rows use different cohorts/checkpoints.

| Study | Native MR | Random control MR | Learned MR | Conclusion |
|---|---:|---:|---:|---|
| [Author-checkpoint context retrieval, all TEST groups](results/portal_full/) | 44.2638% | 44.5649% | **45.3733%** | Positive against both controls in all three conditional repeats |
| [Context retrieval, VALID](results/portal_decoder/) | 57.9101% | 58.0269% | 59.1360% | Positive development mean; later complete-cohort result above |
| [Earlier staged TEST, different upstreams](results/final_staged/) | 43.8168% | 44.0165% | 44.2257% | Criterion failed; negative history retained |

The separate original six-run reproduction remains **inconclusive**:
43.7280% learned-candidate versus 40.1902% random-candidate MR,
paired 95% interval [-1.5415, 8.6171] pp. This random-candidate baseline is
different from the downstream random controls above.

## Current method and evidence

Freeze the **author's released checkpoint** and all generated fake features.
A new contextual dual encoder retrieves a TRAIN-benign connection endpoint,
changing relation evidence while preserving target edges, injected-to-injected
edges, every injected degree and total budgets. Random controls match the
pool, donor, activation and exact exchange dose. Context access and GAGA
surrogate information are broader than original local GCN-only MonTi.

The [complete benchmark](results/portal_full/) covers all 909 TEST groups,
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
| Current method, controls and executable source | [Retrieval decoder](experiments/portal_decoder/), [complete-cohort runner](experiments/portal_full/) |
| Latest results and audits | [Complete retrieval benchmark](results/portal_full/), [development](results/portal_decoder/) |
| Setup and historical reproduction | [Reproducing](docs/REPRODUCING.md) |
| Other attempts, including failures | [Supplementary index](docs/SUPPLEMENTARY_STUDIES.md) |
| Report and presentation drafts | [Report](docs/STAGED_REPORT.md), [slides guide](docs/SLIDES_GUIDE.md) - predate the latest results |
| AI assistance and human contributions | [AI use](AI_USAGE.md), [contributions](CONTRIBUTIONS.md) |

Compact methods/results are public; large raw runs and frozen assets remain
in the recorded archive. Full ML reruns require that runtime and its assets.
Instructor approval, human review, actual contributions, defense and final
submission remain unconfirmed. [License](LICENSE.txt).
