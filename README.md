# MonTi: Frozen Upstream, Local Downstream

CISC8006 reproduction and extension of [MonTi](https://github.com/bdi-lab/MonTi).
We freeze the upstream attack generator and learn downstream connections under
controlled node/edge budgets. **The new context-retrieval downstream exceeds
native and matched random in development mean; complete-cohort transfer remains
to be checked.**

## Results

Higher GAGA miss rate (MR) means a stronger attack. Each row is a separate
matched study; rates across rows use different cohorts/checkpoints.

| Study | Native MR | Random control MR | Learned MR | Conclusion |
|---|---:|---:|---:|---|
| [Staged development](results/downstream_cooperation/) | 53.6486% | 57.0344% | 58.1436% | Native gain; learning advantage over random uncertain |
| [Completed staged TEST](results/final_staged/) | 43.8168% | 44.0165% | 44.2257% | Criterion failed; one seed declined substantially |
| [Author-checkpoint local swap, VALID](results/local_swap/) | 57.9101% | 57.8517% | 57.9101% | No gain over native |
| [Author-checkpoint block swap, VALID](results/block_swap/) | 57.9101% | 57.9685% | 57.9101% | No native gain; below random |
| [Author-checkpoint attention residual, VALID](results/residual_attention/) | 57.9101% | 57.9685% | 57.3263% | Below native and random |
| [Context retrieval, VALID](results/portal_decoder/) | 57.9101% | 58.0269% | 59.1360% | +1.2259 pp native; +1.1092 pp random; full benchmark pending |

The separate original six-run reproduction remains **inconclusive**:
43.7280% learned-candidate versus 40.1902% random-candidate MR,
paired 95% interval [-1.5415, 8.6171] pp. This random-candidate baseline is
different from the downstream random controls above.

## Current method and evidence

The latest local correction retains the **author's released checkpoint**,
generated features, target edges, injected-to-injected edges, individual
injected degrees and total budgets. Its null result revealed that many single
edge swaps leave GAGA's added-neighbor membership unchanged. The registered
[block-recipient follow-up](results/block_swap/) was null; its direct-GCN
follow-up was withdrawn before scoring when the owner changed strategy.
[Feature-residual attention](results/residual_attention/) also declined.
The [context-retrieval decoder](results/portal_decoder/) learns to choose
new TRAIN-benign relation context against same-pool random controls. Its
development mean improves; direct teacher search reaches 59.2528% versus its
58.1436% random control. The random-contrast interval still crosses zero.
Both use broader context access and architecture-matched teacher information
than the original local GCN-only attack. Complete-cohort evaluation will keep
both candidates fixed and report failures as well as gains.

## Read and reproduce

| Purpose | Entry |
|---|---|
| Course scoring points and genuine remaining work | [Assessment mapping](docs/ASSESSMENT.md) |
| Method, controls and executable source | [Staged method](docs/METHOD.md), [local swap](experiments/local_swap/), [block swap](experiments/block_swap/) |
| Latest results and audits | [Full staged TEST](results/final_staged/), [author local correction](results/local_swap/) |
| Setup and historical reproduction | [Reproducing](docs/REPRODUCING.md) |
| Other attempts, including failures | [Supplementary index](docs/SUPPLEMENTARY_STUDIES.md) |
| Report and presentation drafts | [Report](docs/STAGED_REPORT.md), [slides guide](docs/SLIDES_GUIDE.md) - predate the latest results |
| AI assistance and human contributions | [AI use](AI_USAGE.md), [contributions](CONTRIBUTIONS.md) |

Compact methods/results are public; large raw runs and frozen assets remain
in the recorded archive. Full ML reruns require that runtime and its assets.
Instructor approval, human review, actual contributions, defense and final
submission remain unconfirmed. [License](LICENSE.txt).
