# MonTi: Reproduction and Edge Reallocation

A CISC8006 study of [MonTi](https://github.com/bdi-lab/MonTi), with a budget-preserving graph-injection postprocessor.

**For assessment:** [grading points, evidence and remaining gaps](docs/ASSESSMENT.md).

## Current method

After MonTi generates an attack, move some injected-to-candidate edges to the existing target list. Keep injected features, previous target edges, injected-node degrees, and total node/edge budgets unchanged. The rule is deterministic; it uses no LLM or additional training. [Method](docs/METHOD.md).

## Latest results

GossipCop-S, split 0; GCN surrogate and GAGA detector. **Higher miss rate (MR) means a stronger attack.** MR includes targets already misclassified before the attack.

| Development condition | Mean GAGA MR |
|---|---:|
| Original attack | 46.95% |
| Candidate-to-target reallocation | **52.43%** |
| Random target-fill control | 52.53% |
| Candidate-rewiring control | 46.88% |

The measured gain is **+5.48 percentage points**, positive in all three seeds. Its nominal paired 95% interval is **[-0.82, 11.78] pp**, so the frozen stability criterion failed. Random target filling performs similarly; a benefit from the priority rule is unproven. No final TEST evaluation has run for this method.

Scope: 128 validation groups per seed, 1,016 target occurrences and 373 unique targets per seed. The 3,048 pooled occurrences are correlated. [Complete compact results](results/) include negative cases and controls.

The separate original full-test reproduction was 43.73% learned versus 40.19% random, **inconclusive**. It uses a different cohort and is not a paired baseline for the table above.

## Try the standalone rule

Python 3.8+; these commands need no ML dependencies or downloaded data:

```bash
python -m unittest discover -s tests -v
python -m tools.apply_reallocation --input examples/reallocation_input.json --output work/reallocated.json
python -m tools.check_results
```

These run synthetic geometry checks and recount saved results. Full model reruns require the historical experiment source and assets; see [reproduction instructions](docs/REPRODUCING.md).

## Repository guide

| Path | Contents |
|---|---|
| [methods/](methods/) | Current reallocation algorithm |
| [results/](results/) | Seed metrics, all compact case counts, controls and provenance |
| [tools/](tools/) and [tests/](tests/) | JSON interface, result checks and synthetic tests |
| [docs/](docs/) | Method, reproduction, grading evidence and archive guide |
| `models/`, `modules/`, `utils/`, `run.py` | Original MonTi implementation |
| [environment/](environment/) | Recorded ML environment |
| `course/`, `evidence/` | Baseline figure sources and publication verification |

Detailed logs, superseded proposals and outdated presentation drafts remain in the [historical archive](docs/ARCHIVE.md), outside the current working tree.

Upstream authors: Jinhyeok Choi, Heehyeon Kim and Joyce Jiyoung Whang, AAAI 2025. [License](LICENSE.txt) · [AI use](AI_USAGE.md) · [Contributions](CONTRIBUTIONS.md).
