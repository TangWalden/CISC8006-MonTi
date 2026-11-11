# MonTi: Reproduction and Controlled Extensions

A CISC8006 study of [MonTi](https://github.com/bdi-lab/MonTi). **No improvement over the strong random target-fill control is established.** [Grading evidence and remaining gaps](docs/ASSESSMENT.md).

## Latest experiment: feature placement, no measured gain

Keep the randomly target-filled graph and the complete generated feature set fixed; change only which injected position receives each feature vector. A separate frozen GAGA surrogate guides a one-to-one assignment. [Method, controls and limits](experiments/feature_assignment/).

GossipCop-S, split 0, development VALID; **higher miss rate (MR) means a stronger attack**. MR includes targets already missed before injection.

| Condition | Mean official-GAGA MR |
|---|---:|
| Random target-fill graph, original feature positions | 55.0576% |
| Relation-model feature assignment (primary) | 55.0576% |
| GCN feature proposal, common acceptance filter | 55.0576% |
| Cycle-matched random feature permutation, common filter | 55.0576% |

All four conditions have identical class predictions on all 3,213 target occurrences. The primary passes its own-model acceptance filter in 276/384 cases, but changes no own-GAGA or official-GAGA class. **Gain: 0 pp; frozen qualification failed.** A lower surrogate loss is not a detection result. [Seed/case tables and provenance](results/feature_assignment/).

Scope: 128 unchanged groups per seed, 1,071 occurrences and 323 unique targets; frozen generator seeds 3/4/5. Repeated news and previous development inspection limit generalization. The extra own-GAGA model changes the original GCN-only attacker knowledge. No training updates, experimental API calls, GPU work or final TEST queries occurred in this phase.

## Earlier evidence

Edge reallocation previously gained +5.48 pp on a different development cohort, but random target filling was slightly higher and the stability criterion failed. [Earlier method](docs/METHOD.md) · [Preserved results](results/README.md).

The separate full-test reproduction remains 43.73% learned versus 40.19% random, **inconclusive**. It is not the paired baseline for the latest experiment. [Reproduction and paper comparison](docs/REPRODUCING.md).

## Check saved results

These commands use the standard library; assignment numerical tests additionally need the recorded NumPy/SciPy environment.

```bash
python -m tools.check_results
python -m tools.feature_assignment_results check --results results/feature_assignment
python -m unittest discover -s tests -v
```

Missing optional assignment dependencies produce explicit test skips. Full model reruns need the frozen historical runtime and assets; see [reproduction instructions](docs/REPRODUCING.md).

## Repository guide

| Path | Contents |
|---|---|
| [methods/](methods/) | Edge reallocation and bijective feature assignment |
| [experiments/feature_assignment/](experiments/feature_assignment/) | Bounded runner, TRAIN preflight and separate saved-record auditor |
| [results/](results/) | Compact complete comparisons, controls and provenance |
| [tools/](tools/) and [tests/](tests/) | Saved-count checks and synthetic contracts |
| [docs/](docs/) and [environment/](environment/) | Method, setup, assessment mapping and recorded environment |
| `models/`, `modules/`, `utils/`, `run.py` | Original MonTi implementation |

Detailed logs, tensors, superseded proposals and outdated presentation drafts stay outside this working tree. [Archive guide](docs/ARCHIVE.md). Raw attempts are retained; agent checks are not human verification or course completion.

Upstream authors: Jinhyeok Choi, Heehyeon Kim and Joyce Jiyoung Whang, AAAI 2025. [License](LICENSE.txt) · [AI use](AI_USAGE.md) · [Contributions](CONTRIBUTIONS.md).
