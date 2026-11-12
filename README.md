# MonTi: Staged Attack Construction

A CISC8006 reproduction and development study of [MonTi](https://github.com/bdi-lab/MonTi). Our selected extension divides attack construction into a **fixed upstream planner** and a **trained downstream candidate-edge selector**.

The upstream retains the generated auxiliary-node features and plans their target connections. The downstream learns the remaining connections to candidate nodes while the upstream stays frozen. Both stages share each case's original total node and edge budget. [Method](docs/METHOD.md).

## Main development result

GossipCop-S, split 0, three seeds on the same 64 VALID groups per seed. Higher miss rate (MR) means a stronger attack; MR includes fraud targets already missed before injection.

| Condition | Mean official-GAGA MR |
|---|---:|
| Our reproduced native MonTi | 53.6486% |
| Same projected role quotas, random candidate connections | 57.0344% |
| **Fixed upstream + trained downstream** | **58.1436%** |

The staged pipeline improves over our native baseline by **+4.4950 percentage points**. The three seed gains are **+6.1296, +4.3783 and +2.9772 pp**, with a nominal paired 95% interval of **[+0.5715, +8.4185] pp**. It passes the pre-specified development-pilot rule and shows a consistent gain over this native baseline across the three seeds.

The secondary gain over matched random candidate connections is **+1.1092 pp**, with a nominal interval of **[-3.9892, +6.2075] pp**. Superiority over that random control remains uncertain. [All four conditions, seed metrics, per-case counts, quotas and audits](results/downstream_cooperation/).

The 53.6486% baseline comes from our three trained terminal-eight MonTi checkpoints on this exact cohort. Each seed contains 571 target occurrences and 240 unique targets; 233 identities appeared in earlier development contexts. These are validation-stage results for our reproduced baseline, separate from the paper's reported test values and authors' released best checkpoint.

## Read the study

- [English report](docs/STAGED_REPORT.md) and [PDF report](docs/MonTi_Staged_Report.pdf).
- [Staged method and matched controls](docs/METHOD.md), [execution implementation](experiments/downstream_cooperation/) and [frozen protocol](evidence/downstream_cooperation_v1/execution_protocol.json).
- [Complete main-study evidence](results/downstream_cooperation/) and [supplementary research index](docs/SUPPLEMENTARY_STUDIES.md), including all retained positive, null, negative and failed attempts.
- [Reproduction and paper comparison](docs/REPRODUCING.md): the separate formal TEST study remains 43.7280% learned versus 40.1902% random, **inconclusive**. It is a different comparison from the development table above.

## Budget and execution boundaries

The requested half-target, half-candidate allocation is projected to each case's feasible interval. Total injected nodes and undirected edges stay fixed; the allocation is not literally 50:50 in every case. Per-auxiliary-node degrees can change, and the staged quota arms remove auxiliary-to-auxiliary connections retained by native MonTi. The small downstream adds 151,872 trainable parameters and training computation; equal structural budgets do not imply equal total computational cost.

The downstream alone is fitted for four epochs on 128 TRAIN groups per seed, using frozen GCN feedback. Terminal weights and all final topologies are sealed before official-GAGA development scoring. Detailed failures, actual budgets, source identities, saved-record audits and CPU costs are retained with the results. No extra seeds or fits were added to this registered pilot after viewing its outcomes. Separate studies retain their own protocols in the supplementary index.

## Repository guide

| Path | Contents |
|---|---|
| [experiments/downstream_cooperation/](experiments/downstream_cooperation/) | Staged runner, fixed-source registration, TRAIN preflight and saved-record auditors |
| [methods/downstream_quota.py](methods/downstream_quota.py) | Closest-feasible role quotas and legal deterministic hard selection |
| [results/downstream_cooperation/](results/downstream_cooperation/) | Main comparison, all seed/case exports and provenance |
| [docs/SUPPLEMENTARY_STUDIES.md](docs/SUPPLEMENTARY_STUDIES.md) | Index of the other preserved studies and their conclusions |
| [tests/](tests/) and [environment/](environment/) | Synthetic contracts and recorded execution environment |
| `models/`, `modules/`, `utils/`, `run.py` | Original MonTi implementation |

[Setup and saved-result verification](docs/REPRODUCING.md) describe the available checks and their scope. Full ML reruns require the recorded historical runtime and frozen assets; compact exports are not a self-contained fresh-clone model bundle. [Archive guide](docs/ARCHIVE.md).

[Course evidence and remaining requirements](docs/ASSESSMENT.md), [AI use](AI_USAGE.md) and [contributions](CONTRIBUTIONS.md).

Upstream authors: Jinhyeok Choi, Heehyeon Kim and Joyce Jiyoung Whang, AAAI 2025. [License](LICENSE.txt).
