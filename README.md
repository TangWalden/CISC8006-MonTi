# CISC8006: MonTi Reproduction and Improvements

Team: **Junyi Tang** and **Yifan Fan**. This course project studies one component of [MonTi](https://github.com/bdi-lab/MonTi), the AAAI 2025 paper by Jinhyeok Choi, Heehyeon Kim, and Joyce Jiyoung Whang. This is a course reproduction repository. Author credits and the upstream license are preserved.

**Current baseline conclusion: inconclusive.** All six pre-registered runs are complete. On GossipCop-S, split 0, with fixed official GCN surrogate and GAGA victim checkpoints, learned candidate selection achieves **43.73 ± 0.75% MR**, versus **40.19 ± 1.30%** for the defined random selector (sample standard deviation, three seeds). The mean paired difference is **+3.54 percentage points**, with a **95% interval of [−1.54, 8.62]**. Higher MR means a stronger attack. A positive mean alone does not establish the claim.

| Seed | Learned MR | Random MR | Paired difference, pp |
|---|---:|---:|---:|
| 0 | 44.2891% | 39.0870% | +5.2021 |
| 1 | 42.8816% | 41.6262% | +1.2553 |
| 2 | 44.0133% | 39.8573% | +4.1560 |

Paper Table 5 reports 43.70% / 39.13%. Our comparison corrects the candidate flag mapping and explicitly defines random sampling during training and evaluation. Historical ablation semantics remain unconfirmed. The incidence-based budget, unresolved Table 4/5 difference, fixed detectors, and limited uncertainty scope are documented in the protocol. Numerical proximity does not remove these differences.

## Evidence and project status

- [Frozen baseline protocol](course/frozen_protocol.json), [results](evidence/reproduction_summary.json), and [registered baseline runs](evidence/run_registry.json).
- [Independent prediction recount](evidence/baseline_audit.json), [public per-group error counts](evidence/published_baseline), [training logs](evidence/training_logs), and [failed setup and check attempts](evidence/bootstrap).
- [Grading audit](course/GRADING_AUDIT.md), [one-week plan](course/WEEK_PLAN.md), [three improvement proposals](course/IMPROVEMENT_PROPOSAL.md), and [work status](course/PROJECT_STATUS.json).
- [Train/validation diagnosis](course/TRAINING_DIAGNOSIS.md), [pre-run pilot protocol](course/improvements/pilot_protocol_v1.json), and [improvement implementation and tests](course/improvements).
- [Agent-use record](AI_USAGE.md), [planned and actual contributions](CONTRIBUTIONS.md), and [baseline report](docs/baseline_report.html).

The improvement pilots compare the common baseline, structural-coverage candidates, and group-size-proportional sampling using seeds 3/4/5. They use GCN validation only. Full improvement evaluation and mechanism controls require a separate freeze after the registered selection. Course approval remains unconfirmed; technical research pilots are not presented as approved course milestones.

Public documentation and newly generated logs are in English. English translations of earlier agent records identify their original source, hashes, and translation status. Original historical evidence is preserved; translated wording is not presented as verbatim API output.

## Environment and quick check

The verified environment is Linux, Python 3.8.20, PyTorch 1.13.1+cu117, DGL 1.1.2+cu117, NumPy 1.24.4, SciPy 1.10.1, and CUDA 11.7. See the [package lock](environment/pip-freeze.txt) and [runtime record](environment/runtime.json).

The wrapper defaults to the existing server environment. On another matching Linux installation, set `MONTI_PYTHON` to the absolute Python executable path. It sets process-local CUDA library paths and thread limits.

```bash
MONTI_PYTHON=/absolute/path/to/monti/bin/python bash course/run.sh -m course.test_protocol
bash course/run.sh -m course.improvements.test_policies
```

The first command checks the weighted MR known answer, actual selector dispatch, gradients, injection budgets, CPU/GPU execution, and failure handling with synthetic inputs. Six baseline checks and five improvement checks have passed. It does not download the dataset. A fresh-environment installation has not yet been independently verified; the recorded runs reuse the locked server environment.

Obtain the data and pretrained models through the [upstream setup instructions](docs/UPSTREAM_README.md). Preserve the expected directory layout. [Asset records](evidence/bootstrap/selected_assets.json) and the [partition audit](evidence/data/partition_audit.json) record sources and checksums. Use domestic Chinese mirrors when installing dependencies; no new dependencies were installed for these runs.

```bash
# Data audit, after placing the official assets in the expected locations
bash course/run.sh -m course.prepare

# Example of one full baseline run; requires the frozen files and a clean checkout
bash course/run.sh -m course.runner --config course/configs/baseline.json --phase reproduction --mode learned --seed 0 --gpu 0
```

Every reported run links to its commit, exact command, configuration, seed, runtime, GPU wall-clock hours, memory, saved output, and checkpoint checksum. `reproduction-protocol-v1` identifies pre-run commit `9f782dfbb98581e8bbe3c10f5ea5de91237e09ab`; `reproduction-results-v1` archives baseline results. `improvement-pilot-protocol-v1` identifies the improvement pilot freeze. These technical tags are separate from the course's unconfirmed common Week 12 freeze.

## Data, licenses, and remaining work

Large datasets, checkpoints, and node-level predictions remain outside Git. Published group-level counts allow MR recounts; prediction hashes identify the retained originals. The upstream code uses CC BY-NC 4.0, with its [license](LICENSE.txt) retained. A separate data license has not been established, so the code license is not assumed to cover data and the raw dataset is not redistributed here.

Improvement experiments, ablations, the final report, slides, and demo are in progress. Instructor approval, the latest course rules, actual human checks, individual contributions, and the assessed defense remain pending. Planned work and agent-generated work are not recorded as completed human contributions.

## Citation and upstream attribution

Jinhyeok Choi, Heehyeon Kim, and Joyce Jiyoung Whang. *Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks against GNN-Based Fraud Detectors.* AAAI 2025, pp. 16028–16036. [DOI: 10.1609/aaai.v39i15.33760](https://ojs.aaai.org/index.php/AAAI/article/view/33760).

Upstream source revision: `10f233e953679ec3715125dc058ef6d28bee4691`. The original upstream README is retained in [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).
