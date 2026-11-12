# Completed staged TEST comparison

The registered full comparison completed on all 909 TEST groups, with
10,515 target occurrences and 1,221 unique targets per seed. Upstreams are
our frozen reproduction checkpoints, seeds 0/1/2, not the author checkpoint
used in the newer local-swap development study.

| Official GAGA MR | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Native reproduced MonTi | 44.5459% | 43.0623% | 43.8421% | 43.8168% |
| Quota random completion | 43.6519% | 43.9848% | 44.4127% | 44.0165% |
| Trained downstream | 46.1531% | 46.2292% | 40.2948% | 44.2257% |

Trained minus native: **+0.4089 pp**, nominal paired t95
**[-8.3199, 9.1378] pp**. Trained minus random: **+0.2092 pp**, interval
**[-9.1054, 9.5238] pp**. Seed 2 is negative against both controls.
The predeclared criterion **fails**: no stable full-test improvement is
established. The earlier +4.4950 pp development difference does not replace
this result. Different checkpoints and cohorts prevent interpreting their
absolute rates as a paired deterioration.

The root-authored [saved-record audit](saved_audit.json) passed 8,276 file
hashes, 9,090 evaluation records and 11,244 training visits. It did not
deserialize tensors or independently repeat physical-graph evaluation.
Agents are not human reviewers. Main execution took 30,400.43 wall seconds;
GPU/API use was zero. Actual downstream updates: 10,992. The 252 no-choice
training visits remain recorded.

[Metrics, timing, counters and provenance](summary.json) ·
[Frozen protocol](../../docs/FINAL_EVALUATION.md)

This TEST set has now been observed. Future adaptations cannot treat it as
an untouched confirmation set, or repeat it until a favorable result appears.
