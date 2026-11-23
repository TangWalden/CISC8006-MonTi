# MonTi: Frozen Upstream, Relation-Aware Retrieval

CISC8006 reproduction and extension of [MonTi](https://github.com/bdi-lab/MonTi).
We keep the author's attack checkpoint fixed and optimize at most two downstream
candidate endpoints under the same physical node/edge budgets.

## Complete TEST results

The frozen two-dataset mean criterion **passed**. Higher GAGA miss rate (MR)
means a stronger attack; the original checkpoint below already generates attacks.

| Method | GossipCop MR | YelpChi MR |
|---|---:|---:|
| Original author checkpoint | 44.2638% | 30.9098% |
| Frozen two-channel retrieval | 49.1932% | 37.6957% |
| Same-class equal-dose random | 44.1591% | 32.9605% |
| Joint-pool equal-dose random | 44.2194% | 31.1708% |

Current-method gains over the author checkpoint are **+4.9295 / +6.7860 pp**;
over same-class random, **+5.0341 / +4.7353 pp**.
[All repeats, contrasts, negative cases and audit](results/channel_full/).

- Complete original cohorts: GossipCop 909 groups / 10,515 target occurrences;
  YelpChi 548 groups / 2,682 occurrences per repeat. Repetitions 0/1/2 share
  fixed author and teacher checkpoints; repeated target identities are correlated.
- Parameters and cohorts were fixed before this test. Both datasets' actions
  were sealed before either official evaluation. No test-driven retuning occurred.
- GossipCop TEST has historical exposure; it is not an untouched holdout.
  [Development scores](results/channel_retrieval/) use different groups and
  must not be subtracted from these full-test rates.

## Method and contribution

A fixed independent TRAIN-fitted teacher evaluates label-grouped relation
evidence. Its input gradient ranks real endpoints from both known TRAIN classes;
the search then checks shortlisted substitutions. Target memberships/order,
direct target attachments, injected features/degrees and node/edge totals stay
fixed. Same-class random matches chosen classes, slots and exact edge dose;
joint-pool random matches slots/dose without fixing class frequency.

GAGA's existing label grouping, HotFlip-style discrete gradient search and the
typed-message idea behind HGT inform the design. This is a downstream optimizer,
not a new HGT architecture or a first-novelty claim. Expanded endpoint/context
and GAGA-surrogate access exceed original GCN-only MonTi; additional search
compute is disclosed. Equal physical budgets do not mean equal computation.

The separate six-run reproduction remains **inconclusive**: learned candidates
43.7280%, random candidates 40.1902%, paired interval [-1.5415, 8.6171] pp.
That random-candidate baseline differs from these downstream controls. Earlier
null/negative attempts are retained in the supplementary index.

## Project entries

| Purpose | Entry |
|---|---|
| Course scoring points and genuine remaining requirements | [Assessment mapping](docs/ASSESSMENT.md) |
| Fixed method and full runner | [Method](experiments/channel_retrieval/), [full evaluation](experiments/channel_full/) |
| Results, controls and provenance | [Full TEST](results/channel_full/), [development](results/channel_retrieval/) |
| Setup and historical reproduction | [Reproducing](docs/REPRODUCING.md) |
| Other studies and operational checks | [Supplementary index](docs/SUPPLEMENTARY_STUDIES.md), [GPU timing check](results/channel_gpu/) |
| Current slides and planned presentation | [PowerPoint](docs/slides/MonTi_Full_Results.pptx), [editable Bento HTML](docs/slides/MonTi_Full_Results.bento.html), [guide](docs/SLIDES_GUIDE.md) |
| Junyi's eight-minute experiments section | [PowerPoint](docs/slides/MonTi_Experiments_Junyi.pptx), [editable Bento HTML](docs/slides/MonTi_Experiments_Junyi.bento.html). Slide 7 follows the same nine YelpChi reviews under both edits. |
| Report draft | [Dated report](docs/STAGED_REPORT.md) - still needs the latest result addendum |
| AI assistance and actual human contributions | [AI use](AI_USAGE.md), [contributions](CONTRIBUTIONS.md) |

Compact source/results are public; large models, graph cards and raw logs stay
in the recorded archive. Instructor approval, human technical review, verified
contributions, rehearsal/defense and final submission remain separate.
[License](LICENSE.txt).
