# Audited evidence figures

These figures visualize existing saved summaries. They do not add model evaluations, training seeds, or evidence of GAGA test improvement.

## Reproduction

![Paired baseline results and confidence interval](baseline_paired_effect.png)

Learned and random selector MR for every registered seed, followed by the seed differences and their paired Student t(df=2)95% interval. The three observed differences are positive, but the mean-effect interval crosses zero. The pre-registered conclusion remains inconclusive. Uncertainty concerns run randomness under a fixed split and detectors; individual target occurrences are not treated as independent seeds.

## Exploratory pilots

![GCN validation pilot effects](pilot_validation_effects.png)

Paired changes relative to the common small-subset pilot baseline, with exploratory95% t intervals from seeds3/4/5. Both observed means are negative. The rule registered before the pilots selects coverage for the proposed next phase. These GCN validation measurements are not attacked GAGA test results, and no full improvement has been established.

The owner later placed that follow-up proposal on hold and requested results-first diagnosis before choosing an improvement. The figure preserves the historical pilot outcome and rule; it does not indicate owner approval or an active full-evaluation queue.

SVG versions retain editable text and vector elements. [Plotted values](plotted_data.csv) and [source/output hashes](figure_provenance.json) support checking the figures against the run summaries.

## Regeneration

Use a separate Python3.12 plotting environment with [figures-requirements.txt](../../environment/figures-requirements.txt). Keep the frozen Python3.8 ML environment unchanged. Dependencies were obtained through the Tsinghua domestic PyPI mirror, using an isolated local target directory.

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r environment/figures-requirements.txt
python -m course.plot_evidence
```

Run the commands at the repository root in the separate plotting environment. Exact tested versions are in figure_provenance.json. The installed packages are not committed.
