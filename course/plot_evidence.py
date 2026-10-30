"""Render English figures from existing audited summaries; no model evaluation."""
import csv
import hashlib
import json
import pathlib
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'docs/figures'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(figure, name):
    figure.savefig(OUTPUT/(name+'.svg'), metadata={'Date': None, 'Creator': 'MonTi course evidence figures'})
    figure.savefig(OUTPUT/(name+'.png'), dpi=200)
    plt.close(figure)


def style(axis):
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.spines['bottom'].set_color('#94a3b8')
    axis.grid(axis='x', color='#e2e8f0', linewidth=.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis='both', length=0, labelcolor='#334155')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'text.color': '#172b3a', 'axes.labelcolor': '#334155',
                         'svg.fonttype': 'none', 'svg.hashsalt': 'monti-course-evidence-v1',
                         'savefig.facecolor': 'white'})
    baseline_path = ROOT/'evidence/reproduction_summary.json'
    pilot_path = ROOT/'evidence/improvement_pilot_summary.json'
    baseline = json.loads(baseline_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.25), gridspec_kw={'width_ratios': [1, 1.16]})
    fig.subplots_adjust(left=.09, right=.97, top=.73, bottom=.20, wspace=.34)
    fig.text(.06, .94, 'Reproduction: positive seed differences, uncertain mean effect', fontsize=17, weight='bold')
    fig.text(.06, .875, 'GossipCop-S / split 0 / GCN surrogate to GAGA victim / three paired seeds', fontsize=10.5, color='#475569')
    colors = ['#94a3b8', '#2166ac']
    ax = axes[0]
    for index, row in enumerate(baseline['rows']):
        y = 2-index
        ax.plot([row['random_mr_percent'], row['learned_mr_percent']], [y, y], color='#cbd5e1', lw=2.5)
        ax.scatter(row['random_mr_percent'], y, color=colors[0], s=55, label='Random selector' if index == 0 else None, zorder=3)
        ax.scatter(row['learned_mr_percent'], y, color=colors[1], s=55, label='Learned selector' if index == 0 else None, zorder=3)
    ax.set_yticks([2, 1, 0], ['Seed 0', 'Seed 1', 'Seed 2'])
    ax.set_ylim(-.55, 2.9)
    ax.set_xlim(37, 46)
    ax.set_xlabel('Attacked GAGA MR (%)')
    ax.set_title('Each registered seed', loc='left', fontsize=11, weight='bold', pad=12)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    style(ax)
    ax = axes[1]
    deltas = [row['difference_pp'] for row in baseline['rows']]
    ax.scatter(deltas, [3, 2, 1], color=colors[1], s=42, zorder=3)
    mean = baseline['paired_mean_difference_pp']
    low, high = baseline['paired_95ci_pp']
    ax.errorbar(mean, 0, xerr=[[mean-low], [high-mean]], fmt='D', color='#172b3a', ecolor='#2166ac',
                markersize=6, elinewidth=2.2, capsize=5, zorder=4)
    ax.axvline(0, color='#64748b', ls='--', lw=1.2)
    ax.set_yticks([3, 2, 1, 0], ['Seed 0', 'Seed 1', 'Seed 2', 'Paired mean'])
    ax.set_xlim(-3, 10)
    ax.set_ylim(-.65, 3.7)
    ax.set_xlabel('Learned minus random (percentage points)')
    ax.set_title('95% paired t interval crosses zero', loc='left', fontsize=11, weight='bold', pad=12)
    style(ax)
    fig.text(.06, .09, 'Mean +%.2f pp; 95%% CI [%.2f, %.2f]. Pre-registered conclusion: inconclusive.' % (mean, low, high), fontsize=10.5, weight='bold')
    fig.text(.06, .045, 'Interval uses three seed differences and Student t(df=2); target occurrences are not independent seed repetitions.', fontsize=9, color='#475569')
    save(fig, 'baseline_paired_effect')

    fig, ax = plt.subplots(figsize=(10.3, 4.65))
    fig.subplots_adjust(left=.25, right=.95, top=.69, bottom=.24)
    fig.text(.065, .92, 'Pilot selection: both observed validation gains were negative', fontsize=16, weight='bold')
    fig.text(.065, .85, 'GCN validation only / common fixed subsets / paired seeds 3, 4, 5 / no GAGA test inference', fontsize=10, color='#475569')
    for y, key, color in [(1, 'coverage', '#2166ac'), (0, 'size_sampling', '#b45f06')]:
        effect = pilot['effects'][key]
        mean = effect['mean_gain_pp']
        low, high = effect['exploratory_paired_95ci_pp']
        ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt='D', color=color,
                    markersize=7, elinewidth=2.5, capsize=6, zorder=4)
        ax.annotate('%+.3f pp' % mean, (mean, y), xytext=(0, 15), textcoords='offset points',
                    ha='center', fontsize=10, color=color, weight='bold')
    ax.axvline(0, color='#64748b', ls='--', lw=1.2)
    ax.set_yticks([1, 0], ['Coverage anchors', 'Size-proportional sampling'])
    ax.set_xlim(-1.6, .55)
    ax.set_ylim(-.65, 1.65)
    ax.set_xlabel('Change relative to the paired pilot baseline (percentage points)')
    style(ax)
    fig.text(.065, .115, 'The pre-registered rule selects coverage for proposed full evaluation, including when both pilot means are negative.', fontsize=9.4)
    fig.text(.065, .055, 'Bars: exploratory 95% paired t intervals (df=2). These are surrogate-validation results, not demonstrated GAGA test improvements.', fontsize=9, color='#475569')
    save(fig, 'pilot_validation_effects')

    with (OUTPUT/'plotted_data.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=['phase', 'seed', 'condition', 'mr_percent', 'paired_gain_pp'], lineterminator='\n')
        writer.writeheader()
        for row in baseline['rows']:
            for condition in ['learned', 'random']:
                writer.writerow(dict(phase='baseline_test', seed=row['seed'], condition=condition,
                                     mr_percent=row[condition+'_mr_percent'],
                                     paired_gain_pp=row['difference_pp'] if condition == 'learned' else ''))
        for row in pilot['rows']:
            for condition in ['baseline', 'coverage', 'size_sampling']:
                writer.writerow(dict(phase='pilot_GCN_validation', seed=row['seed'], condition=condition,
                                     mr_percent=row[condition],
                                     paired_gain_pp=row[condition]-row['baseline'] if condition != 'baseline' else ''))
    provenance = dict(input_sources=[dict(path=str(path.relative_to(ROOT)), sha256=sha(path)) for path in [baseline_path, pilot_path]],
                      script='course/plot_evidence.py', script_sha256=sha(pathlib.Path(__file__)),
                      environment=dict(python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__),
                      computation='Visualization of existing audited summaries; no model training, prediction, seed addition or parameter selection',
                      figure_units='MR percent; paired effects in percentage points',
                      outputs={path.name: sha(path) for path in sorted(OUTPUT.glob('*')) if path.suffix in ['.png', '.svg', '.csv']})
    (OUTPUT/'figure_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(status='completed', figures=2, output_directory=str(OUTPUT), environment=provenance['environment']), indent=2))


if __name__ == '__main__':
    main()
