import json

import numpy as np
from scipy.stats import t

from course.runner import ROOT, sha256, write_json


def main():
    protocol_path = ROOT / 'course/improvements/pilot_protocol_v1.json'
    protocol = json.loads(protocol_path.read_text())
    registry, cells = [], {}
    for path in sorted((ROOT / 'evidence/improvements').glob('*/registered.json')):
        registration = json.loads(path.read_text())
        if registration['phase'] != 'improvement_pilot':
            continue
        status_path = path.with_name('result.json')
        status = json.loads(status_path.read_text()) if status_path.exists() else dict(status='running_or_interrupted')
        row = dict(registration=registration, status=status, evidence=str(path.parent.relative_to(ROOT)))
        registry.append(row)
        cells.setdefault((registration['variant'], registration['seed']), []).append(row)
    write_json(ROOT / 'evidence/improvement_pilot_registry.json', registry)
    for variant in protocol['variants']:
        for seed in protocol['seeds']:
            attempts = cells.get((variant, seed), [])
            if len(attempts) != 1 or attempts[0]['status']['status'] != 'completed':
                raise RuntimeError('Incomplete, failed or repeated pilot cell requires review: %s/%s' % (variant, seed))
            assert attempts[0]['registration']['protocol_sha256'] == sha256(protocol_path)
    scores = {key: rows[0]['status']['best_validation']['misclf_rate']*100 for key, rows in cells.items()}
    effects = {}
    for variant in ['coverage', 'size_sampling']:
        deltas = np.array([scores[variant, seed]-scores['baseline', seed] for seed in protocol['seeds']])
        mean, sd = float(deltas.mean()), float(deltas.std(ddof=1))
        half = float(t.ppf(.975, 2)*sd/np.sqrt(3))
        effects[variant] = dict(differences_pp=deltas.tolist(), mean_gain_pp=mean, sample_sd_pp=sd,
                               exploratory_paired_95ci_pp=[mean-half, mean+half])
    selected = 'coverage' if effects['coverage']['mean_gain_pp'] >= effects['size_sampling']['mean_gain_pp']-.25 else 'size_sampling'
    summary = dict(selected=selected, effects=effects,
                   rows=[dict(seed=seed, **{variant: scores[variant, seed] for variant in protocol['variants']}) for seed in protocol['seeds']],
                   units='GCN validation MR percent; difference in percentage points',
                   selection_rule=protocol['choice_rule'], test_evaluated=False,
                   classroom_approval=protocol['classroom_approval'],
                   limitation='Small-subset exploratory pilots choose a direction; these scores are not full test improvements.',
                   protocol_sha256=sha256(protocol_path),
                   result_hashes=[dict(evidence=x['evidence'], sha256=sha256(ROOT / x['evidence'] / 'result.json')) for x in registry])
    write_json(ROOT / 'evidence/improvement_pilot_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
