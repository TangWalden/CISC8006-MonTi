import json
import pathlib
import sys

import numpy as np
from scipy.stats import t

from course.runner import ROOT, sha256, write_json


def main():
    runs = []
    for path in sorted((ROOT / 'evidence/runs').glob('*/registered.json')):
        registered = json.loads(path.read_text())
        result_path = path.with_name('result.json')
        status = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'running_or_interrupted'}
        runs.append(dict(registration=registered, status=status, evidence=str(path.parent.relative_to(ROOT))))
    write_json(ROOT / 'evidence/run_registry.json', runs)
    formal = [x for x in runs if x['registration']['phase'] == 'reproduction']
    cells = {}
    for run in formal:
        key = (run['registration']['mode'], run['registration']['seed'])
        cells.setdefault(key, []).append(run)
    failed_attempts = [dict(evidence=x['evidence'], status=x['status']) for x in formal if x['status']['status'] == 'failed']
    cells = {key: [run for run in attempts if run['status']['status'] == 'completed'] for key, attempts in cells.items()}
    seeds = [0, 1, 2]
    if not all(len(cells.get((mode, seed), [])) == 1 and cells[(mode, seed)][0]['status']['status'] == 'completed'
               for mode in ('learned', 'random') for seed in seeds):
        print('Formal three-seed comparison is not complete; no scientific conclusion.', flush=True)
        return
    rows = []
    for seed in seeds:
        learned = cells[('learned', seed)][0]['status']['result']['misclf_rate'] * 100
        random = cells[('random', seed)][0]['status']['result']['misclf_rate'] * 100
        rows.append(dict(seed=seed, learned_mr_percent=learned, random_mr_percent=random, difference_pp=learned-random))
    differences = np.array([row['difference_pp'] for row in rows])
    mean, std = float(differences.mean()), float(differences.std(ddof=1))
    half = float(t.ppf(.975, 2) * std / np.sqrt(3))
    low, high = mean-half, mean+half
    conclusion = 'supported' if low > 0 else ('not supported' if high <= 0 else 'inconclusive')
    result = dict(rows=rows, paired_mean_difference_pp=mean, paired_sample_std_pp=std,
                  failed_attempts=failed_attempts,
                  learned_mean_mr_percent=float(np.mean([r['learned_mr_percent'] for r in rows])),
                  learned_sample_std_percent=float(np.std([r['learned_mr_percent'] for r in rows], ddof=1)),
                  random_mean_mr_percent=float(np.mean([r['random_mr_percent'] for r in rows])),
                  random_sample_std_percent=float(np.std([r['random_mr_percent'] for r in rows], ddof=1)),
                  paired_95ci_pp=[low, high], conclusion=conclusion,
                  scope='fixed split 0 and fixed official surrogate/victim, corrected selector mapping',
                  paper_table5=dict(learned=43.70, random=39.13, difference_pp=4.57),
                  paper_table4_monti='41.34 +/- 1.90; unresolved difference from table 5')
    write_json(ROOT / 'evidence/reproduction_summary.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
