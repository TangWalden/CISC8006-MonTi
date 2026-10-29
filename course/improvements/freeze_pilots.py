import datetime as dt
import json

import numpy as np

from course.runner import ROOT, sha256, write_json


def main():
    destination = ROOT / 'course/improvements/pilot_protocol_v1.json'
    if destination.exists():
        raise RuntimeError('Protocol already exists; preserve it and register a new version if necessary.')
    groups = json.loads((ROOT / 'evidence/runs/20261028T122617Z_reproduction_learned_s0_772a5d/target_groups.json').read_text())
    rng = np.random.RandomState(1729)
    train = np.sort(rng.choice(groups['train'], 256, replace=False))
    valid = np.sort(rng.choice(groups['valid'], 128, replace=False))
    files = ['policies.py', 'sampling.py', 'run.py', 'queue_pilots.py', 'analyze_pilots.py', 'freeze_pilots.py']
    protocol = dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    phase='improvement research pilots', classroom_approval='not confirmed; not claimed as course-approved pilots',
                    user_authorization='2026-10-29 request to complete reproduction/improvements and public evidence within one week',
                    variants=['baseline', 'coverage', 'size_sampling'], seeds=[3, 4, 5], epochs=10, patience=5,
                    train_groups=train.tolist(), valid_groups=valid.tolist(), subset_seed=1729,
                    metric='size-weighted GCN validation MR, percent', test_evaluation=False,
                    choice_rule='Mean paired validation MR gain over corresponding pilot baseline. If gains differ by <=0.25pp choose coverage. Evaluate the selected idea even if both gains are negative.',
                    coverage='Greedy direct-target coverage, at most32 anchors, global-ID ties, learned ST top-k fills to K=128; stop anchors at zero gain',
                    sampling='p(group)=group_size/sum(group_size), with replacement; same number of group updates per epoch',
                    full_evaluation_planned='Selected idea and mechanism control, seeds0/1/2, full frozen baseline training. Separate protocol required before new test evaluation.',
                    controls=dict(coverage='random anchors matching per-group greedy anchor count plus learned complement', size_sampling='uniform sampling with replacement'),
                    pilot_gpu_hour_limit=2, all_additional_research_gpu_hour_limit=12,
                    code=[dict(path='course/improvements/'+f, sha256=sha256(ROOT / 'course/improvements' / f)) for f in files],
                    base_config_sha256=sha256(ROOT / 'course/configs/baseline.json'))
    write_json(destination, protocol)
    print('FROZEN', sha256(destination))


if __name__ == '__main__':
    main()
