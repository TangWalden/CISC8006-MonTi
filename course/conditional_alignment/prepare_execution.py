"""Derive a checksum-frozen alignment execution file from the committed pilot."""
import argparse
import json
import pathlib
import subprocess
from course.conditional_alignment.common import ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference')
    args = parser.parse_args()
    target = pathlib.Path(args.output).resolve()
    if (ROOT / 'runs').resolve() not in target.parents or target.exists():
        raise RuntimeError('Use a new execution file beneath ignored runs/.')
    master = json.loads((ROOT / 'course/NEW_METHODS_EXECUTION_PROTOCOL_V1.json').read_text())
    stage = master['alignment']
    original = json.loads((ROOT / 'evidence/data/input_manifest.json').read_text())
    inputs = []
    for row in original:
        name = row['path']
        if ('Datasets/GossipCop/' in name and '_gaga_' not in name) or ('GCN-gossipcop_0.05-0' in name and '/pretrained_models/' in name):
            inputs.append(dict(path=name, sha256=row['sha256']))
    code_paths = list((ROOT / 'course/conditional_alignment').glob('*.py'))
    code_paths += [ROOT / p for p in ['course/new_methods_guard.py',
        'models/attack_model.py', 'models/victim_model.py', 'modules/data_handler.py',
        'modules/experiment_handler.py', 'modules/result_manager.py', 'modules/schedulers.py',
        'utils/utils.py', 'utils/constants.py', 'course/NEW_METHODS_EXECUTION_PROTOCOL_V1.json']]
    protocol = dict(status='frozen', owner_execution_authorized=True,
        source_commit_at_registration=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        arms=stage['arms'], seeds=stage['seeds'], train_groups=stage['train_groups'], valid_groups=stage['valid_groups'],
        epochs=stage['epochs'], patience=stage['patience'], alignment_lambda=stage['lambda'], job_wall_seconds=stage['maximum_seconds_per_job'],
        export_valid_groups=master['fusion_probe']['groups'],
        physical_gpu_uuid=master['gpu_physical_uuid'],
        reference=dict(min_count=8, gate_quantile=.9, scale_floor=.05, chunk_size=4096, degree_rule='floor_log2_nonself'),
        base_config_sha256=sha256(ROOT / 'course/configs/baseline.json'),
        inputs=inputs, code=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256(p)) for p in sorted(set(code_paths))],
        test_used=False, victim_training_or_query=False)
    if args.reference:
        protocol['reference_sha256'] = sha256(args.reference)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, protocol, exclusive=True)
    print(json.dumps(dict(protocol=str(target.relative_to(ROOT)), sha256=sha256(target), training_ready=bool(args.reference))))


if __name__ == '__main__':
    main()
