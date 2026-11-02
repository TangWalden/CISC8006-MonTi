"""Freeze local input identities for the already authorized validation pilot."""
import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = pathlib.Path(args.output).resolve()
    if (ROOT / 'runs').resolve() not in target.parents:
        raise RuntimeError('Execution configuration must remain under ignored runs/.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise RuntimeError('Configuration already exists; do not silently repeat the pilot.')
    manifest = json.loads((ROOT / 'evidence/data/input_manifest.json').read_text())
    expected = {row['path']: row['sha256'] for row in manifest}
    for name in ['course/fusion_pilot/self_evidence.py', 'models/victim_model.py',
                 'modules/data_handler.py', 'utils/utils.py', 'course/new_methods_guard.py',
                 'course/NEW_METHODS_EXECUTION_PROTOCOL_V1.json']:
        expected[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    config = dict(phase='validation_pilot', graph_model='GAGA', device='cuda:0',
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  max_wall_seconds=600, max_graph_forwards=200, cpu_threads=8,
                  probe_group_ids=[1185, 1267, 1285, 1415], expected_sha256=expected)
    with target.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(path=str(target.relative_to(ROOT)),
                          sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                          source_commit=config['source_commit'])))


if __name__ == '__main__':
    main()
