import json
import pathlib
import platform

import dgl
import numpy as np
import torch

from course.runner import ROOT, sha256, write_json
from utils.utils import load_data


def main():
    data = load_data('gossipcop_0.05', split=0)
    graph = data['graph']
    if not torch.isfinite(graph.ndata['x']).all():
        raise RuntimeError('Nonfinite input features')
    report = dict(nodes=graph.num_nodes(), edges=graph.num_edges(), feature_shape=list(graph.ndata['x'].shape),
                  label_counts=torch.bincount(graph.ndata['y'].long()).tolist(),
                  target_assignment_shape=list(data['target_assignment'].shape), partitions={})
    for split in ('train', 'valid', 'test'):
        nodes = data['idx_' + split]
        groups = data['target_idx_' + split]
        assignment = data['target_assignment'][groups]
        selected_nodes = assignment.nonzero()[1]
        if len(set(nodes.tolist())) != len(nodes):
            raise RuntimeError('Duplicate partition node indices')
        if not np.isin(selected_nodes, nodes).all():
            raise RuntimeError('Target group crosses its registered node partition')
        if not torch.all(graph.ndata['y'][torch.tensor(selected_nodes)] == 1):
            raise RuntimeError('Nonfraud node in target set')
        report['partitions'][split] = dict(nodes=len(nodes), groups=len(groups), target_occurrences=int(assignment.sum()))
    for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        if np.intersect1d(data['idx_' + left], data['idx_' + right]).size:
            raise RuntimeError('Overlapping node partitions')
        if np.intersect1d(data['target_idx_' + left], data['target_idx_' + right]).size:
            raise RuntimeError('Overlapping target group partitions')
    files = []
    for path in sorted((ROOT / 'Datasets/GossipCop').rglob('*')):
        if path.is_file():
            files.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path)))
    for relative in ['models/pretrained_models/surrogate_models/GCN-gossipcop_0.05-0.pickle',
                     'models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle']:
        path = ROOT / relative
        pointer = ROOT / 'downloads/MonTi-10f233e953679ec3715125dc058ef6d28bee4691' / relative
        expected = next(line.split('sha256:')[1] for line in pointer.read_text().splitlines() if line.startswith('oid sha256:'))
        actual = sha256(path)
        if expected != actual:
            raise RuntimeError('Checkpoint hash differs from official LFS pointer: ' + relative)
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=actual, verified_against='official Git LFS pointer'))
    for relative in ['models/pretrained_models/configs/GCN-gossipcop_0.05-0.json',
                     'models/pretrained_models/configs/GAGA-gossipcop_0.05-0.json',
                     'best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json']:
        files.append(dict(path=relative, sha256=sha256(ROOT / relative)))
    (ROOT / 'evidence/data').mkdir(parents=True, exist_ok=True)
    write_json(ROOT / 'evidence/data/input_manifest.json', files)
    write_json(ROOT / 'evidence/data/partition_audit.json', report)
    original = json.loads((ROOT / 'best_models/configs/GCN-GAGA-gossipcop_0.05-best_config.json').read_text())
    if (original['dataset'], original['surrogate_model'], original['victim_model'], original['split']) != ('gossipcop_0.05', 'GCN', 'GAGA', 0):
        raise RuntimeError('Official configuration does not match registered scope')
    (ROOT / 'course/configs').mkdir(exist_ok=True)
    write_json(ROOT / 'course/configs/baseline.json', original)
    print(json.dumps(report, indent=2), flush=True)
    print('OFFICIAL_CONFIG', json.dumps(original), flush=True)


if __name__ == '__main__':
    main()
