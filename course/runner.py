import argparse
import datetime as dt
import functools
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np
import torch

from modules.data_handler import DataHandler
import modules.experiment_handler as upstream
from modules.experiment_handler import ExperimentHandler
from modules.result_manager import ResultManager
from utils.utils import get_n_ids, set_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


class AuditedExperiment(ExperimentHandler):
    def __init__(self, *args, evidence, phase, **kwargs):
        self.evidence, self.phase, self.evaluation_number = evidence, phase, 0
        super().__init__(*args, **kwargs)

    def get_perturbed_graph(self, n_ids, node_budget, edge_budget, training=False, need_sequence=False):
        output = super().get_perturbed_graph(n_ids, node_budget, edge_budget, training, need_sequence)
        if output is None:
            raise RuntimeError('Attack generation failed; refusing to silently remove a target group.')
        graph = output[0] if need_sequence else output
        if graph.num_nodes() != self.data.graph.num_nodes() + node_budget:
            raise RuntimeError('Injected node count violates registered budget.')
        edge_difference = graph.num_edges() - self.data.graph.num_edges() - node_budget
        if edge_difference < 0 or edge_difference % 2 or edge_difference // 2 > edge_budget:
            raise RuntimeError('Injected undirected edge count violates registered budget.')
        return output

    def check_performance(self, model='surrogate', set_type='valid', attack=False):
        if self.phase == 'pilot' and set_type == 'test':
            raise RuntimeError('Pilot may not inspect the test split.')
        self.evaluation_number += 1
        selected = self.data.get_validation_set() if set_type == 'valid' else self.data.get_test_set()
        need_sequence = (self.surrogate_type if model == 'surrogate' else self.victim_type) == 'GAGA'
        name = '%03d_%s_%s_%s' % (self.evaluation_number, model, set_type, 'attack' if attack else 'clean')
        wrong, count, groups, losses = 0, 0, 0, []
        with (self.evidence / (name + '.jsonl')).open('x') as stream, torch.no_grad():
            for target_id in selected:
                node_ids = torch.tensor(get_n_ids(self.data.target_assignment, target_id), dtype=torch.long)
                graph, sequences = self.data.graph, None
                nodes = int(self.data.node_budgets[target_id]) if attack else 0
                edges = int(self.data.edge_budgets[target_id]) if attack else 0
                if attack:
                    output = self.get_perturbed_graph(node_ids, nodes, edges, need_sequence=need_sequence)
                    if need_sequence:
                        graph, sequences = output
                    else:
                        graph = output
                    losses.append(float(self.loss_function(graph, node_ids, training=False, sequences=sequences)))
                elif need_sequence:
                    sequences = self.data.sequences[node_ids].to(self.device)
                labels = graph.ndata['y'][node_ids].cpu().numpy()
                logits = self.get_logits(graph, node_ids, model, training=False, sequences=sequences)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite evaluation logits.')
                predictions = logits.argmax(1).cpu().numpy()
                errors = int((labels != predictions).sum())
                count += len(labels)
                wrong += errors
                groups += 1
                stream.write(json.dumps(dict(target_group=int(target_id), node_ids=node_ids.tolist(),
                                             labels=labels.tolist(), predictions=predictions.tolist(),
                                             incorrect=errors, node_budget=nodes, edge_budget=edges)) + '\n')
                if groups % 100 == 0:
                    print('EVAL', name, groups, '/', len(selected), flush=True)
        if groups != len(selected) or count == 0:
            raise RuntimeError('Incomplete or empty evaluation.')
        result = dict(accuracy=1-wrong/count, misclf_rate=wrong/count)
        if attack:
            result['loss'] = float(np.mean(losses))
        write_json(self.evidence / (name + '.json'), dict(result=result, groups=groups,
                   target_occurrences=count, incorrect=wrong, weighting='target occurrences, size weighted',
                   prediction_sha256=sha256(self.evidence / (name + '.jsonl'))))
        print('EVALUATION', name, json.dumps(result), flush=True)
        return result

    def test(self):
        if self.phase == 'pilot':
            print('PILOT: final test evaluation deliberately skipped.', flush=True)
            return dict(test_skipped=1.0)
        return super().test()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', required=True, choices=['learned', 'random'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--phase', required=True, choices=['pilot', 'reproduction'])
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--valid-limit', type=int, default=0)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    if args.phase == 'reproduction' and (args.train_limit or args.valid_limit or args.epochs):
        parser.error('Reproduction requires a frozen full configuration without CLI scale overrides.')
    torch.set_num_threads(8)
    upstream.tqdm = functools.partial(upstream.tqdm, disable=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + args.phase + '_' + args.mode + '_s' + str(args.seed) + '_' + uuid.uuid4().hex[:6]
    evidence = ROOT / 'evidence' / 'runs' / run_id
    evidence.mkdir(parents=True, exist_ok=False)
    config = json.loads(pathlib.Path(args.config).read_text())
    for key, default in dict(shared_parameters=False, without_candidates=False, pos_encoding=True, ada_budget=True).items():
        config.setdefault(key, default)
    config.update(seed=args.seed, cuda_id=args.gpu, random_candidates=args.mode == 'random', exp_name=run_id)
    if args.epochs:
        config['epochs'] = args.epochs
    config['config_path'] = str(evidence.relative_to(ROOT) / 'config.json')
    write_json(evidence / 'config.json', config)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    registered = dict(run_id=run_id, registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      commit=commit, tracked_changes=dirty, command=[sys.executable, '-m', 'course.runner'] + sys.argv[1:],
                      config_sha256=sha256(evidence / 'config.json'), mode=args.mode,
                      candidate_sampling='uniform without replacement, resampled every forward in train and evaluation' if args.mode == 'random' else 'upstream learned selector',
                      seed=args.seed, phase=args.phase, gpu=args.gpu,
                      torch=torch.__version__, gpu_name=torch.cuda.get_device_name(args.gpu),
                      cost_unit='one GPU wall-clock hour; monetary price unavailable')
    write_json(evidence / 'registered.json', registered)
    started = time.monotonic()
    print('REGISTERED', run_id, flush=True)
    try:
        if args.phase == 'reproduction':
            if dirty:
                raise RuntimeError('Tracked source differs from its registered commit.')
            if not (ROOT / 'course/frozen_protocol.json').exists():
                raise RuntimeError('Formal reproduction requires frozen_protocol.json.')
            frozen = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
            if sha256(args.config) != frozen['source_config_sha256'] or args.seed not in frozen['seeds']:
                raise RuntimeError('Configuration or seed differs from frozen protocol.')
        set_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        torch.cuda.reset_peak_memory_stats(args.gpu)
        data = DataHandler(config)
        if args.phase == 'pilot':
            selection = np.random.RandomState(1729)
            if args.train_limit:
                data.target_idx_train = np.sort(selection.choice(data.target_idx_train, min(args.train_limit, len(data.target_idx_train)), replace=False))
            if args.valid_limit:
                data.target_idx_valid = np.sort(selection.choice(data.target_idx_valid, min(args.valid_limit, len(data.target_idx_valid)), replace=False))
        write_json(evidence / 'target_groups.json', dict(train=data.target_idx_train.tolist(), valid=data.target_idx_valid.tolist(), test_count=len(data.target_idx_test)))
        manager = ResultManager(config, root_dir=str(ROOT / 'runs' / run_id))
        experiment = AuditedExperiment(config, data, manager, evidence=evidence, phase=args.phase)
        clean = experiment.check_performance('victim', 'valid' if args.phase == 'pilot' else 'test', attack=False)
        result = experiment.run()
        status = dict(status='completed', phase=args.phase, result=result, clean=clean,
                      best_epoch=experiment.es_scheduler.b_epoch,
                      train_and_validation_seconds=manager.elapsed_time,
                      completed_epochs=manager.epoch-1,
                      train_groups=len(data.target_idx_train), valid_groups=len(data.target_idx_valid),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.gpu),
                      checkpoint=str(pathlib.Path(manager.best_model_path).relative_to(ROOT)),
                      checkpoint_sha256=sha256(manager.best_model_path))
    except Exception as error:
        traceback.print_exc()
        status = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
    status['elapsed_seconds'] = time.monotonic() - started
    status['gpu_wall_hours'] = status['elapsed_seconds'] / 3600
    write_json(evidence / 'result.json', status)
    print('FINAL', json.dumps(status), flush=True)
    if status['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
