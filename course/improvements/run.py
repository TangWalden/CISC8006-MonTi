"""Pre-registered improvement pilots; reuse the frozen audited evaluator."""
import argparse
import datetime as dt
import functools
import json
import pathlib
import subprocess
import sys
import time
import traceback
import uuid

import dgl
import numpy as np
import scipy
import torch

import modules.experiment_handler as upstream
from modules.result_manager import ResultManager
from course.runner import ROOT, AuditedExperiment, sha256, write_json
from course.improvements.policies import attach_coverage
from course.improvements.sampling import SamplingDataHandler
from utils.utils import set_seed

PROTOCOL = ROOT / 'course/improvements/pilot_protocol_v1.json'


class ImprovedExperiment(AuditedExperiment):
    def initialize_attack_model(self):
        model = super().initialize_attack_model()
        return attach_coverage(model, self.config['candidate_policy'])


def validate(protocol):
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Tracked files differ from registered commit: ' + dirty)
    baseline = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
    if sha256(ROOT / 'course/configs/baseline.json') != protocol['base_config_sha256']:
        raise RuntimeError('Baseline configuration changed.')
    for entry in baseline['inputs'] + baseline['scientific_code'] + protocol['code']:
        if sha256(ROOT / entry['path']) != entry['sha256']:
            raise RuntimeError('Frozen file changed: ' + entry['path'])
    runtime = json.loads((ROOT / 'environment/runtime.json').read_text())
    current = dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__)
    if any(current[k] != runtime[k] for k in current):
        raise RuntimeError('Locked environment changed.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=['baseline', 'coverage', 'size_sampling'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    if args.seed not in protocol['seeds']:
        parser.error('Seed is not pre-registered.')
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_pilot_' + args.variant + '_s%d_' % args.seed + uuid.uuid4().hex[:6]
    evidence = ROOT / 'evidence/improvements' / run_id
    evidence.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'course/configs/baseline.json').read_text())
    for key, default in dict(shared_parameters=False, without_candidates=False, pos_encoding=True, ada_budget=True).items():
        config.setdefault(key, default)
    config.update(seed=args.seed, cuda_id=args.gpu, random_candidates=False, exp_name=run_id,
                  candidate_policy='coverage' if args.variant == 'coverage' else 'baseline',
                  epochs=protocol['epochs'], patience=protocol['patience'])
    config['config_path'] = str((evidence / 'config.json').relative_to(ROOT))
    write_json(evidence / 'config.json', config)
    write_json(evidence / 'registered.json', dict(
        run_id=run_id, phase='improvement_pilot', variant=args.variant, seed=args.seed, gpu=args.gpu,
        registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        command=[sys.executable, '-m', 'course.improvements.run'] + sys.argv[1:],
        config_sha256=sha256(evidence / 'config.json'), protocol_sha256=sha256(PROTOCOL),
        classroom_approval=protocol['classroom_approval'],
        information_boundary='training and GCN validation only; test evaluation prohibited',
        runtime=dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__),
        gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu', '--format=csv,noheader'], text=True),
        cost_unit='GPU wall-clock hours; monetary price unavailable'))
    started = time.monotonic()
    try:
        validate(protocol)
        set_seed(args.seed)
        torch.set_num_threads(8)
        torch.cuda.set_device(args.gpu)
        torch.cuda.reset_peak_memory_stats(args.gpu)
        upstream.tqdm = functools.partial(upstream.tqdm, disable=True)
        data = SamplingDataHandler(config, evidence, 'size_sampling' if args.variant == 'size_sampling' else 'baseline')
        data.target_idx_train = np.array(protocol['train_groups'])
        data.target_idx_valid = np.array(protocol['valid_groups'])
        write_json(evidence / 'target_groups.json', dict(train=data.target_idx_train.tolist(), valid=data.target_idx_valid.tolist(), test_used=False))
        manager = ResultManager(config, root_dir=str(ROOT / 'runs' / run_id))
        experiment = ImprovedExperiment(config, data, manager, evidence=evidence, phase='pilot')
        result = experiment.run()
        assert result == {'test_skipped': 1.0}
        status = dict(status='completed', result=result, best_validation=experiment.es_scheduler.b_results,
                      best_epoch=experiment.es_scheduler.b_epoch, completed_epochs=manager.epoch-1,
                      checkpoint=str(pathlib.Path(manager.best_model_path).relative_to(ROOT)),
                      checkpoint_sha256=sha256(manager.best_model_path),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.gpu))
    except Exception as error:
        traceback.print_exc()
        status = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
    status['elapsed_seconds'] = time.monotonic()-started
    status['gpu_wall_hours'] = status['elapsed_seconds']/3600
    write_json(evidence / 'result.json', status)
    print('FINAL', json.dumps(status), flush=True)
    if status['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
