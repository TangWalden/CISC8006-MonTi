"""One bounded, fresh paired alignment-pilot training job. No victim/test path."""
import argparse
import datetime as dt
import functools
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import traceback

import dgl
import numpy as np
import scipy
import torch

import modules.experiment_handler as upstream
from modules.data_handler import DataHandler
from modules.result_manager import ResultManager
from utils.utils import set_seed
from course.conditional_alignment.common import (
    ROOT, gc_only_config, sha256, validate_protocol, write_json)
from course.conditional_alignment.experiment import AlignmentExperiment


def _timeout(signum, frame):
    raise TimeoutError('Registered wall-clock alarm expired.')


def _check_reference(reference, protocol):
    if reference.get('reference_algorithm_sha256') != sha256(
            pathlib.Path(__file__).with_name('reference.py')):
        raise RuntimeError('Reference algorithm source changed after reference fitting.')
    if reference.get('min_count') != 8 or reference.get('gate_quantile') != 0.9:
        raise RuntimeError('Reference settings differ from frozen protocol.')
    if reference.get('scale_floor') != protocol['reference'].get('scale_floor', 0.05):
        raise RuntimeError('Reference scale differs from frozen protocol.')
    for entry in reference['registered_inputs']:
        if sha256(ROOT/entry['path']) != entry['sha256']:
            raise RuntimeError('An input changed since reference preparation: '+entry['path'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--arm', required=True, choices=['off', 'always', 'gated'])
    parser.add_argument('--seed', required=True, type=int, choices=[3, 4, 5])
    parser.add_argument('--gpu', type=int, default=0, choices=[0])
    parser.add_argument('--output', required=True)
    parser.add_argument('--wall-seconds', type=int)
    args = parser.parse_args()
    if pathlib.Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the registered repository root.')
    output = pathlib.Path(args.output).resolve()
    private_root = (ROOT/'runs').resolve()
    if private_root not in output.parents:
        raise RuntimeError('Pilot outputs/raw artifacts must remain beneath ignored runs/.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'raw').mkdir()
    started = time.monotonic()
    experiment = None
    status = dict(status='failed', arm=args.arm, seed=args.seed, test_used=False,
                  victim_loaded=False, gaga_forwards=0)
    try:
        protocol, provenance = validate_protocol(args.protocol, training=True)
        if float(protocol['alignment_lambda']) != 0.1:
            raise RuntimeError('Frozen alignment coefficient changed.')
        if int(protocol['epochs']) != 10 or int(protocol['patience']) != 5:
            raise RuntimeError('Frozen pilot training limits changed.')
        if len(protocol['train_groups']) != 256 or len(protocol['valid_groups']) != 128:
            raise RuntimeError('Frozen pilot group counts changed.')
        previous = json.loads((ROOT/'course/improvements/pilot_protocol_v1.json').read_text(encoding='utf-8'))
        if (protocol['train_groups'] != previous['train_groups'] or
                protocol['valid_groups'] != previous['valid_groups']):
            raise RuntimeError('Pilot cohort differs from the registered old 256/128 group lists.')
        if protocol.get('export_valid_groups') != [1185, 1267, 1285, 1415]:
            raise RuntimeError('Unexpected registered raw-export validation groups.')
        if sha256(args.reference) != protocol['reference_sha256']:
            raise RuntimeError('Frozen reference checksum mismatch.')
        reference = json.loads(pathlib.Path(args.reference).read_text(encoding='utf-8'))
        _check_reference(reference, protocol)
        limit = int(protocol.get('job_wall_seconds', 480))
        if args.wall_seconds is not None:
            limit = min(limit, args.wall_seconds)
        if not 1 <= limit <= 480:
            raise RuntimeError('Invalid or expanded per-job wall-clock cap.')
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(max(1, int(limit-(time.monotonic()-started))))
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one explicitly visible GPU is required.')
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats(0)
        # Initialize only the visible device and verify the own-process UUID if
        # the guardian registered one. Never change another process or GPU.
        setup = torch.ones(1, device='cuda:0')
        torch.cuda.synchronize(0)
        gpu_apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
        own_uuids = {row.split(',')[1].strip() for row in gpu_apps.splitlines()
                     if row.split(',')[0].strip() == str(os.getpid())}
        expected_uuid = protocol.get('physical_gpu_uuid', protocol.get('gpu_uuid'))
        if len(own_uuids) != 1 or (expected_uuid is not None and own_uuids != {expected_uuid}):
            raise RuntimeError('Own process GPU identity is not the registered single device.')
        del setup
        runtime = dict(torch=torch.__version__, dgl=dgl.__version__,
                       numpy=np.__version__, scipy=scipy.__version__)
        locked = json.loads((ROOT/'environment/runtime.json').read_text(encoding='utf-8'))
        if any(runtime[key] != locked[key] for key in runtime):
            raise RuntimeError('Locked ML runtime changed.')
        config = gc_only_config(protocol, args.seed, 0)
        config.update(exp_name='alignment_%s_seed%d' % (args.arm, args.seed),
                      config_path=str(output/'config.json'))
        write_json(output/'config.json', config, exclusive=True)
        write_json(output/'registered.json', dict(
            **provenance, registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            command=[sys.executable, '-m', 'course.conditional_alignment.run']+sys.argv[1:],
            arm=args.arm, seed=args.seed, runtime=runtime, physical_gpu_uuid=list(own_uuids)[0],
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            reference_sha256=sha256(args.reference), job_wall_seconds=limit,
            config_sha256=sha256(output/'config.json'),
            information_boundary='TRAIN optimization and GCN validation only; no victim/GAGA/test',
            gate='fixed train-reference q90 tolerance, strict greater-than, detached; not a learned/LLM gate',
            training_parameters='original MonTi only; no new alignment/gate trainable parameters'), exclusive=True)
        set_seed(args.seed)
        upstream.tqdm = functools.partial(upstream.tqdm, disable=True)
        data = DataHandler(config)
        original_train = set(int(x) for x in data.target_idx_train)
        original_valid = set(int(x) for x in data.target_idx_valid)
        train = protocol['train_groups']
        valid = protocol['valid_groups']
        if len(set(train)) != len(train) or len(set(valid)) != len(valid):
            raise RuntimeError('Duplicated registered groups.')
        if not set(train) <= original_train or not set(valid) <= original_valid:
            raise RuntimeError('Pilot group is outside its original registered partition.')
        if not set(protocol['export_valid_groups']) <= set(valid):
            raise RuntimeError('An export group is outside the frozen validation cohort.')
        data.target_idx_train = np.array(train, dtype=np.int64)
        data.target_idx_valid = np.array(valid, dtype=np.int64)
        write_json(output/'target_groups.json', dict(train=train, valid=valid, test_used=False), exclusive=True)
        manager = ResultManager(config, root_dir=str(output/'manager'))
        experiment = AlignmentExperiment(config, data, manager, reference=reference,
                                         protocol=protocol, arm=args.arm, evidence=output,
                                         deadline=started+limit, source_commit=provenance['source_commit'])
        result = experiment.run()
        best_epoch = int(experiment.es_scheduler.b_epoch)
        best_summary = json.loads((output/('validation_epoch_%03d.json' % best_epoch)).read_text())
        selected_exports = [row for row in experiment.export_index if row['epoch'] == best_epoch]
        if {row['target_group'] for row in selected_exports} != set(protocol['export_valid_groups']):
            raise RuntimeError('Selected checkpoint has incomplete registered raw attack exports.')
        write_json(output/'selected_attack_exports.json', selected_exports, exclusive=True)
        status.update(status='completed', result=result, best_epoch=best_epoch,
                      completed_epochs=manager.epoch-1,
                      best_validation=experiment.es_scheduler.b_results,
                      best_validation_counts=best_summary,
                      checkpoint=str(pathlib.Path(manager.best_model_path).relative_to(output)),
                      checkpoint_sha256=sha256(manager.best_model_path),
                      reference_sha256=sha256(args.reference),
                      selected_attack_exports=selected_exports,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except BaseException as error:
        traceback.print_exc()
        status.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if experiment is not None:
            status.update(gcn_forwards=experiment.gcn_forwards,
                          generator_forwards=experiment.generator_forwards,
                          optimizer_steps=experiment.optimizer_steps)
        status.update(elapsed_seconds=time.monotonic()-started)
        status['charged_gpu_wall_hours'] = status['elapsed_seconds']/3600.0
        status['cost_note'] = 'Conservative whole-process one-GPU wall time; guardian may charge more.'
        write_json(output/'result.json', status, exclusive=True)
        print('FINAL', json.dumps(status, allow_nan=False), flush=True)
    if status['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
