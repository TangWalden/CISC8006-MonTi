"""One frozen final-epoch-only training job; root owns controller and guardian."""
import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import traceback

import dgl
import numpy as np
import scipy
import torch

from modules.data_handler import DataHandler
from utils.utils import set_seed
from course.conditional_alignment.common import ROOT, gc_only_config, sha256, write_json
from course.alignment_agent.common import validate_protocol
from course.alignment_agent.controller import build_policy
from course.alignment_agent.diagnostics import own_gpu_uuid
from course.alignment_agent.experiment import ARMS, AgentAlignmentExperiment, select_clean_utility


def _load_reference(entry):
    path = ROOT/entry['path']
    if sha256(path) != entry['sha256']:
        raise RuntimeError('Frozen reference checksum mismatch: '+entry['path'])
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--seed', type=int, required=True, choices=[3, 4, 5])
    parser.add_argument('--output', required=True)
    parser.add_argument('--matched-schedule')
    args = parser.parse_args()
    if pathlib.Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the registered repository root.')
    output = pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:
        raise RuntimeError('Training/raw outputs must remain under ignored runs/.')
    output.mkdir(parents=True, exist_ok=False)
    (output/'raw').mkdir()
    started = time.monotonic()
    experiment = None
    result = dict(status='failed', arm=args.arm, seed=args.seed, gaga_forwards=0, test_used=False)
    try:
        protocol, provenance = validate_protocol(args.protocol)
        if protocol['arms'] != list(ARMS) or protocol['seeds'] != [3, 4, 5]:
            raise RuntimeError('Frozen arms/seeds changed.')
        if protocol['epochs'] != 8 or float(protocol['alignment_lambda']) != 0.1 or float(protocol['softplus_beta']) != 1:
            raise RuntimeError('Frozen eight-epoch/objective settings changed.')
        if len(protocol['train_groups']) != 256 or len(protocol['valid_groups']) != 128:
            raise RuntimeError('Frozen TRAIN/VALID cohort sizes changed.')
        previous = json.loads((ROOT/'course/improvements/pilot_protocol_v1.json').read_text(encoding='utf-8'))
        if (protocol['train_groups'] != previous['train_groups'] or
                protocol['valid_groups'] != previous['valid_groups']):
            raise RuntimeError('Cohort differs from the original 256/128 pilot lists.')
        probes = protocol['gradient_probe_groups']
        if len(probes) != 8 or len(set(probes)) != 8 or not set(probes) <= set(protocol['train_groups']):
            raise RuntimeError('Frozen eight TRAIN gradient-probe groups are invalid.')
        expected_exports = {957, 1185, 1198, 1249, 1267, 1285, 1328, 1415}
        if set(protocol['export_valid_groups']) != expected_exports or len(protocol['export_valid_groups']) != 8:
            raise RuntimeError('Frozen final export groups changed.')
        if not expected_exports <= set(protocol['valid_groups']):
            raise RuntimeError('An export group is outside the full evaluation cohort.')
        if not 1 <= int(protocol['job_wall_seconds']) <= 420:
            raise RuntimeError('Per-job wall-clock ceiling expanded.')
        torch.set_num_threads(8)
        own = own_gpu_uuid(protocol['physical_gpu_uuid'])
        torch.cuda.reset_peak_memory_stats(0)
        runtime = dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__)
        locked = json.loads((ROOT/'environment/runtime.json').read_text(encoding='utf-8'))
        if any(runtime[key] != locked[key] for key in runtime):
            raise RuntimeError('Locked runtime changed.')
        reference = _load_reference(protocol['alignment_reference_file'])
        if reference.get('min_count') != 8 or reference.get('gate_quantile') != 0.9:
            raise RuntimeError('Original alignment reference settings changed.')
        if reference.get('reference_algorithm_sha256') != sha256(ROOT/'course/conditional_alignment/reference.py'):
            raise RuntimeError('Original reference algorithm source changed.')
        filter_wrapper = _load_reference(protocol['filter_reference_file'])
        if filter_wrapper.get('status') == 'completed' and 'reference' in filter_wrapper:
            filter_reference = filter_wrapper['reference']
        else:
            filter_reference = filter_wrapper
        config = gc_only_config(dict(protocol, patience=8), args.seed, 0)
        config.update(epochs=8, valid_epoch=8, batch_size=1, early_stopping=False,
                      exp_name='alignment_agent_%s_seed%d' % (args.arm, args.seed))
        write_json(output/'config.json', config, exclusive=True)
        set_seed(args.seed)
        # Full-neighbor sampling is seeded consistently. This is not a promise
        # of cross-device bitwise identity of DGL graph kernels/orderings.
        dgl.seed(args.seed)
        dgl.random.seed(args.seed)
        data = DataHandler(config)
        if not set(protocol['train_groups']) <= set(int(value) for value in data.target_idx_train):
            raise RuntimeError('TRAIN cohort crosses original partition.')
        if not set(protocol['valid_groups']) <= set(int(value) for value in data.target_idx_valid):
            raise RuntimeError('VALID cohort crosses original partition.')
        data.target_idx_train = np.array(protocol['train_groups'], dtype=np.int64)
        data.target_idx_valid = np.array(protocol['valid_groups'], dtype=np.int64)
        utility = select_clean_utility(data)
        write_json(output/'raw/clean_utility_manifest.json', utility, exclusive=True)
        write_json(output/'target_groups.json', dict(train=protocol['train_groups'], valid=protocol['valid_groups'],
                   gradient_probe=probes, export=protocol['export_valid_groups'], test_used=False), exclusive=True)
        # A controller constructor may use only its private RNG; restore all
        # three global streams defensively before MonTi initialization.
        from course.alignment_agent.diagnostics import rng_snapshot, restore_rng
        constructor_rng = rng_snapshot()
        try:
            policy = build_policy(args.arm, args.seed, protocol, output,
                                  matched_schedule_path=args.matched_schedule)
        finally:
            restore_rng(constructor_rng)
        source_commit = provenance.get('source_commit', protocol.get('source_commit', ''))
        write_json(output/'registered.json', dict(
            provenance=provenance, registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            command=[sys.executable, '-m', 'course.alignment_agent.run']+sys.argv[1:],
            source_commit=source_commit, arm=args.arm, seed=args.seed, own_gpu_uuids=own,
            config_sha256=sha256(output/'config.json'),
            alignment_reference_sha256=protocol['alignment_reference_file']['sha256'],
            filter_reference_sha256=protocol['filter_reference_file']['sha256'],
            utility_manifest_sha256=sha256(output/'raw/clean_utility_manifest.json'),
            job_wall_seconds=protocol['job_wall_seconds'],
            counter_caps=dict(generations=2176, gcn_forwards=2562, optimizer_steps=2048,
                              maximum_gradient_calls=3600),
            full_group_clean_baseline='Final8 only; each128 original target group has clean raw and once-filtered-original logits; no controller feedback.',
            gradient_call_rule='128 + 496*protected_ON_epochs; ON gradients reused for probes',
            generation_rng='common.case_seed SHA256 seed|phase|epoch|group|alignment-agent-v2; fork/restore Python/NumPy/Torch CPU/CUDA',
            dgl_seed=args.seed, cross_device_bitwise_identity_claimed=False,
            controller_information='previous completed TRAIN epoch flat state only',
            primary_checkpoint='final epoch8; no interim validation/early stopping',
            matched_schedule_sha256=(sha256(args.matched_schedule) if args.matched_schedule else None),
            runtime=runtime, test_used=False, victim_loaded=False), exclusive=True)
        experiment = AgentAlignmentExperiment(config, data, protocol=protocol, reference=reference,
                       filter_reference=filter_reference, arm=args.arm, outer_policy=policy, evidence=output,
                       deadline=started+int(protocol['job_wall_seconds']), source_commit=source_commit,
                       utility_manifest=utility)
        result['initial_model_state_sha256'] = experiment.initial_model_state_sha256
        write_json(output/'initial_model_state.json', dict(
            arm=args.arm, seed=args.seed, source_commit=source_commit,
            initial_model_state_sha256=experiment.initial_model_state_sha256,
            definition='sorted state_dict keys, dtype, shape and contiguous CPU tensor bytes; length-delimited SHA256',
            captured_before_training=True, optimizer_steps=experiment.optimizer_steps,
            generator_forwards=experiment.generator_forwards, gcn_forwards=experiment.gcn_forwards), exclusive=True)
        outcome = experiment.run()
        result.update(status='completed', outcome=outcome,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    except BaseException as error:
        traceback.print_exc()
        result.update(error=repr(error), traceback=traceback.format_exc())
    finally:
        if experiment is not None:
            result.update(generator_forwards=experiment.generator_forwards,
                          gcn_forwards=experiment.gcn_forwards, optimizer_steps=experiment.optimizer_steps,
                          gradient_calls=experiment.gradient_calls, completed_train_epochs=experiment.completed_epochs,
                          applied_epoch_actions=len(experiment.applied_actions))
        result['elapsed_seconds'] = time.monotonic()-started
        result['charged_gpu_wall_hours'] = result['elapsed_seconds']/3600
        result['cost_policy'] = 'Conservative whole process GPU wall time including API waits; guardian may charge more.'
        write_json(output/'result.json', result, exclusive=True)
        print('TRAINING_FINAL', json.dumps(result, allow_nan=False), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
