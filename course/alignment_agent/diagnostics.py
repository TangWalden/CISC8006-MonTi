"""Bounded TRAIN-only, fixed-checkpoint gradient diagnostic.

One attack generation and one GCN forward per group/checkpoint feed four
autograd.grad calls. There is no backward(), parameter-gradient accumulation,
optimizer step, API request, victim/GAGA access or test prediction read here.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import time
import traceback

import dgl
import numpy as np
import scipy
import torch
import torch.nn.functional as F

from modules.data_handler import DataHandler
from utils.utils import get_n_ids
from course.conditional_alignment.common import (
    ROOT, gc_only_config, sha256, validate_protocol as validate_old_protocol, write_json)
from course.conditional_alignment.experiment import AlignmentExperiment
from course.conditional_alignment.reference import relationship_moments, matched_statistics


STRATA = ('one_edge', 'one_injection_multi_edge', 'multi_injection')
QUOTAS = dict(one_edge=6, one_injection_multi_edge=5, multi_injection=5)
NORM_EPSILON = 1e-12


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.random.get_rng_state().clone(),
                torch_cuda=torch.cuda.get_rng_state(0).clone())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state(state['torch_cuda'], 0)


def rng_fingerprint(state):
    digest = hashlib.sha256()
    digest.update(repr(state['python']).encode('utf-8'))
    value = state['numpy']
    digest.update(value[0].encode('ascii'))
    digest.update(value[1].tobytes())
    digest.update(repr(value[2:]).encode('ascii'))
    digest.update(state['torch_cpu'].numpy().tobytes())
    digest.update(state['torch_cuda'].numpy().tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def fixed_case_rng(seed):
    before = rng_snapshot()
    receipt = dict(case_seed=int(seed), rng_before_sha256=rng_fingerprint(before))
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed(int(seed))
        yield receipt
    finally:
        restore_rng(before)
        receipt['rng_after_restore_sha256'] = rng_fingerprint(rng_snapshot())
        receipt['rng_restored_exactly'] = (
            receipt['rng_after_restore_sha256'] == receipt['rng_before_sha256'])


def select_train_groups(data, eligible, seed=20261103):
    """Outcome-free 6/5/5 sampling; fill shortages by stratum then lowest ID."""
    eligible = [int(value) for value in eligible]
    if len(eligible) != 256 or len(set(eligible)) != 256:
        raise RuntimeError('Expected the frozen 256 unique TRAIN groups.')
    if not set(eligible) <= set(int(value) for value in data.target_idx_train):
        raise RuntimeError('A diagnostic group is outside original TRAIN.')
    pools = {key: [] for key in STRATA}
    for group in sorted(eligible):
        nb, eb = int(data.node_budgets[group]), int(data.edge_budgets[group])
        if nb == 1 and eb == 1:
            stratum = 'one_edge'
        elif nb == 1 and eb > 1:
            stratum = 'one_injection_multi_edge'
        elif nb > 1 and eb >= nb:
            stratum = 'multi_injection'
        else:
            raise RuntimeError('Unexpected original budget regime.')
        pools[stratum].append(group)
    private_rng = np.random.RandomState(seed)
    chosen = {}
    remaining = {}
    for stratum in STRATA:
        shuffled = private_rng.permutation(pools[stratum]).tolist()
        chosen[stratum] = shuffled[:min(QUOTAS[stratum], len(shuffled))]
        remaining[stratum] = sorted(set(pools[stratum])-set(chosen[stratum]))
    fills = []
    while sum(len(value) for value in chosen.values()) < 16:
        progressed = False
        for stratum in STRATA:
            if remaining[stratum] and sum(len(value) for value in chosen.values()) < 16:
                group = remaining[stratum].pop(0)
                chosen[stratum].append(group)
                fills.append(dict(stratum=stratum, target_group=group))
                progressed = True
        if not progressed:
            raise RuntimeError('Insufficient registered TRAIN groups.')
    rows = []
    for stratum in STRATA:
        for group in sorted(chosen[stratum]):
            rows.append(dict(target_group=group, stratum=stratum,
                             group_size=len(get_n_ids(data.target_assignment, group)),
                             node_budget=int(data.node_budgets[group]),
                             edge_budget=int(data.edge_budgets[group])))
    return dict(seed=seed, quotas=QUOTAS, eligible_counts={key: len(value) for key, value in pools.items()},
                actual_counts={key: len(value) for key, value in chosen.items()},
                shortage_fills=fills, groups=rows,
                rule='fixed stratum order; private RandomState sampling; shortage fill round-robin then ID; no outcomes')


def parameter_block(name):
    if name.startswith('transformer_encoder.'):
        return 'transformer'
    if name.startswith('linear_feat.'):
        return 'feature_head'
    if name.startswith('linear_edge.'):
        return 'edge_head'
    if name.startswith('pos_encoding.'):
        return 'role_embedding'
    if name.startswith(('candidate_selector.', 'target_encoder.', 'candidate_encoder.')):
        return name.split('.')[0]
    return 'other'


def _norm_squared(gradients, indices):
    total = 0.0
    used_parameters = used_elements = 0
    for index in indices:
        value = gradients[index]
        if value is not None:
            scalar = float(value.detach().to(torch.float64).square().sum().item())
            if not math.isfinite(scalar):
                raise RuntimeError('Nonfinite parameter gradient.')
            total += scalar
            used_parameters += 1
            used_elements += value.numel()
    return total, used_parameters, used_elements


def gradient_norms(gradients, named_parameters):
    groups = {'all': list(range(len(named_parameters)))}
    for index, (name, _) in enumerate(named_parameters):
        groups.setdefault(parameter_block(name), []).append(index)
    result = {}
    for block, indices in groups.items():
        squared, used_parameters, used_elements = _norm_squared(gradients, indices)
        elements = sum(named_parameters[index][1].numel() for index in indices)
        norm = math.sqrt(squared)
        result[block] = dict(l2=norm, rms=norm/math.sqrt(elements),
                             total_parameters=len(indices), total_elements=elements,
                             used_parameters=used_parameters, used_elements=used_elements,
                             near_zero=norm <= NORM_EPSILON)
    return result


def compare_gradients(attack, alignment, named_parameters, attack_norms, alignment_norms):
    groups = {'all': list(range(len(named_parameters)))}
    for index, (name, _) in enumerate(named_parameters):
        groups.setdefault(parameter_block(name), []).append(index)
    result = {}
    for block, indices in groups.items():
        dot = 0.0
        shared_parameters = shared_elements = 0
        for index in indices:
            left, right = attack[index], alignment[index]
            if left is not None and right is not None:
                dot += float((left.detach().to(torch.float64)*right.detach().to(torch.float64)).sum().item())
                shared_parameters += 1
                shared_elements += left.numel()
        an, rn = attack_norms[block]['l2'], alignment_norms[block]['l2']
        if not math.isfinite(dot):
            raise RuntimeError('Nonfinite gradient dot product.')
        defined = an > NORM_EPSILON and rn > NORM_EPSILON
        result[block] = dict(
            dot=dot, cosine=max(-1.0, min(1.0, dot/(an*rn))) if defined else None,
            alignment_to_attack_norm_ratio=rn/an if an > NORM_EPSILON else None,
            attack_near_zero=an <= NORM_EPSILON, alignment_near_zero=rn <= NORM_EPSILON,
            shared_non_none_parameters=shared_parameters, shared_non_none_elements=shared_elements,
            missing_gradient_convention='None contributes zero in full-parameter coordinates')
    return result


def margin_summary(margins):
    values = margins.detach().to(dtype=torch.float64, device='cpu').reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64))
    return dict(target_occurrences=values.numel(), positive_margin_nodes=int((values > 0).sum()),
                zero_margin_nodes=int((values == 0).sum()), negative_margin_nodes=int((values < 0).sum()),
                mean=float(values.mean()), min=float(quantiles[0]), q10=float(quantiles[1]),
                median=float(quantiles[2]), q90=float(quantiles[3]), max=float(quantiles[4]),
                minimum_absolute_margin=float(values.abs().min()),
                definition='GCN logit1 minus logit0; TRAIN targets are known fraud; no raw target IDs released')


class DiagnosticExperiment(AlignmentExperiment):
    def get_perturbed_graph(self, *args, **kwargs):
        if self.generator_forwards >= 32:
            raise RuntimeError('Generation call cap reached before another call.')
        return super().get_perturbed_graph(*args, **kwargs)

    def get_logits(self, *args, **kwargs):
        if self.gcn_forwards >= 32:
            raise RuntimeError('GCN forward cap reached before another call.')
        return super().get_logits(*args, **kwargs)


def verify_registration(path):
    protocol_path = pathlib.Path(path).resolve()
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol.get('status') != 'frozen':
        raise RuntimeError('A committed, frozen diagnostic protocol is required.')
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=ROOT, text=True).strip()
    if dirty:
        raise RuntimeError('Tracked source differs from committed source: '+dirty)
    required = {'course/alignment_agent/__init__.py', 'course/alignment_agent/diagnostics.py'}
    code = protocol.get('code', [])
    if not required <= {row['path'] for row in code}:
        raise RuntimeError('New diagnostic source hashes are incomplete.')
    for row in code:
        subprocess.check_output(['git', 'ls-files', '--error-unmatch', '--', row['path']], cwd=ROOT)
        if sha256(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('Diagnostic source hash mismatch: '+row['path'])
    for key in ('old_execution_protocol', 'reference'):
        if sha256(ROOT/protocol[key]['path']) != protocol[key]['sha256']:
            raise RuntimeError('Frozen '+key+' hash mismatch.')
    old, _ = validate_old_protocol(ROOT/protocol['old_execution_protocol']['path'], training=True)
    if protocol['reference']['sha256'] != old['reference_sha256']:
        raise RuntimeError('Diagnostic reference differs from original training reference.')
    if protocol.get('max_generations') != 32 or protocol.get('max_gcn_forwards') != 32:
        raise RuntimeError('Diagnostic forward bounds changed.')
    if not 0 < protocol.get('max_wall_seconds', 0) <= 180 or not 0 < protocol.get('max_gpu_hours', 0) <= 0.05:
        raise RuntimeError('Diagnostic time cap expanded or absent.')
    selection = protocol['selection']
    if selection != dict(seed=20261103, count=16, quotas=QUOTAS):
        raise RuntimeError('Registered outcome-free selection rule changed.')
    checkpoints = protocol['checkpoints']
    if [row['name'] for row in checkpoints] != ['off_seed3', 'gated_seed3']:
        raise RuntimeError('Only old OFF/GATED seed3 best checkpoints are permitted.')
    for row, arm in zip(checkpoints, ['off', 'gated']):
        if row['arm'] != arm or row['seed'] != 3 or row['best_epoch'] != 7:
            raise RuntimeError('Unexpected checkpoint identity or epoch.')
        if sha256(ROOT/row['path']) != row['sha256']:
            raise RuntimeError('Frozen attack checkpoint checksum mismatch.')
    return protocol, old, dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        protocol_sha256=sha256(protocol_path), old_execution_protocol_sha256=protocol['old_execution_protocol']['sha256'],
        reference_sha256=protocol['reference']['sha256'], tracked_changes=dirty)


def own_gpu_uuid(expected):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise RuntimeError('Only physical ordinal zero may be visible.')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch._C._cuda_getDeviceCount() != 1:
        raise RuntimeError('Torch/native visible GPU counts must equal one.')
    torch.cuda.set_device(0)
    torch.zeros(1, device='cuda:0').sum().item()
    for _ in range(5):
        rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                        '--format=csv,noheader'], text=True)
        own = {row.split(',')[1].strip() for row in rows.splitlines()
               if row.split(',')[0].strip() == str(os.getpid())}
        if own:
            if own != {expected}:
                raise RuntimeError('Process GPU UUID differs from the frozen physical device.')
            return sorted(own)
        time.sleep(0.1)
    raise RuntimeError('Own process GPU UUID was not visible.')


def measure_case(experiment, group, named_parameters):
    if any(value.grad is not None for _, value in named_parameters):
        raise RuntimeError('Diagnostic requires untouched parameter .grad buffers.')
    versions = [value._version for _, value in named_parameters]
    before_generation, before_gcn = experiment.generator_forwards, experiment.gcn_forwards
    ids = torch.tensor(get_n_ids(experiment.data.target_assignment, group['target_group']), dtype=torch.long)
    graph = experiment.get_perturbed_graph(ids, group['node_budget'], group['edge_budget'], training=True)
    logits = experiment.get_logits(graph, ids, model='surrogate', training=True)
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError('Nonfinite diagnostic logits.')
    unused_penalty, old_stats = experiment.consume_alignment(graph)
    del unused_penalty
    margins = logits[:, 1]-logits[:, 0]
    hinge = torch.relu(margins).mean()
    smooth = F.softplus(margins, beta=1.0, threshold=20.0).mean()
    injected = torch.arange(experiment.data.graph.num_nodes(), graph.num_nodes(), device=graph.device)
    signatures, degrees = relationship_moments(graph, injected, chunk_size=4096, use_weights=True)
    center, scale, tolerance, _, _ = matched_statistics(
        degrees, experiment.reference, dtype=signatures.dtype, device=signatures.device)
    distance = ((signatures-center)/scale).square().mean(dim=1)
    mask = distance.detach() > tolerance
    ungated = 0.1*distance.mean()
    q90 = 0.1*(distance*mask.to(distance.dtype)).mean()
    losses = dict(hinge=hinge, smooth=smooth, ungated_alignment=ungated, q90_alignment=q90)
    parameters = tuple(value for _, value in named_parameters)
    before_grad_rng = rng_fingerprint(rng_snapshot())
    gradients = {}
    for name, loss in losses.items():
        experiment.check_deadline()
        gradients[name] = torch.autograd.grad(loss, parameters, retain_graph=True,
                                               create_graph=False, allow_unused=True)
    after_grad_rng = rng_fingerprint(rng_snapshot())
    if before_grad_rng != after_grad_rng:
        raise RuntimeError('Gradient-only instrumentation consumed RNG state.')
    norms = {name: gradient_norms(value, named_parameters) for name, value in gradients.items()}
    comparisons = {}
    for attack in ['hinge', 'smooth']:
        for align in ['ungated_alignment', 'q90_alignment']:
            comparisons[attack+'_vs_'+align] = compare_gradients(
                gradients[attack], gradients[align], named_parameters, norms[attack], norms[align])
    if versions != [value._version for _, value in named_parameters]:
        raise RuntimeError('Model parameter storage changed during gradient diagnostic.')
    if any(value.grad is not None for _, value in named_parameters):
        raise RuntimeError('autograd.grad unexpectedly accumulated parameter .grad buffers.')
    if experiment.optimizer_steps or len(experiment.optimizer.state):
        raise RuntimeError('An optimizer update/state mutation occurred during diagnosis.')
    if experiment.generator_forwards-before_generation != 1 or experiment.gcn_forwards-before_gcn != 1:
        raise RuntimeError('A case did not use exactly one generation and one GCN forward.')
    experiment.check_deadline()
    experiment._capture = None
    return dict(**group, losses={name: float(value.detach().item()) for name, value in losses.items()},
                margin=margin_summary(margins), alignment=old_stats,
                measured_ungated_mean_distance=float(distance.detach().mean().item()),
                measured_q90_active_nodes=int(mask.sum().item()),
                gradient_norms=norms, comparisons=comparisons,
                gradient_calls=4, gradient_only_rng_unchanged=True,
                parameter_versions_unchanged=True, parameter_grad_buffers_untouched=True,
                optimizer_steps=0, generation_calls=1, gcn_forwards=1,
                norm_epsilon=NORM_EPSILON,
                caveat='Fixed-checkpoint local derivatives, not historical training causality or an Adam-step guarantee.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if pathlib.Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from the registered repository root.')
    output = pathlib.Path(args.output).resolve()
    if (ROOT/'runs').resolve() not in output.parents:
        raise RuntimeError('Diagnostic output must be exclusive beneath ignored runs/.')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    experiment = None
    outer_rng = None
    completed = 0
    result = dict(status='failed', optimizer_steps=0, api_calls=0, gaga_forwards=0,
                  test_used=False, new_training=False)
    try:
        protocol, old, provenance = verify_registration(args.protocol)
        limit = min(float(protocol['max_wall_seconds']), 3600*float(protocol['max_gpu_hours']))
        deadline = started+limit
        torch.set_num_threads(8)
        own = own_gpu_uuid(protocol['physical_gpu_uuid'])
        outer_rng = rng_snapshot()
        runtime = dict(torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__)
        locked = json.loads((ROOT/'environment/runtime.json').read_text(encoding='utf-8'))
        if any(runtime[key] != locked[key] for key in runtime):
            raise RuntimeError('Locked ML runtime changed.')
        reference = json.loads((ROOT/protocol['reference']['path']).read_text(encoding='utf-8'))
        config = gc_only_config(old, 3, 0)
        data = DataHandler(config)
        manifest = select_train_groups(data, old['train_groups'])
        write_json(output/'selected_train_groups.json', manifest, exclusive=True)
        write_json(output/'registered.json', dict(
            **provenance, registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            runtime=runtime, own_gpu_uuids=own,
            selected_train_groups_sha256=sha256(output/'selected_train_groups.json'),
            selection_before_model_loading=True, maximum_generations=32, maximum_gcn_forwards=32,
            scheduler_rule='restore initial schedulers then seven step calls after checkpoint loading; next-training-step state',
            smooth_objective='mean softplus(logit1-logit0), beta=1, numerical threshold=20',
            information_boundary='original TRAIN groups only; gradients but no optimizer updates; no victim/GAGA/test/API'), exclusive=True)
        experiment = DiagnosticExperiment(config, data, None, reference=reference,
                                           protocol=old, arm='off', evidence=output,
                                           deadline=deadline, source_commit=provenance['source_commit'])
        named_parameters = tuple((name, value) for name, value in experiment.model.named_parameters()
                                 if value.requires_grad)
        if not named_parameters:
            raise RuntimeError('No trainable MonTi parameters for differentiation.')
        checkpoint_rows = []
        with (output/'cases.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for checkpoint in protocol['checkpoints']:
                experiment.check_deadline()
                state = torch.load(ROOT/checkpoint['path'], map_location='cpu')
                experiment.model.load_state_dict(state, strict=True)
                del state
                experiment.arm = checkpoint['arm']
                experiment.model.temp_scheduler.current_temp = float(config['initial_temp'])
                experiment.model.eps_scheduler.current_eps = float(config['initial_eps'])
                for _ in range(7):
                    experiment.model.step()
                experiment.model.train(True)
                experiment.surrogate_model.eval()
                checkpoint_rows.append(dict(name=checkpoint['name'], checkpoint_sha256=checkpoint['sha256'],
                                             best_epoch=7, scheduler_step_calls=7,
                                             temperature=experiment.model.temp_scheduler.get_temperature(),
                                             epsilon=experiment.model.eps_scheduler.get_epsilon()))
                for group in manifest['groups']:
                    experiment.check_deadline()
                    seed = (int(protocol.get('case_seed_root', 20261103))+group['target_group']) % (2**32)
                    with fixed_case_rng(seed) as rng_receipt:
                        row = measure_case(experiment, group, named_parameters)
                    if not rng_receipt['rng_restored_exactly']:
                        raise RuntimeError('A case did not restore RNG state.')
                    row.update(checkpoint=checkpoint['name'], rng=rng_receipt)
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                    stream.flush()
                    completed += 1
                    write_json(output/'progress.json', dict(completed_cases=completed,
                               generator_forwards=experiment.generator_forwards,
                               gcn_forwards=experiment.gcn_forwards, optimizer_steps=0))
                    print('GRADIENT_CASE', checkpoint['name'], group['target_group'], completed, flush=True)
        if completed != 32 or experiment.generator_forwards != 32 or experiment.gcn_forwards != 32:
            raise RuntimeError('Incomplete bounded diagnostic matrix.')
        experiment.check_deadline()
        result.update(status='completed', cases=completed, checkpoints=checkpoint_rows,
                      groups_per_checkpoint=16, gradient_calls=4*completed,
                      cases_sha256=sha256(output/'cases.jsonl'),
                      selected_train_groups_sha256=sha256(output/'selected_train_groups.json'),
                      parameter_count=sum(value.numel() for _, value in named_parameters),
                      interpretation='Exploratory TRAIN-only derivatives at two selected checkpoints; no causal or method-efficacy claim.')
    except BaseException as error:
        traceback.print_exc()
        result.update(error=repr(error), traceback=traceback.format_exc(), completed_cases=completed)
    finally:
        if outer_rng is not None:
            restore_rng(outer_rng)
            result['outer_rng_restored'] = rng_fingerprint(outer_rng) == rng_fingerprint(rng_snapshot())
        if experiment is not None:
            result.update(generator_forwards=experiment.generator_forwards,
                          gcn_forwards=experiment.gcn_forwards, optimizer_steps=experiment.optimizer_steps)
        result['elapsed_seconds'] = time.monotonic()-started
        result['charged_gpu_wall_hours'] = result['elapsed_seconds']/3600
        result['cost_policy'] = 'Whole process lifetime conservatively charged; outer guardian may charge more.'
        write_json(output/'result.json', result, exclusive=True)
        print('DIAGNOSTIC_FINAL', json.dumps(result, allow_nan=False), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
