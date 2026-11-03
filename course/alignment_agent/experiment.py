"""Fixed-eight-epoch attack training; outer decisions see lagged TRAIN only.

All scientific v1 modules remain unchanged. The outer action multiplies the
original nodewise q90 penalty. Final raw/filtered validation is never controller
feedback, checkpoint selection or a reason to alter a frozen action schedule.
"""
import contextlib
import hashlib
import json
import math
import random
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import get_n_ids
from course.conditional_alignment.common import sha256, write_json
from course.conditional_alignment.experiment import AlignmentExperiment
from course.alignment_agent.diagnostics import (
    parameter_block, NORM_EPSILON)
from course.alignment_agent.filter import apply_filter, MAX_WORKING_BYTES
from course.alignment_agent.common import case_seed


ARMS = ('legacy_hinge_off', 'smooth_off', 'smooth_threshold', 'smooth_rule',
        'smooth_agent', 'smooth_random')
MAX_GENERATIONS = 8*256+128
MAX_GCN_FORWARDS = 8*256+4*128+2
PROTECTED_ARMS = ('smooth_rule', 'smooth_agent', 'smooth_random')


def model_state_sha256(state):
    """Content hash independent of Torch serialization paths/ZIP metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().to(device='cpu').contiguous()
        header = json.dumps(dict(name=name, dtype=str(value.dtype), shape=list(value.shape)),
                            sort_keys=True, separators=(',', ':')).encode('utf-8')
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, 'big'))
        digest.update(raw)
    return digest.hexdigest()


def generation_seed(seed, phase, epoch, group):
    return case_seed(seed, phase, epoch, group)


@contextlib.contextmanager
def fork_generation_rng(seed, *, cuda_device=None):
    """Restore exact caller RNG; CPU-only tests need no CUDA state/access."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_state = None if cuda_device is None else torch.cuda.get_rng_state(cuda_device).clone()
    receipt = dict(case_seed=int(seed), cuda_device=cuda_device)
    try:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if cuda_device is not None:
            torch.cuda.manual_seed(int(seed))
        yield receipt
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_device)
        after_np = np.random.get_state()
        receipt['rng_restored_exactly'] = (
            random.getstate() == python_state and after_np[0] == numpy_state[0]
            and np.array_equal(after_np[1], numpy_state[1]) and after_np[2:] == numpy_state[2:]
            and torch.equal(cpu_state, torch.random.get_rng_state())
            and (cuda_state is None or torch.equal(cuda_state, torch.cuda.get_rng_state(cuda_device))))


def attack_losses(logits):
    margin = logits[:, 0]-logits[:, 1]
    return margin, torch.relu(-margin).mean(), F.softplus(-margin, beta=1.0, threshold=20.0).mean()


def training_objective(hinge, smooth, q90_penalty, arm, outer_enabled):
    if arm not in ARMS or type(outer_enabled) is not bool:
        raise ValueError('Frozen arm and Boolean action are required.')
    if arm in ('legacy_hinge_off', 'smooth_off') and outer_enabled:
        raise ValueError('OFF controls may never activate the penalty.')
    base = hinge if arm == 'legacy_hinge_off' else smooth
    # Preserve the ordinary base graph/None-gradient support exactly when OFF.
    return base, base+q90_penalty if outer_enabled else base


def protect_alignment_gradients(named_parameters, attack_gradients, alignment_gradients,
                                epsilon=NORM_EPSILON):
    """Project/cap q90 gradients in fixed parameter blocks; no .grad writes.

    None is zero in a block's coordinates. A parameter with both gradients None
    is excluded from the proxy, preserving unused-parameter Adam semantics.
    Orthogonality/capping are local gradient statements, not an Adam-step bound.
    """
    if len(named_parameters) != len(attack_gradients) or len(named_parameters) != len(alignment_gradients):
        raise ValueError('Gradient/parameter tuple lengths differ.')
    blocks = {}
    for index, (name, _) in enumerate(named_parameters):
        blocks.setdefault(parameter_block(name), []).append(index)
    total = [None]*len(named_parameters)
    records = {}
    for block, indices in blocks.items():
        # Concatenate once per fixed block. All reductions/projection/capping
        # remain on-device in float64; one packed transfer reports the block.
        a = torch.cat([(torch.zeros_like(named_parameters[index][1]) if attack_gradients[index] is None
                        else attack_gradients[index].detach()).reshape(-1) for index in indices]).to(torch.float64)
        r = torch.cat([(torch.zeros_like(named_parameters[index][1]) if alignment_gradients[index] is None
                        else alignment_gradients[index].detach()).reshape(-1) for index in indices]).to(torch.float64)
        a2, r2, dot = a.square().sum(), r.square().sum(), (a*r).sum()
        anorm, rnorm = a2.sqrt(), r2.sqrt()
        valid = a2 > epsilon**2
        coefficient = torch.where(valid, torch.minimum(dot/a2.clamp_min(epsilon**2),
                                                        torch.zeros_like(dot)), torch.zeros_like(dot))
        projected = torch.where(valid, r-coefficient*a, torch.zeros_like(r))
        projected_norm = projected.square().sum().sqrt()
        factor = torch.where(projected_norm > 0,
                             torch.minimum(torch.ones_like(anorm), anorm/projected_norm.clamp_min(torch.finfo(torch.float64).tiny)),
                             torch.ones_like(anorm))
        adjusted = projected*factor
        packed = torch.stack((anorm, rnorm, dot, coefficient, factor,
                              adjusted.square().sum().sqrt(), (a*adjusted).sum())).cpu().tolist()
        if not all(math.isfinite(value) for value in packed):
            raise RuntimeError('Nonfinite block gradient statistics.')
        an, rn, raw_dot, coeff, cap, adjusted_norm, adjusted_dot = packed
        records[block] = dict(attack_norm=an, raw_alignment_norm=rn, raw_dot=raw_dot,
                              projection_applied=coeff < 0, projection_coefficient=coeff,
                              suppressed_attack_near_zero=an <= epsilon, cap_factor=cap,
                              cap_applied=cap < 1, protected_alignment_norm=adjusted_norm,
                              protected_dot=adjusted_dot, fixed_cap_ratio=1.0,
                              reduction='float64 per-block vectors; one packed CPU transfer; cast total to parameter dtype')
        combined = a+adjusted
        offset = 0
        for index in indices:
            _, parameter = named_parameters[index]
            size = parameter.numel()
            if attack_gradients[index] is not None or alignment_gradients[index] is not None:
                total[index] = combined[offset:offset+size].reshape(parameter.shape).to(parameter.dtype)
            offset += size
    return tuple(total), records


def protected_scalar_proxy(reported_value, named_parameters, total_gradients):
    """Return the reporting value with exactly the prescribed detached gradient."""
    linear = None
    for (_, parameter), gradient in zip(named_parameters, total_gradients):
        if gradient is not None:
            term = (parameter*gradient.detach()).sum()
            linear = term if linear is None else linear+term
    return reported_value if linear is None else reported_value.detach()+(linear-linear.detach())


def select_clean_utility(data):
    """Choose 64 original VALID nodes per class by fixed hash, before outputs."""
    valid = [int(value) for value in data.idx_valid]
    if len(valid) != len(set(valid)):
        raise RuntimeError('Original VALID node IDs are duplicated.')
    values = data.graph.ndata['y'][torch.tensor(valid, device=data.graph.device)].detach().cpu().tolist()
    rows = []
    for label in [0, 1]:
        pool = [node for node, truth in zip(valid, values) if int(truth) == label]
        ordered = sorted(pool, key=lambda node: (
            hashlib.sha256(('20261103|clean_utility|%d' % node).encode('ascii')).hexdigest(), node))
        if len(ordered) < 64:
            raise RuntimeError('Fewer than 64 original VALID nodes in a utility class.')
        rows.extend(dict(node_id=node, label=label) for node in ordered[:64])
    rows.sort(key=lambda row: row['node_id'])
    return dict(seed=20261103, classes=[0, 1], per_class=64, nodes=rows,
                selection='fixed hash of original VALID IDs within class; no model outcomes')


def binary_metrics(labels, predictions):
    labels = torch.as_tensor(labels, device='cpu').reshape(-1)
    predictions = torch.as_tensor(predictions, device='cpu').reshape(-1)
    if labels.shape != predictions.shape or not len(labels):
        raise RuntimeError('Invalid metric denominator.')
    benign, fraud = labels == 0, labels == 1
    return dict(cases=len(labels), correct=int((labels == predictions).sum()),
                accuracy=float((labels == predictions).double().mean()),
                benign_count=int(benign.sum()), fraud_count=int(fraud.sum()),
                benign_false_positive_rate=(float((predictions[benign] == 1).double().mean())
                                           if bool(benign.any()) else None),
                fraud_recall=(float((predictions[fraud] == 1).double().mean())
                              if bool(fraud.any()) else None))


def transition_counts(labels, clean_predictions, attacked_predictions):
    labels = torch.as_tensor(labels, device='cpu').reshape(-1)
    clean = torch.as_tensor(clean_predictions, device='cpu').reshape(-1)
    attacked = torch.as_tensor(attacked_predictions, device='cpu').reshape(-1)
    if not len(labels) or clean.shape != labels.shape or attacked.shape != labels.shape:
        raise RuntimeError('Clean/attacked transition denominators differ.')
    cw, aw = clean != labels, attacked != labels
    result = dict(target_occurrences=len(labels), clean_wrong=int(cw.sum()), attacked_wrong=int(aw.sum()),
                  new_wrong=int((~cw & aw).sum()), recoveries=int((cw & ~aw).sum()),
                  retained_correct=int((~cw & ~aw).sum()), retained_wrong=int((cw & aw).sum()))
    if (result['new_wrong']+result['recoveries']+result['retained_correct']+result['retained_wrong'] != len(labels)
            or result['clean_wrong'] != result['recoveries']+result['retained_wrong']
            or result['attacked_wrong'] != result['new_wrong']+result['retained_wrong']):
        raise RuntimeError('Transition counts do not reconcile.')
    return result


class AgentAlignmentExperiment(AlignmentExperiment):
    def __init__(self, config, data, *, protocol, reference, filter_reference,
                 arm, outer_policy, evidence, deadline, source_commit, utility_manifest):
        if arm not in ARMS:
            raise ValueError('Unknown frozen training arm.')
        self.training_arm = arm
        self.outer_policy = outer_policy
        self.filter_reference = filter_reference
        self.utility_manifest = utility_manifest
        self.phase = 'not_started'
        self.current_group = None
        self.outer_enabled = False
        self.applied_actions = []
        self.action_receipts = []
        self.gradient_calls = 0
        self.completed_epochs = 0
        self.probe_groups = set(int(value) for value in protocol['gradient_probe_groups'])
        self.last_generation_rng = None
        self._clean_filter_cache = None
        super().__init__(config, data, None, reference=reference, protocol=protocol,
                         arm='gated', evidence=evidence, deadline=deadline,
                         source_commit=source_commit)
        self.named_parameters_for_probe = tuple((name, value) for name, value in self.model.named_parameters()
                                                if value.requires_grad)
        self.initial_model_state_sha256 = model_state_sha256(self.model.state_dict())

    def get_perturbed_graph(self, *args, **kwargs):
        if self.generator_forwards >= MAX_GENERATIONS:
            raise RuntimeError('Generation cap reached before another call.')
        if self.current_group is None or self.phase not in ('train', 'validation'):
            raise RuntimeError('Generation phase/group is not registered.')
        seed = generation_seed(self.config['seed'], self.phase, self.current_epoch, self.current_group)
        with fork_generation_rng(seed, cuda_device=0) as receipt:
            graph = super().get_perturbed_graph(*args, **kwargs)
        if not receipt['rng_restored_exactly']:
            raise RuntimeError('Generation did not restore Torch/NumPy/Python RNG.')
        self.last_generation_rng = dict(receipt, phase=self.phase, epoch=self.current_epoch,
                                        target_group=self.current_group)
        return graph

    def get_logits(self, *args, **kwargs):
        if self.gcn_forwards >= MAX_GCN_FORWARDS:
            raise RuntimeError('GCN forward cap reached before another call.')
        return super().get_logits(*args, **kwargs)

    def _loss_gradients(self, smooth, q90):
        parameters = tuple(value for _, value in self.named_parameters_for_probe)
        before = [(id(value.grad), None if value.grad is None else value.grad._version)
                  for value in parameters]
        self.gradient_calls += 1
        gs = torch.autograd.grad(smooth, parameters, retain_graph=True,
                                 create_graph=False, allow_unused=True)
        self.gradient_calls += 1
        gr = torch.autograd.grad(q90, parameters, retain_graph=True,
                                 create_graph=False, allow_unused=True)
        after = [(id(value.grad), None if value.grad is None else value.grad._version)
                 for value in parameters]
        if before != after:
            raise RuntimeError('Gradient probe mutated parameter .grad buffers.')
        self.check_deadline()
        return gs, gr

    def _probe_summary(self, gs, gr):
        named = self.named_parameters_for_probe
        a = torch.cat([(torch.zeros_like(parameter) if value is None else value.detach()).reshape(-1)
                       for (_, parameter), value in zip(named, gs)]).to(torch.float64)
        r = torch.cat([(torch.zeros_like(parameter) if value is None else value.detach()).reshape(-1)
                       for (_, parameter), value in zip(named, gr)]).to(torch.float64)
        an, rn, dot = torch.stack((a.square().sum().sqrt(), r.square().sum().sqrt(), (a*r).sum())).cpu().tolist()
        if not all(math.isfinite(value) for value in (an, rn, dot)):
            raise RuntimeError('Nonfinite probe gradient statistics.')
        elements = sum(parameter.numel() for _, parameter in named)
        def norm_record(values, norm):
            return dict(l2=norm, rms=norm/math.sqrt(elements), total_parameters=len(named), total_elements=elements,
                        used_parameters=sum(value is not None for value in values),
                        used_elements=sum(value.numel() for value in values if value is not None),
                        near_zero=norm <= NORM_EPSILON)
        ns, nr = {'all':norm_record(gs, an)}, {'all':norm_record(gr, rn)}
        comparison = {'all':dict(dot=dot,
            cosine=max(-1.0,min(1.0,dot/(an*rn))) if an>NORM_EPSILON and rn>NORM_EPSILON else None,
            alignment_to_attack_norm_ratio=rn/an if an>NORM_EPSILON else None,
            attack_near_zero=an<=NORM_EPSILON, alignment_near_zero=rn<=NORM_EPSILON,
            shared_non_none_parameters=sum(left is not None and right is not None for left,right in zip(gs,gr)),
            shared_non_none_elements=sum(left.numel() for left,right in zip(gs,gr) if left is not None and right is not None),
            missing_gradient_convention='None is zero in full-parameter coordinates',
            reduction='one float64 full-parameter vector reduction and one packed CPU transfer')}
        self.check_deadline()
        return dict(smooth_norms=ns, q90_norms=nr, comparison=comparison,
                    gradient_calls=2, optimizer_steps_during_probe=0,
                    objective='smooth vs actual lambda-weighted nodewise q90 penalty')

    def _decision(self, epoch, previous_state):
        self.check_deadline()
        decision = self.outer_policy(epoch, previous_state)
        if not isinstance(decision, dict) or type(decision.get('align')) is not bool:
            raise RuntimeError('Outer policy must return a Boolean align action.')
        if type(decision.get('fallback')) is not bool or not isinstance(decision.get('reason'), str):
            raise RuntimeError('Outer policy decision lacks an auditable fallback/reason.')
        if (epoch == 1 or self.training_arm in ('legacy_hinge_off', 'smooth_off')) and decision['align']:
            raise RuntimeError('Warmup/OFF arm attempted to enable alignment.')
        if self.training_arm == 'smooth_rule' and epoch > 1 and not decision['align']:
            raise RuntimeError('Protected rule control is deterministically ON after warmup.')
        if epoch > 1 and previous_state.get('completed_epoch') != epoch-1:
            raise RuntimeError('Controller input is not the previous completed TRAIN epoch.')
        digest = hashlib.sha256(json.dumps(previous_state, sort_keys=True,
                                           allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        row = dict(epoch=epoch, previous_train_state_sha256=digest,
                   decision=decision, information='lagged TRAIN only; no current/validation outcomes')
        write_json(self.evidence/('outer_action_epoch_%03d.json' % epoch), row, exclusive=True)
        self.action_receipts.append(row)
        self.outer_enabled = decision['align']
        self.applied_actions.append(self.outer_enabled)

    def _train_one_epoch(self):
        self.phase = 'train'
        self.model.train(True)
        self.surrogate_model.eval()
        rows, margins, probes = [], [], []
        before_steps = self.optimizer_steps
        path = self.evidence/'raw'/('train_epoch_%03d_cases.jsonl' % self.current_epoch)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for t_ids, node_budgets, edge_budgets in self.data.get_train_loader():
                self.check_deadline()
                if len(t_ids) != 1:
                    raise RuntimeError('Frozen training keeps one group per optimizer step.')
                group = int(t_ids[0].item())
                self.current_group = group
                ids = torch.tensor(get_n_ids(self.data.target_assignment, group), dtype=torch.long)
                nb, eb = int(node_budgets[0]), int(edge_budgets[0])
                graph = self.get_perturbed_graph(ids, nb, eb, training=True)
                logits = self.get_logits(graph, ids, model='surrogate', training=True)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Nonfinite TRAIN logits.')
                q90, alignment = self.consume_alignment(graph)
                attack_margin, hinge, smooth = attack_losses(logits)
                objective, total = training_objective(hinge, smooth, q90, self.training_arm, self.outer_enabled)
                # Same default zero_grad behavior as the original optimizer;
                # probes below do not accumulate .grad or update optimizer state.
                self.optimizer.zero_grad()
                needs_protection = self.outer_enabled and self.training_arm in PROTECTED_ARMS
                gs = gr = None
                if needs_protection or group in self.probe_groups:
                    gs, gr = self._loss_gradients(smooth, q90)
                probe = self._probe_summary(gs, gr) if group in self.probe_groups else None
                if probe is not None:
                    probes.append(probe)
                protection = None
                if needs_protection:
                    adjusted, protection = protect_alignment_gradients(self.named_parameters_for_probe, gs, gr)
                    total = protected_scalar_proxy(total, self.named_parameters_for_probe, adjusted)
                total.backward()
                self.optimizer.step()
                values = attack_margin.detach().to(dtype=torch.float64, device='cpu')
                margins.append(values)
                row = dict(target_group=group, target_occurrences=len(ids),
                           hinge=float(hinge.detach()), softplus=float(smooth.detach()),
                           objective=float(objective.detach()), total_loss=float(total.detach()),
                           outer_enabled=self.outer_enabled,
                           applied_alignment_penalty=float(q90.detach()) if self.outer_enabled else 0.0,
                           diagnostic_nodewise_q90_penalty=float(q90.detach()),
                           protected_update=needs_protection, gradient_protection=protection,
                           loss_value_semantics=('unmodified smooth+q90 value; gradient is block-protected'
                                                 if needs_protection else 'ordinary objective derivative'),
                           mean_attack_margin=float(values.mean()),
                           margin_positive_nodes=int((values > 0).sum()),
                           generation_rng=self.last_generation_rng, alignment=alignment,
                           gradient_probe=probe)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                rows.append(row)
                self._capture = None
                del graph, logits, q90, hinge, smooth, total, objective, attack_margin, gs, gr
                if needs_protection:
                    del adjusted
                if len(rows) % 64 == 0:
                    print('TRAIN_GROUPS', self.current_epoch, len(rows), flush=True)
        if len(rows) != 256 or self.optimizer_steps-before_steps != 256 or len(probes) != 8:
            raise RuntimeError('Incomplete epoch, optimizer count or fixed eight-group probe.')
        pooled = torch.cat(margins)
        valid = [row['comparison']['all'] for row in probes
                 if row['comparison']['all']['cosine'] is not None
                 and row['comparison']['all']['alignment_to_attack_norm_ratio'] is not None]
        outside = sum(row['alignment']['outside_tolerance_nodes'] for row in rows)
        injected = sum(row['alignment']['injected_nodes'] for row in rows)
        state = dict(
            completed_epoch=self.current_epoch, train_groups=len(rows), target_occurrences=int(pooled.numel()),
            margin_mean=float(pooled.mean()), margin_median=float(torch.quantile(pooled, 0.5)),
            margin_positive_fraction=float((pooled > 0).double().mean()),
            hinge_mean=float(np.mean([row['hinge'] for row in rows])),
            softplus_mean=float(np.mean([row['softplus'] for row in rows])),
            distance_mean=float(np.mean([row['alignment']['mean_distance'] for row in rows])),
            outside_fraction=outside/injected, reference_quantile=float(self.reference['gate_quantile']),
            gradient_probe_groups=len(probes),
            gradient_cosine_median=statistics.median(row['cosine'] for row in valid) if valid else None,
            gradient_norm_ratio_median=statistics.median(row['alignment_to_attack_norm_ratio'] for row in valid) if valid else None,
            gradient_missing_count=len(probes)-len(valid), prior_actions=list(self.applied_actions))
        write_json(self.evidence/('train_state_epoch_%03d.json' % self.current_epoch), state, exclusive=True)
        summary = dict(epoch=self.current_epoch, optimizer_steps=256,
                       applied_outer_action=self.outer_enabled,
                       mean_total_loss=float(np.mean([row['total_loss'] for row in rows])),
                       injected_nodes=injected, outside_tolerance_nodes=outside,
                       scheduled_penalty_active_nodes=(outside if self.outer_enabled else 0),
                       activation_interpretation='q90 eligibility; block protection can suppress/project/cap its gradient',
                       protected_update_groups=sum(row['protected_update'] for row in rows),
                       protected_blocks_projected=sum(block['projection_applied'] for row in rows
                           for block in (row['gradient_protection'] or {}).values()),
                       protected_blocks_capped=sum(block['cap_applied'] for row in rows
                           for block in (row['gradient_protection'] or {}).values()),
                       protected_blocks_suppressed_zero_attack=sum(block['suppressed_attack_near_zero'] for row in rows
                           for block in (row['gradient_protection'] or {}).values()),
                       fallback_nodes=sum(row['alignment']['fallback_nodes'] for row in rows),
                       state_file_sha256=sha256(self.evidence/('train_state_epoch_%03d.json' % self.current_epoch)),
                       margin_convention='logit0-logit1; positive supports attack; all-target-occurrence margin summaries',
                       loss_distance_weighting='equal group means; outside_fraction is injected-node weighted')
        write_json(self.evidence/('train_epoch_%03d.json' % self.current_epoch), summary, exclusive=True)
        self.model.step()
        print('TRAIN_EPOCH', json.dumps(dict(state=state, summary=summary)), flush=True)
        return state

    def _export_attack(self, target_group):
        if int(target_group) not in self.protocol['export_valid_groups']:
            return
        if self.phase != 'validation' or self.current_epoch != 8 or self._capture is None:
            raise RuntimeError('New-method exports require the final validation generation.')
        path = self.evidence/'raw'/('attack_epoch_008_group_%d.pt' % int(target_group))
        if path.exists():
            raise FileExistsError('Preserve the existing final attack export.')
        payload = {key:value.detach().cpu() for key,value in self._capture.items()}
        payload.update(target_group=int(target_group), seed=int(self.config['seed']), epoch=8,
                       phase='validation', arm=self.training_arm, training_arm=self.training_arm,
                       nodewise_alignment_arm='gated', source_commit=self.source_commit,
                       protocol_sha256=self.protocol['_protocol_sha256'],
                       node_budget=int(payload['x_atk'].shape[0]),
                       edge_budget=int(payload['adj_attack'].count_nonzero().item()),
                       original_num_nodes=self.data.graph.num_nodes(),
                       checkpoint_rule='final_epoch8_only',
                       interpretation='Exact raw final validation perturbation; keep private; no victim/test evaluation.')
        torch.save(payload, path)
        self.export_index.append(dict(epoch=8, target_group=int(target_group), phase='validation',
                                      arm=self.training_arm, nodewise_alignment_arm='gated',
                                      source_commit=self.source_commit,
                                      protocol_sha256=self.protocol['_protocol_sha256'],
                                      path=str(path.relative_to(self.evidence)), sha256=sha256(path)))

    def _filter_and_audit(self, graph):
        original = self.data.graph.num_nodes()
        before_nodes, before_edges = graph.num_nodes(), graph.num_edges()
        feature_version = graph.ndata['x']._version
        filtered = apply_filter(graph, self.filter_reference, max_working_bytes=MAX_WORKING_BYTES)
        if graph.num_nodes() != before_nodes or graph.num_edges() != before_edges or graph.ndata['x']._version != feature_version:
            raise RuntimeError('Evaluation filter mutated the source attack graph.')
        flags = filtered['flagged']
        # Origin identities are introduced AFTER the identity-blind scorer.
        src, dst = graph.edges(order='eid')
        removed = (src != dst) & (flags[src.long()] | flags[dst.long()])
        old_old = (src < original) & (dst < original)
        audit = dict(flagged_injected_nodes=int(flags[original:].sum()),
                     injected_nodes=before_nodes-original,
                     flagged_original_nodes=int(flags[:original].sum()), original_nodes=original,
                     removed_original_original_nonself_arcs=int((removed & old_old).sum()),
                     removed_injected_incident_nonself_arcs=int((removed & ~old_old).sum()),
                     removed_total_nonself_arcs=int(removed.sum()),
                     mean_original_anomaly_score=float(filtered['scores'][:original].mean()),
                     mean_injected_anomaly_score=(float(filtered['scores'][original:].mean())
                                                  if before_nodes > original else None),
                     origin_identity_used_for_scoring=False,
                     interpretation='Separate unoptimized anomaly diagnostic; not full camouflage or adaptive robustness.')
        if audit['removed_total_nonself_arcs'] != filtered['counters']['removed_nonself_incident_arcs']:
            raise RuntimeError('Offline removed-edge audit does not reconcile.')
        return filtered, audit

    def evaluate_final(self):
        self.phase = 'validation'
        self.current_epoch = 8
        self.model.eval()
        self.surrogate_model.eval()
        if self._clean_filter_cache is not None:
            raise RuntimeError('Clean filter baseline was already prepared; preserve the original attempt.')
        # One identity-blind isolation of the ORIGINAL graph is reused for all
        # clean target baselines and the separate balanced utility cohort.
        with torch.no_grad():
            self._clean_filter_cache = self._filter_and_audit(self.data.graph)
        clean_filtered_graph = self._clean_filter_cache[0]['post_graph']
        public_rows = []
        wrong = {'raw': 0, 'filtered': 0}
        transition_totals = {'raw':{}, 'filtered':{}}
        exposures = 0
        with (self.evidence/'raw/final_validation_predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream, torch.no_grad():
            for group_value in self.data.target_idx_valid:
                self.check_deadline()
                group = int(group_value)
                self.current_group = group
                ids = torch.tensor(get_n_ids(self.data.target_assignment, group), dtype=torch.long)
                nb, eb = int(self.data.node_budgets[group]), int(self.data.edge_budgets[group])
                clean_raw_logits = self.get_logits(self.data.graph, ids, model='surrogate', training=False)
                clean_filtered_logits = self.get_logits(clean_filtered_graph, ids, model='surrogate', training=False)
                graph = self.get_perturbed_graph(ids, nb, eb, training=False)
                raw_logits = self.get_logits(graph, ids, model='surrogate', training=False)
                _, alignment = self.consume_alignment(graph)
                filtered, filter_audit = self._filter_and_audit(graph)
                post_logits = self.get_logits(filtered['post_graph'], ids, model='surrogate', training=False)
                if not bool(torch.isfinite(raw_logits).all() and torch.isfinite(post_logits).all()
                            and torch.isfinite(clean_raw_logits).all() and torch.isfinite(clean_filtered_logits).all()):
                    raise RuntimeError('Nonfinite final validation logits.')
                labels = graph.ndata['y'][ids.to(graph.device)].detach().cpu()
                raw = raw_logits.argmax(1).detach().cpu()
                post = post_logits.argmax(1).detach().cpu()
                clean_raw = clean_raw_logits.argmax(1).detach().cpu()
                clean_post = clean_filtered_logits.argmax(1).detach().cpu()
                raw_transitions = transition_counts(labels, clean_raw, raw)
                filtered_transitions = transition_counts(labels, clean_post, post)
                counts = dict(raw_incorrect=int((raw != labels).sum()),
                              filtered_incorrect=int((post != labels).sum()),
                              clean_raw_incorrect=raw_transitions['clean_wrong'],
                              clean_filtered_incorrect=filtered_transitions['clean_wrong'],
                              changed_classes=int((raw != post).sum()))
                row = dict(target_group=group, target_occurrences=len(ids), alignment=alignment,
                           **counts, raw_transitions=raw_transitions, filtered_transitions=filtered_transitions,
                           anomaly_filter=filter_audit, filter_counters=filtered['counters'])
                public_rows.append(row)
                private = dict(row, node_ids=ids.tolist(), labels=labels.tolist(),
                               raw_predictions=raw.tolist(), filtered_predictions=post.tolist(),
                               clean_raw_predictions=clean_raw.tolist(), clean_filtered_predictions=clean_post.tolist(),
                               clean_raw_logits=clean_raw_logits.detach().cpu().tolist(),
                               clean_filtered_logits=clean_filtered_logits.detach().cpu().tolist(),
                               raw_logits=raw_logits.detach().cpu().tolist(),
                               filtered_logits=post_logits.detach().cpu().tolist(),
                               generation_rng=self.last_generation_rng)
                stream.write(json.dumps(private, allow_nan=False)+'\n')
                stream.flush()
                self._export_attack(group)
                wrong['raw'] += counts['raw_incorrect']
                wrong['filtered'] += counts['filtered_incorrect']
                for name, values in [('raw',raw_transitions), ('filtered',filtered_transitions)]:
                    for key, value in values.items():
                        transition_totals[name][key] = transition_totals[name].get(key, 0)+value
                exposures += len(ids)
                self._capture = None
                if len(public_rows) % 32 == 0:
                    print('FINAL_VALIDATION_GROUPS', len(public_rows), flush=True)
        if len(public_rows) != 128 or not exposures:
            raise RuntimeError('Final validation denominator is incomplete.')
        write_json(self.evidence/'final_validation_groups.json', public_rows, exclusive=True)
        summary = dict(epoch=8, groups=128, target_occurrences=exposures,
                       raw_incorrect=wrong['raw'], filtered_incorrect=wrong['filtered'],
                       raw_mr_percent=100*wrong['raw']/exposures,
                       filtered_mr_percent=100*wrong['filtered']/exposures,
                       clean_raw_mr_percent=100*transition_totals['raw']['clean_wrong']/exposures,
                       clean_filtered_mr_percent=100*transition_totals['filtered']['clean_wrong']/exposures,
                       raw_transitions=transition_totals['raw'], filtered_transitions=transition_totals['filtered'],
                       clean_filter_counters=self._clean_filter_cache[0]['counters'],
                       clean_filter_audit=self._clean_filter_cache[1],
                       transition_baseline='Raw attack vs original clean graph; filtered attack vs once-filtered original clean graph; all same 128 groups after training.',
                       group_equal_mean_distance=float(np.mean([row['alignment']['mean_distance'] for row in public_rows])),
                       injected_nodes=sum(row['alignment']['injected_nodes'] for row in public_rows),
                       outside_tolerance_nodes=sum(row['alignment']['outside_tolerance_nodes'] for row in public_rows),
                       flagged_injected_nodes=sum(row['anomaly_filter']['flagged_injected_nodes'] for row in public_rows),
                       flagged_original_node_exposures=sum(row['anomaly_filter']['flagged_original_nodes'] for row in public_rows),
                       removed_original_original_nonself_arcs=sum(row['anomaly_filter']['removed_original_original_nonself_arcs'] for row in public_rows),
                       raw_predictions_sha256=sha256(self.evidence/'raw/final_validation_predictions.jsonl'),
                       weighting='Original full 128-group target occurrences; filter does not remove evaluation cases.',
                       checkpoint_rule='Final epoch8 only; no interim validation or best checkpoint selection.')
        write_json(self.evidence/'final_validation.json', summary, exclusive=True)
        write_json(self.evidence/'final_attack_exports.json', self.export_index, exclusive=True)
        return summary

    def evaluate_clean_utility(self):
        ids = torch.tensor([row['node_id'] for row in self.utility_manifest['nodes']], dtype=torch.long)
        labels = torch.tensor([row['label'] for row in self.utility_manifest['nodes']], dtype=torch.long)
        graph = self.data.graph
        self.model.eval()
        if self._clean_filter_cache is None:
            raise RuntimeError('Full-group clean baseline must precede balanced utility evaluation.')
        with torch.no_grad():
            raw_logits = self.get_logits(graph, ids, training=False)
            filtered, audit = self._clean_filter_cache
            post_logits = self.get_logits(filtered['post_graph'], ids, training=False)
            if not bool(torch.isfinite(raw_logits).all() and torch.isfinite(post_logits).all()):
                raise RuntimeError('Nonfinite clean utility logits.')
            raw = raw_logits.argmax(1).detach().cpu()
            post = post_logits.argmax(1).detach().cpu()
        write_json(self.evidence/'raw/clean_utility_predictions.json', dict(
            nodes=self.utility_manifest['nodes'], raw_logits=raw_logits.detach().cpu().tolist(),
            filtered_logits=post_logits.detach().cpu().tolist(), raw_predictions=raw.tolist(),
            filtered_predictions=post.tolist()), exclusive=True)
        summary = dict(cases=128, raw=binary_metrics(labels, raw), filtered=binary_metrics(labels, post),
                       filter_audit=audit, filter_counters=filtered['counters'],
                       selected_before_outputs=True, agent_feedback=False,
                       interpretation='Balanced selected original VALID utility cohort; not population prevalence or test utility.')
        write_json(self.evidence/'clean_utility.json', summary, exclusive=True)
        return summary

    def run(self):
        previous = {}
        for epoch in range(1, 9):
            self.current_epoch = epoch
            self._decision(epoch, previous)
            previous = self._train_one_epoch()
            self.completed_epochs = epoch
            write_json(self.evidence/'progress.json', dict(completed_epochs=epoch,
                       gcn_forwards=self.gcn_forwards, generator_forwards=self.generator_forwards,
                       optimizer_steps=self.optimizer_steps, gradient_calls=self.gradient_calls,
                       actual_prior_actions=list(self.applied_actions), validation_performed=False))
        protected_on_epochs = (sum(self.applied_actions) if self.training_arm in PROTECTED_ARMS else 0)
        expected_gradients = 128+496*protected_on_epochs
        if self.optimizer_steps != 2048 or self.gradient_calls != expected_gradients:
            raise RuntimeError('Frozen eight-epoch update/gradient counts differ.')
        model_path = self.evidence/'final_model.pt'
        optimizer_path = self.evidence/'final_optimizer.pt'
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)
        write_json(self.evidence/'applied_actions.json', dict(arm=self.training_arm,
                   seed=int(self.config['seed']), actions=self.applied_actions,
                   protocol_sha256=self.protocol['_protocol_sha256'],
                   decisions=self.action_receipts, epochs=8,
                   action_semantics='Outer Boolean multiplies original nodewise q90 penalty; lambda=.1'), exclusive=True)
        write_json(self.evidence/'outer_actions.json', dict(actions=self.applied_actions,
                   seed=int(self.config['seed']), arm=self.training_arm,
                   protocol_sha256=self.protocol['_protocol_sha256']), exclusive=True)
        final = self.evaluate_final()
        clean = self.evaluate_clean_utility()
        if self.generator_forwards != MAX_GENERATIONS or self.gcn_forwards != MAX_GCN_FORWARDS:
            raise RuntimeError('Final generation/GCN counters do not reconcile.')
        if {row['target_group'] for row in self.export_index} != set(self.protocol['export_valid_groups']):
            raise RuntimeError('Final epoch eight-group exports are incomplete.')
        self.check_deadline()
        return dict(final_validation=final, clean_utility=clean,
                    final_checkpoint=str(model_path.relative_to(self.evidence)),
                    final_checkpoint_sha256=sha256(model_path),
                    optimizer_checkpoint=str(optimizer_path.relative_to(self.evidence)),
                    optimizer_checkpoint_sha256=sha256(optimizer_path),
                    final_epoch=8, temperature=self.model.temp_scheduler.get_temperature(),
                    initial_model_state_sha256=self.initial_model_state_sha256,
                    epsilon=self.model.eps_scheduler.get_epsilon(), actions=self.applied_actions,
                    export_sha256=sha256(self.evidence/'final_attack_exports.json'),
                    optimizer_steps=self.optimizer_steps, gradient_calls=self.gradient_calls,
                    protected_on_epochs=protected_on_epochs, expected_gradient_calls=expected_gradients,
                    gcn_forwards=self.gcn_forwards, generator_forwards=self.generator_forwards,
                    api_calls_counted_by='root controller/bridge physical request ledger',
                    gaga_forwards=0, test_used=False)
