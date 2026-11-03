"""Independent stdlib-only audit of saved alignment-agent validation outputs.

No project imports, tensor deserialization, subprocess, network, training or
inference. Missing binary/data artifacts are reported explicitly, never treated
as independently verified hashes. The output contains aggregates, not node IDs.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys


ARMS = ('legacy_hinge_off', 'smooth_off', 'smooth_threshold', 'smooth_rule',
        'smooth_agent', 'smooth_random')
PROTECTED = {'smooth_rule', 'smooth_agent', 'smooth_random'}
SOURCE = '5a1d9e57a99541006c6d749adda9602217575421'
PROTOCOL_SHA = '1f7098358dd6f625642100de0242ccc25ae62389f75cef94ec9e0d777efa5ee3'
STATE_FIELDS = {'completed_epoch', 'train_groups', 'target_occurrences', 'margin_mean',
    'margin_median', 'margin_positive_fraction', 'hinge_mean', 'softplus_mean',
    'distance_mean', 'outside_fraction', 'reference_quantile', 'gradient_probe_groups',
    'gradient_cosine_median', 'gradient_norm_ratio_median', 'gradient_missing_count',
    'prior_actions'}


class AuditFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def near(actual, expected, message, *, rel=2e-6, absolute=1e-7):
    require(type(actual) in (int, float) and math.isfinite(actual), message + ': nonfinite/non-numeric')
    require(math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute),
            '%s: %r versus %r' % (message, actual, expected))


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))


def object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def load(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=object_no_duplicates,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def lines(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line, object_pairs_hook=object_no_duplicates,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                for line in stream if line.strip()]


def argmax_rows(values):
    result = []
    for value in values:
        require(isinstance(value, list) and len(value) == 2, 'Expected two class logits')
        require(all(type(x) in (float, int) and math.isfinite(x) for x in value), 'Invalid saved logit')
        # torch.argmax returns the first index on an exact tie.
        result.append(0 if value[0] >= value[1] else 1)
    return result


def transitions(labels, clean, attacked):
    require(len(labels) == len(clean) == len(attacked) and labels, 'Transition length mismatch')
    cw = [a != b for a, b in zip(labels, clean)]
    aw = [a != b for a, b in zip(labels, attacked)]
    return dict(target_occurrences=len(labels), clean_wrong=sum(cw), attacked_wrong=sum(aw),
        new_wrong=sum(not a and b for a, b in zip(cw, aw)),
        recoveries=sum(a and not b for a, b in zip(cw, aw)),
        retained_correct=sum(not a and not b for a, b in zip(cw, aw)),
        retained_wrong=sum(a and b for a, b in zip(cw, aw)))


def case_seed(seed, phase, epoch, group):
    raw = ('%s|%s|%s|%s|alignment-agent-v2' % (seed, phase, epoch, group)).encode()
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


def metrics(labels, predictions):
    n = len(labels)
    benign = sum(v == 0 for v in labels)
    fraud = sum(v == 1 for v in labels)
    correct = sum(a == b for a, b in zip(labels, predictions))
    return dict(cases=n, correct=correct, accuracy=correct/n, benign_count=benign, fraud_count=fraud,
        benign_false_positive_rate=sum(y == 0 and p == 1 for y, p in zip(labels, predictions))/benign if benign else None,
        fraud_recall=sum(y == 1 and p == 1 for y, p in zip(labels, predictions))/fraud if fraud else None)


class Auditor:
    def __init__(self, args):
        self.args = args
        self.repo = pathlib.Path(args.repo_root).resolve()
        self.stage = pathlib.Path(args.stage_root).resolve()
        self.matrix = pathlib.Path(args.matrix_root).resolve()
        self.file_map = load(args.file_map) if args.file_map else {}
        self.errors, self.unavailable, self.verified = [], [], []
        self.jobs, self.initial_hashes = [], collections.defaultdict(set)
        self.train_budgets, self.clean_vectors, self.utility_manifest = {}, {}, None
        self.protocol = load(args.protocol)
        self.protocol_sha = sha(args.protocol)
        self.reference = None
        self.budget_meta = {}

    def resolve(self, relative, *, job=None):
        relative = str(relative).replace('\\', '/')
        if relative in self.file_map:
            return pathlib.Path(self.file_map[relative]).resolve()
        candidates = []
        if job is not None:
            candidates.append(job / relative)
        candidates.append(self.repo / relative)
        marker = '/'+str(self.protocol['stage_root']).rstrip('/').split('/')[-1]+'/'
        normalized = '/'+relative.lstrip('/')
        if marker in normalized:
            candidates.append(self.stage / normalized.split(marker, 1)[1])
        candidates += [self.stage / pathlib.PurePosixPath(relative).name,
                       self.matrix / pathlib.PurePosixPath(relative).name]
        existing = [p for p in candidates if p.is_file()]
        return existing[0] if existing else candidates[0]

    def hash_check(self, relative, expected, role, *, job=None):
        require(isinstance(expected, str) and len(expected) == 64 and
                all(c in '0123456789abcdef' for c in expected), role + ': invalid recorded SHA256')
        path = self.resolve(relative, job=job)
        if not path.is_file():
            self.unavailable.append(dict(role=role, artifact=str(relative), recorded_sha256=expected))
            return None
        actual = sha(path)
        require(actual == expected, role + ': artifact byte hash mismatch: ' + str(relative))
        self.verified.append(dict(role=role, artifact=str(relative), sha256=actual))
        return path

    def section(self, label, function):
        try:
            return function()
        except (AuditFailure, OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            self.errors.append(dict(section=label, error=type(error).__name__, detail=str(error)))
            return None

    def provenance(self):
        p = self.protocol
        require(self.protocol_sha == self.args.expected_protocol_sha256, 'Execution protocol SHA changed')
        require(p['source_commit'] == self.args.expected_source, 'Execution source changed')
        require(p['status'] == 'frozen' and p['owner_execution_authorized'] is True, 'Protocol not frozen/authorized')
        require(p['arms'] == list(ARMS) and p['seeds'] == [3, 4, 5], 'Matrix membership changed')
        require(p['epochs'] == 8 and p['terminal_epoch'] == 8 and p['validation_epochs'] == [8], 'Terminal rule changed')
        require(p['phase_gpu_hours'] == 2 and p['job_wall_seconds'] == 420, 'Compute ceilings changed')
        require(p['test_queries_allowed'] is False and p['gaga_queries_allowed'] is False, 'Query boundary changed')
        require(len(set(p['train_groups'])) == 256 and len(set(p['valid_groups'])) == 128, 'Group denominator changed')
        require(not set(p['train_groups']) & set(p['valid_groups']), 'TRAIN/VALID group overlap')
        for name, wanted in p['source_files'].items():
            self.hash_check(name, wanted, 'frozen_source')
        for row in p['inputs']:
            self.hash_check(row['path'], row['sha256'], 'frozen_input')
        self.hash_check('course/configs/baseline.json', p['base_config_sha256'], 'frozen_baseline_config')
        for key in ('alignment_reference_file', 'filter_reference_file'):
            row = p[key]
            path = self.hash_check(row['path'], row['sha256'], key)
            require(path is not None, 'Reference must be transferred: ' + key)
            wrapper = load(path)
            if key == 'filter_reference_file':
                require(wrapper['status'] == 'completed' and wrapper['model_forwards'] == 0 and
                        wrapper['optimizer_steps'] == 0 and wrapper['gpu_hours'] == 0, 'Filter fit boundary changed')
                self.reference = wrapper['reference']
                require(self.reference['fraud_labels_used'] is False and self.reference['model_calls'] == 0,
                        'Filter reference used labels or model calls')
                require(self.reference['flag_rule'] == 'strict_score_greater_than_threshold' and
                        self.reference['calibration_quantile'] == .99 and self.reference['split_seed'] == 20261103,
                        'Filter calibration definition changed')
        path = pathlib.Path(self.args.budget_metadata) if self.args.budget_metadata else self.repo/'evidence/results_first_diagnosis_v1/group_outcomes.csv'
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                group = int(row['target_group'])
                item = tuple(int(row[k]) for k in ('group_size', 'node_budget', 'edge_budget'))
                require(group not in self.budget_meta or self.budget_meta[group] == item, 'Metadata budget inconsistent')
                self.budget_meta[group] = item
        require(set(p['valid_groups']) <= set(self.budget_meta), 'Frozen VALID budget metadata incomplete')
        self.verified.append(dict(role='saved_validation_metadata', artifact=str(path), sha256=sha(path)))
        old = load(self.repo/'course/improvements/pilot_protocol_v1.json')
        require(p['train_groups'] == old['train_groups'] and p['valid_groups'] == old['valid_groups'], 'Old pilot membership changed')

    def rng(self, record, seed, phase, epoch, group):
        require(record['rng_restored_exactly'] is True and record['case_seed'] == case_seed(seed, phase, epoch, group),
                'Generation RNG key/restoration mismatch')
        require(record['phase'] == phase and record['epoch'] == epoch and record['target_group'] == group,
                'Generation RNG context mismatch')

    def physical(self, row, *, validation):
        group, n = row['target_group'], row['target_occurrences']
        a = row['alignment']
        nb, eb = a['node_budget'], a['edge_budget']
        require(type(nb) is int and type(eb) is int and nb >= 1 and eb >= nb, 'Invalid physical budget')
        require(a['injected_nodes'] == nb and a['actual_undirected_injected_edges'] == eb, 'Physical budget counter mismatch')
        require(0 <= a['outside_tolerance_nodes'] <= nb and 0 <= a['fallback_nodes'] <= nb, 'Invalid nodewise signature count')
        key = (n, nb, eb)
        if validation:
            require(key == self.budget_meta[group], 'VALID budget differs from frozen metadata')
        else:
            require(group not in self.train_budgets or self.train_budgets[group] == key, 'TRAIN budget differs across epochs/arms')
            self.train_budgets[group] = key
        return nb, eb

    def filter_counts(self, counters, audit, nb, eb):
        ref = self.reference
        nodes = ref['original_graph_nodes'] + nb
        arcs = ref['original_graph_arcs'] + nb + 2*eb
        require(counters['input_nodes'] == counters['output_nodes'] == counters['self_loops_preserved'] == nodes,
                'Filter altered node/self-loop denominator')
        require(counters['input_arcs'] == arcs and counters['input_nonself_arcs'] == arcs-nodes, 'Filter input physical arcs mismatch')
        removed = counters['removed_nonself_incident_arcs']
        require(counters['output_arcs'] == arcs-removed and counters['output_nonself_arcs'] == arcs-removed-nodes,
                'Filter removed-arc counts mismatch')
        require(0 <= removed <= arcs-nodes and removed % 2 == 0, 'Filter broke symmetric physical-arc count')
        require(counters['input_node_edge_counts_preserved'] is True and counters['current_injected_id_or_origin_input'] is False
                and counters['fraud_labels_used'] is False and audit['origin_identity_used_for_scoring'] is False,
                'Filter information/mutation guard failed')
        require(audit['original_nodes'] == ref['original_graph_nodes'] and audit['injected_nodes'] == nb,
                'Offline origin denominator mismatch')
        require(0 <= audit['flagged_injected_nodes'] <= nb and 0 <= audit['flagged_original_nodes'] <= audit['original_nodes'],
                'Offline flag count invalid')
        require(counters['flagged_nodes'] == audit['flagged_injected_nodes']+audit['flagged_original_nodes'], 'Flag counts do not reconcile')
        require(audit['removed_total_nonself_arcs'] == removed ==
                audit['removed_original_original_nonself_arcs']+audit['removed_injected_incident_nonself_arcs'],
                'Offline removed-edge partition mismatch')
        near(counters['flagged_fraction'], counters['flagged_nodes']/nodes, 'Flag fraction')
        near(counters['threshold'], ref['threshold'], 'Filter threshold', rel=0, absolute=0)

    def job(self, arm, seed):
        p = self.protocol
        directory = self.matrix/('%s_seed%d' % (arm, seed))
        result, reg, config = (load(directory/name) for name in ('result.json', 'registered.json', 'config.json'))
        require(result['status'] == 'completed' and result['arm'] == arm and result['seed'] == seed, 'Job failed or identity mismatch')
        require(reg['arm'] == arm and reg['seed'] == seed and reg['source_commit'] == p['source_commit'], 'Registration identity mismatch')
        require(reg['provenance']['protocol_sha256'] == self.protocol_sha and reg['provenance']['tracked_changes'] == '', 'Registration protocol/clean-source mismatch')
        require(reg['own_gpu_uuids'] and set(reg['own_gpu_uuids']) == {p['physical_gpu_uuid']}, 'Physical GPU mismatch')
        require(reg['config_sha256'] == sha(directory/'config.json'), 'Configuration byte hash mismatch')
        for key, expected in [('epochs',8), ('batch_size',1), ('valid_epoch',8), ('early_stopping',False),
                              ('random_candidates',False), ('surrogate_model','GCN'), ('victim_model','GCN'), ('seed',seed)]:
            require(config[key] == expected, 'Configuration changed: '+key)
        runtime = load(self.repo/'environment/runtime.json')
        require(all(reg['runtime'][key] == runtime[key] for key in reg['runtime']), 'Locked runtime mismatch')
        require(reg['alignment_reference_sha256'] == p['alignment_reference_file']['sha256'] and
                reg['filter_reference_sha256'] == p['filter_reference_file']['sha256'], 'Job reference mismatch')
        require(reg['counter_caps'] == dict(generations=2176,gcn_forwards=2562,optimizer_steps=2048,maximum_gradient_calls=3600), 'Recorded caps changed')
        require(reg['test_used'] is False and reg['victim_loaded'] is False and result['gaga_forwards'] == 0 and result['test_used'] is False,
                'Forbidden model/query declaration')
        groups = load(directory/'target_groups.json')
        require(groups['train'] == p['train_groups'] and groups['valid'] == p['valid_groups'] and
                groups['gradient_probe'] == p['gradient_probe_groups'] and groups['export'] == p['export_valid_groups'], 'Job cohorts changed')
        initial = load(directory/'initial_model_state.json')
        require(initial['arm'] == arm and initial['seed'] == seed and initial['source_commit'] == p['source_commit'] and
                initial['captured_before_training'] is True and
                initial['optimizer_steps'] == initial['generator_forwards'] == initial['gcn_forwards'] == 0,
                'Initial-weight capture was not before all work')
        initial_sha = initial['initial_model_state_sha256']
        require(result['initial_model_state_sha256'] == result['outcome']['initial_model_state_sha256'] == initial_sha,
                'Initial-weight hash receipts disagree')
        self.initial_hashes[seed].add(initial_sha)
        actions = load(directory/'outer_actions.json')
        applied = load(directory/'applied_actions.json')
        flags = actions['actions']
        require(len(flags) == 8 and all(type(v) is bool for v in flags) and flags[0] is False, 'Incomplete Boolean action schedule')
        require(actions['arm'] == applied['arm'] == arm and actions['seed'] == applied['seed'] == seed and
                actions['protocol_sha256'] == applied['protocol_sha256'] == self.protocol_sha and applied['actions'] == flags,
                'Applied schedule identity mismatch')
        if arm in ('legacy_hinge_off','smooth_off'):
            require(not any(flags), 'OFF arm enabled penalty')
        if arm in ('smooth_threshold','smooth_rule'):
            require(flags == [False]+[True]*7, 'Fixed ON control changed')
        decisions = []
        for epoch in range(1, 9):
            state = load(directory/('train_state_epoch_%03d.json'%epoch))
            summary = load(directory/('train_epoch_%03d.json'%epoch))
            rows = lines(directory/'raw'/('train_epoch_%03d_cases.jsonl'%epoch))
            decision = load(directory/('outer_action_epoch_%03d.json'%epoch))
            previous = {} if epoch == 1 else load(directory/('train_state_epoch_%03d.json'%(epoch-1)))
            require(decision['epoch'] == epoch and decision['decision']['align'] == flags[epoch-1] and
                    decision['previous_train_state_sha256'] == hashlib.sha256(canonical(previous).encode()).hexdigest(), 'Action lagged-state mismatch')
            decisions.append(decision)
            require(set(state) == STATE_FIELDS and state['completed_epoch'] == epoch and state['prior_actions'] == flags[:epoch], 'TRAIN state allowlist/history mismatch')
            require(len(rows) == 256 and {r['target_group'] for r in rows} == set(p['train_groups']), 'TRAIN epoch omitted/duplicated a group')
            require(summary['optimizer_steps'] == 256 and summary['applied_outer_action'] == flags[epoch-1], 'TRAIN summary steps/action mismatch')
            require(summary['state_file_sha256'] == sha(directory/('train_state_epoch_%03d.json'%epoch)), 'TRAIN state byte hash mismatch')
            probes = []
            for row in rows:
                group = row['target_group']
                self.physical(row, validation=False)
                self.rng(row['generation_rng'], seed, 'train', epoch, group)
                require(row['outer_enabled'] == flags[epoch-1], 'Per-case outer action mismatch')
                protected = arm in PROTECTED and flags[epoch-1]
                require(row['protected_update'] == protected, 'Per-case protected update mismatch')
                base = row['hinge'] if arm == 'legacy_hinge_off' else row['softplus']
                near(row['objective'], base, 'TRAIN base objective')
                q90 = row['diagnostic_nodewise_q90_penalty']
                require(q90 >= 0, 'Negative candidate penalty')
                near(row['applied_alignment_penalty'], q90 if flags[epoch-1] else 0, 'Applied penalty value')
                near(row['total_loss'], base+(q90 if flags[epoch-1] else 0), 'Reported loss value')
                if protected:
                    require(row['gradient_protection'], 'Missing block protection receipt')
                    for block in row['gradient_protection'].values():
                        an, qn = block['attack_norm'], block['protected_alignment_norm']
                        require(math.isfinite(an) and math.isfinite(qn) and an >= 0 and qn >= 0, 'Invalid protected norm')
                        require(qn <= an+max(1e-12,an*1e-9), 'Alignment gradient exceeds block cap')
                        require(block['protected_dot'] >= -max(1e-18,an*qn*1e-9), 'Protected gradient still opposes attack')
                        if block['suppressed_attack_near_zero']:
                            require(qn == 0, 'Near-zero attack block not suppressed')
                else:
                    require(row['gradient_protection'] is None, 'Unexpected protected-gradient receipt')
                if row['gradient_probe'] is not None:
                    probes.append(row)
            require({r['target_group'] for r in probes} == set(p['gradient_probe_groups']) and len(probes) == 8, 'Gradient-probe cohort changed')
            n = sum(r['target_occurrences'] for r in rows)
            injected = sum(r['alignment']['injected_nodes'] for r in rows)
            outside = sum(r['alignment']['outside_tolerance_nodes'] for r in rows)
            require(state['train_groups'] == 256 and state['target_occurrences'] == n and state['gradient_probe_groups'] == 8, 'TRAIN state denominator mismatch')
            near(state['margin_mean'], sum(r['target_occurrences']*r['mean_attack_margin'] for r in rows)/n, 'TRAIN pooled mean margin')
            near(state['margin_positive_fraction'], sum(r['margin_positive_nodes'] for r in rows)/n, 'TRAIN positive-margin fraction')
            near(state['outside_fraction'], outside/injected, 'TRAIN outside fraction')
            for field, rowfield in [('hinge_mean','hinge'),('softplus_mean','softplus')]:
                near(state[field], statistics.mean(r[rowfield] for r in rows), 'TRAIN '+field)
            near(state['distance_mean'], statistics.mean(r['alignment']['mean_distance'] for r in rows), 'TRAIN group-equal distance')
            valid = [r['gradient_probe']['comparison']['all'] for r in probes
                     if r['gradient_probe']['comparison']['all']['cosine'] is not None and
                     r['gradient_probe']['comparison']['all']['alignment_to_attack_norm_ratio'] is not None]
            require(state['gradient_missing_count'] == 8-len(valid), 'Gradient missing denominator mismatch')
            if valid:
                near(state['gradient_cosine_median'], statistics.median(r['cosine'] for r in valid), 'Gradient cosine median')
                near(state['gradient_norm_ratio_median'], statistics.median(r['alignment_to_attack_norm_ratio'] for r in valid), 'Gradient ratio median')
            else:
                require(state['gradient_cosine_median'] is None and state['gradient_norm_ratio_median'] is None, 'Missing diagnostics not nullable')
            if arm == 'smooth_agent' and epoch > 1:
                controller = load(directory/('applied_controller_epoch_%02d.json'%epoch))
                require(controller == decision['decision'], 'Applied controller receipt differs from training action')
                state_sha = hashlib.sha256(canonical(previous).encode()).hexdigest()
                require(controller['state_sha256'] == state_sha and controller['protocol_sha256'] == self.protocol_sha, 'Controller context mismatch')
                request_path = self.stage/'requests'/(controller['request_id']+'.json')
                envelope = load(request_path)
                require(envelope['state'] == previous and envelope['epoch'] == epoch and envelope['seed'] == seed and
                        envelope['protocol_sha256'] == self.protocol_sha, 'Controller request contains wrong state/context')
        require(applied['decisions'] == decisions, 'Final applied decisions differ from per-epoch receipts')
        outcome = result['outcome']
        expected_gradients = 128+496*(sum(flags) if arm in PROTECTED else 0)
        for record in (result, outcome):
            require(record['optimizer_steps'] == 2048 and record['generator_forwards'] == 2176 and
                    record['gcn_forwards'] == 2562 and record['gradient_calls'] == expected_gradients, 'Terminal counters mismatch')
        require(result['completed_train_epochs'] == outcome['final_epoch'] == 8 and result['applied_epoch_actions'] == 8 and
                outcome['actions'] == flags and outcome['gaga_forwards'] == 0 and outcome['test_used'] is False, 'Terminal epoch/query status mismatch')
        for key in ('final_checkpoint','optimizer_checkpoint'):
            self.hash_check(outcome[key], outcome[key+'_sha256'], 'job_'+key, job=directory)
        require(not list(directory.glob('validation_epoch_*')) and not list((directory/'raw').glob('validation_epoch_*')),
                'Unexpected interim validation artifact')
        progress = load(directory/'progress.json')
        require(progress['completed_epochs'] == 8 and progress['validation_performed'] is False and
                progress['optimizer_steps'] == 2048 and progress['generator_forwards'] == progress['gcn_forwards'] == 2048,
                'Pre-final progress does not confirm no interim validation')
        validation = self.validation(directory, arm, seed)
        require(outcome['final_validation'] == validation, 'Result embeds a different final validation summary')
        utility = self.utility(directory)
        require(outcome['clean_utility'] == utility, 'Result embeds a different clean utility summary')
        exports = load(directory/'final_attack_exports.json')
        require(len(exports) == 8 and {r['target_group'] for r in exports} == set(p['export_valid_groups']), 'Final export cohort incomplete')
        require(sha(directory/'final_attack_exports.json') == outcome['export_sha256'], 'Export index byte hash mismatch')
        for row in exports:
            require(row['arm'] == arm and row['nodewise_alignment_arm'] == 'gated' and row['epoch'] == 8 and
                    row['phase'] == 'validation' and row['source_commit'] == p['source_commit'] and
                    row['protocol_sha256'] == self.protocol_sha, 'Export method/protocol identity mismatch')
            self.hash_check(row['path'], row['sha256'], 'raw_attack_export', job=directory)
        self.jobs.append(dict(arm=arm,seed=seed,raw_mr_percent=validation['raw_mr_percent'],
            filtered_mr_percent=validation['filtered_mr_percent'],
            clean_raw_mr_percent=validation['clean_raw_mr_percent'],clean_filtered_mr_percent=validation['clean_filtered_mr_percent'],
            raw_transitions=validation['raw_transitions'],filtered_transitions=validation['filtered_transitions'],
            injected_nodes=validation['injected_nodes'],outside_tolerance_nodes=validation['outside_tolerance_nodes'],
            flagged_injected_nodes=validation['flagged_injected_nodes'],
            flagged_original_node_exposures=validation['flagged_original_node_exposures'],
            actual_on_epochs=sum(flags),protected_on_epochs=sum(flags) if arm in PROTECTED else 0,
            optimizer_steps=2048,generator_forwards=2176,gcn_forwards=2562,gradient_calls=expected_gradients,
            initial_model_state_sha256=initial_sha))

    def validation(self, directory, arm, seed):
        summary = load(directory/'final_validation.json')
        rows = lines(directory/'raw/final_validation_predictions.jsonl')
        public = load(directory/'final_validation_groups.json')
        require(len(rows) == len(public) == 128 and [r['target_group'] for r in rows] == self.protocol['valid_groups'], 'VALID full cohort/order mismatch')
        totals = {'raw':collections.Counter(), 'filtered':collections.Counter()}
        n = 0
        for row, pub in zip(rows, public):
            require(all(row[key] == value for key, value in pub.items()), 'Public/private VALID row differs')
            group, labels, ids = row['target_group'], row['labels'], row['node_ids']
            require(len(labels) == len(ids) == row['target_occurrences'] and len(set(ids)) == len(ids) and
                    all(v == 1 for v in labels), 'Target identity/label denominator mismatch')
            nb, eb = self.physical(row, validation=True)
            self.filter_counts(row['filter_counters'], row['anomaly_filter'], nb, eb)
            self.rng(row['generation_rng'], seed, 'validation', 8, group)
            for prefix in ('raw','filtered','clean_raw','clean_filtered'):
                require(argmax_rows(row[prefix+'_logits']) == row[prefix+'_predictions'], 'Saved argmax mismatch: '+prefix)
                require(len(row[prefix+'_predictions']) == len(labels), 'Prediction count mismatch')
            for name, cp, ap in [('raw','clean_raw_predictions','raw_predictions'),
                                 ('filtered','clean_filtered_predictions','filtered_predictions')]:
                derived = transitions(labels, row[cp], row[ap])
                require(derived == row[name+'_transitions'], 'Per-group transition identity mismatch')
                totals[name].update(derived)
                require(row[name+'_incorrect'] == derived['attacked_wrong'], 'Per-group MR count mismatch')
                require(row['clean_'+name+'_incorrect'] == derived['clean_wrong'], 'Per-group clean count mismatch')
            require(row['changed_classes'] == sum(a != b for a,b in zip(row['raw_predictions'],row['filtered_predictions'])), 'Filter class changes mismatch')
            key = (group, tuple(ids))
            clean_vector = (tuple(row['clean_raw_predictions']), tuple(row['clean_filtered_predictions']))
            require(key not in self.clean_vectors or self.clean_vectors[key] == clean_vector,
                    'Fixed same-group clean classes differ across new jobs; investigate numerical/context effects')
            self.clean_vectors[key] = clean_vector
            n += len(labels)
        require(n == 1091 and summary['target_occurrences'] == 1091 and summary['groups'] == 128 and summary['epoch'] == 8, 'VALID exposure denominator mismatch')
        for name in ('raw','filtered'):
            require(dict(totals[name]) == summary[name+'_transitions'], 'Pooled transitions mismatch')
            require(summary[name+'_incorrect'] == totals[name]['attacked_wrong'], 'Pooled MR count mismatch')
            near(summary[name+'_mr_percent'], 100*totals[name]['attacked_wrong']/n, 'Pooled MR percent')
            near(summary['clean_'+name+'_mr_percent'], 100*totals[name]['clean_wrong']/n, 'Pooled clean MR percent')
        require(summary['raw_predictions_sha256'] == sha(directory/'raw/final_validation_predictions.jsonl'), 'Raw VALID byte hash mismatch')
        for field, section, key in [('injected_nodes','alignment','injected_nodes'),
            ('outside_tolerance_nodes','alignment','outside_tolerance_nodes'),
            ('flagged_injected_nodes','anomaly_filter','flagged_injected_nodes'),
            ('flagged_original_node_exposures','anomaly_filter','flagged_original_nodes'),
            ('removed_original_original_nonself_arcs','anomaly_filter','removed_original_original_nonself_arcs')]:
            require(summary[field] == sum(r[section][key] for r in rows), 'Pooled field mismatch: '+field)
        near(summary['group_equal_mean_distance'], statistics.mean(r['alignment']['mean_distance'] for r in rows), 'VALID group-equal distance')
        self.filter_counts(summary['clean_filter_counters'],summary['clean_filter_audit'],0,0)
        return summary

    def utility(self, directory):
        summary = load(directory/'clean_utility.json')
        raw = load(directory/'raw/clean_utility_predictions.json')
        manifest = load(directory/'raw/clean_utility_manifest.json')
        reg = load(directory/'registered.json')
        require(reg['utility_manifest_sha256'] == sha(directory/'raw/clean_utility_manifest.json'), 'Utility manifest hash mismatch')
        require(manifest['nodes'] == raw['nodes'] and len(raw['nodes']) == 128, 'Utility cohort mismatch')
        require(len({r['node_id'] for r in raw['nodes']}) == 128, 'Utility duplicate nodes')
        if self.utility_manifest is None:
            self.utility_manifest = manifest
        require(self.utility_manifest == manifest, 'Utility manifest differs across jobs')
        labels = [r['label'] for r in raw['nodes']]
        require(labels.count(0) == labels.count(1) == 64, 'Utility class denominator mismatch')
        for name, logitkey, predictionkey in [('raw','raw_logits','raw_predictions'),('filtered','filtered_logits','filtered_predictions')]:
            require(argmax_rows(raw[logitkey]) == raw[predictionkey], 'Utility argmax mismatch')
            require(metrics(labels,raw[predictionkey]) == summary[name], 'Utility metrics mismatch')
        require(summary['agent_feedback'] is False and summary['selected_before_outputs'] is True, 'Utility feedback boundary changed')
        return summary

    def schedules_and_cost(self):
        p = self.protocol
        queue = load(self.matrix/'queue_complete.json')
        progress = load(self.matrix/'queue_progress.json')
        require(queue['status'] == 'completed' and queue['protocol_sha256'] == self.protocol_sha and
                queue['test_queries'] == queue['gaga_forwards'] == 0, 'Queue did not complete within boundary')
        require(queue['jobs'] == progress['jobs'] and progress['pending'] == 0, 'Final queue receipts disagree')
        expected = [(arm,seed) for seed in (3,4,5) for arm in ARMS]
        require([(r['arm'],r['seed']) for r in queue['jobs']] == expected, 'Queue omitted/duplicated/reordered a job')
        for seed in (3,4,5):
            agent_path = self.matrix/('smooth_agent_seed%d'%seed)/'outer_actions.json'
            agent = load(agent_path)['actions']
            schedule_path = self.matrix/('matched_schedule_seed%d.json'%seed)
            schedule = load(schedule_path)
            tail = list(agent[1:]); random.Random(20261103+seed).shuffle(tail)
            require(schedule['actions'] == [False]+tail and schedule['source_enabled_epochs'] == sum(agent) and
                    schedule['seed'] == seed and schedule['protocol_sha256'] == self.protocol_sha and
                    schedule['source_actions_sha256'] == sha(agent_path), 'Matched random derivation mismatch')
            actual = load(self.matrix/('smooth_random_seed%d'%seed)/'outer_actions.json')['actions']
            require(actual == schedule['actions'] and sum(actual) == sum(agent), 'Random applied ON count/permutation differs')
            require(load(self.matrix/('smooth_random_seed%d'%seed)/'registered.json')['matched_schedule_sha256'] == sha(schedule_path), 'Matched schedule byte hash mismatch')
            require(len(self.initial_hashes[seed]) == 1, 'Initial weight content hashes differ within seed')
        diagnostic = self.stage/'gradient_diagnostic_guardian.json'
        require(sha(diagnostic) == p['diagnostic_guardian_sha256'], 'Diagnostic guardian hash mismatch')
        dg = load(diagnostic)
        require(dg['status'] == 'completed', 'Diagnostic did not complete')
        near(p['diagnostic_gpu_hours'],dg['charged_gpu_wall_hours'],'Diagnostic charge')
        total = p['diagnostic_gpu_hours']
        costs = []
        for row in queue['jobs']:
            require(row['exit_code'] == 0, 'Nonzero completed job exit')
            guardian_path = self.matrix/('%s_seed%d_guardian.json'%(row['arm'],row['seed']))
            require(sha(guardian_path) == row['guardian_sha256'], 'Job guardian byte hash mismatch')
            guardian = load(guardian_path)
            require(guardian['status'] == 'completed' and guardian['physical_gpu_uuid'] == p['physical_gpu_uuid'], 'Guardian status/GPU mismatch')
            charge = max(row['elapsed_process_seconds']/3600,guardian['charged_gpu_wall_hours'])
            near(row['charged_gpu_wall_hours'],charge,'Conservative queue outer charge')
            result = load(self.matrix/('%s_seed%d'%(row['arm'],row['seed']))/'result.json')
            near(result['charged_gpu_wall_hours'],result['elapsed_seconds']/3600,'Inner process charge')
            require(charge+1e-9 >= result['charged_gpu_wall_hours'], 'Outer charge smaller than inner charge')
            total += charge
            costs.append(dict(arm=row['arm'],seed=row['seed'],charged_gpu_wall_hours=charge,
                              guardian_gpu_wall_hours=guardian['charged_gpu_wall_hours'],outer_process_seconds=row['elapsed_process_seconds']))
        near(queue['charged_gpu_hours_including_diagnostic'],total,'Final queue total')
        near(progress['charged_gpu_hours_including_diagnostic'],total,'Final progress total')
        require(total <= 2, 'Aggregate two-GPU-hour phase ceiling exceeded')
        return dict(charged_gpu_hours_including_diagnostic=total,diagnostic_gpu_hours=p['diagnostic_gpu_hours'],jobs=costs,
                    policy='Each matrix job=max(outer process elapsed/3600,guardian); diagnostic guardian charged once as frozen.')

    def run(self):
        self.section('provenance_and_references',self.provenance)
        for seed in (3,4,5):
            for arm in ARMS:
                self.section('%s_seed%d'%(arm,seed),lambda arm=arm,seed=seed:self.job(arm,seed))
        cost = self.section('matrix_schedules_initial_weights_and_cost',self.schedules_and_cost)
        require_count = len(self.jobs) == 18
        if not require_count:
            self.errors.append(dict(section='matrix_completeness',error='AuditFailure',detail='Only %d/18 jobs passed all saved-output checks'%len(self.jobs)))
        status = 'SAVED_OUTPUT_CHECKS_FAILED' if self.errors else ('PASS_WITH_EXPLICIT_UNVERIFIED_ARTIFACTS' if self.unavailable else 'ALL_SAVED_OUTPUT_CHECKS_PASS')
        return dict(schema='monti_alignment_agent_independent_stdlib_saved_audit_v1',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=status,
            auditor='Codex rubric_review agent; not human verification or instructor approval',
            source_commit=self.protocol['source_commit'],protocol_sha256=self.protocol_sha,
            audit_script_sha256=sha(__file__),completed_jobs_passing_checks=len(self.jobs),expected_jobs=18,
            groups_per_job=128,target_occurrences_per_job=1091,train_epochs=8,
            jobs=self.jobs,cost=cost,errors=self.errors,
            verified_artifact_count=len(self.verified),verified_artifacts=self.verified,
            unavailable_artifact_count=len(self.unavailable),unavailable_artifacts=self.unavailable,
            scope=dict(training_inference_API_calls=0,network_calls=0,project_imports=0,tensor_deserialization=0,
                       checked='Saved JSON/logit argmax/counts, transitions, frozen membership, counters, byte hashes of available artifacts, controller schedules and runtime charge'),
            limitations=[
                'Recorded physical counts are independently reconciled with saved budgets and input/output arc counters; tensor graph topology is not deserialized or reconstructed.',
                'Initial state content hashes are compared as captured before work; the underlying initial tensors are not independently rehashed by this stdlib audit.',
                'TRAIN budget consistency is checked across all epochs/arms; VALID budgets are also checked against the previous saved metadata table.',
                'TRAIN margin medians cannot be reconstructed from per-group means; saved state hashes and other available aggregates are checked.',
                'No-interim-validation and no-forbidden-query conclusions concern saved counters/artifacts and frozen inspected code, not external process tracing.',
                'The frozen diagnostic charge is its complete guardian lifetime; no absent outer diagnostic timing is invented.',
                'Repeated target occurrences are separate attack contexts, not independent nodes; these exploratory validation results are not fresh final-test evidence.',
                'Unoptimized anomaly signatures are not statistically independent of graph normalization and do not establish adaptive robustness.',
                'Unavailable artifact bytes remain explicitly unverified; a recorded SHA string is not a byte-level verification.' ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--matrix-root',required=True,help='Directory with the 18 job directories and queue receipts')
    parser.add_argument('--stage-root',required=True,help='Transferred stage holding references, requests and diagnostic guardian')
    parser.add_argument('--repo-root',required=True,help='Local canonical public source tree, read only')
    parser.add_argument('--file-map',help='Optional JSON mapping original repository-relative artifact names to transferred local files')
    parser.add_argument('--budget-metadata',help='Saved group_outcomes.csv; defaults to the public diagnosis table')
    parser.add_argument('--expected-source',default=SOURCE)
    parser.add_argument('--expected-protocol-sha256',default=PROTOCOL_SHA)
    parser.add_argument('--output',required=True,help='New exclusive private JSON receipt; never overwrites')
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    require(not output.exists(),'Preserve the previous audit receipt; choose a new output')
    auditor = Auditor(args)
    receipt = auditor.run()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=receipt['status'],jobs_passed=receipt['completed_jobs_passing_checks'],
                         errors=len(receipt['errors']),unavailable_artifacts=receipt['unavailable_artifact_count'],
                         output=str(output),sha256=sha(output))))
    # Incomplete byte coverage must not silently look like a complete pass.
    return 1 if receipt['errors'] else (2 if receipt['unavailable_artifact_count'] else 0)


if __name__ == '__main__':
    sys.exit(main())
