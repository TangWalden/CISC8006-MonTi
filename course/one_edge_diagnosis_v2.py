"""Finite CPU-only GCN endpoint counterfactuals; no training or victim access."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import pathlib
import signal
import statistics
import subprocess
import time
import traceback

import dgl
import numpy as np
import scipy
import torch

from modules.data_handler import DataHandler
from modules.experiment_handler import ExperimentHandler
from models.victim_model import GCN
from utils.utils import get_n_ids, set_seed

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def csv_write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


class SurrogateOnly(ExperimentHandler):
    def load_pretrained_models(self):
        assert self.surrogate_type == 'GCN'
        self.surrogate_model = self.load_pretrained_model('GCN', flag='surrogate')
        self.surrogate_model.to(self.device)
        self.surrogate_model.eval()
        self.sampler_surrogate = dgl.dataloading.MultiLayerFullNeighborSampler(self.surrogate_model.num_layers)
        assert self.victim_model is None

    def load_pretrained_model(self, model_name, flag='surrogate'):
        assert model_name == 'GCN' and flag == 'surrogate', 'Victim loading is prohibited.'
        setting = 'GCN-%s-%s' % (self.data.data_name, self.data.split)
        folder = ROOT / 'models/pretrained_models'
        config = json.loads((folder / 'configs' / (setting + '.json')).read_text())
        model = GCN(self.data.feat_dim, config['emb_sizes'], num_classes=self.data.num_classes)
        model.load_state_dict(torch.load(folder / 'surrogate_models' / (setting + '.pickle'), map_location='cpu'))
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model

    def get_logits(self, graph, n_ids, model='surrogate', training=False, sequences=None):
        assert model == 'surrogate' and not training
        return super().get_logits(graph, n_ids, model, training=False, sequences=None)

    def run(self):
        raise RuntimeError('Training is prohibited in this diagnostic.')

    train_epoch = run
    test = run
    check_performance = run
    check_clean_performance = run
    calculate_node_representations = run


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def attach(graph, feature, endpoint):
    total = graph.num_nodes()
    output = dgl.add_nodes(graph, 1, {'x': feature})
    output.add_edges([total, endpoint], [endpoint, total])
    output = dgl.add_self_loop(dgl.remove_self_loop(output))
    assert output.num_nodes() == total + 1
    assert output.num_edges() == graph.num_edges() + 3
    assert sorted(x for x in output.successors(total).tolist() if x != total) == [endpoint]
    assert torch.equal(output.ndata['x'][total:total+1], feature)
    return output


def equivalent_graph(left, right):
    assert torch.equal(left.ndata['x'], right.ndata['x'])
    ls, ld = left.edges(order='eid')
    rs, rd = right.edges(order='eid')
    ordered = torch.equal(ls, rs) and torch.equal(ld, rd)
    if not ordered:
        count = left.num_nodes()
        assert torch.equal(torch.sort(ls*count+ld).values, torch.sort(rs*count+rd).values)
    return ordered


def margins(value, labels):
    other = value.clone()
    other.scatter_(1, labels[:, None], float('-inf'))
    return value.gather(1, labels[:, None]).squeeze(1) - other.max(1).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evidence/one_edge_diagnosis_v2')
    parser.add_argument('--protocol', default='course/one_edge_diagnosis_protocol_v2.json')
    args = parser.parse_args()
    output = ROOT / args.output
    assert str(output.resolve()).startswith(str(ROOT.resolve()) + os.sep)
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / 'raw'
    raw_dir.mkdir()
    protocol_path = ROOT / args.protocol
    assert str(protocol_path.resolve()).startswith(str(ROOT.resolve()) + os.sep)
    protocol = json.loads(protocol_path.read_text())
    started = time.monotonic()
    calls = 0
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()
    write(output / 'registered.json', dict(source_commit=source, tracked_changes=dirty,
          registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
          protocol=args.protocol, protocol_sha256=sha(protocol_path), script_sha256=sha(pathlib.Path(__file__)),
          command=['env', 'CUDA_VISIBLE_DEVICES=', 'bash', 'course/run.sh', '-m', 'course.one_edge_diagnosis_v2'] + __import__('sys').argv[1:],
          scope='Reconstructed one-edge attacks and exhaustive fixed-feature target endpoints; GCN validation only',
          torch=torch.__version__, dgl=dgl.__version__, numpy=np.__version__, scipy=scipy.__version__,
          gpu_hours=0, device='CPU only'))

    def alarm(signum, frame):
        raise TimeoutError('Registered diagnostic wall-time limit reached.')

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(protocol['wall_seconds_limit'])
    input_hashes = {}
    rows, endpoint_rows = [], []
    try:
        assert not dirty and not torch.cuda.is_available()
        assert protocol['groups'] == sorted(set(protocol['groups'])) and len(protocol['groups']) == 16
        assert protocol['attacker_seeds'] == [0, 1, 2]
        torch.set_num_threads(8)
        frozen = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
        required = frozen['inputs'] + frozen['scientific_code'] + frozen['caches'] + protocol['source_files']
        for item in required:
            assert sha(ROOT / item['path']) == item['sha256'], item['path']
            input_hashes[item['path']] = item['sha256']
        for run in protocol['runs']:
            for key in ['config', 'checkpoint', 'source_result', 'source_prediction', 'source_prediction_summary']:
                item = run[key]
                assert '_test_' not in item['path']
                assert sha(ROOT / item['path']) == item['sha256'], item['path']
                input_hashes[item['path']] = item['sha256']
            metadata = json.loads((ROOT / run['source_result']['path']).read_text())
            assert metadata['status'] == 'completed' and metadata['best_epoch'] == run['best_epoch']
            assert metadata['checkpoint'] == run['checkpoint']['path'] and metadata['checkpoint_sha256'] == run['checkpoint']['sha256']
            folder = (ROOT / run['source_result']['path']).parent
            selected = sorted(folder.glob('*_surrogate_valid_attack.json'))[run['best_epoch']-1]
            assert selected.relative_to(ROOT).as_posix() == run['source_prediction_summary']['path']
        runtime = json.loads((ROOT / 'environment/runtime.json').read_text())
        for key, value in [('torch',torch.__version__),('dgl',dgl.__version__),('numpy',np.__version__),('scipy',scipy.__version__)]:
            assert value == runtime[key], (key,value,runtime[key])
        write(output / 'verified_inputs.json', input_hashes)
        config = json.loads((ROOT / protocol['runs'][0]['config']['path']).read_text())
        config['cuda_id'] = 'cpu'
        data = DataHandler(config)
        groups = protocol['groups']
        assert set(groups).issubset(set(data.target_idx_valid.tolist()))
        assert not set(groups).intersection(data.target_idx_train.tolist() + data.target_idx_test.tolist())
        assert all(int(data.node_budgets[g]) == int(data.edge_budgets[g]) == 1 for g in groups)
        assert sum(len(get_n_ids(data.target_assignment, g)) for g in groups) == protocol['target_occurrences_per_seed'] == 81
        clean = {}
        tolerance = protocol['margin_change_tolerance']
        raw_path = raw_dir / 'predictions.jsonl'
        progress_path = raw_dir / 'progress.jsonl'
        with raw_path.open('x') as raw, progress_path.open('x') as progress, torch.no_grad():
            def progress_record(value):
                progress.write(json.dumps(value) + '\n')
                progress.flush()

            for run in protocol['runs']:
                config = json.loads((ROOT / run['config']['path']).read_text())
                assert config['ada_budget'] and not config['random_candidates'] and config['surrogate_model'] == 'GCN'
                config['cuda_id'] = 'cpu'
                experiment = SurrogateOnly(config, data)
                experiment.model.load_state_dict(torch.load(ROOT / run['checkpoint']['path'], map_location='cpu'))
                for parameter in experiment.model.parameters():
                    parameter.requires_grad = False
                experiment.model.eval()
                experiment.surrogate_model.eval()
                assert experiment.victim_model is None and experiment.model.ada_budget
                assert all(not parameter.requires_grad for parameter in experiment.surrogate_model.parameters())
                source_predictions = {}
                for line in (ROOT / run['source_prediction']['path']).read_text().splitlines():
                    record = json.loads(line)
                    if record['target_group'] in groups:
                        source_predictions[record['target_group']] = record
                assert set(source_predictions) == set(groups)
                source_summary = json.loads((ROOT / run['source_prediction_summary']['path']).read_text())
                assert source_summary['prediction_sha256'] == run['source_prediction']['sha256']

                def logits(graph, n_ids, phase, group, position=None):
                    nonlocal calls
                    assert calls < protocol['surrogate_forward_limit']
                    assert time.monotonic() - started < protocol['wall_seconds_limit']
                    calls += 1
                    context = dict(call=calls,phase=phase,target_group=group,attacker_seed=run['seed'],endpoint_position=position)
                    progress_record(dict(event='forward_started',**context))
                    value = experiment.get_logits(graph, n_ids, training=False).detach().cpu()
                    assert torch.isfinite(value).all()
                    progress_record(dict(event='forward_completed',logits=value.tolist(),**context))
                    return value

                def metrics(value, labels, clean_value):
                    predicted = value.argmax(1)
                    correct = predicted.eq(labels)
                    clean_correct = clean_value.argmax(1).eq(labels)
                    other = value.clone()
                    other.scatter_(1, labels[:, None], float('-inf'))
                    margin = value.gather(1, labels[:, None]).squeeze(1) - other.max(1).values
                    clean_other = clean_value.clone()
                    clean_other.scatter_(1, labels[:, None], float('-inf'))
                    clean_margin = clean_value.gather(1, labels[:, None]).squeeze(1) - clean_other.max(1).values
                    return dict(residuals=int(correct.sum()), incorrect=int((~correct).sum()),
                                clean_correct_denominator=int(clean_correct.sum()),
                                clean_correct_to_wrong=int((clean_correct & ~correct).sum()),
                                clean_wrong_to_correct=int((~clean_correct & correct).sum()),
                                mean_correct_margin=float(margin.mean()), min_correct_margin=float(margin.min()),
                                max_correct_margin=float(margin.max()),
                                targets_margin_changed_above_tolerance=int(((margin-clean_margin).abs()>tolerance).sum()))

                for group in groups:
                    n_ids = torch.tensor(get_n_ids(data.target_assignment, group), dtype=torch.long)
                    labels = data.labels[n_ids].cpu()
                    saved = source_predictions[group]
                    assert saved['node_ids'] == n_ids.tolist() and saved['labels'] == labels.tolist()
                    assert saved['incorrect'] == sum(a!=b for a,b in zip(saved['labels'], saved['predictions']))
                    if group not in clean:
                        progress_record(dict(event='group_identity',target_group=group,node_ids=n_ids.tolist(),labels=labels.tolist()))
                        clean[group] = logits(data.graph, n_ids, 'clean', group)
                    clean_logits = clean[group]
                    set_seed(protocol['reconstruction_seed'])
                    perturbed = experiment.get_perturbed_graph(n_ids, 1, 1, training=False, need_sequence=False)
                    assert perturbed is not None
                    total = data.graph.num_nodes()
                    assert perturbed.num_nodes() == total + 1 and perturbed.num_edges() == data.graph.num_edges() + 3
                    attached = [x for x in perturbed.successors(total).tolist() if x != total]
                    assert len(attached) == 1 and attached[0] in n_ids.tolist()
                    endpoint = attached[0]
                    endpoint_position = n_ids.tolist().index(endpoint)
                    feature = perturbed.ndata['x'][total:total+1].detach().clone()
                    progress_record(dict(event='reconstruction_generated',target_group=group,attacker_seed=run['seed'],
                                         feature=feature.tolist(),endpoint_node_id=endpoint,endpoint_position=endpoint_position))
                    baseline_logits = logits(perturbed, n_ids, 'reconstruction', group)
                    baseline = metrics(baseline_logits, labels, clean_logits)
                    set_seed(protocol['repeat_seed'])
                    repeat_graph = experiment.get_perturbed_graph(n_ids, 1, 1, training=False, need_sequence=False)
                    assert repeat_graph is not None
                    repeat_logits = logits(repeat_graph, n_ids, 'repeat', group)
                    repeat_feature = repeat_graph.ndata['x'][total:total+1]
                    repeat_endpoint = [x for x in repeat_graph.successors(total).tolist() if x != total]
                    endpoints = []
                    anchor_logit_difference = anchor_margin_difference = None
                    anchor_edge_order_match = None
                    for position, candidate in enumerate(n_ids.tolist()):
                        counter_graph = attach(data.graph, feature, candidate)
                        if candidate == endpoint:
                            anchor_edge_order_match = equivalent_graph(counter_graph, perturbed)
                        value = logits(counter_graph, n_ids, 'endpoint', group, position)
                        measured = metrics(value, labels, clean_logits)
                        if candidate == endpoint:
                            anchor_logit_difference = float((value-baseline_logits).abs().max())
                            anchor_margin_difference = float((margins(value,labels)-margins(baseline_logits,labels)).abs().max())
                            assert value.argmax(1).tolist() == baseline_logits.argmax(1).tolist()
                        endpoint_rows.append(dict(target_group=group, attacker_seed=run['seed'],
                              endpoint_position=position, reconstructed_selected_endpoint=position==endpoint_position,
                              group_size=len(n_ids), **measured))
                        endpoints.append(dict(endpoint_position=position, logits=value.tolist(), **measured))
                    residuals = [x['residuals'] for x in endpoints]
                    best = min(endpoints, key=lambda x: (x['residuals'], x['endpoint_position']))
                    assert min(residuals) <= baseline['residuals']
                    best_logits = torch.tensor(best['logits'],dtype=baseline_logits.dtype)
                    gained = baseline_logits.argmax(1).eq(labels) & best_logits.argmax(1).ne(labels)
                    lost = baseline_logits.argmax(1).ne(labels) & best_logits.argmax(1).eq(labels)
                    assert int(gained.sum()-lost.sum()) == baseline['residuals']-min(residuals)
                    baseline_margins, best_margins = margins(baseline_logits,labels), margins(best_logits,labels)
                    row = dict(target_group=group, attacker_seed=run['seed'], group_size=len(n_ids),
                          node_budget=1, edge_budget=1, clean_residuals=int(clean_logits.argmax(1).eq(labels).sum()),
                          saved_original_residuals=len(n_ids)-saved['incorrect'],
                          reconstructed_residuals=baseline['residuals'],
                          source_prediction_match=baseline_logits.argmax(1).tolist()==saved['predictions'],
                          source_prediction_disagreements=sum(a!=b for a,b in zip(baseline_logits.argmax(1).tolist(),saved['predictions'])),
                          reconstructed_endpoint_position=endpoint_position, legal_endpoints=len(endpoints),
                          best_residuals=min(residuals), median_residuals=statistics.median(residuals), worst_residuals=max(residuals),
                          best_clean_correct_to_wrong=max(x['clean_correct_to_wrong'] for x in endpoints),
                          worst_clean_correct_to_wrong=min(x['clean_correct_to_wrong'] for x in endpoints),
                          restricted_oracle_clean_headroom=max(x['clean_correct_to_wrong'] for x in endpoints)-baseline['clean_correct_to_wrong'],
                          strictly_better_endpoints=sum(x<baseline['residuals'] for x in residuals),
                          oracle_additional_incorrect=baseline['residuals']-min(residuals),
                          oracle_gained_occurrences=int(gained.sum()),oracle_lost_occurrences=int(lost.sum()),
                          gained_min_signed_margin_before=float(baseline_margins[gained].min()) if gained.any() else None,
                          gained_max_signed_margin_after=float(best_margins[gained].max()) if gained.any() else None,
                          best_endpoint_position=best['endpoint_position'],
                          clean_correct_to_wrong=baseline['clean_correct_to_wrong'],
                          clean_wrong_to_correct=baseline['clean_wrong_to_correct'],
                          mean_correct_margin=baseline['mean_correct_margin'],
                          targets_margin_changed_above_tolerance=baseline['targets_margin_changed_above_tolerance'],
                          targets_zero_direct_edges=len(n_ids)-1,
                          repeat_prediction_match=repeat_logits.argmax(1).tolist()==baseline_logits.argmax(1).tolist(),
                          repeat_endpoint_match=repeat_endpoint==[endpoint],
                          repeat_feature_exact_match=torch.equal(feature,repeat_feature),
                          repeat_max_logit_difference=float((repeat_logits-baseline_logits).abs().max()),
                          reattachment_max_logit_difference=anchor_logit_difference,
                          reattachment_max_margin_difference=anchor_margin_difference,
                          reattachment_edge_order_match=anchor_edge_order_match,
                          reconstructed_feature_sha256=tensor_sha(feature))
                    rows.append(row)
                    raw.write(json.dumps(dict(target_group=group,attacker_seed=run['seed'],node_ids=n_ids.tolist(),
                              labels=labels.tolist(),clean_logits=clean_logits.tolist(),source_saved_predictions=saved['predictions'],
                              reconstructed_logits=baseline_logits.tolist(),feature=feature.tolist(),endpoint_node_id=endpoint,
                              repeat_logits=repeat_logits.tolist(),counterfactuals=endpoints))+'\n')
                    raw.flush()
                    print('GROUP', run['seed'], group, 'residuals', baseline['residuals'], 'oracle_best', min(residuals), flush=True)
        assert len(rows) == 48 and len(endpoint_rows) == 243 and calls == 355
        csv_write(output / 'groups.csv', rows)
        csv_write(output / 'endpoints.csv', endpoint_rows)
        per_seed = []
        for seed in protocol['attacker_seeds']:
            chosen = [row for row in rows if row['attacker_seed'] == seed]
            per_seed.append(dict(attacker_seed=seed,groups=len(chosen),target_occurrences=sum(row['group_size'] for row in chosen),
                clean_residuals=sum(row['clean_residuals'] for row in chosen),
                saved_original_residuals=sum(row['saved_original_residuals'] for row in chosen),
                reconstructed_residuals=sum(row['reconstructed_residuals'] for row in chosen),
                restricted_oracle_residuals=sum(row['best_residuals'] for row in chosen),
                worst_endpoint_residuals=sum(row['worst_residuals'] for row in chosen),
                oracle_additional_incorrect=sum(row['oracle_additional_incorrect'] for row in chosen),
                conditional_oracle_additional_clean_flips=sum(row['restricted_oracle_clean_headroom'] for row in chosen),
                groups_with_strict_endpoint_opportunity=sum(row['strictly_better_endpoints']>0 for row in chosen),
                source_prediction_concordant_groups=sum(row['source_prediction_match'] for row in chosen),
                clean_correct_to_wrong=sum(row['clean_correct_to_wrong'] for row in chosen),
                clean_wrong_to_correct=sum(row['clean_wrong_to_correct'] for row in chosen)))
        summary = dict(status='completed',scope='Selected validation cohort; reconstructed attacks, not original perturbation identity',
             groups=16,attacker_checkpoints=3,group_checkpoint_cases=len(rows),endpoint_configurations=len(endpoint_rows),
             target_occurrences=243,surrogate_forward_calls=calls,per_seed=per_seed,
             strict_endpoint_opportunity_cases=sum(row['strictly_better_endpoints']>0 for row in rows),
             source_prediction_concordant_cases=sum(row['source_prediction_match'] for row in rows),
             repeat_prediction_concordant_cases=sum(row['repeat_prediction_match'] for row in rows),
             repeat_feature_exact_cases=sum(row['repeat_feature_exact_match'] for row in rows),
             repeat_endpoint_concordant_cases=sum(row['repeat_endpoint_match'] for row in rows),
             largest_reattachment_logit_difference=max(row['reattachment_max_logit_difference'] for row in rows),
             largest_reattachment_margin_difference=max(row['reattachment_max_margin_difference'] for row in rows),
             numerical_caution='Nominal 1e-5 margin-change flags may be sensitive to observed numerical discrepancies. Observed maxima are empirical references, not guaranteed error bounds; primary comparisons require identical-graph prediction equality.',
             mean_restricted_oracle_headroom_pp=100*sum(row['oracle_additional_incorrect'] for row in rows)/243,
             raw_prediction_sha256=sha(raw_path),raw_progress_sha256=sha(progress_path),
             cpu_elapsed_seconds=time.monotonic()-started,gpu_hours=0,
             margin_change_tolerance=tolerance,
             interpretation='Restricted endpoint oracle is optimistic by construction, not a trained method or held-out gain. Zero headroom does not rule out different features. No victim or test queries.')
        write(output / 'summary.json', summary)
        write(output / 'output_checksums.json', {name:sha(output/name) for name in ['registered.json','verified_inputs.json','groups.csv','endpoints.csv','summary.json']})
        print(json.dumps(summary,indent=2),flush=True)
    except Exception:
        write(output / 'failure.json', dict(status='failed',traceback=traceback.format_exc(),surrogate_forward_calls=calls,
              cpu_elapsed_seconds=time.monotonic()-started,gpu_hours=0))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
