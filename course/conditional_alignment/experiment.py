"""GCN-only MonTi adapter. Original scientific source is never edited."""
import json
import pathlib
import time

import dgl
import numpy as np
import torch

from modules.experiment_handler import ExperimentHandler
from utils.utils import get_n_ids
from course.conditional_alignment.common import sha256, write_json
from course.conditional_alignment.reference import alignment_penalty


class AlignmentExperiment(ExperimentHandler):
    def __init__(self, config, data, manager, *, reference, protocol, arm, evidence,
                 deadline, source_commit):
        self.reference, self.protocol, self.arm = reference, protocol, arm
        self.evidence = pathlib.Path(evidence)
        self.deadline, self.source_commit = deadline, source_commit
        self.current_epoch = 0
        self.gcn_forwards = self.generator_forwards = self.optimizer_steps = 0
        self._pending = None
        self._capture = None
        self.export_index = []
        self._train_alignment = []
        super().__init__(config, data, manager)
        original_step = self.optimizer.step
        def counted_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.optimizer_steps += 1
            return result
        self.optimizer.step = counted_step
        self._observer = self.model.register_forward_hook(self._observe_attack)

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Registered per-job wall-clock deadline reached.')

    def load_pretrained_models(self):
        # Do not call the superclass method: it would also load a victim model.
        if self.surrogate_type != 'GCN' or self.victim_type != 'GCN':
            raise RuntimeError('Alignment pilot only permits the GCN surrogate.')
        self.surrogate_model = self.load_pretrained_model('GCN', flag='surrogate').to(self.device)
        self.sampler_surrogate = dgl.dataloading.MultiLayerFullNeighborSampler(
            self.surrogate_model.num_layers)
        self.victim_model = self.sampler_victim = None

    def load_pretrained_model(self, model_name, flag='surrogate'):
        if model_name != 'GCN' or flag != 'surrogate':
            raise RuntimeError('Victim/non-GCN checkpoint loading is prohibited.')
        return super().load_pretrained_model(model_name, flag=flag)

    def get_node_representations(self):
        # Refuse cache regeneration, which would add unregistered model calls.
        path = pathlib.Path('Datasets/GossipCop/h_GCN.npy')
        if not path.is_file():
            raise RuntimeError('The frozen original GCN representation cache is required.')
        return super().get_node_representations()

    def _observe_attack(self, model, inputs, output):
        graph, targets = inputs[:2]
        features, adjacency, candidates = output
        if features is None or adjacency is None:
            raise RuntimeError('Attack generation returned an empty output.')
        self._capture = dict(x_atk=features.detach(), adj_attack=adjacency.detach(),
                             target_ids=graph.ndata[dgl.NID][targets].detach(),
                             candidate_ids=(torch.empty(0, dtype=torch.long, device=graph.device)
                                            if candidates is None else
                                            graph.ndata[dgl.NID][candidates].detach()))
        # A forward hook returning None preserves the upstream outputs.

    def get_perturbed_graph(self, n_ids, node_budget, edge_budget,
                            training=False, need_sequence=False):
        self.check_deadline()
        if need_sequence:
            raise RuntimeError('GAGA sequence construction is prohibited in this pilot.')
        if self._pending is not None:
            raise RuntimeError('Previous alignment penalty was not consumed.')
        self.generator_forwards += 1
        graph = super().get_perturbed_graph(n_ids, node_budget, edge_budget,
                                             training=training, need_sequence=False)
        if graph is None:
            raise RuntimeError('A failed attack must not silently remove a group.')
        original = self.data.graph.num_nodes()
        if graph.num_nodes() != original + int(node_budget):
            raise RuntimeError('Injected node budget mismatch.')
        added_arcs = graph.num_edges()-self.data.graph.num_edges()-int(node_budget)
        if added_arcs < 0 or added_arcs % 2 or added_arcs//2 != int(edge_budget):
            raise RuntimeError('Actual undirected injected edge budget mismatch.')
        expected_targets = torch.as_tensor(n_ids, device=graph.device)
        if self._capture is None or not torch.equal(self._capture['target_ids'], expected_targets):
            raise RuntimeError('Observed target sequence differs from the requested group.')
        injected = torch.arange(original, graph.num_nodes(), device=graph.device)
        penalty, stats = alignment_penalty(
            graph, injected, self.reference, arm=self.arm,
            coefficient=float(self.protocol['alignment_lambda']),
            chunk_size=int(self.protocol['reference'].get('chunk_size', 4096)), training=training)
        stats.update(node_budget=int(node_budget), edge_budget=int(edge_budget),
                     actual_undirected_injected_edges=added_arcs//2)
        self._pending = (id(graph), penalty, stats)
        return graph

    def get_logits(self, graph, n_ids, model='surrogate', training=True, sequences=None):
        self.check_deadline()
        if model != 'surrogate' or sequences is not None:
            raise RuntimeError('Victim/non-GCN logits are prohibited.')
        requested = torch.as_tensor(n_ids, dtype=graph.idtype, device=graph.device)
        _, output_nodes, blocks = self.sampler_surrogate.sample_blocks(graph, requested)
        if not torch.equal(output_nodes, requested) or not torch.equal(
                blocks[-1].dstdata[dgl.NID], requested):
            raise RuntimeError('GCN sampler changed requested target order.')
        blocks = [block.to(self.device) for block in blocks]
        self.gcn_forwards += 1
        return self.surrogate_model(blocks, edge_weight=training)

    def consume_alignment(self, graph):
        if self._pending is None or self._pending[0] != id(graph):
            raise RuntimeError('Alignment context does not match the generated graph.')
        _, penalty, stats = self._pending
        self._pending = None
        return penalty, stats

    def loss_function(self, graph_perturbed, n_ids, training=True, sequences=None):
        base = super().loss_function(graph_perturbed, n_ids, training=training, sequences=sequences)
        penalty, stats = self.consume_alignment(graph_perturbed)
        stats.update(base_attack_loss=float(base.detach().item()),
                     total_loss=float((base+penalty).detach().item()))
        if training:
            self._train_alignment.append(stats)
        return base + penalty

    def train_epoch(self):
        self.check_deadline()
        self._train_alignment = []
        before_steps = self.optimizer_steps
        super().train_epoch()
        steps = self.optimizer_steps-before_steps
        rows = self._train_alignment
        summary = dict(epoch=self.current_epoch, groups=len(rows), optimizer_steps=steps,
                       mean_base_attack_loss=float(np.mean([x['base_attack_loss'] for x in rows])),
                       mean_total_loss=float(np.mean([x['total_loss'] for x in rows])),
                       mean_alignment_distance=float(np.mean([x['mean_distance'] for x in rows])),
                       active_nodes=sum(x['active_nodes'] for x in rows),
                       injected_nodes=sum(x['injected_nodes'] for x in rows),
                       fallback_nodes=sum(x['fallback_nodes'] for x in rows))
        write_json(self.evidence/('train_epoch_%03d.json' % self.current_epoch), summary, exclusive=True)
        print('TRAIN_ALIGNMENT', json.dumps(summary), flush=True)

    def _export_attack(self, target_group):
        if int(target_group) not in self.protocol.get('export_valid_groups', []):
            return
        path = self.evidence/'raw'/('attack_epoch_%03d_group_%d.pt' %
                                   (self.current_epoch, int(target_group)))
        payload = {key: value.detach().cpu() for key, value in self._capture.items()}
        payload.update(target_group=int(target_group), epoch=self.current_epoch,
                       arm=self.arm, source_commit=self.source_commit,
                       node_budget=int(payload['x_atk'].shape[0]),
                       edge_budget=int(payload['adj_attack'].count_nonzero().item()),
                       original_num_nodes=self.data.graph.num_nodes(),
                       interpretation='Raw validation perturbation; keep private; no victim/test evaluation.')
        torch.save(payload, path)
        self.export_index.append(dict(epoch=self.current_epoch, target_group=int(target_group),
                                      path=str(path.relative_to(self.evidence)), sha256=sha256(path)))

    def check_performance(self, model='surrogate', set_type='valid', attack=True):
        if model != 'surrogate' or set_type != 'valid' or not attack:
            raise RuntimeError('Only attacked GCN validation evaluation is permitted.')
        self.check_deadline()
        self.model.eval()
        stem = 'validation_epoch_%03d' % self.current_epoch
        raw_path = self.evidence/'raw'/(stem+'.jsonl')
        wrong = exposures = 0
        public_rows = []
        with raw_path.open('x', encoding='utf-8', newline='\n') as stream, torch.no_grad():
            for target_group in self.data.target_idx_valid:
                self.check_deadline()
                ids = torch.tensor(get_n_ids(self.data.target_assignment, target_group), dtype=torch.long)
                nb = int(self.data.node_budgets[target_group])
                eb = int(self.data.edge_budgets[target_group])
                graph = self.get_perturbed_graph(ids, nb, eb, training=False)
                logits = self.get_logits(graph, ids, training=False)
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Nonfinite GCN validation logits.')
                labels = graph.ndata['y'][ids.to(graph.device)].detach().cpu()
                prediction = logits.argmax(dim=1).detach().cpu()
                if logits.shape[0] != len(ids):
                    raise RuntimeError('Validation output length mismatch.')
                errors = int((labels != prediction).sum().item())
                base = torch.relu(logits[:, 1]-logits[:, 0]).mean()
                penalty, stats = self.consume_alignment(graph)
                row = dict(target_group=int(target_group), target_occurrences=len(ids),
                           incorrect=errors, gcn_correct_after_attack=len(ids)-errors,
                           base_attack_loss=float(base.item()), total_loss=float((base+penalty).item()),
                           **stats)
                public_rows.append(row)
                private = dict(row, node_ids=ids.tolist(), labels=labels.tolist(),
                               predictions=prediction.tolist(), logits=logits.detach().cpu().tolist())
                stream.write(json.dumps(private, allow_nan=False)+'\n')
                stream.flush()
                self._export_attack(target_group)
                wrong += errors
                exposures += len(ids)
                self._capture = None
        if len(public_rows) != len(self.data.target_idx_valid) or not exposures:
            raise RuntimeError('Incomplete validation evaluation.')
        write_json(self.evidence/(stem+'_groups.json'), public_rows, exclusive=True)
        summary = dict(epoch=self.current_epoch, groups=len(public_rows), target_occurrences=exposures,
                       incorrect=wrong, mr_percent=100.0*wrong/exposures,
                       mean_alignment_distance=float(np.mean([x['mean_distance'] for x in public_rows])),
                       active_nodes=sum(x['active_nodes'] for x in public_rows),
                       injected_nodes=sum(x['injected_nodes'] for x in public_rows),
                       fallback_nodes=sum(x['fallback_nodes'] for x in public_rows),
                       raw_prediction_sha256=sha256(raw_path),
                       weighting='target occurrences; overlapping groups are separate attack scenarios')
        write_json(self.evidence/(stem+'.json'), summary, exclusive=True)
        print('VALIDATION', json.dumps(summary), flush=True)
        return dict(accuracy=1.0-wrong/exposures, misclf_rate=wrong/exposures,
                    loss=float(np.mean([x['total_loss'] for x in public_rows])))

    def test(self):
        raise RuntimeError('This adapter has no test or victim evaluation entry point.')

    def run(self):
        self.result_manager.start_train(1)
        for epoch in range(1, int(self.config['epochs'])+1):
            self.current_epoch = epoch
            self.train_epoch()
            stop = self.validation(epoch)
            write_json(self.evidence/'progress.json', dict(
                completed_epochs=epoch, best_epoch=self.es_scheduler.b_epoch,
                best_validation=self.es_scheduler.b_results,
                gcn_forwards=self.gcn_forwards, generator_forwards=self.generator_forwards,
                optimizer_steps=self.optimizer_steps))
            if stop:
                break
        self.result_manager.end_train()
        write_json(self.evidence/'attack_exports.json', self.export_index, exclusive=True)
        return dict(test_skipped=True, victim_loaded=False, gaga_forwards=0,
                    gcn_forwards=self.gcn_forwards, generator_forwards=self.generator_forwards,
                    optimizer_steps=self.optimizer_steps)
