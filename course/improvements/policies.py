import types

import dgl
import numpy as np
import torch

from models.attack_model import MonTi
from utils.utils import get_complment_indices


def coverage_anchors(graph, targets, pool, limit=32):
    """Greedy direct-target coverage, with global node ID as a deterministic tie break."""
    target_list = targets.detach().cpu().tolist()
    target_bits = {node: 1 << i for i, node in enumerate(target_list)}
    candidates = pool.detach().cpu().tolist()
    covers = {node: 0 for node in candidates}
    sources, destinations = graph.out_edges(targets.to(graph.device))
    for source, destination in zip(sources.cpu().tolist(), destinations.cpu().tolist()):
        if destination in covers:
            covers[destination] |= target_bits[source]
    ids = graph.ndata[dgl.NID].cpu().tolist() if dgl.NID in graph.ndata else list(range(graph.num_nodes()))
    available = sorted((node for node in candidates if covers[node]), key=lambda node: ids[node])
    uncovered = (1 << len(target_list))-1
    selected = []
    for _ in range(min(limit, len(available))):
        best = max(available, key=lambda node: bin(covers[node] & uncovered).count('1'))
        if not covers[best] & uncovered:
            break
        selected.append(best)
        uncovered &= ~covers[best]
        available.remove(best)
        if not uncovered:
            break
    return torch.tensor(selected, dtype=torch.long, device=graph.device)


def _select_with_coverage(self, graph, targets, cand_selection=True):
    pool = get_complment_indices(graph.num_nodes(), targets).to(graph.device)
    if not cand_selection or len(pool) <= self.max_candidates:
        return MonTi.select_candidates(self, graph, targets, cand_selection)
    anchors = coverage_anchors(graph, targets, pool, min(32, self.max_candidates))
    if self.coverage_policy == 'random_anchors' and len(anchors):
        # Match the number of structural anchors for this exact group, not always 32.
        anchors = pool[torch.tensor(np.random.choice(len(pool), len(anchors), replace=False),
                                    dtype=torch.long, device=graph.device)]
    remaining_mask = torch.ones(graph.num_nodes(), dtype=torch.bool, device=graph.device)
    remaining_mask[targets.to(graph.device)] = False
    remaining_mask[anchors] = False
    remaining = remaining_mask.nonzero(as_tuple=True)[0]
    slots = self.max_candidates-len(anchors)
    if slots:
        original_k = self.candidate_selector.max_candidates
        self.candidate_selector.max_candidates = slots
        try:
            gates, chosen = self.candidate_selector(graph, targets, remaining,
                self.temp_scheduler.get_temperature(), self.eps_scheduler.get_epsilon())
        finally:
            self.candidate_selector.max_candidates = original_k
    else:
        gates = torch.empty(0, device=graph.device)
        chosen = torch.empty(0, dtype=torch.long, device=graph.device)
    return torch.cat([torch.ones(len(anchors), dtype=gates.dtype, device=graph.device), gates]), torch.cat([anchors, chosen])


def attach_coverage(model, policy):
    if policy == 'baseline':
        return model
    if policy not in ('coverage', 'random_anchors'):
        raise ValueError('Unknown candidate policy: ' + policy)
    model.coverage_policy = policy
    model.select_candidates = types.MethodType(_select_with_coverage, model)
    return model


def group_sampling_weights(sizes, mode):
    sizes = torch.as_tensor(sizes, dtype=torch.double)
    if sizes.ndim != 1 or len(sizes) == 0 or (sizes <= 0).any():
        raise ValueError('Every training group must have positive size.')
    if mode == 'size_sampling':
        return sizes / sizes.sum()
    if mode == 'uniform_replacement':
        return torch.full_like(sizes, 1/len(sizes))
    raise ValueError('Unknown sampling policy: ' + mode)
