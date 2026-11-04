"""Change only the representation gathered as an injected edge-score source.

The legacy mode calls the frozen implementation directly. The corrected mode
retains its softmax, randomness, target-first allocation and lower-triangular
edge space. In particular, later injected nodes may connect to earlier injected
nodes under that original edge space; this module does not change that policy.
"""
import torch

from models.attack_model import MonTi
from utils.utils import gumbel_softmax, st_top_k


SOURCE_MODES = ('legacy', 'source_fixed')


class SourceRoleMonTi(MonTi):
    def __init__(self, *args, source_mode='legacy', **kwargs):
        if source_mode not in SOURCE_MODES:
            raise ValueError('Unknown edge-score source mode.')
        # A plain string adds no parameter, state-dict key or RNG consumption.
        self.source_mode = source_mode
        super().__init__(*args, **kwargs)

    def edge_generation(self, z, subgraph, num_targets, num_candidates,
                        node_budget, edge_budget, random_edge=False):
        if self.source_mode == 'legacy':
            return super().edge_generation(z, subgraph, num_targets, num_candidates,
                                           node_budget, edge_budget, random_edge)

        adj_attack_shape = (node_budget, z.size(0))
        adj_attack = torch.zeros(adj_attack_shape, device=subgraph.device)
        mask_tril = torch.tril(torch.ones(adj_attack_shape, dtype=torch.bool),
                               diagonal=num_targets + num_candidates - 1).to(subgraph.device)

        if random_edge:
            rand = torch.rand_like(adj_attack)
            _, indices = rand[mask_tril].topk(edge_budget)
            src, dst = mask_tril.nonzero(as_tuple=True)
            adj_attack[src[indices], dst[indices]] = 1
            return adj_attack

        e = self.linear_edge(z)
        e = torch.nn.functional.normalize(e, dim=-1)
        src, dst = mask_tril.nonzero(as_tuple=True)
        # The sole scientific change: src is injection-relative, whereas e is
        # ordered [targets, candidates, injected]. Physical graph IDs do not
        # supply this representation offset elsewhere in the scorer.
        edge_scores = torch.sum(e[num_targets + num_candidates + src] * e[dst], dim=-1)

        temperature = self.temp_scheduler.get_temperature()
        epsilon = self.eps_scheduler.get_epsilon()
        edge_scores = gumbel_softmax(edge_scores, temperature, epsilon, dim=0,
                                     randomness=self.training)

        edge_scores_mat = torch.zeros(adj_attack_shape, device=subgraph.device)
        edge_scores_mat[src, dst] = edge_scores
        if self.ada_budget:
            adj_attack[:, :num_targets], _ = st_top_k(edge_scores_mat[:, :num_targets], 1, dim=-1)
            edge_budget -= node_budget
            if edge_budget > 0:
                mask_empty = (adj_attack == 0)
                mask = mask_empty & mask_tril
                edge_budget = min(edge_budget, mask.sum())
                adj_attack_remaining, _ = st_top_k(edge_scores_mat[mask], edge_budget, dim=0)
                adj_attack[mask] = adj_attack_remaining
        else:
            assert edge_budget % node_budget == 0
            degree_budget = edge_budget // node_budget
            adj_attack, _ = st_top_k(edge_scores_mat, degree_budget, dim=-1)
        return adj_attack
