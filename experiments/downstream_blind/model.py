"""Matched scorer without realized upstream target-attachment information.

Only the conditioning summary differs from the visible scorer. Auxiliary
features, the target list, candidate inputs, parameter shapes and scoring
operations are retained. The blind interface accepts a registered integer
quota, never an actual attachment matrix. Upstream auxiliary features may still
contain information correlated with upstream actions; this is not a claim of
removing all indirect information about those actions.
"""

import torch

from experiments.downstream_cooperation.model import DownstreamCandidateScorer


class ActionBlindCandidateScorer(DownstreamCandidateScorer):
    """Replace attachment mean/count with whole-list mean and quota/B."""

    def conditioning_inputs(self, x_aux, x_targets, target_quota):
        if (x_aux.ndim != 2 or x_targets.ndim != 2
                or x_aux.shape[0] < 1 or x_targets.shape[0] < 1
                or x_aux.shape[1] != self.feature_dim
                or x_targets.shape[1] != self.feature_dim):
            raise ValueError('Upstream feature shapes differ.')
        if not x_aux.is_floating_point() or not x_targets.is_floating_point():
            raise ValueError('Raw upstream features must be floating point.')
        if x_aux.device != x_targets.device or x_aux.dtype != x_targets.dtype:
            raise ValueError('Raw upstream features must share dtype and device.')
        b, t = len(x_aux), len(x_targets)
        if type(target_quota) is not int or not b <= target_quota <= b * t:
            raise ValueError('Registered integer target quota must satisfy B <= quota <= B*T.')
        auxiliary, targets = x_aux.detach(), x_targets.detach()
        if not bool(torch.isfinite(auxiliary).all() and torch.isfinite(targets).all()):
            raise ValueError('Upstream features must be finite.')
        # This neutral stencil contains no realized upstream decision. Mirror
        # the visible branch's matrix multiplication and rowwise division so
        # a full target attachment has bit-identical float32 conditioning.
        neutral = torch.ones((b, t), dtype=auxiliary.dtype, device=auxiliary.device)
        neutral_counts = neutral.sum(dim=1, keepdim=True)
        mean = neutral.matmul(targets) / neutral_counts
        expected_counts = torch.full((b, 1), target_quota / b,
                                     dtype=auxiliary.dtype, device=auxiliary.device)
        return auxiliary, mean, torch.log1p(expected_counts)

    def forward(self, x_aux, x_targets, target_quota, x_candidates, h_candidates):
        # The parent's forward delegates conditioning to this override; its
        # candidate branches, parameter names and ranking scores stay unchanged.
        return super().forward(x_aux, x_targets, target_quota, x_candidates, h_candidates)
