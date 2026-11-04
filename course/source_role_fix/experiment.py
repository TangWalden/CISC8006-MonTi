"""Reuse the frozen GCN-only training harness with one isolated source choice."""
from course.dual_surrogate.experiment import DualExperiment
from course.source_role_fix.model import SOURCE_MODES, SourceRoleMonTi


class SourceExperiment(DualExperiment):
    def __init__(self, config, data, protocol, arm, teacher_directory, output, deadline):
        if arm not in SOURCE_MODES:
            raise ValueError('Only legacy and source_fixed are registered.')
        self.source_arm = arm
        # This keeps ROLE_MAP, GCN feedback, RNG keys, eight terminal epochs,
        # scheduler, optimizer, h_GCN and physical-budget checks unchanged.
        # No new teacher is loaded by the inherited gcn_only branch.
        super().__init__(config, data, protocol, 'gcn_only', teacher_directory,
                         output, deadline)

    def initialize_attack_model(self):
        # Match ExperimentHandler.initialize_attack_model constructor order and
        # arguments. Construct once: replacing an already initialized model
        # would consume another random initialization and invalidate pairing.
        features = self.data.features
        feat_dim = features.shape[1]
        min_feat, _ = features.min(dim=0)
        max_feat, _ = features.max(dim=0)
        feat_budget = max(int(features.sum(dim=1).mean() + 0.5), 1)
        return SourceRoleMonTi(
            feat_dim=feat_dim, min_feat=min_feat.to(self.device),
            max_feat=max_feat.to(self.device), discrete=self.data.discrete,
            feat_budget=feat_budget, emb_dim=self.config['emb_dim'],
            ff_dim=self.config['ff_dim'], max_candidates=self.config['max_candidates'],
            num_layers=self.config['num_layers'], num_heads=self.config['num_heads'],
            initial_temp=self.config['initial_temp'], min_temp=self.config['min_temp'],
            temp_decay_rate=self.config['temp_decay_rate'],
            initial_eps=self.config['initial_eps'], min_eps=self.config['min_eps'],
            eps_decay_rate=self.config['eps_decay_rate'], dropout=self.config['dropout'],
            shared_parameters=self.config['shared_parameters'],
            without_candidates=self.config['without_candidates'],
            pos_encoding=self.config['pos_encoding'], ada_budget=self.config['ada_budget'],
            source_mode=self.source_arm)

    def run(self):
        result = super().run()
        if result['arm'] != 'gcn_only' or self.arm != 'gcn_only':
            raise RuntimeError('Source comparison changed its single-GCN feedback.')
        result.update(arm=self.source_arm, feedback_arm='gcn_only',
                      source_mode=self.model.source_mode,
                      scorer_source=('e[src]' if self.source_arm == 'legacy'
                                     else 'e[num_targets+num_candidates+src]'),
                      consistent_training=True,
                      fixed_generator_conditioning='original h_GCN')
        return result
