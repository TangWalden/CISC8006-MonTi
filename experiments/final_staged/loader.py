"""Load exact formal best MonTi weights for frozen, freshly paired CPU replay.

Imports of the locked ML runtime occur only when load_engine is explicitly
called by the registered preflight or main process. Importing this file performs
no model initialization, data reads, checkpoint deserialization or inference.
"""
import copy
import hashlib
import json
import os
import pathlib


FORMAL_SOURCE = '9f782dfbb98581e8bbe3c10f5ea5de91237e09ab'
EXPECTED = {
    0: (4, 14, 'd1b566626db88b56637fc171c39dee312735a1252608b1c3181bf42cccdcb190'),
    1: (3, 13, '628b1790e09d01a599d5e9907afe5f271df4108588cafde7b78fadf2af331b11'),
    2: (5, 15, '93d3515e9a96132bb05201f4815a0bb67976e47a63aeb11633a1627cf6df1fc0'),
}


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_engine(p, seed, output, deadline):
    """Return (frozen capture engine, original CPU data, receipt, state hashes)."""
    if seed not in EXPECTED or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Only the three registered CPU formal upstreams may load.')
    runtime = pathlib.Path(p['runtime_root']).resolve(strict=True)
    reference = p['formal_baselines'][str(seed)]

    def frozen_file(name):
        ref = reference[name]
        relative = pathlib.PurePosixPath(ref['path'])
        if ref['root'] not in ('asset', 'runtime') or relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe formal checkpoint/configuration reference.')
        path = (runtime / str(relative)).resolve(strict=True)
        if runtime not in path.parents or sha(path) != ref['sha256']:
            raise RuntimeError('Frozen formal file bytes changed: ' + name)
        return path

    config = json.loads(frozen_file('config_ref').read_text(encoding='utf-8'))
    registered = json.loads(frozen_file('registered_ref').read_text(encoding='utf-8'))
    historical = json.loads(frozen_file('result_ref').read_text(encoding='utf-8'))
    checkpoint = frozen_file('checkpoint_ref')
    best, terminal, wanted = EXPECTED[seed]
    if (reference['execution_source'] != FORMAL_SOURCE or registered['commit'] != FORMAL_SOURCE
            or registered['seed'] != seed or registered['mode'] != 'learned'
            or registered['phase'] != 'reproduction' or registered['tracked_changes'] != ''
            or historical['status'] != 'completed' or historical['best_epoch'] != best
            or historical['completed_epochs'] != terminal
            or historical['checkpoint_sha256'] != wanted or reference['checkpoint_ref']['sha256'] != wanted
            or reference['best_epoch'] != best or reference['scheduler_restored_steps'] != terminal
            or config['seed'] != seed or config['random_candidates'] is not False
            or config['without_candidates'] is not False or config['ada_budget'] is not True):
        raise RuntimeError('The exact frozen formal learned identity or candidate flags differ.')

    # Only the fixed original GCN is constructed here. Changing this loader-only
    # victim declaration avoids loading GAGA or its sequence caches during fit;
    # it does not change the MonTi architecture or its saved parameter mapping.
    cpu_config = copy.deepcopy(config)
    cpu_config.update(cuda_id=0, surrogate_model='GCN', victim_model='GCN')
    import numpy as np
    import torch
    from modules.data_handler import DataHandler
    from course.dual_surrogate.experiment import DualExperiment
    from course.dual_surrogate.runtime import load_classifier, model_hash

    class FrozenFormalExperiment(DualExperiment):
        def load_pretrained_model(self, name, flag='surrogate'):
            if name != 'GCN' or flag != 'surrogate':
                raise RuntimeError('Official GAGA and fresh teachers are prohibited during fitting.')
            return load_classifier('gcn_base', self.data)

        def get_node_representations(self):
            path = runtime / 'Datasets/GossipCop/h_GCN.npy'
            if not path.is_file():
                raise RuntimeError('No surrogate representation regeneration is permitted.')
            self.data.graph.ndata['h'] = torch.from_numpy(np.load(path, allow_pickle=False)).to(self.device)

        def run(self):
            raise RuntimeError('The formal upstream is fixed; only the downstream may train.')

        def train_epoch(self):
            raise RuntimeError('Upstream training is prohibited.')

        def validation(self, *args, **kwargs):
            raise RuntimeError('No VALID scoring or checkpoint selection is authorized.')

        def test(self):
            raise RuntimeError('Final TEST scoring belongs to the separately sealed runner.')

        def check_performance(self, *args, **kwargs):
            raise RuntimeError('Unregistered evaluation through the upstream engine is prohibited.')

    data = DataHandler(cpu_config)
    if (data.device.type != 'cpu' or list(map(int, data.target_idx_train)) != p['train_groups']
            or list(map(int, data.target_idx_test)) != p['test_groups']):
        raise RuntimeError('The complete original CPU TRAIN/TEST partitions changed.')
    # Retain the full TEST cohort but remove the unused VALID selection from the
    # engine interface. The original graph and source incidence budgets remain.
    data.target_idx_valid = np.empty(0, dtype=np.int64)
    engine = FrozenFormalExperiment(cpu_config, data, p, 'gcn_only', runtime / 'runs', output, deadline)
    state = torch.load(checkpoint, map_location='cpu')
    engine.model.load_state_dict(state, strict=True)
    del state
    for _ in range(terminal):
        engine.model.step()
    engine.capture_epoch = best
    engine.current_epoch = best
    engine.phase = 'not_started'
    if engine.victim_model is not None or set(engine.teachers) != {'gcn_base'}:
        raise RuntimeError('Only the original frozen GCN may be present during fitting.')
    hashes = {}
    for role, model in (('upstream', engine.model), ('teacher', engine.teachers['gcn_base'])):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        hashes[role] = model_hash(model)

    def prohibit_upstream_update(*args, **kwargs):
        raise RuntimeError('The frozen upstream optimizer may never update.')
    engine.optimizer.step = prohibit_upstream_update
    if engine.optimizer_steps != 0:
        raise RuntimeError('An upstream update occurred during loading.')
    receipt = dict(seed=seed, run_id=reference['run_id'], execution_source=FORMAL_SOURCE,
        checkpoint_sha256=wanted, final_model_sha256=wanted,
        config_sha256=reference['config_ref']['sha256'], best_epoch=best, completed_epochs=terminal,
        capture_epoch=best, scheduler_restored_steps=terminal,
        restored_temperature=engine.model.temp_scheduler.get_temperature(),
        restored_epsilon=engine.model.eps_scheduler.get_epsilon(),
        upstream_state_sha256=hashes['upstream'], original_GCN_state_sha256=hashes['teacher'],
        upstream_optimizer_steps=0, official_GAGA_loaded=False,
        config_adaptation='CPU device and GCN-only loader declaration; original saved MonTi architecture and learned-candidate flags retained.',
        scheduler_scope='Historical final TEST reloaded only model.state_dict, leaving the terminal training scheduler objects; restore terminal14/13/15 steps here.',
        replay_scope='Fresh per-case fork_rng with selected-best capture_epoch4/3/5. Historical GPU prediction or graph identity is not promised.',
        evaluation_schedule_scope='The frozen randomness=False softmax ignores temperature and epsilon; injected latent noise for multiple auxiliaries remains seeded.')
    return engine, data, receipt, hashes
