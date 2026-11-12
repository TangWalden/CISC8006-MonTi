"""One official released MonTi state, frozen for three downstream/noise repeats.

Importing this adapter reads no checkpoint and imports no ML package. The
registered TRAIN preflight must verify strict keys/shapes and functional
compatibility before a main invocation. This is not a paper-number replay.
"""
import copy
import hashlib
import json
import os
import pathlib

AUTHOR_CHECKPOINT_SHA = '9f5dc7dd594a8e8927449665ef3a07b777199c2436d6fd1cc1f7adf0fdb6c23d'
AUTHOR_CONFIG_SHA = 'e27fd6959f35ab1786f35e4e70c89bec5156da1a6577f4481287310ebf552c2a'
CONFIG = dict(exp_name='best_models', model='MonTi', surrogate_model='GCN', victim_model='GAGA',
    dataset='gossipcop_0.05', multi_target=True, split=0, seed=3, node_budget=.05,
    edge_budget=.1, k_exp=2, masking=False, max_candidates=128, num_layers=6,
    num_heads=4, emb_dim=64, ff_dim=2048, initial_temp=10, min_temp=.01,
    temp_decay_rate=.63, initial_eps=10, min_eps=.01, eps_decay_rate=.63,
    lr=.001, weight_decay=.0001, dropout=.1, epochs=100, valid_epoch=1,
    patience=10, batch_size=1, batch_size_pre=1024, cuda_id=0)
DEFAULTS = dict(random_candidates=False, shared_parameters=False, without_candidates=False,
                pos_encoding=True, ada_budget=True)


def sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def load_engine(p, seed, output, deadline):
    if p['schema']!='monti_local_swap_v1' or seed not in (0,1,2) or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Only registered author-upstream CPU repetitions0/1/2 may load.')
    runtime=pathlib.Path(p['runtime_root']).resolve(strict=True);reference=p['author_upstream']
    def checked(name,wanted):
        ref=reference[name];relative=pathlib.PurePosixPath(ref['path'])
        if ref['root']!='asset' or relative.is_absolute() or '..' in relative.parts:
            raise RuntimeError('Unsafe separately staged official-author asset.')
        path=(runtime/str(relative)).resolve(strict=True)
        if (runtime/'runs') not in path.parents or ref['sha256']!=wanted or sha(path)!=wanted:
            raise RuntimeError('The exact separately staged official-author asset changed.')
        return path
    config_path=checked('config_ref',AUTHOR_CONFIG_SHA)
    checkpoint=checked('checkpoint_ref',AUTHOR_CHECKPOINT_SHA)
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config!=CONFIG:raise RuntimeError('The published737-byte author configuration differs.')
    config.update(DEFAULTS)
    cpu_config=copy.deepcopy(config);cpu_config.update(cuda_id=0,victim_model='GCN')
    import numpy as np
    import torch
    from modules.data_handler import DataHandler
    from modules.experiment_handler import ExperimentHandler
    from course.dual_surrogate.experiment import DualExperiment
    from course.dual_surrogate.common import case_seed
    from course.dual_surrogate.runtime import load_classifier,model_hash,fork_rng
    class FrozenAuthorExperiment(DualExperiment):
        capture_noise_seed=seed
        def load_pretrained_model(self,name,flag='surrogate'):
            if name!='GCN' or flag!='surrogate':raise RuntimeError('Only original GCN may load before sealing.')
            return load_classifier('gcn_base',self.data)
        def get_node_representations(self):
            path=runtime/'Datasets/GossipCop/h_GCN.npy'
            if not path.is_file():raise RuntimeError('No surrogate-representation regeneration is allowed.')
            self.data.graph.ndata['h']=torch.from_numpy(np.load(path,allow_pickle=False)).to(self.device)
        def get_perturbed_graph(self,targets,node_budget,edge_budget,training=False,need_sequence=False):
            # Preserve the native construction and inherited capture hook. Only
            # the explicitly registered repetition-noise key differs from the
            # author's constructor/configuration seed3, which stays truthful.
            if training or need_sequence:raise RuntimeError('Author upstream is capture-only in eval mode.')
            self.check_deadline()
            noise=case_seed(self.capture_noise_seed,self.phase,self.current_epoch,self.current_group)
            with fork_rng(noise):
                graph=ExperimentHandler.get_perturbed_graph(self,targets,node_budget,edge_budget,False,False)
            self.generator_forwards+=1
            if graph is None:raise RuntimeError('An author-generation failure cannot remove a case.')
            original=self.data.graph.num_nodes();added=graph.num_edges()-self.data.graph.num_edges()-int(node_budget)
            if (added<0 or added%2 or graph.num_nodes()!=original+int(node_budget) or added//2!=int(edge_budget)
                    or not torch.equal(self.capture['target_ids'],targets.to(graph.device))):
                raise RuntimeError('Native author capture changed target order or actual structural budgets.')
            return graph
        def run(self):raise RuntimeError('The author upstream may not train.')
        def train_epoch(self):raise RuntimeError('The author upstream may not train.')
        def validation(self,*args,**kwargs):raise RuntimeError('No VALID selection is authorized.')
        def test(self):raise RuntimeError('TEST evaluation is prohibited in this development study.')
        def check_performance(self,*args,**kwargs):raise RuntimeError('Unregistered upstream evaluation prohibited.')
    data=DataHandler(cpu_config)
    if (data.device.type!='cpu' or not set(p['train_groups'])<=set(map(int,data.target_idx_train))
            or not set(p['valid_groups'])<=set(map(int,data.target_idx_valid))):
        raise RuntimeError('Registered development groups cross the original partitions.')
    data.target_idx_train=np.asarray(p['train_groups'],dtype=np.int64)
    data.target_idx_valid=np.asarray(p['valid_groups'],dtype=np.int64)
    data.target_idx_test=np.empty(0,dtype=np.int64)
    engine=FrozenAuthorExperiment(cpu_config,data,p,'gcn_only',runtime/'runs',output,deadline)
    state=torch.load(checkpoint,map_location='cpu')
    expected=engine.model.state_dict()
    if (not isinstance(state,dict) or set(state)!=set(expected)
            or any(not torch.is_tensor(state[k]) or state[k].shape!=expected[k].shape
                   or not bool(torch.isfinite(state[k]).all()) for k in expected)):
        raise RuntimeError('Official checkpoint must be a finite, strict plain MonTi state_dict with matching shapes.')
    engine.model.load_state_dict(state,strict=True);del state,expected
    engine.capture_epoch=0;engine.current_epoch=0;engine.phase='not_started'
    if engine.victim_model is not None or set(engine.teachers)!={'gcn_base'} or engine.config['seed']!=3:
        raise RuntimeError('Author seed/frozen original-GCN boundary changed.')
    hashes={}
    for role,model in (('upstream',engine.model),('teacher',engine.teachers['gcn_base'])):
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False);parameter.grad=None
        hashes[role]=model_hash(model)
    if engine.model.temp_scheduler.get_temperature()!=10 or engine.model.eps_scheduler.get_epsilon()!=10:
        raise RuntimeError('Official inference must begin with fresh initial schedulers, without restoration steps.')
    def prohibit(*args,**kwargs):raise RuntimeError('The frozen author upstream optimizer may never update.')
    engine.optimizer.step=prohibit
    if engine.optimizer_steps:raise RuntimeError('An upstream update occurred during loading.')
    receipt=dict(downstream_seed=seed,capture_noise_seed=seed,author_configuration_seed=3,
        checkpoint_sha256=AUTHOR_CHECKPOINT_SHA,final_model_sha256=AUTHOR_CHECKPOINT_SHA,
        config_sha256=AUTHOR_CONFIG_SHA,upstream_origin='Official released best_models/checkpoints/GCN-GAGA-gossipcop_0.05-best_model.pickle',
        distinct_author_checkpoint_count=1,capture_epoch=0,capture_epoch_semantics='Synthetic RNG key only, not an author training/best epoch',
        scheduler_steps_applied=0,restored_temperature=10,restored_epsilon=10,
        upstream_state_sha256=hashes['upstream'],original_GCN_state_sha256=hashes['teacher'],
        upstream_optimizer_steps=0,official_GAGA_loaded=False,strict_state_dictionary_compatible=True,
        configuration_adaptation='CPU device and GCN-only loader declaration; author seed3 and all MonTi architecture/default flags retained.',
        RNG_scope='Native eval keeps torch.randn latent noise when node_budget>1. Repetition0/1/2 keys are explicit per-case fork_rng, shared across arms; not historical CLI bitwise replay.',
        scheduler_scope='Official run.py inference constructs initial scheduler objects; eval randomness=False ignores temperature/epsilon. No historical epoch is fabricated.',
        replication_scope='One fixed author state, three downstream/capture-noise repetitions; not three independently trained author models or a paper-table equivalence claim.')
    return engine,data,receipt,hashes
