"""TRAIN-only MonTi fits against one or two independently fixed teachers."""
import json
import pathlib
import time

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from modules.experiment_handler import ExperimentHandler
from utils.utils import get_n_ids
from course.dual_surrogate.common import ROOT, ARMS, case_seed, sha256, write_json
from course.dual_surrogate.runtime import model_hash, fork_rng, gcn_logits, load_classifier, load_fit_view,checked_logits
from course.dual_surrogate.sequence_adapter import updated_sequences

ROLE_MAP = {'gcn_only':('gcn_base',),'gaga_only':('gaga_new',),
    'two_gcn':('gcn_base','gcn_new'),'gcn_gaga':('gcn_base','gaga_new')}


class DualExperiment(ExperimentHandler):
    def __init__(self,config,data,protocol,arm,teacher_directory,output,deadline):
        if arm not in ARMS: raise ValueError('Unknown frozen arm.')
        self.protocol,self.arm,self.output,self.deadline=protocol,arm,pathlib.Path(output),deadline
        self.generator_forwards=self.optimizer_steps=0
        self.classifier_forwards={r:0 for r in ('gcn_base','gcn_new','gaga_new','gaga_victim')}
        self.current_group,self.current_epoch,self.phase=None,0,'not_started'
        self.capture=self.live_capture=None
        super().__init__(config,data,None,False)
        self.teacher_directory=pathlib.Path(teacher_directory)
        self.teachers={'gcn_base':self.surrogate_model}
        # New classifier constructors must not change the subsequent TRAIN
        # loader/dropout RNG streams differently across paired arms.
        with fork_rng(20261103):
            for role in ROLE_MAP[arm]:
                if role not in self.teachers: self.teachers[role]=load_classifier(role,data,self.teacher_directory)
            self.fit_view=None
            if 'gaga_new' in ROLE_MAP[arm]: self.fit_view,_=load_fit_view(data,self.teacher_directory,protocol)
        self.teacher_initial_hashes={r:model_hash(m) for r,m in self.teachers.items()}
        self.initial_model_hash=model_hash(self.model)
        self.model.register_forward_hook(self.observe)
        original_step=self.optimizer.step
        def step(*args,**kwargs):
            result=original_step(*args,**kwargs);self.optimizer_steps+=1;return result
        self.optimizer.step=step

    def check_deadline(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Registered per-job deadline reached.')

    def load_pretrained_models(self):
        self.surrogate_model=self.load_pretrained_model('GCN',flag='surrogate').to(self.device)
        self.sampler_surrogate=dgl.dataloading.MultiLayerFullNeighborSampler(2)
        self.victim_model=self.sampler_victim=None

    def load_pretrained_model(self,name,flag='surrogate'):
        if name!='GCN' or flag!='surrogate':raise RuntimeError('Victim loading is prohibited during training.')
        return super().load_pretrained_model(name,flag)

    def get_node_representations(self):
        path=ROOT/'Datasets/GossipCop/h_GCN.npy'
        if not path.is_file():raise RuntimeError('No representation regeneration allowed.')
        self.data.graph.ndata['h']=torch.from_numpy(np.load(path)).to(self.device)

    def observe(self,model,inputs,output):
        subgraph,targets=inputs[:2];features,adj,candidates=output
        if features is None or adj is None:raise RuntimeError('No attack may be silently omitted.')
        c=(torch.empty(0,dtype=torch.long,device=subgraph.device) if candidates is None
           else subgraph.ndata[dgl.NID][candidates])
        self.live_capture=(features,adj)
        self.capture=dict(x_atk=features.detach(),adj_attack=adj.detach(),
            target_ids=subgraph.ndata[dgl.NID][targets].detach(),candidate_ids=c.detach())

    def get_perturbed_graph(self,targets,node_budget,edge_budget,training=False,need_sequence=False):
        self.check_deadline()
        seed=case_seed(self.config['seed'],self.phase,self.current_epoch,self.current_group)
        with fork_rng(seed):
            graph=super().get_perturbed_graph(targets,node_budget,edge_budget,training,False)
        self.generator_forwards+=1
        if graph is None:raise RuntimeError('A failed generation cannot remove a case.')
        original=self.data.graph.num_nodes()
        added=graph.num_edges()-self.data.graph.num_edges()-int(node_budget)
        actual=added//2
        if added<0 or added%2 or graph.num_nodes()!=original+int(node_budget) or actual!=int(edge_budget):
            raise RuntimeError('Actual injection budgets differ from registered caps.')
        if not torch.equal(self.capture['target_ids'],targets.to(graph.device)):
            raise RuntimeError('Captured target order differs.')
        return graph

    def teacher_logits(self,role,graph,targets,training=True):
        self.check_deadline();self.classifier_forwards[role]+=1
        if role=='gaga_new':
            original=self.data.graph.num_nodes()
            seq=updated_sequences(self.fit_view,graph,targets,self.capture['candidate_ids'],
                torch.arange(original,graph.num_nodes(),device=graph.device),training=training)
            return checked_logits(self.teachers[role](seq),len(targets))
        return gcn_logits(self.teachers[role],graph,targets,training=training,full_graph=role=='gcn_new')

    def run(self):
        roles=ROLE_MAP[self.arm]
        for epoch in range(1,9):
            self.current_epoch,self.phase=epoch,'train'; self.model.train(True)
            rows=[]
            with (self.output/'raw'/('train_epoch_%03d.jsonl'%epoch)).open('x',encoding='utf-8',newline='\n') as stream:
                for values,nodes,edges in self.data.get_train_loader():
                    self.current_group=int(values[0]);nb,eb=int(nodes[0]),int(edges[0])
                    ids=torch.tensor(get_n_ids(self.data.target_assignment,self.current_group),device=self.device,dtype=torch.long)
                    graph=self.get_perturbed_graph(ids,nb,eb,True)
                    losses=[];record={}
                    for role in roles:
                        logits=self.teacher_logits(role,graph,ids,True)
                        if not bool(torch.isfinite(logits).all()):raise RuntimeError('Nonfinite teacher logits.')
                        margin=logits[:,0]-logits[:,1]
                        loss=F.softplus(-margin).mean();losses.append(loss)
                        record[role]=dict(loss=float(loss.detach()),mean_attack_margin=float(margin.detach().mean()),wrong=int((logits.argmax(1)==0).sum()))
                    total=sum(losses)/len(losses)
                    self.optimizer.zero_grad();total.backward();self.optimizer.step()
                    row=dict(group=self.current_group,targets=len(ids),node_budget=nb,edge_budget=eb,
                        actual_injected_nodes=graph.num_nodes()-self.data.graph.num_nodes(),
                        actual_injected_edges=(graph.num_edges()-self.data.graph.num_edges()-nb)//2,
                        losses=record,total_loss=float(total.detach()),generation_seed=case_seed(self.config['seed'],'train',epoch,self.current_group))
                    stream.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                    self.capture=self.live_capture=None
                    del graph,logits,total,losses,loss
                    if len(rows)%64==0:print('TRAIN_GROUPS',self.arm,self.config['seed'],epoch,len(rows),flush=True)
                stream.flush()
            if len(rows)!=256:raise RuntimeError('Incomplete TRAIN epoch.')
            self.model.step()
            write_json(self.output/('epoch_%03d.json'%epoch),dict(epoch=epoch,groups=256,
                optimizer_steps=self.optimizer_steps,generator_forwards=self.generator_forwards,
                classifier_forwards=self.classifier_forwards,mean_loss=float(np.mean([r['total_loss'] for r in rows])),
                validation_performed=False,victim_loaded=False),exclusive=True)
        if self.optimizer_steps!=2048 or self.generator_forwards!=2048:raise RuntimeError('Training count mismatch.')
        for role,model in self.teachers.items():
            if model_hash(model)!=self.teacher_initial_hashes[role] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('Teacher parameters or gradient buffers changed.')
        model_path=self.output/'final_model.pt';optimizer_path=self.output/'final_optimizer.pt'
        torch.save(self.model.state_dict(),model_path);torch.save(self.optimizer.state_dict(),optimizer_path)
        return dict(status='completed',arm=self.arm,seed=self.config['seed'],epochs=8,
            optimizer_steps=self.optimizer_steps,generator_forwards=self.generator_forwards,
            classifier_forwards=self.classifier_forwards,initial_model_state_sha256=self.initial_model_hash,
            final_model_state_sha256=model_hash(self.model),final_model_sha256=sha256(model_path),
            final_optimizer_sha256=sha256(optimizer_path),teacher_initial_hashes=self.teacher_initial_hashes,
            teacher_final_hashes={r:model_hash(m) for r,m in self.teachers.items()},
            victim_loaded=False,validation_performed=False,test_queries=0)
