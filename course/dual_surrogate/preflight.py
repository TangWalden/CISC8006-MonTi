"""Bounded real-TRAIN functional checks; no victim deserialization or updates."""
import argparse
import json
import pathlib
import time
import traceback
from types import SimpleNamespace,MethodType

import dgl
import torch
from utils.utils import get_n_ids,set_seed
from modules.data_handler import DataHandler
from course.dual_surrogate.common import ROOT,private_output,sha256,validate_protocol,write_json
from course.dual_surrogate.runtime import (config_for,load_data,load_classifier,load_original_view,
    model_hash,checked_logits,gcn_logits,verify_teacher_directory)
from course.dual_surrogate.experiment import DualExperiment
from course.dual_surrogate.sequence_adapter import updated_sequences
from course.dual_surrogate.verify import verify_teacher_partition,verify_gaga_label_masks


def original_update(view,graph,ids,candidates,injected):
    copied=graph.local_var();original=view.features.shape[0]
    for name,value in view.group_masks.items():
        suffix=torch.ones(len(injected),device=graph.device) if name=='train_masked' else torch.zeros(len(injected),device=graph.device)
        copied.ndata[name]=torch.cat((value.to(graph.device),suffix))
    oracle=SimpleNamespace(**vars(view))
    oracle.group_agg_tensor=MethodType(DataHandler.group_agg_tensor,oracle)
    oracle.feat_agg_tensor=MethodType(DataHandler.feat_agg_tensor,oracle)
    return DataHandler.get_updated_sequences(oracle,copied,ids,candidates,injected,edge_weight=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=private_output(args.output,create=True);start=time.monotonic()
    result=dict(status='failed',optimizer_steps=0,sealed_victim_loaded=False,sealed_victim_forwards=0,test_queries=0)
    try:
        p,provenance=validate_protocol(args.protocol);teacher=ROOT/p['teacher_directory'];verify_teacher_directory(teacher)
        split=json.loads((teacher/'train_split_private.json').read_text())
        partition=verify_teacher_partition(split['original_train_nodes'],split['fit_nodes'],split['stop_nodes'])
        masks=torch.load(teacher/'fit_group_masks.pt',map_location='cpu')
        maskcheck=verify_gaga_label_masks(split['fit_nodes'],split['stop_nodes'],
            torch.nonzero(masks['train_neg']).reshape(-1).tolist(),torch.nonzero(masks['train_pos']).reshape(-1).tolist())
        torch.set_num_threads(8);set_seed(20261103);dgl.seed(20261103);dgl.random.seed(20261103)
        data=load_data(p,20261103)
        engine=DualExperiment(config_for(20261103),data,p,'gcn_gaga',teacher,out,start+p['preflight_wall_seconds'])
        engine.phase='preflight';engine.current_epoch=0
        full=load_original_view(data,engine.fit_view.adj,engine.fit_view.adj2)
        gcn2=load_classifier('gcn_new',data,teacher);gcn2_initial=model_hash(gcn2)
        model_before=model_hash(engine.model);rows=[]
        for group in p['preflight']['train_groups']:
            engine.current_group=group
            ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
            graph=engine.get_perturbed_graph(ids,int(data.node_budgets[group]),int(data.edge_budgets[group]),True)
            candidates=engine.capture['candidate_ids'];injected=torch.arange(data.graph.num_nodes(),graph.num_nodes(),device=data.device)
            seq,trace=updated_sequences(engine.fit_view,graph,ids,candidates,injected,training=True,return_trace=True)
            ref=original_update(engine.fit_view,graph,ids,candidates,injected)
            if not torch.allclose(seq,ref,atol=2e-5,rtol=1e-5):raise RuntimeError('FIT80 hard-forward sequence differs from frozen updater.')
            other=updated_sequences(full,graph,ids,candidates,injected,training=True)
            other_ref=original_update(full,graph,ids,candidates,injected)
            if not torch.allclose(other,other_ref,atol=2e-5,rtol=1e-5):raise RuntimeError('Original-TRAIN hard-forward differs.')
            a=checked_logits(engine.teachers['gaga_new'](seq),len(ids));b=checked_logits(engine.teachers['gaga_new'](ref),len(ids))
            if not torch.allclose(a,b,atol=2e-5,rtol=1e-5):raise RuntimeError('New GAGA hard-forward logits differ.')
            loss=torch.nn.functional.softplus(a[:,1]-a[:,0]).mean()
            gradients=torch.autograd.grad(loss,engine.live_capture,retain_graph=True,allow_unused=True)
            norms=[0.0 if g is None else float(g.detach().double().norm()) for g in gradients]
            if not all(__import__('math').isfinite(v) for v in norms):raise RuntimeError('Nonfinite feature/edge gradient.')
            head_gradients=torch.autograd.grad(loss,(engine.model.linear_feat.weight,engine.model.linear_edge.weight),
                retain_graph=True,allow_unused=True)
            head_norms=[0.0 if g is None else float(g.detach().double().norm()) for g in head_gradients]
            if not all(__import__('math').isfinite(v) for v in head_norms):raise RuntimeError('Nonfinite feature/edge head gradient.')
            g1=engine.teacher_logits('gcn_base',graph,ids,True)
            g2=gcn_logits(gcn2,graph,ids,training=True,full_graph=True)
            rows.append(dict(group=group,targets=len(ids),sequence_max_error=float((seq-ref).abs().max()),
                original_mask_sequence_max_error=float((other-other_ref).abs().max()),
                logits_max_error=float((a-b).abs().max()),feature_gradient_norm=norms[0],edge_gradient_norm=norms[1],trace=trace,
                feature_head_gradient_norm=head_norms[0],edge_head_gradient_norm=head_norms[1],
                gcn_new_inference='full graph, matches teacher fit convention',gcn_base_shape=list(g1.shape),gcn_new_shape=list(g2.shape)))
            engine.capture=engine.live_capture=None
        if not any(r['feature_gradient_norm']>0 for r in rows) or not any(r['edge_gradient_norm']>0 for r in rows):
            raise RuntimeError('All real-TRAIN feature or edge gradients vanish.')
        if not any(r['feature_head_gradient_norm']>0 for r in rows) or not any(r['edge_head_gradient_norm']>0 for r in rows):
            raise RuntimeError('GAGA feedback does not reach a generator output head.')
        if engine.optimizer_steps or model_hash(engine.model)!=model_before or model_hash(gcn2)!=gcn2_initial:
            raise RuntimeError('Preflight mutated parameters or performed updates.')
        for role,model in engine.teachers.items():
            if model_hash(model)!=engine.teacher_initial_hashes[role] or any(v.grad is not None for v in model.parameters()):
                raise RuntimeError('Frozen teacher changed in preflight.')
        result.update(status='passed',source_commit=provenance['source_commit'],protocol_sha256=provenance['protocol_sha256'],
            partition_check=partition,mask_check=maskcheck,cases=rows,generator_forwards=4,
            classifier_forwards=dict(gcn_base=4,gcn_new=4,gaga_new=8,gaga_victim=0),autograd_calls=8,
            hard_forward_tolerance=dict(atol=2e-5,rtol=1e-5),backward_is_st_approximation=True)
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result['elapsed_seconds']=time.monotonic()-start;write_json(out/'result.json',result,exclusive=True)
        print('DUAL_PREFLIGHT_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
