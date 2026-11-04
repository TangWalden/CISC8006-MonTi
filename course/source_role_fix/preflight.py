"""Four fixed TRAIN gradient/budget checks, without optimizer or victim calls."""
import argparse
import json
import math
import time
import traceback

import dgl
import torch

from utils.utils import get_n_ids,set_seed
from course.source_role_fix.common import ROOT,ARMS,private_output,validate_protocol,write_json
from course.dual_surrogate.runtime import config_for,load_data,model_hash
from course.source_role_fix.experiment import SourceExperiment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=private_output(args.output,create=True);start=time.monotonic()
    result=dict(status='failed',optimizer_steps=0,sealed_victim_loaded=False,sealed_victim_forwards=0,test_queries=0)
    rows=[];engines=[];autograd_calls=0
    try:
        p,provenance=validate_protocol(args.protocol)
        torch.set_num_threads(8);set_seed(20261104);dgl.seed(20261104);dgl.random.seed(20261104)
        data=load_data(p,20261104)
        for arm in ARMS:
            engine=SourceExperiment(config_for(20261104),data,p,arm,ROOT/p['teacher_directory'],out,start+p['preflight_wall_seconds'])
            engines.append(engine);engine.phase='preflight';engine.current_epoch=0
            initial=model_hash(engine.model);arm_rows=[]
            for group in p['preflight']['train_groups']:
                engine.current_group=group
                ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
                graph=engine.get_perturbed_graph(ids,nb,eb,True)
                logits=engine.teacher_logits('gcn_base',graph,ids,True)
                loss=torch.nn.functional.softplus(logits[:,1]-logits[:,0]).mean()
                autograd_calls+=1
                gradients=torch.autograd.grad(loss,tuple(engine.live_capture)+(engine.model.linear_feat.weight,engine.model.linear_edge.weight),allow_unused=True)
                norms=[0. if value is None else float(value.detach().double().norm()) for value in gradients]
                if not all(math.isfinite(x) for x in norms):raise RuntimeError('Nonfinite GCN-to-generator gradient.')
                row=dict(arm=arm,group=group,targets=len(ids),node_budget=nb,edge_budget=eb,
                    actual_injected_nodes=graph.num_nodes()-data.graph.num_nodes(),
                    actual_injected_edges=(graph.num_edges()-data.graph.num_edges()-nb)//2,
                    feature_gradient_norm=norms[0],edge_gradient_norm=norms[1],
                    feature_head_gradient_norm=norms[2],edge_head_gradient_norm=norms[3])
                rows.append(row);arm_rows.append(row);engine.capture=engine.live_capture=None
            for key in ('feature_gradient_norm','edge_gradient_norm','feature_head_gradient_norm','edge_head_gradient_norm'):
                if not any(row[key]>0 for row in arm_rows):raise RuntimeError('All real-TRAIN gradients vanish: '+arm+' '+key)
            if engine.optimizer_steps or model_hash(engine.model)!=initial:raise RuntimeError('Preflight updated generator.')
            for role,model in engine.teachers.items():
                if model_hash(model)!=engine.teacher_initial_hashes[role] or any(v.grad is not None for v in model.parameters()):
                    raise RuntimeError('Frozen GCN teacher changed.')
        if len({engine.initial_model_hash for engine in engines})!=1:raise RuntimeError('Paired initial generator states differ.')
        result.update(status='passed',source_commit=provenance['source_commit'],protocol_sha256=provenance['protocol_sha256'],cases=rows,
            paired_initial_state_sha256=engines[0].initial_model_hash)
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(generator_forwards=sum(e.generator_forwards for e in engines),
            classifier_forwards={role:sum(e.classifier_forwards[role] for e in engines) for role in ('gcn_base','gcn_new','gaga_new','gaga_victim')},
            autograd_calls=autograd_calls,elapsed_seconds=time.monotonic()-start)
        write_json(out/'result.json',result,exclusive=True);print('SOURCE_ROLE_PREFLIGHT_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
