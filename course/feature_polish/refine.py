"""Polish injected features on one shared fixed hard graph per saved generator.

Only generated features are optimized. The whole-group guard inspects original
GCN predictions only, never official-victim outcomes or evaluation labels.
Every branch, including failed/no-change refinements, retains every VALID case.
"""
import argparse
import hashlib
import json
import math
import pathlib
import time
import traceback

from course.feature_polish.common import ROOT,private_output,sha256,validate_protocol,write_json


ARMS=('baseline','gcn_pgd','gaga_pgd','gaga_guarded','random')


def tensor_sha256(value):
    value=value.detach().cpu().contiguous()
    header=json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode()
    return hashlib.sha256(len(header).to_bytes(8,'big')+header+value.numpy().tobytes()).hexdigest()


def make_baseline_engine(protocol,seed,output,deadline):
    """Reuse only the explicitly frozen legacy terminal8 generator, without fit."""
    import torch
    from course.dual_surrogate.experiment import DualExperiment
    from course.dual_surrogate.runtime import config_for,load_data,load_classifier,model_hash
    class CpuSafeBaselineExperiment(DualExperiment):
        def load_pretrained_model(self,name,flag='surrogate'):
            if name!='GCN' or flag!='surrogate':
                raise RuntimeError('Feature-only baseline loading permits the original GCN surrogate only.')
            # The runtime helper creates the same single GCN(feat_dim,[64,64],2)
            # and verifies its frozen bytes, but deserializes with map_location
            # CPU before moving to the registered execution device. No victim
            # load, second initialization or historical module edit is needed.
            return load_classifier('gcn_base',self.data)
    reference=protocol['baseline_runs'][str(seed)]
    directory=pathlib.Path(reference['directory'])
    if directory.is_absolute() or '..' in directory.parts:
        raise RuntimeError('Baseline run must remain repository-relative.')
    directory=ROOT/directory
    if sha256(directory/'result.json')!=reference['result_sha256']:
        raise RuntimeError('Frozen legacy terminal receipt changed.')
    receipt=json.loads((directory/'result.json').read_text(encoding='utf-8'))
    if (receipt['status']!='completed' or receipt['arm']!='legacy' or receipt['seed']!=seed or
            receipt['epochs']!=8 or receipt['optimizer_steps']!=2048 or
            sha256(directory/'final_model.pt')!=reference['final_model_sha256'] or
            receipt['final_model_sha256']!=reference['final_model_sha256']):
        raise RuntimeError('Baseline must be the specified completed legacy terminal8 model.')
    data=load_data(protocol,seed)
    # The inherited gcn_only branch never reads or loads a fresh GAGA teacher.
    engine=CpuSafeBaselineExperiment(config_for(seed),data,protocol,'gcn_only',ROOT/protocol['teacher_directory'],output,deadline)
    engine.model.load_state_dict(torch.load(directory/'final_model.pt',map_location=data.device))
    if model_hash(engine.model)!=receipt['final_model_state_sha256']:
        raise RuntimeError('Restored baseline generator content differs from its terminal receipt.')
    for _ in range(8):engine.model.step()
    engine.model.eval();engine.phase='validation';engine.current_epoch=8
    return engine,data,receipt


def graph_with_features(graph,original,features):
    import torch
    if features.shape!=(graph.num_nodes()-original,graph.ndata['x'].shape[1]):
        raise RuntimeError('Injected feature shape differs from fixed graph.')
    local=graph.local_var()
    local.ndata['x']=torch.cat((graph.ndata['x'][:original].detach(),features),dim=0)
    return local


def preserves_baseline_gcn_zeros(baseline_logits,candidate_logits):
    import torch
    if baseline_logits.shape!=candidate_logits.shape or baseline_logits.ndim!=2 or baseline_logits.shape[1]!=2:
        raise RuntimeError('Guard requires parallel binary logits.')
    baseline=baseline_logits.argmax(1);candidate=candidate_logits.argmax(1)
    protected=baseline==0
    lost=protected & (candidate!=0)
    return bool(torch.logical_not(lost).all()),dict(protected_target_count=int(protected.sum()),
        lost_protected_target_count=int(lost.sum()),candidate_benign_count=int((candidate==0).sum()),
        empty_protected_set=not bool(protected.any()))


def projected_features(x0,minimum,maximum,forward,steps,alpha,check,counters):
    """Terminal sign-PGD in normalized coordinates; no best-iterate selection."""
    import torch
    import torch.nn.functional as F
    from course.feature_polish.affine import features_from_unit_box
    span=maximum-minimum;movable=span>0
    safe_span=torch.where(movable,span,torch.ones_like(span))
    unit0=torch.where(movable,(x0-minimum)/safe_span,torch.zeros_like(x0)).clamp(0,1).detach()
    unit=unit0.clone();history=[]
    for step in range(1,steps+1):
        check();unit=unit.detach().requires_grad_(True)
        x=features_from_unit_box(unit,minimum,maximum)
        # Enforce the exact original raw box despite final floating-point rounding.
        x=torch.maximum(minimum,torch.minimum(maximum,x))
        logits=forward(x);loss=F.softplus(logits[:,1]-logits[:,0]).mean()
        counters['feature_autograd_calls']+=1
        gradient,=torch.autograd.grad(loss,unit)
        if not bool(torch.isfinite(loss) and torch.isfinite(gradient).all()):
            raise RuntimeError('Nonfinite injected-feature PGD value/gradient.')
        direction=torch.where(movable,gradient.sign(),torch.zeros_like(gradient))
        updated=(unit.detach()-alpha*direction).clamp(0,1)
        counters['feature_pgd_steps']+=1
        history.append(dict(step=step,loss_before_update=float(loss.detach()),
            gradient_linf=float(gradient.detach().abs().max()),gradient_l2=float(gradient.detach().double().norm()),
            zero_gradient_coordinates=int((gradient.detach()==0).sum()),
            normalized_update_linf=float((updated-unit.detach()).abs().max())))
        unit=updated
    value=features_from_unit_box(unit.detach(),minimum,maximum)
    value=torch.maximum(minimum,torch.minimum(maximum,value)).detach()
    if not torch.equal(value[:,~movable],x0[:,~movable]):
        raise RuntimeError('Zero-range feature dimensions changed.')
    return value,history


def random_features(x0,minimum,maximum,seed,steps,alpha):
    import torch
    from course.feature_polish.affine import features_from_unit_box
    span=maximum-minimum;movable=span>0;safe=torch.where(movable,span,torch.ones_like(span))
    unit=torch.where(movable,(x0-minimum)/safe,torch.zeros_like(x0)).clamp(0,1).detach()
    generator=torch.Generator(device='cpu');generator.manual_seed(seed)
    direction=(torch.randint(0,2,x0.shape,generator=generator,dtype=torch.int64)*2-1).to(device=x0.device,dtype=x0.dtype)
    direction=torch.where(movable,direction,torch.zeros_like(direction))
    for _ in range(steps):unit=(unit+alpha*direction).clamp(0,1)
    value=features_from_unit_box(unit,minimum,maximum)
    return torch.maximum(minimum,torch.minimum(maximum,value)).detach()


def feature_statistics(features,x0,minimum,maximum):
    import torch
    if features.shape!=x0.shape or features.dtype!=x0.dtype or not bool(torch.isfinite(features).all()):
        raise RuntimeError('A retained branch has invalid features.')
    if not bool(((features>=minimum)&(features<=maximum)).all()):
        raise RuntimeError('A retained branch violates the original feature box.')
    span=maximum-minimum;movable=span>0;safe=torch.where(movable,span,torch.ones_like(span))
    delta=torch.where(movable,(features-x0)/safe,torch.zeros_like(features))
    if not torch.equal(features[:,~movable],x0[:,~movable]):
        raise RuntimeError('Zero-range feature dimensions changed.')
    unit=torch.where(movable,(features-minimum)/safe,torch.zeros_like(features))
    mask=movable.unsqueeze(0).expand_as(features)
    saturated=((unit<=1e-6)|(unit>=1-1e-6)) & mask
    denominator=int(mask.sum())
    return dict(content_sha256=tensor_sha256(features),normalized_displacement_linf=float(delta.abs().max()),
        normalized_displacement_l2=float(delta.double().norm()),normalized_displacement_rms=float(delta.double().square().mean().sqrt()),
        raw_displacement_linf=float((features-x0).abs().max()),movable_coordinates=denominator,
        zero_range_dimensions=int((~movable).sum()),saturation_fraction=(int(saturated.sum())/denominator if denominator else 0.),
        saturation_rule='normalized coordinate within1e-6of0or1; zero-range dimensions excluded')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seed',required=True,type=int,choices=[3,4,5]);args=parser.parse_args()
    output=private_output(args.output,create=True);(output/'captures').mkdir();(output/'histories').mkdir()
    started,cpu_started=time.monotonic(),time.process_time()
    counters=dict(baseline_generations=0,gcn_forwards=0,gaga_teacher_forwards=0,feature_autograd_calls=0,
        feature_pgd_steps=0,generator_optimizer_steps=0,teacher_optimizer_steps=0,official_victim_forwards=0,API_calls=0,TEST_queries=0)
    result=dict(status='failed',seed=args.seed,official_victim_loaded=False,counters=counters);engine=None
    try:
        protocol,provenance=validate_protocol(args.protocol)
        gate_reference=protocol['preflight_ref']
        gate_path=ROOT/gate_reference['path']
        if sha256(gate_path)!=gate_reference['sha256']:
            raise RuntimeError('The passed preflight receipt bytes changed.')
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        if (gate['status']!='passed' or gate['counts']['optimizer_steps']!=0 or
                gate['counts']['official_victim_forwards']!=0 or
                gate['provenance']['source_commit']!=provenance['source_commit']):
            raise RuntimeError('A same-source passed zero-update/no-victim preflight is required.')
        if protocol['polish_settings']!={'steps':20,'alpha':.02}:
            raise RuntimeError('Fixed terminal20/alpha.02 polishing settings changed.')
        deadline=started+protocol['refine_job_wall_seconds']
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('Frozen per-seed feature-polishing deadline reached.')
        import dgl
        import torch
        import torch.nn.functional as F
        from utils.utils import get_n_ids,set_seed
        from course.dual_surrogate.common import case_seed
        from course.dual_surrogate.runtime import checked_logits,gcn_logits,load_original_view,model_hash
        from course.feature_polish.affine import build_affine
        from course.feature_polish.teacher import load_teacher
        torch.set_num_threads(8);set_seed(args.seed);dgl.seed(args.seed);dgl.random.seed(args.seed)
        engine,data,baseline_receipt=make_baseline_engine(protocol,args.seed,output,deadline)
        if data.discrete:raise RuntimeError('This box-feature refinement is registered for continuous GossipCop features only.')
        model_before=model_hash(engine.model);gcn=engine.teachers['gcn_base'];gcn_before=model_hash(gcn)
        teacher,teacher_manifest=load_teacher(protocol,data);teacher_before=model_hash(teacher)
        view=load_original_view(data)
        minimum=data.features.min(dim=0).values;maximum=data.features.max(dim=0).values
        write_json(output/'registered.json',dict(provenance=provenance,seed=args.seed,arms=list(ARMS),
            generator_checkpoint_sha256=baseline_receipt['final_model_sha256'],generator_normalized_state_sha256=model_before,
            teacher_ref=protocol['teacher_ref'],teacher_state_sha256=teacher_before,
            original_gcn_state_sha256=gcn_before,polish_settings=protocol['polish_settings'],
            baseline_generation='Exactly once per group, unchanged legacy generator in eval mode, validation phase, epoch8 per-case RNG',
            fixed_graph_all_arms=True,official_victim_loaded=False,TEST_queries=0),exclusive=True)
        exports=[];guard_counts=dict(accepted=0,fallback=0,empty_protected_set=0)
        with (output/'groups.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for group in protocol['valid_groups']:
                check();engine.current_group=group
                targets=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
                with torch.no_grad():graph=engine.get_perturbed_graph(targets,nb,eb,False)
                counters['baseline_generations']+=1
                capture={key:value.detach().clone() for key,value in engine.capture.items()}
                engine.capture=engine.live_capture=None
                original=data.graph.num_nodes();injected=torch.arange(original,graph.num_nodes(),device=data.device)
                x0=capture['x_atk'];adjacency_hash=tensor_sha256(capture['adj_attack'])
                affine=build_affine(view,graph,targets,capture['candidate_ids'],injected,x0)
                def gcn_forward(features):
                    check();counters['gcn_forwards']+=1
                    return checked_logits(gcn_logits(gcn,graph_with_features(graph,original,features),targets,training=False),len(targets))
                def gaga_forward(features):
                    check();counters['gaga_teacher_forwards']+=1
                    return checked_logits(teacher(affine.sequence(features)),len(targets))
                with torch.no_grad():
                    baseline_gcn=gcn_forward(x0);baseline_gaga=gaga_forward(x0)
                features={'baseline':x0}
                features['gcn_pgd'],gcn_history=projected_features(x0,minimum,maximum,gcn_forward,20,.02,check,counters)
                features['gaga_pgd'],gaga_history=projected_features(x0,minimum,maximum,gaga_forward,20,.02,check,counters)
                random_seed=case_seed(args.seed,'feature-polish-random',20,group)
                features['random']=random_features(x0,minimum,maximum,random_seed,20,.02)
                logits={'baseline':dict(gcn=baseline_gcn,gaga_teacher=baseline_gaga)}
                with torch.no_grad():
                    for arm in ('gcn_pgd','gaga_pgd','random'):
                        logits[arm]=dict(gcn=gcn_forward(features[arm]),gaga_teacher=gaga_forward(features[arm]))
                accepted,guard=preserves_baseline_gcn_zeros(baseline_gcn,logits['gaga_pgd']['gcn'])
                guard_counts['accepted' if accepted else 'fallback']+=1
                guard_counts['empty_protected_set']+=int(guard['empty_protected_set'])
                features['gaga_guarded']=features['gaga_pgd'] if accepted else x0
                logits['gaga_guarded']=logits['gaga_pgd'] if accepted else logits['baseline']
                guard.update(accepted=accepted,action='accept entire terminal GAGA-PGD candidate' if accepted else 'fallback entire group to baseline',
                    uses_evaluation_labels=False,uses_official_victim=False,step_selection=False)
                if tensor_sha256(capture['adj_attack'])!=adjacency_hash or graph.num_nodes()!=original+nb:
                    raise RuntimeError('Fixed perturbation topology/budget changed while refining features.')
                statistics={arm:feature_statistics(features[arm],x0,minimum,maximum) for arm in ARMS}
                payload=dict(capture={key:value.cpu() for key,value in capture.items()},
                    features={arm:features[arm].detach().cpu() for arm in ARMS})
                path=output/'captures'/('group_%d.pt'%group)
                with path.open('xb') as handle:torch.save(payload,handle)
                export=dict(group=group,path=path.relative_to(output).as_posix(),sha256=sha256(path))
                exports.append(export)
                own_logits={arm:{name:value.cpu().tolist() for name,value in logits[arm].items()} for arm in ARMS}
                own_losses={arm:{name:float(F.softplus(value[:,1]-value[:,0]).mean()) for name,value in logits[arm].items()} for arm in ARMS}
                row=dict(group=group,seed=args.seed,node_ids=targets.cpu().tolist(),labels=data.labels[targets].cpu().tolist(),
                    targets=len(targets),node_budget=nb,edge_budget=eb,actual_injected_nodes=nb,
                    actual_injected_nonself_undirected_edges=(graph.num_edges()-data.graph.num_edges()-nb)//2,
                    generation_seed=case_seed(args.seed,'validation',8,group),random_seed=random_seed,
                    graph_capture=export,adjacency_content_sha256=adjacency_hash,feature_statistics=statistics,
                    final_own_logits=own_logits,final_own_losses=own_losses,guard=guard,affine_trace=affine.trace,
                    counters=dict(counters))
                write_json(output/'histories'/('group_%d.json'%group),dict(group=group,gcn_pgd=gcn_history,gaga_pgd=gaga_history,
                    terminal_only=True,random_rule='One fixed Rademacher direction,20projected alpha.02steps; movement not matched to PGD'),exclusive=True)
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                if len(exports)%16==0:print('FEATURE_POLISH_GROUPS',args.seed,len(exports),flush=True)
                del graph,affine,capture,x0,features,logits,payload,baseline_gcn,baseline_gaga
        if (len(exports)!=128 or engine.generator_forwards!=128 or counters['baseline_generations']!=128 or
                counters['gcn_forwards']!=3072 or counters['gaga_teacher_forwards']!=3072 or
                counters['feature_autograd_calls']!=5120 or counters['feature_pgd_steps']!=5120):
            raise RuntimeError('Frozen per-seed refinement cohort/counters differ.')
        if model_hash(engine.model)!=model_before or model_hash(gcn)!=gcn_before or model_hash(teacher)!=teacher_before:
            raise RuntimeError('Feature refinement changed generator or classifier parameters.')
        if engine.optimizer_steps or any(parameter.grad is not None for model in (gcn,teacher) for parameter in model.parameters()):
            raise RuntimeError('Unexpected generator update or frozen-classifier gradient buffers.')
        result.update(status='completed',source_commit=provenance['source_commit'],protocol_sha256=provenance['protocol_sha256'],
            arms=list(ARMS),groups=128,exports=exports,groups_jsonl_sha256=sha256(output/'groups.jsonl'),guard_counts=guard_counts,
            generator_checkpoint_sha256=baseline_receipt['final_model_sha256'],teacher_ref=protocol['teacher_ref'],
            generator_state_sha256=model_before,original_gcn_state_sha256=gcn_before,teacher_state_sha256=teacher_before,
            refinement='Terminal20 sign-PGD on injected normalized continuous features only; one shared hard graph per group',
            official_victim_loaded=False)
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        if engine is not None:result['recorded_engine_generator_forwards']=engine.generator_forwards
        result.update(counters=counters,elapsed_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu_started)
        write_json(output/'result.json',result,exclusive=True);print('FEATURE_POLISH_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
