"""Four fixed TRAIN-only construction/derivative checks; no efficacy selection."""
import argparse
import copy
import hashlib
import json
import math
import pathlib
import random
import signal
import time
import traceback

from runtime import (TRAIN_PREFLIGHT,verify_protocol,check_device,load_engine,
    load_own_teacher,make_case,canonical,softplus,source_state,assert_frozen,
    tensor_hash,data_hashes,write_json,check_permuted)


COUNTER_CAPS = dict(generations=4,generator_completed_forwards=4,
    proposal_generation_GCN_forwards=0,native_canonical_updates=8,
    own_GAGA_float32_forwards=16,own_GCN_float32_forwards=8,
    own_GAGA_float64_forwards=12,own_GCN_float64_forwards=12,
    autograd_calls=16,GCN_block_topologies=4,classifier_float64_clones=2,
    official_GAGA_forwards=0,optimizer_updates=0,API_calls=0,TEST_queries=0)


def finite_difference_check(role,case,model32,model64,settings,counters):
    """Same fixed graph and raw-feature direction for F32/F64/central FD.

    Float64 copies retain the actual frozen weights and fixed GCN blocks; no
    model constructor or optimizer is invoked. The GAGA double check uses the
    affine sequence operator, whose baseline/cyclic-swap agreement with native
    float32 aggregation is checked separately; it is not independent native
    float64 aggregation derivative validation. The derivative of Softplus is
    checked directly, including when a direction is numerically uninformative.
    A zero/small directional derivative is reported rather than replaced with
    a raw-margin objective. Raw probes are numerical construction checks, not
    legal accepted feature proposals or measurements of attack efficacy.
    """
    import torch
    from course.dual_surrogate.runtime import checked_logits
    x0=case['x0'];count=len(case['ids'])
    encoded=json.dumps([3,int(case['info']['group']),role,'assignment-preflight-direction-v1'],
        separators=(',',':')).encode('ascii')
    key=hashlib.sha256(encoded).hexdigest();rng=random.Random(int(key,16))
    values=[-1. if rng.getrandbits(1)==0 else 1. for _ in range(x0.numel())]
    direction=torch.tensor(values,dtype=torch.float64,device=x0.device).reshape(x0.shape)/math.sqrt(x0.numel())
    def forward(model,x):
        dtype='float64' if x.dtype==torch.float64 else 'float32'
        counters['own_'+role.upper()+'_'+dtype+'_forwards']+=1
        value=model(case['affine'].sequence(x)) if role=='gaga' else case['gcn_forward'](model,x)
        return checked_logits(value,count)
    center32=x0.detach().clone().requires_grad_(True)
    logits32=forward(model32,center32);loss32=softplus(logits32)
    counters['autograd_calls']+=1
    gradient32,=torch.autograd.grad(loss32,center32)
    center64=x0.detach().double().requires_grad_(True)
    logits64=forward(model64,center64);loss64=softplus(logits64)
    counters['autograd_calls']+=1
    gradient64,=torch.autograd.grad(loss64,center64)
    if not bool(torch.isfinite(gradient32).all() and torch.isfinite(gradient64).all()):
        raise RuntimeError('Nonfinite '+role+' feature derivative in fixed TRAIN check.')
    auto32=float((gradient32.double()*direction).sum())
    auto64=float((gradient64*direction).sum())
    h=settings['FD_epsilon']
    with torch.no_grad():
        plus=float(softplus(forward(model64,center64.detach()+h*direction)))
        minus=float(softplus(forward(model64,center64.detach()-h*direction)))
    estimate=(plus-minus)/(2*h)
    tolerance=settings['FD_atol']+settings['FD_rtol']*abs(auto64)
    error=abs(auto64-estimate);precision_error=abs(auto32-auto64)
    receipt=dict(role=role,objective='mean Softplus(logit1-logit0)',gradient_point='baseline_x0',
        direction_key_sha256=key,direction_tensor_sha256=tensor_hash(direction),
        direction_raw_l2=float(direction.norm()),epsilon=h,
        float32_loss=float(loss32.detach()),float64_loss=float(loss64.detach()),
        float32_directional_autograd=auto32,float64_directional_autograd=auto64,
        plus_float64_loss=plus,minus_float64_loss=minus,central_float64_difference=estimate,
        finite_difference_absolute_error=error,float32_float64_derivative_absolute_error=precision_error,
        original_atol=settings['FD_atol'],original_rtol=settings['FD_rtol'],acceptance_tolerance=tolerance,
        finite_difference_passed=bool(error<=tolerance),float32_float64_derivative_passed=bool(precision_error<=tolerance),
        direction_informative=bool(abs(auto64)>settings['FD_atol']),
        zero_directional_derivative=bool(auto64==0.),
        gradient_float32_linf=float(gradient32.detach().abs().max()),gradient_float64_linf=float(gradient64.detach().abs().max()),
        gradient_float32_sha256=tensor_hash(gradient32),gradient_float64_sha256=tensor_hash(gradient64),
        frozen_GCN_blocks=(case['block_signature'] if role=='gcn' else None),
        float64_input_map=('Existing fixed-topology affine.sequence cast to float64; native F32 aggregation parity is checked separately'
            if role=='gaga' else 'Same sampled fixed DGL blocks and source-NID feature gather, cast to float64'),
        limitation='A small directional derivative is an uninformative construction check, not feature-assignment efficacy evidence.')
    return logits32.detach(),receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    start,cpu_start=time.monotonic(),time.process_time();deadline=None
    counters={key:0 for key in COUNTER_CAPS};counters.update(dataset_loads=0,generator_models_loaded=0,
        own_GAGA_models_loaded=0,proposal_GCN_models_loaded=0)
    result=dict(status='failed',counters=counters,counter_caps=COUNTER_CAPS,cases=[],official_victim_loaded=False)
    engine=data=view=teacher=gcn=None;handles=[]
    def check():
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('Frozen TRAIN construction deadline reached.')
    def stop(signum,frame):raise TimeoutError('TRAIN construction guardian/inner signal received.')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop)
    try:
        p,provenance=verify_protocol(args.protocol);result['provenance']=provenance
        if pathlib.Path(p['asset_root'])/'runs' not in out.parents:raise RuntimeError('TRAIN receipts must remain in ignored historical runs/.')
        settings=p['preflight']
        if (tuple(settings['groups'])!=TRAIN_PREFLIGHT or settings['sequence_atol']!=1e-5
                or settings['logit_atol']!=1e-4 or settings['FD_epsilon']!=1e-4
                or settings['FD_atol']!=1e-6 or settings['FD_rtol']!=5e-3):
            raise RuntimeError('The fixed TRAIN groups or construction tolerances changed.')
        deadline=start+p['resource']['preflight_inner_seconds'];signal.alarm(max(1,int(deadline-time.monotonic())))
        result['device']=check_device(p)
        import dgl
        import torch
        from utils.utils import set_seed
        from course.dual_surrogate.runtime import load_original_view,checked_logits
        set_seed(3);dgl.seed(3);dgl.random.seed(3)
        engine,data,receipt=load_engine(p,3,out,deadline)
        counters['dataset_loads']+=1;counters['generator_models_loaded']+=1;counters['proposal_GCN_models_loaded']+=1
        if set(engine.teachers)!={'gcn_base'} or engine.victim_model is not None:
            raise RuntimeError('TRAIN construction loaded an unexpected teacher/victim.')
        if not set(TRAIN_PREFLIGHT)<=set(map(int,p['train_groups'])):
            raise RuntimeError('A preflight group is outside registered original TRAIN.')
        view=load_original_view(data);gcn=engine.teachers['gcn_base'];teacher,teacher_manifest=load_own_teacher(p,data)
        counters['own_GAGA_models_loaded']+=1
        hashes=dict(generator=source_state(engine.model),gcn=source_state(gcn),own_teacher=source_state(teacher))
        original=data_hashes(data,view);result['immutable_content_before']=original
        double_models={}
        for role,model in (('gaga',teacher),('gcn',gcn)):
            clone=copy.deepcopy(model)
            if source_state(clone)!=hashes['own_teacher' if role=='gaga' else 'gcn']:
                raise RuntimeError('A numerical clone differs before dtype conversion.')
            double_models[role]=clone.double().eval();counters['classifier_float64_clones']+=1
        result['model_states_before']=hashes
        result['float64_clone_states']={role:source_state(model) for role,model in double_models.items()}
        def completed(model,inputs,outputs):counters['generator_completed_forwards']+=1
        handles.append(engine.model.register_forward_hook(completed))
        def forbidden(model,inputs):
            counters['proposal_generation_GCN_forwards']+=1
            raise RuntimeError('Baseline generation unexpectedly queried GCN instead of h_GCN cache.')
        forbidden_handle=gcn.register_forward_pre_hook(forbidden)
        for group in TRAIN_PREFLIGHT:
            check();entry=dict(group=group,status='started');result['cases'].append(entry)
            case=make_case(p,engine,data,view,3,group,phase='train')
            counters['generations']+=1;counters['native_canonical_updates']+=1;counters['GCN_block_topologies']+=1
            forbidden_handle.remove()
            x0=case['x0'];identity=torch.arange(len(x0),dtype=torch.long,device=x0.device)
            permutation=torch.roll(identity,shifts=1) if len(identity)>1 else identity
            permuted=x0.index_select(0,permutation).detach()
            entry.update(targets=len(case['ids']),injected_nodes=len(x0),feature_width=x0.shape[1],
                node_budget=case['info']['node_budget'],edge_budget=case['info']['edge_budget'],
                baseline_feature_sha256=tensor_hash(x0),permutation=permutation.cpu().tolist(),
                permutation_invariants=check_permuted(case,permuted,permutation),block_signature=case['block_signature'])
            with torch.no_grad():
                native0=case['baseline_sequence'];native1=canonical(data,view,case['graph'],case['ids'],case['candidates'],case['injected'],permuted)
                counters['native_canonical_updates']+=1
                affine0=case['affine'].sequence(x0);affine1=case['affine'].sequence(permuted)
                sequence_errors=[float((native-affine).abs().max()) for native,affine in ((native0,affine0),(native1,affine1))]
                native_logits=[]
                for sequence in (native0,native1):
                    counters['own_GAGA_float32_forwards']+=1
                    native_logits.append(checked_logits(teacher(sequence),len(case['ids'])))
                counters['own_GAGA_float32_forwards']+=1
                affine_permuted_logits=checked_logits(teacher(affine1),len(case['ids']))
                counters['own_GCN_float32_forwards']+=1
                permuted_gcn=checked_logits(case['gcn_forward'](gcn,permuted),len(case['ids']))
            entry['derivatives']={}
            for role,model in (('gaga',teacher),('gcn',gcn)):
                center_logits,derivative=finite_difference_check(role,case,model,double_models[role],settings,counters)
                entry['derivatives'][role]=derivative
                if role=='gaga':affine_center_logits=center_logits
                else:entry['GCN_baseline_and_permuted_argmax_equal']=bool(center_logits.argmax(1).equal(permuted_gcn.argmax(1)))
            logit_errors=[float((native_logits[0]-affine_center_logits).abs().max()),
                float((native_logits[1]-affine_permuted_logits).abs().max())]
            entry.update(sequence_absolute_errors=sequence_errors,logit_absolute_errors=logit_errors,
                native_sequence_hashes=[tensor_hash(native0),tensor_hash(native1)],
                affine_sequence_hashes=[tensor_hash(affine0),tensor_hash(affine1)])
            passed=(max(sequence_errors)<=settings['sequence_atol'] and max(logit_errors)<=settings['logit_atol']
                and all(value['finite_difference_passed'] and value['float32_float64_derivative_passed'] for value in entry['derivatives'].values()))
            entry['status']='passed' if passed else 'failed'
            if not passed:raise RuntimeError('A fixed TRAIN sequence/logit/Softplus derivative construction check failed.')
            assert_frozen(gcn,hashes['gcn']);assert_frozen(teacher,hashes['own_teacher'])
            if source_state(engine.model)!=hashes['generator'] or engine.optimizer_steps:
                raise RuntimeError('TRAIN construction changed the frozen generator.')
            forbidden_handle=gcn.register_forward_pre_hook(forbidden)
            del case,x0,native0,native1,affine0,affine1,native_logits,affine_permuted_logits,permuted_gcn
        forbidden_handle.remove()
        if any(counters[key]!=value for key,value in COUNTER_CAPS.items()):
            raise RuntimeError('Complete TRAIN construction counters differ from the source-frozen caps.')
        result['immutable_content_after']=data_hashes(data,view)
        result['model_states_after']=dict(generator=source_state(engine.model),gcn=source_state(gcn),own_teacher=source_state(teacher))
        if result['immutable_content_after']!=original or result['model_states_after']!=hashes:
            raise RuntimeError('TRAIN construction changed source features/caches/classifier state.')
        if result['float64_clone_states']!={role:source_state(model) for role,model in double_models.items()}:
            raise RuntimeError('Numerical derivative checking changed a frozen float64 clone.')
        result.update(status='passed',no_VALID_or_TEST_queries=True,settings=settings,
            actual_teacher_manifest_sha256=p['own_teacher_manifest']['sha256'],baseline_terminal_receipt=receipt,
            limitation='Four fixed TRAIN construction cases; no claim of assignment efficacy, transfer or optimal nonlinear decisions.')
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        signal.alarm(0)
        for handle in handles:handle.remove()
        result.update(counters=counters,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu_start)
        write_json(out/'result.json',result)
        print('ASSIGNMENT_PREFLIGHT_END',json.dumps(dict(status=result['status'],counters=counters)),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
