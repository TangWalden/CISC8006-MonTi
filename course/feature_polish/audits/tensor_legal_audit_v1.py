"""Independent CPU-only structural audit of verified saved attack captures.

Allowed scientific reads: the frozen original graph's raw features; labels only
at the original TRAIN indices; that index array; saved fresh-teacher TRAIN JSON;
384 byte-verified capture tensors and JSON receipts. Never load a classifier
checkpoint, construct/call a model, optimize, inspect non-TRAIN truth labels,
query the official victim/API, or allocate a CUDA tensor.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import time
import traceback

ARMS=('baseline','gcn_pgd','gaga_pgd','gaga_guarded','random')
SEEDS=(3,4,5)
FLOAT_ATOL=2e-6
FLOAT_RTOL=2e-6
STAT_ATOL=5e-5
STAT_RTOL=2e-5
MOVEMENT_TOL=1e-5


def digest(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


class Checks:
    def __init__(self,root):self.root=root;self.files={};self.counts=collections.Counter();self.warnings=[]
    def check(self,value,name):
        self.counts[name]+=1
        if not value:raise AssertionError('Structural audit failed: '+name)
    def path(self,name):
        raw=pathlib.Path(name);p=raw.resolve() if raw.is_absolute() else (self.root/raw).resolve()
        self.check(self.root in p.parents and p.is_file(),'allowed_repository_file')
        return p
    def file(self,name,wanted=None):
        p=self.path(name);actual=digest(p);key=p.relative_to(self.root).as_posix()
        self.check(self.files.setdefault(key,actual)==actual,'input_bytes_unchanged')
        if wanted is not None:self.check(actual==wanted,'recorded_byte_sha256')
        return p
    def json(self,name,wanted=None):return json.loads(self.file(name,wanted).read_text(encoding='utf-8'))
    def rows(self,name,wanted=None):
        with self.file(name,wanted).open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
    def close(self,x,y,name,atol=STAT_ATOL,rtol=STAT_RTOL):
        self.check(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=atol+rtol*abs(y),name)


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    header=json.dumps([str(value.dtype),list(value.shape)],separators=(',',':')).encode()
    return hashlib.sha256(len(header).to_bytes(8,'big')+header+value.numpy().tobytes()).hexdigest()


def random_key(seed,group):
    text='%d|feature-polish-random|20|%d|dual-surrogate-v1'%(seed,group)
    return int(hashlib.sha256(text.encode('ascii')).hexdigest()[:8],16)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.');parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--matrix',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--audit-source-sha256',required=True)
    args=parser.parse_args();root=pathlib.Path(args.root).resolve();out=pathlib.Path(args.output)
    out=out.resolve() if out.is_absolute() else (root/out).resolve()
    if (root/'runs').resolve() not in out.parents:raise RuntimeError('Fresh audit output must stay in ignored runs/.')
    out.mkdir(parents=True,exist_ok=False);started,cpu=time.monotonic(),time.process_time();c=Checks(root)
    counters=dict(original_graph_loads=0,train_index_array_loads=0,verified_capture_tensor_loads=0,
        classifier_checkpoint_deserializations=0,classifier_constructions=0,model_forwards=0,
        optimizer_updates=0,nonTRAIN_label_indexes=0,API_calls=0,TEST_queries=0)
    result=dict(status='failed',independent_of_method_implementation_authorship=True,human_verification=False,
        GPU_hours=0,counters=counters)
    records=[]
    try:
        own=digest(__file__);c.check(own==args.audit_source_sha256,'externally_frozen_structural_audit_source')
        write(out/'registered_scope.json',dict(audit_source_sha256=own,expected_protocol_sha256=args.protocol_sha256,
            registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),new_output_reads_started=False,
            allowed_scientific_reads=['Frozen graph raw features; truth indexed only at original TRAIN IDs, all other truth discarded',
                'Frozen original TRAIN-index array and saved teacher TRAIN prediction JSON',
                'Byte-verified saved capture tensors and saved JSON metadata for all384cases'],
            forbidden_operations=['Classifier weight deserialization, construction or inference','CUDA tensor allocation',
                'Any VALID/test truth-label indexing','Victim/API queries or model/feature updates'],
            tolerance_registration=dict(random_feature_atol=FLOAT_ATOL,random_feature_rtol=FLOAT_RTOL,
                metadata_statistic_atol=STAT_ATOL,metadata_statistic_rtol=STAT_RTOL,normalized_movement_atol=MOVEMENT_TOL)))
        p=c.json(args.protocol,args.protocol_sha256);matrix=pathlib.Path(args.matrix)
        matrix=matrix.resolve() if matrix.is_absolute() else (root/matrix).resolve()
        c.check(root in matrix.parents and matrix.is_dir(),'matrix_boundary')
        c.check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['polish_settings']==dict(steps=20,alpha=.02),'fixed_method_scope')
        registered_inputs={row['path']:row['sha256'] for row in p['inputs']}
        graph_path=c.file(p['data']['graph'],registered_inputs[p['data']['graph']])
        index_path=c.file(p['data']['train_indices'],registered_inputs[p['data']['train_indices']])
        captures=[];all_rows={}
        for seed in SEEDS:
            directory=matrix/('seed%d'%seed);receipt=c.json(directory/'result.json')
            c.check(receipt['status']=='completed' and receipt['seed']==seed and receipt['groups']==128,'refinement_complete_before_tensor_reads')
            rows=c.rows(directory/'groups.jsonl',receipt['groups_jsonl_sha256']);indexed={row['group']:row for row in rows}
            c.check(len(indexed)==len(rows)==128 and set(indexed)==set(p['valid_groups']),'all_fixed_groups_before_tensor_reads')
            exports={row['group']:row for row in receipt['exports']}
            c.check(len(exports)==len(receipt['exports'])==128 and set(exports)==set(indexed),'capture_manifest_cover')
            for group in p['valid_groups']:
                export=exports[group];c.check(indexed[group]['graph_capture']==export,'capture_row_manifest_binding')
                relative=pathlib.Path(export['path']);c.check(not relative.is_absolute() and '..' not in relative.parts,'safe_capture_relative_path')
                capture=c.file(directory/relative,export['sha256'])
                captures.append(dict(seed=seed,group=group,path=capture,sha256=export['sha256'],row=indexed[group]))
            all_rows[seed]=indexed
        c.check(len(captures)==384,'all_capture_bytes_verified_before_deserialization')
        teacher_manifest=c.json(p['teacher_ref']['path'],p['teacher_ref']['sha256'])
        teacher_dir=c.path(p['teacher_ref']['path']).parent
        teacher_train=c.rows(teacher_dir/'raw/final_in_sample_train_predictions.jsonl')
        write(out/'authorized_input_manifest.json',[dict(path=name,sha256=value) for name,value in sorted(c.files.items())])
        # All permitted scientific files have been named/hashed before ML-library
        # imports or any graph/array/capture deserialization below.
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU structural audit requires empty CUDA visibility.')
        import dgl
        import numpy as np
        import torch
        torch.set_num_threads(8)
        c.check(not torch.cuda.is_available() and torch.cuda.device_count()==0 and torch._C._cuda_getDeviceCount()==0,'cpu_zero_visible_cuda_devices')
        c.check(torch.__version__=='1.13.1+cu117' and np.__version__=='1.24.4' and dgl.__version__.startswith('1.1.2'),'locked_libraries')
        graphs,_=dgl.load_graphs(str(graph_path));counters['original_graph_loads']+=1
        original=graphs[0].to('cpu');del graphs
        values=np.load(index_path,allow_pickle=False).reshape(-1);counters['train_index_array_loads']+=1
        ids_np=values.astype(np.int64)
        c.check(np.array_equal(values.astype(np.float64),ids_np.astype(np.float64)) and len(ids_np)==len(set(ids_np.tolist()))==6595,'exact_original_train_ids')
        train_ids=torch.tensor(ids_np,dtype=torch.long)
        all_truth=original.ndata.pop('y');train_truth=all_truth[train_ids].to(torch.long).clone();del all_truth
        train_set=set(ids_np.tolist());truth={node:int(value) for node,value in zip(ids_np.tolist(),train_truth.tolist())}
        expected_ids=[node for row in teacher_train for node in row['node_ids']]
        c.check(len(expected_ids)==len(set(expected_ids))==6595 and set(expected_ids)==train_set,'teacher_only_full_train_truth_cover')
        for row in teacher_train:
            c.check(row['labels']==[truth[node] for node in row['node_ids']],'saved_teacher_train_labels_match_original_train_only')
        raw=original.ndata['x'];n,width=raw.shape
        c.check(raw.dtype==torch.float32 and width==768 and bool(torch.isfinite(raw).all()),'original_raw_feature_scope')
        minimum=raw.min(dim=0).values;maximum=raw.max(dim=0).values;span=maximum-minimum;movable=span>0
        safe=torch.where(movable,span,torch.ones_like(span));zero=~movable
        box_hash=dict(minimum_sha256=tensor_digest(minimum),maximum_sha256=tensor_digest(maximum),zero_width_dimensions=int(zero.sum()))
        c.check(bool(movable.any()),'nonconstant_feature_space')
        for entry in captures:
            # Rehash directly before the allowed CPU-only torch.load operation.
            c.file(entry['path'],entry['sha256']);payload=torch.load(entry['path'],map_location='cpu')
            counters['verified_capture_tensor_loads']+=1;row=entry['row'];group=entry['group'];seed=entry['seed']
            c.check(set(payload)=={'capture','features'} and set(payload['features'])==set(ARMS),'strict_saved_tensor_schema')
            capture=payload['capture'];c.check(set(capture)=={'x_atk','adj_attack','target_ids','candidate_ids'},'strict_original_capture_schema')
            x0,adj,targets,candidates=(capture[k] for k in ('x_atk','adj_attack','target_ids','candidate_ids'))
            c.check(targets.ndim==candidates.ndim==1 and targets.dtype==candidates.dtype==torch.int64,'integer_target_candidate_ids')
            target_list=targets.tolist();candidate_list=candidates.tolist();nb,eb=row['node_budget'],row['edge_budget']
            c.check(target_list==row['node_ids'] and len(target_list)==row['targets'] and len(set(target_list+candidate_list))==len(target_list)+len(candidate_list)
                and all(0<=node<n for node in target_list+candidate_list),'exact_target_order_and_distinct_original_ids')
            c.check(not set(target_list)&train_set,'validation_targets_not_teacher_train_labels')
            c.check(x0.dtype==torch.float32 and tuple(x0.shape)==(nb,width) and adj.dtype==torch.float32
                and tuple(adj.shape)==(nb,len(target_list)+len(candidate_list)+nb),'capture_shapes_dtypes')
            c.check(bool(torch.isfinite(x0).all() and torch.isfinite(adj).all()) and bool(((adj==0)|(adj==1)).all()),'finite_binary_capture_adjacency')
            boundary=len(target_list)+len(candidate_list)
            legal=torch.tril(torch.ones_like(adj,dtype=torch.bool),diagonal=boundary-1)
            c.check(not bool((adj[~legal]!=0).any()) and int(adj.count_nonzero())==eb,'original_legal_edge_space_cardinality')
            c.check(bool((adj[:,:len(target_list)].sum(1)>=1).all()),'mandatory_target_attachment_per_injection')
            src,dst=adj.nonzero(as_tuple=True);index=torch.cat((targets,candidates,torch.arange(n,n+nb,dtype=torch.long)))
            physical_src=src+n;physical_dst=index[dst]
            logical=torch.stack((torch.minimum(physical_src,physical_dst),torch.maximum(physical_src,physical_dst)),dim=1)
            c.check(not bool((physical_src==physical_dst).any()) and len(torch.unique(logical,dim=0))==eb,'logical_nonself_undirected_edges_unique')
            c.check(row['actual_injected_nodes']==nb and row['actual_injected_nonself_undirected_edges']==eb,'recorded_physical_budget_matches_capture')
            c.check(tensor_digest(adj)==row['adjacency_content_sha256'],'hard_adjacency_content_hash')
            c.check(torch.equal(payload['features']['baseline'],x0),'baseline_features_are_original_capture')
            baseline_pred=[0 if z[0]>=z[1] else 1 for z in row['final_own_logits']['baseline']['gcn']]
            candidate_pred=[0 if z[0]>=z[1] else 1 for z in row['final_own_logits']['gaga_pgd']['gcn']]
            accepted=all(a!=0 or b==0 for a,b in zip(baseline_pred,candidate_pred));alias='gaga_pgd' if accepted else 'baseline'
            c.check(accepted==row['guard']['accepted'] and torch.equal(payload['features']['gaga_guarded'],payload['features'][alias]),'actual_guarded_feature_alias')
            case=dict(seed=seed,group=group,injected_nodes=nb,nonself_undirected_edges=eb,guard_accepted=accepted,branches={})
            for arm in ARMS:
                feature=payload['features'][arm];stats=row['feature_statistics'][arm]
                c.check(feature.device.type=='cpu' and feature.dtype==torch.float32 and feature.shape==x0.shape
                    and bool(torch.isfinite(feature).all()),'actual_feature_branch_shape_device_finite')
                c.check(bool(((feature>=minimum)&(feature<=maximum)).all()),'actual_original_raw_feature_box')
                c.check(torch.equal(feature[:,zero],x0[:,zero]),'actual_zero_width_dimensions_unchanged')
                c.check(tensor_digest(feature)==stats['content_sha256'],'actual_feature_content_hash')
                delta=torch.where(movable,(feature-x0)/safe,torch.zeros_like(feature))
                linf=float(delta.abs().max());l2=float(delta.double().norm());rms=float(delta.double().square().mean().sqrt())
                c.check(linf<=.4+MOVEMENT_TOL,'actual_twenty_step_normalized_movement_bound')
                for key,value in [('normalized_displacement_linf',linf),('normalized_displacement_l2',l2),
                    ('normalized_displacement_rms',rms),('raw_displacement_linf',float((feature-x0).abs().max()))]:
                    c.close(stats[key],value,'recorded_feature_displacement_recount')
                unit=torch.where(movable,(feature-minimum)/safe,torch.zeros_like(feature));mask=movable.unsqueeze(0).expand_as(feature)
                saturated=((unit<=1e-6)|(unit>=1-1e-6))&mask;denominator=int(mask.sum())
                c.check(stats['movable_coordinates']==denominator and stats['zero_range_dimensions']==int(zero.sum()),'feature_statistic_denominators')
                c.close(stats['saturation_fraction'],int(saturated.sum())/denominator if denominator else 0.,'recorded_saturation_fraction_recount')
                case['branches'][arm]=dict(normalized_linf=linf,normalized_l2=l2,saturation_fraction=int(saturated.sum())/denominator if denominator else 0.)
            # Independent reimplementation of the fixed random-direction route.
            c.check(row['random_seed']==random_key(seed,group),'random_key_from_frozen_namespace')
            generator=torch.Generator(device='cpu');generator.manual_seed(row['random_seed'])
            direction=(torch.randint(0,2,x0.shape,generator=generator,dtype=torch.int64)*2-1).float()
            direction=torch.where(movable,direction,torch.zeros_like(direction))
            unit=torch.where(movable,(x0-minimum)/safe,torch.zeros_like(x0)).clamp(0,1)
            for _ in range(20):unit=(unit+.02*direction).clamp(0,1)
            expected=torch.where(movable,minimum+unit*span,minimum)
            expected=torch.maximum(minimum,torch.minimum(maximum,expected))
            random_actual=payload['features']['random']
            c.check(bool(torch.allclose(random_actual,expected,atol=FLOAT_ATOL,rtol=FLOAT_RTOL)),'actual_random_direction_twenty_step_reconstruction')
            case['random_reconstruction_max_absolute_error']=float((random_actual-expected).abs().max())
            records.append(case);del payload,capture,x0,adj,targets,candidates,feature,expected
        c.check(counters['verified_capture_tensor_loads']==384 and len(records)==384,'complete_saved_tensor_scope')
        c.check(all(counters[key]==0 for key in ('classifier_checkpoint_deserializations','classifier_constructions','model_forwards',
            'optimizer_updates','nonTRAIN_label_indexes','API_calls','TEST_queries')),'no_forbidden_scientific_operations')
        write(out/'private_capture_checks.json',records)
        result.update(status='passed',original_nodes=n,feature_width=width,feature_box=box_hash,
            captures=384,feature_branches=1920,registered_protocol_sha256=args.protocol_sha256,
            shared_physical_budget_per_seed=dict(injected_nodes=7776,nonself_undirected_edges=358491),
            actual_random_max_absolute_error=max(row['random_reconstruction_max_absolute_error'] for row in records),
            limitations=['Original-original arcs are not included in capture payloads; their immutability follows the frozen input hash and producer/reconstruction schema, not an independently saved full graph per case.',
                'This audit validates saved geometry/features and stored guard choices; it does not replay PGD gradients or classifier predictions.',
                'Original TRAIN label truth is checked only for the teacher; no VALID/test truth labels are indexed.',
                'Random-route comparison permits the preregistered float32 CPU/GPU arithmetic tolerance; actual terminal movement is not claimed equal across PGD/random conditions.',
                'Agent structural verification is not human review, an efficacy/novelty proof or final TEST evaluation.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(counters=counters,check_counts=dict(c.counts),actual_byte_hashed_files=len(c.files),warnings=c.warnings,
            process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started)
        write(out/'input_hashes.json',[dict(path=name,sha256=value) for name,value in sorted(c.files.items())])
        write(out/'result.json',result);print(json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
