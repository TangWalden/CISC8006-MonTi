"""Uniform official-GAGA development evaluation after all refinements finish.

The attacked original-GCN channel reuses the exact frozen refinement logits;
only its clean graph is queried here. Official GAGA always uses the canonical
hard-graph sequence updater, never the affine optimization shortcut.
"""
import argparse
import json
import pathlib
import time
import traceback

from course.feature_polish.common import ROOT,ARMS,private_output,sha256,validate_protocol,write_json


def read_rows(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--matrix',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=private_output(args.output,create=True)
    start,cpu_start=time.monotonic(),time.process_time()
    counts=dict(official_gaga_forwards=0,original_gcn_clean_forwards=0,
        attacked_gcn_rows_reused=0,hard_graph_reconstructions=0,canonical_sequence_updates=0,generator_forwards=0)
    result=dict(status='failed',TEST_queries=0,experimental_API_calls=0,counts=counts)
    try:
        p,provenance=validate_protocol(args.protocol);matrix=private_output(args.matrix)
        marker_path=pathlib.Path(p['stage_root'])/'all3_refinements_complete.json'
        marker=json.loads(marker_path.read_text(encoding='utf-8'))
        runs={}
        for seed in p['seeds']:
            directory=matrix/('seed%d'%seed)
            r=json.loads((directory/'result.json').read_text(encoding='utf-8'))
            if (r['status']!='completed' or r['seed']!=seed or r['groups']!=128 or
                    marker['refinement_results'][str(seed)]!=sha256(directory/'result.json')):
                raise RuntimeError('All three complete refinement jobs are required before victim loading.')
            registered=json.loads((directory/'registered.json').read_text(encoding='utf-8'))
            if registered['provenance']!=provenance or tuple(registered['arms'])!=ARMS:
                raise RuntimeError('Refinement source/protocol/arms differ.')
            if sha256(directory/'groups.jsonl')!=r['groups_jsonl_sha256']:
                raise RuntimeError('Retained refinement rows changed.')
            rows=read_rows(directory/'groups.jsonl');indexed={row['group']:row for row in rows}
            if len(rows)!=128 or set(indexed)!=set(p['valid_groups']) or len(r['exports'])!=128:
                raise RuntimeError('Refinement group denominator differs.')
            if {export['group'] for export in r['exports']}!=set(p['valid_groups']):
                raise RuntimeError('Refinement export group coverage differs.')
            for export in r['exports']:
                if indexed[export['group']]['graph_capture']!=export:
                    raise RuntimeError('Refinement rows and capture receipts disagree.')
                relative=pathlib.Path(export['path'])
                if relative.is_absolute() or '..' in relative.parts or sha256(directory/relative)!=export['sha256']:
                    raise RuntimeError('A shared graph/feature capture changed.')
            runs[seed]=(directory,r,indexed)
        import dgl
        import numpy as np
        import torch
        from utils.utils import get_n_ids,set_seed
        from course.dual_surrogate.runtime import load_data,load_original_view,load_classifier,gcn_logits,checked_logits,transitions
        from course.dual_surrogate.sequence_adapter import updated_sequences
        from course.source_role_fix.saved_teacher_diagnostic import reconstruct
        from course.feature_polish.refine import tensor_sha256,graph_with_features,preserves_baseline_gcn_zeros
        torch.set_num_threads(8);set_seed(20261104);dgl.seed(20261104);dgl.random.seed(20261104)
        deadline=start+p['evaluation_wall_seconds']
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('Final development evaluation deadline reached.')
        data=load_data(p,20261104);view=load_original_view(data)
        gcn=load_classifier('gcn_base',data)
        # No official victim constructor/checkpoint load occurs before the
        # immutable three-job/capture gates above have all passed.
        victim=load_classifier('gaga_victim',data)
        sequence=np.load(ROOT/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',allow_pickle=False,mmap_mode='r')
        write_json(out/'registered.json',dict(provenance=provenance,all3_refinements_complete_before_victim_load=True,
            marker_sha256=sha256(marker_path),final_TEST_used=False,official_victim_channel='Canonical hard sequences only',
            attacked_gcn_channel='Exact stored original-GCN refinement logits; no fresh attacked GCN queries',
            refinement_result_sha256s={str(seed):sha256(directory/'result.json') for seed,(directory,_,_) in runs.items()}),exclusive=True)
        minimum=data.features.min(dim=0).values;maximum=data.features.max(dim=0).values
        clean={}
        with torch.no_grad(),(out/'clean_predictions.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for group in p['valid_groups']:
                check();ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                ga=gcn_logits(gcn,data.graph,ids);counts['original_gcn_clean_forwards']+=1
                batch=np.array(sequence[ids.cpu().numpy()],copy=True)
                gb=checked_logits(victim(torch.from_numpy(batch).to(data.device)),len(ids));counts['official_gaga_forwards']+=1
                labels=data.labels[ids].cpu();clean[group]=(ga.argmax(1).cpu(),gb.argmax(1).cpu())
                stream.write(json.dumps(dict(group=group,node_ids=ids.cpu().tolist(),labels=labels.tolist(),
                    gcn_logits=ga.cpu().tolist(),gaga_logits=gb.cpu().tolist()),allow_nan=False)+'\n')
        summaries=[]
        for seed in p['seeds']:
            directory,receipt,rows=runs[seed]
            streams={arm:(out/('%s_seed%d_predictions.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n') for arm in ARMS}
            totals={arm:{name:dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0) for name in ('gcn','gaga')} for arm in ARMS}
            metadata={arm:[] for arm in ARMS}
            try:
                with torch.no_grad():
                    for group in p['valid_groups']:
                        check();saved=rows[group];path=directory/saved['graph_capture']['path']
                        if sha256(path)!=saved['graph_capture']['sha256']:
                            raise RuntimeError('A graph capture changed during evaluation.')
                        payload=torch.load(path,map_location='cpu')
                        if set(payload)!=set(('capture','features')) or set(payload['features'])!=set(ARMS):
                            raise RuntimeError('Capture schema or feature branch coverage differs.')
                        base=payload['capture'];saved_for_graph=dict(saved,
                            actual_injected_edges=saved['actual_injected_nonself_undirected_edges'])
                        clean_for_graph=dict(group=group,node_ids=saved['node_ids'],labels=saved['labels'])
                        graph,ids,candidates,injected=reconstruct(data,base,saved_for_graph,clean_for_graph)
                        counts['hard_graph_reconstructions']+=1
                        original=data.graph.num_nodes();x0=base['x_atk'].to(data.device)
                        if tensor_sha256(base['adj_attack'])!=saved['adjacency_content_sha256']:
                            raise RuntimeError('Shared hard adjacency identity differs.')
                        if not torch.equal(payload['features']['baseline'],base['x_atk']):
                            raise RuntimeError('Baseline feature branch changed from the shared generator output.')
                        allowed,guard=preserves_baseline_gcn_zeros(
                            torch.tensor(saved['final_own_logits']['baseline']['gcn']),
                            torch.tensor(saved['final_own_logits']['gaga_pgd']['gcn']))
                        if allowed!=saved['guard']['accepted']:
                            raise RuntimeError('Final guard does not recount from its original-GCN decisions.')
                        guarded_source='gaga_pgd' if allowed else 'baseline'
                        if saved['final_own_logits']['gaga_guarded']!=saved['final_own_logits'][guarded_source]:
                            raise RuntimeError('Guarded stored logits differ from their selected whole-group source.')
                        if not torch.equal(payload['features']['gaga_guarded'],payload['features'][guarded_source]):
                            raise RuntimeError('Guarded features differ from their registered whole-group choice.')
                        labels=data.labels[ids].cpu()
                        for arm in ARMS:
                            features=payload['features'][arm].to(data.device)
                            if features.shape!=x0.shape or features.dtype!=x0.dtype or not bool(((features>=minimum)&(features<=maximum)).all()):
                                raise RuntimeError('A retained feature branch violates the original box or shape.')
                            if tensor_sha256(features)!=saved['feature_statistics'][arm]['content_sha256']:
                                raise RuntimeError('Refined feature content differs from its receipt.')
                            local=graph_with_features(graph,original,features)
                            seq=updated_sequences(view,local,ids,candidates,injected,training=False)
                            counts['canonical_sequence_updates']+=1
                            logits=checked_logits(victim(seq),len(ids));counts['official_gaga_forwards']+=1
                            stored=torch.tensor(saved['final_own_logits'][arm]['gcn'],dtype=torch.float32)
                            checked_logits(stored,len(ids));counts['attacked_gcn_rows_reused']+=1
                            prediction=(stored.argmax(1),logits.argmax(1).cpu())
                            row=dict(group=group,seed=seed,arm=arm,node_ids=ids.cpu().tolist(),labels=labels.tolist(),
                                gcn_logits=stored.tolist(),gaga_logits=logits.cpu().tolist(),gcn_logits_reused=True,
                                node_budget=saved['node_budget'],edge_budget=saved['edge_budget'],
                                actual_injected_nodes=len(injected),actual_injected_edges=(local.num_edges()-data.graph.num_edges()-len(injected))//2,
                                common_capture_sha256=saved['graph_capture']['sha256'],
                                hard_adjacency_content_sha256=saved['adjacency_content_sha256'],
                                injected_feature_content_sha256=saved['feature_statistics'][arm]['content_sha256'])
                            for index,name in enumerate(('gcn','gaga')):
                                transition=transitions(labels,clean[group][index],prediction[index]);row[name+'_transitions']=transition
                                for key,value in transition.items():totals[arm][name][key]+=value
                            streams[arm].write(json.dumps(row,allow_nan=False)+'\n')
                            metadata[arm].append({key:value for key,value in row.items() if key not in ('node_ids','labels','gcn_logits','gaga_logits')})
                        del graph,payload,base,local,seq,features
            finally:
                for stream in streams.values():stream.close()
            for arm in ARMS:
                if len(metadata[arm])!=128 or any(v['occurrences']!=1091 for v in totals[arm].values()):
                    raise RuntimeError('Complete fixed validation denominator differs.')
                nodes=sum(row['actual_injected_nodes'] for row in metadata[arm]);edges=sum(row['actual_injected_edges'] for row in metadata[arm])
                if (nodes,edges)!=(7776,358491):raise RuntimeError('Fixed physical scenario budgets differ.')
                summary=dict(arm=arm,seed=seed,groups=128,totals=totals[arm],
                    gcn_mr_percent=100*totals[arm]['gcn']['attacked_wrong']/1091,
                    gaga_mr_percent=100*totals[arm]['gaga']['attacked_wrong']/1091,
                    injected_nodes=nodes,injected_edges=edges,stored_gcn_channel=True)
                write_json(out/('%s_seed%d_groups.json'%(arm,seed)),metadata[arm],exclusive=True)
                summaries.append(summary);print('FEATURE_POLISH_EVAL_RUN',json.dumps(summary),flush=True)
        expected=dict(official_gaga_forwards=2048,original_gcn_clean_forwards=128,
            attacked_gcn_rows_reused=1920,hard_graph_reconstructions=384,canonical_sequence_updates=1920,generator_forwards=0)
        if counts!=expected:raise RuntimeError('Complete evaluation counter totals differ.')
        result.update(status='completed',summaries=summaries,provenance=provenance,
            final_TEST_used=False,gcn_channel='Clean queried once; attacked predictions reuse stored refinement logits',
            official_gaga_channel='Fresh canonical hard-graph inference, never optimization affine shortcut')
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(counts=counts,elapsed_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu_start)
        write_json(out/'result.json',result,exclusive=True);print('FEATURE_POLISH_EVAL_END',json.dumps(result,allow_nan=False),flush=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
