"""Development evaluation only after all twelve terminal checkpoints exist."""
import argparse
import json
import pathlib
import time
import traceback

import dgl
import torch

from utils.utils import get_n_ids,set_seed
from course.dual_surrogate.common import ARMS,ROOT,case_seed,private_output,sha256,validate_protocol,write_json
from course.dual_surrogate.runtime import config_for,load_data,load_classifier,load_original_view,gcn_logits,fork_rng,transitions,checked_logits
from course.dual_surrogate.sequence_adapter import updated_sequences
from course.dual_surrogate.experiment import DualExperiment


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--protocol',required=True)
    parser.add_argument('--matrix',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=private_output(args.output,create=True);start=time.monotonic()
    result=dict(status='failed',test_queries=0);counts=dict(gcn_base=0,gaga_victim=0,generations=0)
    try:
        p,provenance=validate_protocol(args.protocol);matrix=private_output(args.matrix)
        runs={}
        for seed in p['seeds']:
            for arm in ARMS:
                directory=matrix/('%s_seed%d'%(arm,seed));r=json.loads((directory/'result.json').read_text())
                if r['status']!='completed' or r['optimizer_steps']!=2048 or r['epochs']!=8:
                    raise RuntimeError('All twelve fixed-epoch trainings must complete before victim loading.')
                if sha256(directory/'final_model.pt')!=r['final_model_sha256']:
                    raise RuntimeError('A terminal checkpoint changed.')
                runs[(arm,seed)]=(directory,r)
        deadline=start+p['evaluation_wall_seconds'];torch.set_num_threads(8)
        def check():
            if time.monotonic()>=deadline:raise TimeoutError('Evaluation deadline reached.')
        set_seed(20261103);dgl.seed(20261103);dgl.random.seed(20261103)
        data=load_data(p,20261103);view=load_original_view(data)
        gcn=load_classifier('gcn_base',data);victim=load_classifier('gaga_victim',data)
        original_sequence=torch.from_numpy(__import__('numpy').load(ROOT/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy'))
        # The original official victim is first deserialized only at this point.
        write_json(out/'registered.json',dict(provenance=provenance,all12_trainings_complete_before_victim_load=True,
            original_valid_groups=p['valid_groups'],final_test_used=False,
            checkpoint_sha256s={a+'_s'+str(s):r['final_model_sha256'] for (a,s),(_,r) in runs.items()}),exclusive=True)
        clean={}
        with torch.no_grad(),(out/'clean_predictions.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for group in p['valid_groups']:
                check();ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                a=gcn_logits(gcn,data.graph,ids);b=checked_logits(victim(original_sequence[ids.cpu()].to(data.device)),len(ids))
                counts['gcn_base']+=1;counts['gaga_victim']+=1
                if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise RuntimeError('Nonfinite clean logits.')
                clean[group]=(a.argmax(1).cpu(),b.argmax(1).cpu())
                stream.write(json.dumps(dict(group=group,node_ids=ids.cpu().tolist(),labels=data.labels[ids].cpu().tolist(),
                    gcn_logits=a.cpu().tolist(),gaga_logits=b.cpu().tolist()))+'\n')
        summaries=[]
        for seed in p['seeds']:
            for arm in ARMS:
                check();directory,r=runs[(arm,seed)];set_seed(seed)
                engine=DualExperiment(config_for(seed),data,p,'gcn_only',ROOT/p['teacher_directory'],out,deadline)
                engine.model.load_state_dict(torch.load(directory/'final_model.pt',map_location=data.device))
                for _ in range(8):engine.model.step()
                engine.model.eval();engine.phase='validation';engine.current_epoch=8
                totals={'gcn':dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0),
                    'gaga':dict(occurrences=0,clean_wrong=0,attacked_wrong=0,new_wrong=0,recoveries=0)}
                rows=[];exports=[]
                with torch.no_grad(),(out/('%s_seed%d_predictions.jsonl'%(arm,seed))).open('x',encoding='utf-8',newline='\n') as stream:
                    for group in p['valid_groups']:
                        check();engine.current_group=group
                        ids=torch.tensor(get_n_ids(data.target_assignment,group),device=data.device,dtype=torch.long)
                        nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
                        graph=engine.get_perturbed_graph(ids,nb,eb,False)
                        counts['generations']+=1
                        ga=gcn_logits(gcn,graph,ids)
                        atk=torch.arange(data.graph.num_nodes(),graph.num_nodes(),device=data.device)
                        seq=updated_sequences(view,graph,ids,engine.capture['candidate_ids'],atk,training=False)
                        gb=checked_logits(victim(seq),len(ids));counts['gcn_base']+=1;counts['gaga_victim']+=1
                        if not bool(torch.isfinite(ga).all() and torch.isfinite(gb).all()):raise RuntimeError('Nonfinite attacked logits.')
                        labels=data.labels[ids].cpu();pred=(ga.argmax(1).cpu(),gb.argmax(1).cpu())
                        row=dict(group=group,node_ids=ids.cpu().tolist(),labels=labels.tolist(),
                            gcn_logits=ga.cpu().tolist(),gaga_logits=gb.cpu().tolist(),
                            node_budget=nb,edge_budget=eb,actual_injected_nodes=graph.num_nodes()-data.graph.num_nodes(),
                            actual_injected_edges=(graph.num_edges()-data.graph.num_edges()-nb)//2)
                        for index,name in enumerate(('gcn','gaga')):
                            transition=transitions(labels,clean[group][index],pred[index]);row[name+'_transitions']=transition
                            for key,value in transition.items():totals[name][key]+=value
                        stream.write(json.dumps(row,allow_nan=False)+'\n')
                        rows.append({k:v for k,v in row.items() if k not in ('node_ids','labels','gcn_logits','gaga_logits')})
                        if group in p['export_valid_groups']:
                            export=out/('%s_seed%d_group%d.pt'%(arm,seed,group));torch.save({k:v.detach().cpu() for k,v in engine.capture.items()},export)
                            exports.append(dict(group=group,path=export.name,sha256=sha256(export)))
                        engine.capture=engine.live_capture=None
                if len(rows)!=128 or any(v['occurrences']!=1091 for v in totals.values()):raise RuntimeError('Full validation denominator differs.')
                summary=dict(arm=arm,seed=seed,groups=128,totals=totals,
                    gcn_mr_percent=100*totals['gcn']['attacked_wrong']/1091,
                    gaga_mr_percent=100*totals['gaga']['attacked_wrong']/1091,
                    injected_nodes=sum(x['actual_injected_nodes'] for x in rows),
                    injected_edges=sum(x['actual_injected_edges'] for x in rows),
                    checkpoint_sha256=r['final_model_sha256'],exports=exports)
                write_json(out/('%s_seed%d_groups.json'%(arm,seed)),rows,exclusive=True)
                summaries.append(summary);print('DUAL_EVAL_RUN',json.dumps(summary),flush=True)
                del engine
        if counts!=dict(gcn_base=1664,gaga_victim=1664,generations=1536):raise RuntimeError('Evaluation forward counters differ.')
        result.update(status='completed',summaries=summaries,counts=counts,source_commit=provenance['source_commit'],
            protocol_sha256=provenance['protocol_sha256'],test_queries=0,raw_records_private=True)
    except BaseException as error:
        traceback.print_exc();result.update(error=repr(error),traceback=traceback.format_exc(),counts=counts)
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        write_json(out/'result.json',result,exclusive=True)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
