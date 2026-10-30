"""Descriptive extraction of saved GCN validation outcomes; no model inference."""
import argparse
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import traceback

import numpy as np
import torch

from modules.data_handler import DataHandler

ROOT=pathlib.Path(__file__).resolve().parents[1]
FEATURES=['group_size','node_budget','edge_budget','nodes_per_target','edges_per_target',
          'edges_per_injected','mean_target_degree','min_target_degree','max_target_degree']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def to_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def scheduled(initial, rate, minimum, epoch):
    result=initial
    for _ in range(epoch):
        result=max(minimum,result*rate)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='evidence/results_first_diagnosis_v1')
    args=parser.parse_args()
    output=ROOT/args.output
    assert str(output.resolve()).startswith(str(ROOT.resolve())+'/')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    write(output/'registered.json',dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
          source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          script_sha256=sha(pathlib.Path(__file__)),command=['env','CUDA_VISIBLE_DEVICES=','bash','course/run.sh','-m','course.results_first_diagnosis']+__import__('sys').argv[1:],
          scope='Post-hoc saved GCN validation outcomes and fixed metadata only; no test predictions or model inference',
          plan='course/RESULTS_FIRST_DIAGNOSIS_PLAN.md',plan_sha256=sha(ROOT/'course/RESULTS_FIRST_DIAGNOSIS_PLAN.md'),
          features=FEATURES,gpu_hours=0))
    inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    try:
        assert not torch.cuda.is_available(), 'Run this extraction with CUDA_VISIBLE_DEVICES empty.'
        torch.set_num_threads(8)
        config=read(ROOT/'course/configs/baseline.json')
        data=DataHandler(dict(config,cuda_id='cpu'))
        all_needed=set(data.target_idx_train.tolist()+data.target_idx_valid.tolist())
        meta={}
        members={}
        truth={}
        for group in sorted(all_needed):
            nodes=data.target_assignment[group].nonzero()[1].tolist()
            size=len(nodes)
            degrees=data.graph.ndata['degree'][nodes].numpy()
            nb,eb=int(data.node_budgets[group]),int(data.edge_budgets[group])
            meta[group]=dict(group_size=size,node_budget=nb,edge_budget=eb,nodes_per_target=nb/size,
                             edges_per_target=eb/size,edges_per_injected=eb/nb,
                             mean_target_degree=float(degrees.mean()),min_target_degree=float(degrees.min()),max_target_degree=float(degrees.max()))
            members[group]=nodes
            truth[group]=data.labels[nodes].tolist()
        baseline=[x for x in read(ROOT/'evidence/run_registry.json') if x['registration']['phase']=='reproduction']
        pilots=read(ROOT/'evidence/improvement_pilot_registry.json')
        assert len(baseline)==6 and len(pilots)==9
        cohorts={'full_baseline':baseline,'pilot':pilots}
        snapshots={}
        cohort_groups={}
        run_rows=[]
        curve_rows=[]
        for cohort,runs in cohorts.items():
            for run in runs:
                reg=run['registration']
                condition=reg['mode'] if cohort=='full_baseline' else reg['variant']
                folder=ROOT/run['evidence']
                result=read(folder/'result.json')
                groups=read(folder/'target_groups.json')
                cfg=read(folder/'config.json')
                read(folder/'registered.json')
                assert result['status']=='completed' and cfg['valid_epoch']==1
                if cohort=='pilot':
                    assert result['result']=={'test_skipped':1.0}
                valid=groups['valid']
                train=groups['train']
                if cohort not in cohort_groups:
                    cohort_groups[cohort]=dict(train=train,valid=valid)
                assert cohort_groups[cohort]==dict(train=train,valid=valid)
                paths=sorted(folder.glob('*_surrogate_valid_attack.json'))
                assert len(paths)==result['completed_epochs']
                curves=[]
                best_snapshot=None
                for epoch,path in enumerate(paths,1):
                    assert '_test_' not in path.name
                    summary=read(path)
                    raw=path.with_suffix('.jsonl')
                    assert sha(raw)==summary['prediction_sha256']
                    inputs[str(raw.relative_to(ROOT))]=summary['prediction_sha256']
                    rows=[json.loads(line) for line in raw.read_text().splitlines()]
                    seen=set()
                    errors=count=0
                    snapshot={}
                    for row in rows:
                        group=row['target_group']
                        assert group in valid and group not in seen
                        seen.add(group)
                        assert row['node_ids']==members[group] and row['labels']==truth[group]
                        assert len(row['predictions'])==len(row['labels'])
                        assert row['node_budget']==meta[group]['node_budget'] and row['edge_budget']==meta[group]['edge_budget']
                        wrong=sum(a!=b for a,b in zip(row['labels'],row['predictions']))
                        assert wrong==row['incorrect']
                        errors+=wrong
                        count+=len(row['labels'])
                        if epoch==result['best_epoch']:
                            for node,label,prediction in zip(row['node_ids'],row['labels'],row['predictions']):
                                snapshot[group,node]=int(label==prediction)
                    assert seen==set(valid) and len(rows)==summary['groups']
                    assert errors==summary['incorrect'] and count==summary['target_occurrences']
                    assert abs(errors/count-summary['result']['misclf_rate'])<1e-14
                    curve=dict(cohort=cohort,condition=condition,seed=reg['seed'],epoch=epoch,
                               mr_percent=100*errors/count,gcn_correct_after_attack=count-errors,
                               target_occurrences=count,group_mean_hinge_loss=summary['result']['loss'])
                    curves.append(curve)
                    curve_rows.append(curve)
                    if epoch==result['best_epoch']:
                        best_snapshot=snapshot
                assert best_snapshot is not None
                counts=[row['gcn_correct_after_attack'] for row in curves]
                earliest_best=counts.index(min(counts))+1
                assert earliest_best==result['best_epoch']
                assert sha(ROOT/result['checkpoint'])==result['checkpoint_sha256']
                inputs[result['checkpoint']]=result['checkpoint_sha256']
                key=cohort,condition,reg['seed']
                assert key not in snapshots
                snapshots[key]=best_snapshot
                best=curves[result['best_epoch']-1]
                final=curves[-1]
                run_rows.append(dict(cohort=cohort,condition=condition,seed=reg['seed'],run_id=reg['run_id'],
                    groups=len(valid),target_occurrences=best['target_occurrences'],best_epoch=result['best_epoch'],
                    completed_epochs=result['completed_epochs'],best_mr_percent=best['mr_percent'],
                    final_mr_percent=final['mr_percent'],final_minus_best_pp=final['mr_percent']-best['mr_percent'],
                    best_residual_count=best['gcn_correct_after_attack'],final_residual_count=final['gcn_correct_after_attack'],
                    tied_best_epoch_count=sum(value==min(counts) for value in counts),
                    best_epoch_hinge_loss=best['group_mean_hinge_loss'],final_hinge_loss=final['group_mean_hinge_loss'],
                    epoch_lag=result['completed_epochs']-result['best_epoch'],
                    scheduled_temperature_at_best=scheduled(cfg['initial_temp'],cfg['temp_decay_rate'],cfg['min_temp'],result['best_epoch']),
                    scheduled_temperature_at_end=scheduled(cfg['initial_temp'],cfg['temp_decay_rate'],cfg['min_temp'],result['completed_epochs'])))
        aggregate=[]
        persistence=[]
        by_condition={}
        concentration=[]
        for cohort in cohorts:
            conditions=sorted({key[1] for key in snapshots if key[0]==cohort})
            for condition in conditions:
                seed_keys=sorted(key for key in snapshots if key[:2]==(cohort,condition))
                assert len(seed_keys)==3
                maps=[snapshots[key] for key in seed_keys]
                keys=set(maps[0])
                assert all(set(item)==keys for item in maps)
                residual_counts=collections.Counter({key:sum(item[key] for item in maps) for key in keys})
                by_condition[cohort,condition]=residual_counts
                for frequency in range(4):
                    persistence.append(dict(cohort=cohort,condition=condition,gcn_correct_in_n_seeds=frequency,
                                            occurrence_keys=sum(value==frequency for value in residual_counts.values()),
                                            total_occurrence_keys=len(keys)))
                node_total=collections.Counter()
                node_residual=collections.Counter()
                for (group,node),residual in residual_counts.items():
                    node_total[node]+=3
                    node_residual[node]+=residual
                top=[node for node,_ in node_residual.most_common(10)]
                total_residual=sum(node_residual.values())
                for rank,node in enumerate(top,1):
                    concentration.append(dict(cohort=cohort,condition=condition,rank_by_residual=rank,
                         node_key=hashlib.sha256(str(node).encode()).hexdigest()[:12],
                         evaluated_occurrences=node_total[node],gcn_correct_after_attack=node_residual[node],
                         residual_rate=node_residual[node]/node_total[node],
                         share_of_all_residuals=node_residual[node]/total_residual if total_residual else 0))
                for group in cohort_groups[cohort]['valid']:
                    values=[residual_counts[group,node] for node in members[group]]
                    denominator=3*meta[group]['group_size']
                    aggregate.append(dict(cohort=cohort,condition=condition,target_group=group,seeds=3,
                         **meta[group],target_occurrences=denominator,gcn_correct_after_attack=sum(values),
                         residual_rate=sum(values)/denominator,mean_mr_percent=100*(1-sum(values)/denominator),
                         persistent_residual_occurrence_keys=sum(value==3 for value in values)))
        bin_rows=[]
        cuts={}
        for cohort,groups in cohort_groups.items():
            cuts[cohort]={}
            for feature in FEATURES:
                boundaries=sorted(set(float(x) for x in np.quantile([meta[group][feature] for group in groups['train']],[.25,.5,.75])))
                cuts[cohort][feature]=boundaries
                for condition in sorted({row['condition'] for row in aggregate if row['cohort']==cohort}):
                    selected=[row for row in aggregate if row['cohort']==cohort and row['condition']==condition]
                    for index in range(len(boundaries)+1):
                        rows=[row for row in selected if bisect.bisect_right(boundaries,row[feature])==index]
                        count=sum(row['target_occurrences'] for row in rows)
                        remaining=sum(row['gcn_correct_after_attack'] for row in rows)
                        unique_nodes={node for row in rows for node in members[row['target_group']]}
                        bin_rows.append(dict(cohort=cohort,condition=condition,feature=feature,bin=index,
                            lower_inclusive=boundaries[index-1] if index else None,
                            upper_exclusive=boundaries[index] if index<len(boundaries) else None,
                            groups=len(rows),unique_nodes=len(unique_nodes),target_occurrences=count,
                            gcn_correct_after_attack=remaining,residual_rate=remaining/count if count else None,
                            mr_percent=100*(1-remaining/count) if count else None))
        paired=[]
        for cohort,reference,variants in [('full_baseline','random',['learned']),('pilot','baseline',['coverage','size_sampling'])]:
            for variant in variants:
                for seed in sorted(key[2] for key in snapshots if key[:2]==(cohort,reference)):
                    left=snapshots[cohort,reference,seed]
                    right=snapshots[cohort,variant,seed]
                    assert set(left)==set(right)
                    gains=sum(left[key]==1 and right[key]==0 for key in left)
                    losses=sum(left[key]==0 and right[key]==1 for key in left)
                    paired.append(dict(cohort=cohort,variant=variant,reference=reference,seed=seed,
                        target_occurrences=len(left),additional_misclassified_occurrences=gains,
                        fewer_misclassified_occurrences=losses,net_misclassified_occurrences=gains-losses,
                        difference_pp=100*(gains-losses)/len(left)))
        summary=[]
        for cohort,condition in sorted(by_condition):
            rows=[row for row in run_rows if (row['cohort'],row['condition'])==(cohort,condition)]
            node_rows=[row for row in concentration if (row['cohort'],row['condition'])==(cohort,condition)]
            residuals=sum(row['best_residual_count'] for row in rows)
            summary.append(dict(cohort=cohort,condition=condition,seeds=[row['seed'] for row in rows],
               mean_best_gcn_validation_mr_percent=statistics.mean(row['best_mr_percent'] for row in rows),
               mean_final_minus_best_pp=statistics.mean(row['final_minus_best_pp'] for row in rows),
               residual_occurrences_across_seeds=residuals,
               persistent_residual_occurrence_keys=sum(value==3 for value in by_condition[cohort,condition].values()),
               top10_nodes_share_of_residuals=sum(row['gcn_correct_after_attack'] for row in node_rows)/residuals if residuals else None,
               top10_nodes_share_of_all_evaluated_occurrences=sum(row['evaluated_occurrences'] for row in node_rows)/sum(row['target_occurrences'] for row in rows)))
        to_csv(output/'runs.csv',run_rows)
        to_csv(output/'validation_curves.csv',curve_rows)
        to_csv(output/'group_outcomes.csv',aggregate)
        to_csv(output/'training_derived_bins.csv',bin_rows)
        to_csv(output/'persistence.csv',persistence)
        to_csv(output/'node_concentration.csv',concentration)
        to_csv(output/'paired_changes.csv',paired)
        write(output/'bin_boundaries.json',cuts)
        report=dict(status='completed',conditions=summary,runs=run_rows,paired_changes=paired,
             information_boundary='Saved GCN validation predictions only; fixed metadata; no test predictions or new inference',
             interpretation='Post-hoc descriptive; selected validation snapshots are not independent confirmation; no causal conclusion or chosen improvement',
             unit='Group-node occurrence, repeated across three seeds; overlapping nodes and groups are not independent experiments',
             cpu_elapsed_seconds=time.monotonic()-started,gpu_hours=0)
        write(output/'summary.json',report)
        write(output/'input_checksums.json',inputs)
        print(json.dumps(dict(status='completed',output=str(output),conditions=summary,paired_changes=paired),indent=2),flush=True)
    except Exception:
        write(output/'failure.json',dict(status='failed',traceback=traceback.format_exc(),cpu_elapsed_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
