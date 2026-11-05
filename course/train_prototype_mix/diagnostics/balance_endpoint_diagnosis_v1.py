"""POST HOC saved-JSON-only target-level balance failure diagnosis.

The64gains/19losses and negative overall qualification were already known.
This source/scope is frozen before reading raw retained prediction JSONL.
No tensor/dataset/model/weight loading, inference, API, training or GPU work.
All3273 correlated occurrences remain; no VALID-derived deployment gate is fit.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import time
import traceback

ARMS=('baseline','balanced_target','matched_random')
SEEDS=(3,4,5)
CASE_GROUPS=(1071,1001,938,1130)
QUANTILES=(0.,.1,.25,.5,.75,.9,1.)


def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def write(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')


def quantile(values,p):
    ordered=sorted(values)
    if not ordered:return None
    position=(len(ordered)-1)*p;lo=int(position);hi=math.ceil(position)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def distribution(values):
    return dict(count=len(values),mean=statistics.fmean(values) if values else None,
        quantiles={str(p):quantile(values,p) for p in QUANTILES})


def pred_margin(logits):
    if len(logits)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in logits):
        raise ValueError('Invalid frozen binary logits.')
    return (0 if logits[0]>=logits[1] else 1),logits[0]-logits[1]


def outcome(base,final,label):
    if base==label and final!=label:return 'gain'
    if base!=label and final==label:return 'loss'
    return 'unchanged_correct' if final==label else 'unchanged_wrong'


def degree_status(delta):return 'increased' if delta>0 else 'decreased' if delta<0 else 'unchanged'
def coverage_status(before,after):
    if before==0 and after>0:return 'newly_covered'
    if before>0 and after>0:return 'already_covered'
    return 'still_uncovered' if after==0 and before==0 else 'previously_covered_lost'


def describe(rows):
    count=len(rows);events=collections.Counter(row['balanced_outcome'] for row in rows)
    return dict(occurrences=count,unique_target_ids=len({row['node_id'] for row in rows}),
        group_seed_cases=len({(row['seed'],row['group']) for row in rows}),
        balanced_outcomes=dict(events),net_wrong_change=events['gain']-events['loss'],
        gain_rate_percent=100*events['gain']/count if count else None,
        loss_rate_percent=100*events['loss']/count if count else None,
        baseline_signed_benign_margin=distribution([row['baseline_margin'] for row in rows]),
        baseline_absolute_benign_margin=distribution([abs(row['baseline_margin']) for row in rows]),
        clean_signed_benign_margin=distribution([row['clean_margin'] for row in rows]),
        baseline_target_injected_degree=distribution([row['baseline_degree'] for row in rows]),
        balanced_target_injected_degree=distribution([row['balanced_degree'] for row in rows]),
        balanced_degree_delta=distribution([row['balanced_degree']-row['baseline_degree'] for row in rows]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--source-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=pathlib.Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    wall,cpu=time.monotonic(),time.process_time();files={};checks=collections.Counter()
    result=dict(status='failed',post_hoc=True,ML_imports=0,tensor_loads=0,dataset_loads=0,
        model_calls=0,optimizer_updates=0,API_calls=0,GPU_hours=0)
    def check(value,name):
        checks[name]+=1
        if not value:raise AssertionError('Saved endpoint diagnosis failed: '+name)
    try:
        check(sha(__file__)==args.source_sha256,'frozen_source')
        check(sha(args.manifest)==args.manifest_sha256,'frozen_scope_input_manifest')
        manifest=json.loads(pathlib.Path(args.manifest).read_text(encoding='utf-8'))
        write(out/'execution_registration.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256=args.source_sha256,manifest_sha256=args.manifest_sha256,
            original_aggregate_outcomes_already_known=True,new_raw_prediction_reads_started=False,
            scope='Target-level saved JSON diagnosis only; no gate selection, classifier calls, tensors or new experiments.'))
        for name,ref in manifest['inputs'].items():
            path=pathlib.Path(ref['path']);check(sha(path)==ref['sha256'],'frozen_input_bytes');files[name]=path
        ledger=json.loads(files['cost_ledger'].read_text(encoding='utf-8'))
        check(ledger['measured_or_conservatively_charged']+ledger['unmetered_setup_reserve']<=ledger['limit'],'existing_budget_not_exhausted')
        audit=json.loads(files['independent_audit'].read_text(encoding='utf-8'))
        check(audit['status']=='passed' and audit['provenance']==manifest['producer_provenance'],'completed_audited_scientific_snapshot')
        def rows(name):
            with files[name].open(encoding='utf-8') as stream:return [json.loads(line) for line in stream if line.strip()]
        geometry_rows=rows('geometry');geometry={(r['seed'],r['group']):r for r in geometry_rows}
        check(len(geometry)==len(geometry_rows)==384,'all_geometry_384cases')
        groups=sorted({r['group'] for r in geometry_rows});check(len(groups)==128,'all128groups')
        clean_rows=rows('clean');clean={r['group']:r for r in clean_rows}
        check(len(clean)==len(clean_rows)==128 and set(clean)==set(groups),'complete_clean_saved_prediction_cover')
        predictions={}
        for seed in SEEDS:
            for arm in ARMS:
                rr=rows('%s_seed%d'%(arm,seed));indexed={r['group']:r for r in rr}
                check(len(indexed)==len(rr)==128 and set(indexed)==set(groups),'all9prediction_streams_complete')
                predictions[seed,arm]=indexed
        occurrences=[]
        for seed in SEEDS:
            for group in groups:
                geom=geometry[seed,group];ids=geom['node_ids'];labels=geom['labels'];g=geom['geometry']
                check(ids==clean[group]['node_ids'] and labels==clean[group]['labels'] and len(ids)==len(set(ids))
                    and all(y==1 for y in labels),'shared_full_fraud_target_identity')
                before=g['target_incoming_injected_degrees_before'];after=g['target_incoming_injected_degrees_after']
                check(len(before)==len(ids) and set(after)==set(ARMS) and after['baseline']==before
                    and all(len(v)==len(ids) for v in after.values()),'per_target_degree_vector_alignment')
                indexed={arm:predictions[seed,arm][group] for arm in ARMS}
                for arm,row in indexed.items():
                    check(row['node_ids']==ids and row['labels']==labels and row['source_capture_sha256']==geom['source_capture_sha256']
                        and row['prepared_artifact_sha256']==geom['artifact']['sha256']
                        and row['variant_adjacency_content_sha256']==geom['variants'][arm]['adjacency_content_sha256'],
                        'frozen_prediction_geometry_binding')
                for position,(node,label) in enumerate(zip(ids,labels)):
                    decoded={arm:pred_margin(row['gaga_logits'][position]) for arm,row in indexed.items()}
                    cp,cm=pred_margin(clean[group]['gaga_logits'][position]);bp,bm=decoded['baseline'];ap,am=decoded['balanced_target'];rp,rm=decoded['matched_random']
                    bdeg,adeg,rdeg=before[position],after['balanced_target'][position],after['matched_random'][position]
                    check(all(type(x) is int and x>=0 for x in (bdeg,adeg,rdeg)),'integer_nonnegative_target_incidence')
                    check(not (bdeg>0 and adeg==0),'balanced_does_not_uncover_previously_covered_targets')
                    row=dict(seed=seed,group=group,target_position=position,node_id=node,label=label,
                        group_targets=len(ids),node_budget=geom['node_budget'],edge_budget=geom['edge_budget'],
                        clean_prediction=cp,clean_margin=cm,baseline_prediction=bp,balanced_prediction=ap,random_prediction=rp,
                        baseline_margin=bm,balanced_margin=am,random_margin=rm,
                        baseline_degree=bdeg,balanced_degree=adeg,random_degree=rdeg,
                        balanced_degree_change=degree_status(adeg-bdeg),random_degree_change=degree_status(rdeg-bdeg),
                        balanced_coverage=coverage_status(bdeg,adeg),random_coverage=coverage_status(bdeg,rdeg),
                        balanced_outcome=outcome(bp,ap,label),random_outcome=outcome(bp,rp,label),
                        balanced_vs_random_outcome=outcome(rp,ap,label))
                    occurrences.append(row)
        check(len(occurrences)==3273 and len({(r['seed'],r['group'],r['node_id']) for r in occurrences})==3273,'complete_correlated_occurrence_denominator')
        check(len({r['node_id'] for r in occurrences})==374,'unique_target_denominator')
        balanced=collections.Counter(r['balanced_outcome'] for r in occurrences);random=collections.Counter(r['random_outcome'] for r in occurrences)
        check(balanced['gain']==64 and balanced['loss']==19,'known_aggregate64gain19loss_reconciled')
        expected=audit['paired_contrasts']['balanced_target-minus-baseline_gaga']['paired_counts']
        check((balanced['gain'],balanced['loss'])==(expected['gained_wrong'],expected['lost_wrong']),'independent_audit_count_identity')
        event_keys=lambda category,arm:{(r['seed'],r['group'],r['node_id']) for r in occurrences if r[arm+'_outcome']==category}
        bg,bl,rg,rl=(event_keys(category,arm) for arm,category in [('balanced','gain'),('balanced','loss'),('random','gain'),('random','loss')])
        table=collections.Counter((r['balanced_outcome'],r['random_outcome']) for r in occurrences)
        overlap=dict(balanced_gains=len(bg),random_gains=len(rg),shared_gains=len(bg&rg),balanced_only_gains=len(bg-rg),random_only_gains=len(rg-bg),
            balanced_losses=len(bl),random_losses=len(rl),shared_losses=len(bl&rl),balanced_only_losses=len(bl-rl),random_only_losses=len(rl-bl),
            exact_same_gain_occurrences=bg==rg,exact_same_loss_occurrences=bl==rl,
            full_cross_table=[dict(balanced=x,random=y,occurrences=n) for (x,y),n in sorted(table.items())])
        summary=dict(all_occurrences=describe(occurrences),balanced_gains=describe([r for r in occurrences if r['balanced_outcome']=='gain']),
            balanced_losses=describe([r for r in occurrences if r['balanced_outcome']=='loss']),
            unchanged_correct=describe([r for r in occurrences if r['balanced_outcome']=='unchanged_correct']),
            unchanged_wrong=describe([r for r in occurrences if r['balanced_outcome']=='unchanged_wrong']))
        strata=[]
        for field in ('balanced_coverage','balanced_degree_change','random_coverage','random_degree_change'):
            for value in sorted({r[field] for r in occurrences}):
                subset=[r for r in occurrences if r[field]==value]
                strata.append(dict(field=field,value=value,**describe(subset)))
        for coverage in sorted({r['balanced_coverage'] for r in occurrences}):
            for change in sorted({r['balanced_degree_change'] for r in occurrences}):
                subset=[r for r in occurrences if r['balanced_coverage']==coverage and r['balanced_degree_change']==change]
                if subset:strata.append(dict(field='balanced_coverage_x_degree_change',value=coverage+' / '+change,**describe(subset)))
        thresholds=[quantile([abs(r['baseline_margin']) for r in occurrences],q) for q in (.25,.5,.75)]
        margin_bins=[]
        for bucket in range(4):
            subset=[r for r in occurrences if sum(abs(r['baseline_margin'])>t for t in thresholds)==bucket]
            margin_bins.append(dict(bin=bucket,thresholds_from_all3273_VALID_baseline_absolute_margins=thresholds,**describe(subset)))
        cases=[]
        for group in CASE_GROUPS:
            for seed in SEEDS:
                subset=[r for r in occurrences if r['seed']==seed and r['group']==group];g=geometry[seed,group]['geometry']
                cases.append(dict(seed=seed,group=group,selection='Previously discussed fixed groups1071/1001; already reported posthoc negative938 and positive1130. All3seeds retained, not selected by this new diagnosis.',
                    targets=len(subset),baseline_wrong=sum(r['baseline_prediction']!=r['label'] for r in subset),
                    balanced_wrong=sum(r['balanced_prediction']!=r['label'] for r in subset),random_wrong=sum(r['random_prediction']!=r['label'] for r in subset),
                    phi_before=g['sum_squared_target_degree_before'],phi_balanced=g['sum_squared_target_degree_balanced'],
                    outcome_and_geometry=describe(subset),changed_occurrences=[r for r in subset if r['balanced_outcome'] in ('gain','loss') or r['random_outcome'] in ('gain','loss')]))
        repetitions=collections.Counter(r['node_id'] for r in occurrences);memberships=collections.defaultdict(set)
        for row in occurrences:memberships[row['node_id']].add(row['group'])
        write(out/'event_occurrences.json',[r for r in occurrences if r['balanced_outcome'] in ('gain','loss') or r['random_outcome'] in ('gain','loss')])
        with (out/'all3273_occurrences.csv').open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(occurrences[0]),lineterminator='\n');writer.writeheader();writer.writerows(occurrences)
        result.update(status='completed_saved_json_diagnosis',producer_provenance=manifest['producer_provenance'],
            analysis_registration='Source/scope frozen before these raw target-level records were read, after aggregate outcomes already known; POST HOC, not efficacy preregistration.',
            denominator=dict(correlated_occurrences=3273,group_seed_cases=384,unique_targets=374,
                maximum_seed_group_occurrences_per_unique_target=max(repetitions.values()),
                maximum_group_memberships_per_unique_target=max(map(len,memberships.values())),
                repeated_across_distinct_groups=sum(len(v)>1 for v in memberships.values())),
            distributions=summary,balanced_random_occurrence_overlap=overlap,descriptive_structural_strata=strata,
            descriptive_VALID_margin_quartile_bins=margin_bins,prior_named_case_followups=cases,
            budget_context=dict(charged_GPU_hours=ledger['measured_or_conservatively_charged'],reserve=ledger['unmetered_setup_reserve'],new_GPU_hours=0),
            limitations=['All3273 occurrences remain; target IDs, groups and seeds repeat. Rates/quantiles are descriptive, not IID estimates or causal effects.',
                'Degree means incoming edges from injected nodes only, not total graph degree, edge influence or neighboring-news trust. Feature allocation, GCN normalization and multi-hop context may confound degree/outcome associations.',
                'Margins are uncalibrated frozen official-GAGA logit differences, not probabilities or certified reliability. Quartile cutpoints use the same already-inspected VALID population solely for description.',
                'No validation-derived gate/rule is selected, fitted or evaluated here. Any next method needs a separate frozen bounded comparison; no new experiment has been selected by this diagnosis.',
                'Only saved JSON/CSV arithmetic is added. No tensor, dataset, checkpoint or model is loaded; zero inference/training/API/GPU work.',
                'Overall primary criterion still failed. Per-case or stratum associations do not override the negative primary comparison or authorize finalTEST/tag.'])
    except BaseException as error:
        traceback.print_exc();result.update(error_type=type(error).__name__,error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(check_counts=dict(checks),input_files_read=len(files),process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall)
        write(out/'input_hashes.json',{name:dict(path=str(path),sha256=sha(path)) for name,path in files.items()})
        write(out/'summary.json',result);print(json.dumps(dict(status=result['status'],output=str(out),checks=sum(checks.values()),CPU_seconds=result['process_cpu_seconds'],wall_seconds=result['wall_seconds']),flush=True))
    if result['status']!='completed_saved_json_diagnosis':raise SystemExit(1)


if __name__=='__main__':main()
