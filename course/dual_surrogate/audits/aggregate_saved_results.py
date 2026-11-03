"""Post-hoc saved-summary arithmetic; preliminary values were already inspected.

No model, tensor deserialization, dataset loading, network call, or new efficacy
experiment. Three-seed t intervals condition on the fixed teachers and cohort.
"""
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import time


def summarize(values, paired=False):
    mean=statistics.mean(values);sd=statistics.stdev(values)
    half=4.302652729911275*sd/math.sqrt(3)
    return dict(values=values,mean=mean,sd=sd,
        **({'paired_seed_t95':[mean-half,mean+half]} if paired else {'seed_mean_t95':[mean-half,mean+half]}),
        n_training_seeds=3,scope='Conditional on the fixed two new teachers and development cohort, not independent-news uncertainty')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--result',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.process_time();source=pathlib.Path(args.result)
    result=json.loads(source.read_text(encoding='utf-8'))
    if result['status']!='completed' or result['test_queries']!=0:raise RuntimeError('Completed development-only saved result required.')
    by={(x['arm'],x['seed']):x for x in result['summaries']};arms=('gcn_only','gaga_only','two_gcn','gcn_gaga')
    if set(by)!={(a,s) for a in arms for s in (3,4,5)}:raise RuntimeError('Exact twelve fixed runs required.')
    output=dict(schema='monti_dual_surrogate_saved_aggregate_v1',status='completed_negative_result',
        post_hoc_saved_only=True,preliminary_arithmetic_inspected_before_script=True,
        source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),arms={},contrasts={},test_queries=0,new_model_forwards=0)
    for arm in arms:
        rows=[by[arm,s] for s in (3,4,5)]
        for row in rows:
            if row['groups']!=128:raise RuntimeError('Group denominator mismatch.')
            if row['injected_nodes']!=7776 or row['injected_edges']!=358491:raise RuntimeError('Physical-budget sum mismatch.')
            for role in ('gaga','gcn'):
                totals=row['totals'][role]
                if totals['clean_wrong']!=dict(gaga=398,gcn=498)[role]:raise RuntimeError('Common clean baseline mismatch.')
                if totals['occurrences']!=1091 or totals['clean_wrong']+totals['new_wrong']-totals['recoveries']!=totals['attacked_wrong']:
                    raise RuntimeError('Transition/occurrence mismatch.')
                if not math.isclose(row[role+'_mr_percent'],100*totals['attacked_wrong']/1091,abs_tol=1e-12):raise RuntimeError('Saved rate mismatch.')
        output['arms'][arm]=dict(gaga_mr_percent=summarize([r['gaga_mr_percent'] for r in rows]),
            gcn_mr_percent=summarize([r['gcn_mr_percent'] for r in rows]),
            gaga_new_wrong=[r['totals']['gaga']['new_wrong'] for r in rows],gaga_recoveries=[r['totals']['gaga']['recoveries'] for r in rows],
            gaga_attacked_wrong=[r['totals']['gaga']['attacked_wrong'] for r in rows],occurrences_per_seed=1091,
            physical_nodes_per_run=7776,physical_edges_per_run=358491)
    for a,b in (('gcn_gaga','gcn_only'),('gaga_only','gcn_only'),('two_gcn','gcn_only'),('gcn_gaga','gaga_only'),('gcn_gaga','two_gcn')):
        output['contrasts'][a+'_minus_'+b]=summarize([by[a,s]['gaga_mr_percent']-by[b,s]['gaga_mr_percent'] for s in (3,4,5)],True)
    output['clean']=dict(gaga_wrong=398,gcn_wrong=498,occurrences=1091,gaga_mr_percent=100*398/1091,
        note='Fraud-target occurrences, not all-news accuracy; repeated nodes are retained.')
    output['execution']=dict(script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
        cpu_seconds_through_serialization_start=time.process_time()-started,model_calls=0,API_calls=0,tensor_deserialization=0)
    with pathlib.Path(args.output).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(output,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__=='__main__':main()
