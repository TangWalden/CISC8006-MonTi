"""Compact author-study export after completion and both passed saved audits.

Standard library only. Every registered case is retained; no tensor, model,
per-target prediction, feature, checkpoint object or demo-edge decoding occurs.
Original counts/geometry remain the scientific evidence; this is curation.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

SOURCE_ROOT=Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:sys.path.insert(0,str(SOURCE_ROOT))
from experiments.author_staged import audit_saved as helper

ARMS=helper.ARMS
SEEDS=helper.SEEDS
FIELDS=('clean_wrong','attacked_wrong','new_wrong','recoveries')
RELATIVE_SOURCE='experiments/author_staged/export_saved.py'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)
def opaque(seed,group):
    return hashlib.sha256(json.dumps([seed,group,'author_staged_tensor_audit_v1']).encode('ascii')).hexdigest()[:20]
def intervals(metrics):
    lookup={(r['seed'],r['arm']):r for r in metrics};contrasts={}
    for control in ARMS[:-1]:
        values=[lookup[s,'quota_learned']['gaga_mr_percent']-lookup[s,control]['gaga_mr_percent'] for s in SEEDS]
        mean=statistics.mean(values)
        # Retain the exact critical constant frozen in the scientific runner
        # and independently authored audit; export does not redefine its rule.
        radius=4.302652729911275*statistics.stdev(values)/math.sqrt(3)
        contrasts[control]=dict(seed_differences_pp=values,mean_pp=mean,nominal_paired_t95_pp=[mean-radius,mean+radius])
    primary=contrasts['native_monti']
    qualified=primary['mean_pp']>0 and all(v>0 for v in primary['seed_differences_pp']) and primary['nominal_paired_t95_pp'][0]>0
    return contrasts,bool(qualified)
def case_metrics(rows):
    require(len(rows)==8181,'Every8181 arm-case count row must remain.')
    index={(v['seed'],v['group'],v['arm']):v for v in rows}
    require(len(index)==len(rows),'Duplicate arm-case counts.')
    metrics=[]
    for seed in SEEDS:
        for arm in ARMS:
            selected=[v for v in rows if v['seed']==seed and v['arm']==arm]
            n=sum(v['targets'] for v in selected)
            require(len(selected)==909 and n==10515,'Full909-group/10515-occurrence denominator differs.')
            value=dict(seed=seed,arm=arm,groups=909,occurrences=n)
            for detector in ('gcn','gaga'):
                for row in selected:
                    count=row[detector]
                    require(count['occurrences']==row['targets'] and all(type(count[k]) is int and count[k]>=0 for k in FIELDS)
                        and count['clean_wrong']<=row['targets'] and count['attacked_wrong']<=row['targets']
                        and count['new_wrong']<=row['targets']-count['clean_wrong'] and count['recoveries']<=count['clean_wrong']
                        and count['attacked_wrong']==count['clean_wrong']+count['new_wrong']-count['recoveries'],
                        'Invalid per-case clean/new/recovery count identity.')
                value.update({detector+'_'+k:sum(v[detector][k] for v in selected) for k in FIELDS})
                value[detector+'_mr_percent']=100*value[detector+'_attacked_wrong']/n
                value[detector+'_group_macro_mr_percent']=statistics.mean(100*v[detector]['attacked_wrong']/v['targets'] for v in selected)
            metrics.append(value)
    return metrics


def guardian_binding(guardian,p,protocol_path,run,result_sha):
    expected=['/home/tjy/miniconda3/envs/monti/bin/python',str(Path(p['source_root'])/'experiments/final_staged/run.py'),
        '--protocol',str(Path(protocol_path).resolve()),'--output',str(run),
        '--preflight',str(run.parent/'preflight_attempt1/result.json')]
    require(guardian['phase']=='run' and guardian['status']=='completed' and guardian['exit_code']==0
        and guardian['child_result_sha256']==result_sha and guardian['protocol_sha256']==sha(protocol_path)
        and guardian['execution_profile']=='author_upstream' and guardian['guardian_lock']=='author_staged_exclusive.lock'
        and guardian['command']==expected and guardian['output']==str(run) and guardian['selected_device']==p['device']
        and guardian['inner_seconds']==p['resource']['main_inner_seconds']
        and guardian['outer_seconds']==p['resource']['main_inner_seconds']+p['resource']['outer_extra_seconds']
        and guardian['whole_phase_GPU_hours']==0
        and guardian['whole_phase_wall_seconds_including_review_wait']<=p['resource']['CPU_phase_seconds']
        and 0<=guardian['outer_wall_seconds']<=guardian['outer_seconds']+30
        and guardian['guardian_wall_seconds']>=guardian['outer_wall_seconds'],
        'Completed exact-command guardian/profile/result/cap binding differs; relocated copies cannot claim original-path command verification.')
    return {k:guardian[k] for k in ('outer_wall_seconds','guardian_wall_seconds','child_process_cpu_seconds',
        'whole_phase_wall_seconds_including_review_wait','whole_phase_GPU_hours','inner_seconds','outer_seconds')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run-root','protocol','saved-audit','tensor-audit','post-source-manifest','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--guardian',help='Optional exact original completed main guardian; omit for relocated copies.')
    args=parser.parse_args(argv);started,cpu=time.monotonic(),time.process_time()
    root=Path(args.run_root).resolve(strict=True);output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    inputs={};generated=[];receipt=dict(schema='monti_author_staged_export_receipt_v1',status='failed',
        source_sha256=sha(__file__),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        counters=dict(model_queries=0,tensor_deserializations=0,dataset_loads=0,checkpoint_object_loads=0,
            optimizer_updates=0,API_calls=0,TEST_queries=0),GPU_hours=0)
    def read(path,key):
        raw=Path(path).read_bytes();inputs[key]=hashlib.sha256(raw).hexdigest();return json.loads(raw.decode('utf-8'))
    def bound(name):
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe run artifact path.')
        path=(root/relative).resolve(strict=True);require(root in path.parents and path.is_file(),'Run artifact escapes its root.');return path
    def write_json(name,value):
        with (output/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,ensure_ascii=True,allow_nan=False);f.write('\n')
        generated.append(name)
    def write_csv(name,rows):
        with (output/name).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        generated.append(name)
    error=None
    try:
        post=read(args.post_source_manifest,'postprocessing_manifest')
        require(re.fullmatch('[0-9a-f]{40}',post.get('source_commit','')) is not None
            and post.get('files',{}).get(RELATIVE_SOURCE)==receipt['source_sha256'],'Committed postprocessing manifest does not bind exporter bytes.')
        p=read(args.protocol,'protocol');helper.author_protocol_check(p)
        require(p['status']=='frozen' and p['owner_execution_authorized'] is True and p['final_TEST_allowed'] is True
            and tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS
            and len(p['train_groups'])==len(set(p['train_groups']))==937 and len(p['test_groups'])==len(set(p['test_groups']))==909
            and not set(p['train_groups'])&set(p['test_groups'])
            and sum(map(len,p['train_members'].values()))==11211 and sum(map(len,p['test_members'].values()))==10515,
            'Wrong frozen full-cohort author study.')
        helper_source=sha(Path(helper.__file__))
        require(p['source_manifest']['files']['experiments/author_staged/audit_saved.py']==helper_source,
            'Imported standard-library audit helpers differ from the scientific source manifest.')
        provenance=dict(source_commit=p['source_manifest']['source_commit'],runtime_commit=p['runtime_commit'],protocol_sha256=inputs['protocol'])
        result=read(bound('result.json'),'result.json');saved=read(args.saved_audit,'saved_audit');tensor=read(args.tensor_audit,'tensor_audit')
        require(result['status']=='completed' and result['provenance']==provenance and result['full_TEST_evaluated'] is True
            and result['GPU_hours']==0,'Completed same-protocol author result required.')
        require(saved['status']==tensor['status']=='passed' and saved['schema']=='monti_author_staged_saved_logit_recount_v1'
            and tensor['schema']=='monti_author_staged_saved_tensor_geometry_audit_v1'
            and saved['recount']['provenance']==tensor['provenance']==provenance
            and saved['auditor_source_sha256']==helper_source
            and tensor['audit_source_binding']['auditor_source_sha256']==p['source_manifest']['files']['experiments/author_staged/audit_tensors.py'],
            'Both passed same-protocol author audits and their committed source bindings are required.')
        audited=saved['recount'];seal=read(bound('sealed_before_victim.json'),'sealed_before_victim.json')
        cache=read(bound('upstream_cache_manifest.json'),'upstream_cache_manifest.json')
        require(inputs['result.json']==saved['input_file_hashes']['result.json']==tensor['inputs']['completed_result_sha256']
            and inputs['sealed_before_victim.json']==result['pre_victim_seal_sha256']==tensor['inputs']['sealed_plan_manifest_sha256']
            and inputs['upstream_cache_manifest.json']==seal['upstream_cache_sha256']==tensor['inputs']['upstream_cache_manifest_sha256']
            and seal['provenance']==cache['provenance']==provenance,'Result/audits/seal/cache byte binding differs.')
        author=helper.author_receipts_check(cache['receipts'],p)
        require(author==audited['author_upstream_binding']==tensor['author_upstream_binding']
            and cache['receipts']==result['upstream_receipts'],'The declared single author state differs across audited receipts.')
        require(audited['occurrence_denominator_per_arm_seed']==10515 and audited['attacked_cases']==8181
            and audited['saved_rows']==9090 and result['counters']==audited['producer_logical_counters'],
            'Audited cohort/counters differ.')
        require(audited['unique_test_targets']==p['cohort']['unique_test_targets']
            ==len({v for values in p['test_members'].values() for v in values}),
            'Audited unique-target metadata differs; this is not an original dataset deduplication.')
        counters=result['counters']
        require(counters['upstream_generations']==5538 and counters['evaluation_GCN_forwards']==counters['official_GAGA_forwards']==9090
            and counters['TEST_queries']==18180 and counters['TEST_upstream_generations']==counters['TEST_downstream_policy_forwards']==2727
            and counters['upstream_optimizer_updates']==counters['API_calls']==counters['validation_queries']==0
            and 0<counters['training_GCN_forwards']==counters['autograd_calls']==counters['downstream_optimizer_updates']<=11244
            and counters['downstream_optimizer_updates']+counters['degenerate_training_cases']==11244,'Full execution counter inventory differs.')
        for seed in SEEDS:
            name='downstream/s%d_final.pt'%seed;actual=sha(bound(name));inputs[name]=actual
            require(actual==seal['downstream_checkpoint_hashes'][str(seed)]==tensor['inputs']['terminal_checkpoint_byte_hashes'][str(seed)],
                'Terminal downstream checkpoint byte binding differs; objects are never deserialized.')
        plans={(r['seed'],r['group']):r for r in seal['plans']};geometry={r['opaque_case_key']:r for r in tensor['test_role_counts']}
        expected={(s,g,a) for s in SEEDS for g in p['test_groups'] for a in ARMS}
        declared={(r['seed'],r['group'],r['arm']):r for r in result['all_case_counts']}
        require(len(declared)==len(result['all_case_counts'])==8181 and set(declared)==expected
            and len(plans)==len(seal['plans'])==len(geometry)==len(tensor['test_role_counts'])==2727
            and set(plans)=={(s,g) for s in SEEDS for g in p['test_groups']}
            and set(geometry)=={opaque(s,g) for s in SEEDS for g in p['test_groups']},'Full count/geometry Cartesian inventory differs.')
        metrics=case_metrics(result['all_case_counts']);metric_lookup={(r['seed'],r['arm']):r for r in metrics}
        require(len(audited['seed_metrics'])==9,'Audited seed metric inventory differs.')
        for original in audited['seed_metrics']:
            value=metric_lookup[original['seed'],original['arm']]
            require(all(value[k]==original[k] for k in ('seed','arm','groups','occurrences')),'Audited metric keys differ.')
            for detector in ('gcn','gaga'):helper.close(value[detector+'_mr_percent'],original[detector+'_mr_percent'])
        contrasts,qualified=intervals(metrics)
        for arm,value in contrasts.items():
            original=audited['contrasts'][arm]
            helper.close(value['mean_pp'],original['mean_pp'])
            for field in ('seed_differences_pp','nominal_paired_t95_pp'):
                require(len(value[field])==len(original[field]),'Audited contrast dimensions differ.')
                for x,y in zip(value[field],original[field]):helper.close(x,y)
        require(qualified==audited['qualification_met']==result['qualification_met'] and result['controls_promoted'] is False,
            'Registered primary decision/control policy differs.')
        cases=[];quotas=[]
        for seed in SEEDS:
            for group in p['test_groups']:
                key=opaque(seed,group);meta=geometry[key];plan=plans[seed,group];q=meta['planned_role_allocation'];n=len(p['test_members'][str(group)])
                helper.quota_check(q,n)
                require(meta['seed']==seed and meta['targets']==n and plan['quota']==q
                    and plan['frozen_upstream_feature_sha256']==meta['frozen_upstream_feature_sha256']
                    and plan['frozen_upstream_target_block_sha256']==meta['frozen_upstream_target_block_sha256'],'Audited native-input/quota bindings differ.')
                quota=dict(opaque_case_key=key,seed=seed,targets=n,candidates=meta['candidates'],node_budget=q['injected_count'],edge_budget=q['edge_budget'])
                quota.update({k:q[k] for k in ('ideal_target_quota','ideal_candidate_quota','target_quota','candidate_quota',
                    'target_fraction','candidate_fraction','projection_applied','exact_half','odd_budget')})
                baseline=declared[seed,group,'native_monti']
                for arm in ARMS:
                    row=declared[seed,group,arm];roles=meta['arms'][arm]
                    require(row['targets']==n and row['quota']==q and row['actual_nodes']==q['injected_count'] and row['actual_edges']==q['edge_budget']
                        and row['feature_sha256']==meta['frozen_upstream_feature_sha256'] and row['topology_sha256']==plan['topology_sha256'][arm]
                        and all(row[d]['clean_wrong']==baseline[d]['clean_wrong'] for d in ('gcn','gaga')),
                        'Count/sealed-geometry/shared-cohort binding differs.')
                    require(roles['target_edges']+roles['candidate_edges']+roles['injected_to_injected_owned_edges']==roles['total_owned_undirected_edges']==q['edge_budget'],
                        'Actual audited role totals differ.')
                    if arm!='native_monti':require(roles['target_edges']==q['target_quota'] and roles['candidate_edges']==q['candidate_quota'] and roles['injected_to_injected_owned_edges']==0,'Completion role allocation differs.')
                    compact=dict(opaque_case_key=key,seed=seed,arm=arm,targets=n,node_budget=q['injected_count'],edge_budget=q['edge_budget'])
                    compact.update({d+'_'+k:row[d][k] for d in ('gcn','gaga') for k in FIELDS});cases.append(compact)
                    quota.update({arm+'_'+k:roles[k] for k in ('target_edges','candidate_edges','injected_to_injected_owned_edges',
                        'total_owned_undirected_edges','target_fraction','injected_physical_degree_min','injected_physical_degree_max','injected_physical_degree_sum')})
                quotas.append(quota)
        means={arm:statistics.mean(r['gaga_mr_percent'] for r in metrics if r['arm']==arm) for arm in ARMS}
        for arm,value in means.items():helper.close(value,audited['gaga_mean_mr_percent'][arm])
        costs=dict(main_wall_seconds=result['wall_seconds'],main_process_cpu_seconds=result['process_cpu_seconds'],GPU_hours=0,counters=counters)
        if args.guardian:
            guardian=read(args.guardian,'main_guardian');costs['exact_original_guardian']=guardian_binding(guardian,p,args.protocol,root,inputs['result.json'])
        latest=dict(schema='monti_author_staged_compact_v1',status='completed',scientific_provenance=provenance,
            seeds=list(SEEDS),arms=list(ARMS),full_TEST_evaluated=True,controls_promoted=False,qualification_met=qualified,
            gaga_mean_mr_percent=means,contrasts=contrasts,
            replication=dict(distinct_author_generators=1,author_configuration_seed=3,downstream_capture_noise_repetitions=list(SEEDS),
                interpretation='Conditional on one fixed official generator; not three independently trained author upstreams or a paper-performance claim'),
            cohort=dict(train_groups=937,test_groups_per_repetition=909,target_occurrences_per_arm_repetition=10515,
                pooled_correlated_occurrences_per_arm=31545,unique_test_targets=audited['unique_test_targets']),
            quota_summary=tensor['quota_summary'],costs=costs,
            limits=['Native, random and trained arms use freshly paired author captures. Paper percentages and own-trained baselines are context only.',
                'The complete scheme changes quotas and auxiliary-edge policy. Structural budgets do not equal training cost or isolate one causal component.',
                'Three-repetition/df2 intervals are conditional and do not erase repeated prior TEST inspection or correlated target occurrences.',
                'Random comparison is secondary; negative and inconclusive outcomes stay visible. No control is promoted or new seed added.'])
        write_json('latest.json',latest);write_csv('seed_metrics.csv',metrics);write_csv('case_counts.csv',cases);write_csv('quota_by_case.csv',quotas)
        write_json('provenance.json',dict(schema='monti_author_staged_export_provenance_v1',scientific_provenance=provenance,
            postprocessing_source_commit=post['source_commit'],exporter_source_sha256=receipt['source_sha256'],auditor_helper_source_sha256=helper_source,
            original_input_sha256=inputs,author_upstream_binding=author,payload_sha256={name:sha(output/name) for name in generated},
            exact_original_guardian_verified=bool(args.guardian),opaque_key_scheme='SHA256(JSON[seed, original_group, author_staged_tensor_audit_v1])[:20]',
            scope='Saved audited count/geometry curation; zero tensor/model/dataset/parameter-object decoding. No raw per-target outputs or private absolute paths exported.'))
        receipt.update(status='completed',scientific_provenance=provenance,postprocessing_source_commit=post['source_commit'],
            seed_metric_rows=9,arm_case_rows=8181,geometry_case_rows=2727,output_sha256={name:sha(output/name) for name in generated})
    except BaseException as caught:
        error=caught;receipt.update(error=repr(caught),traceback=traceback.format_exc());traceback.print_exc()
    finally:
        receipt.update(wall_seconds=time.monotonic()-started,process_cpu_seconds=time.process_time()-cpu)
        with (output/'export_receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump(receipt,stream,indent=2,ensure_ascii=True,allow_nan=False);stream.write('\n')
        print(json.dumps(dict(status=receipt['status'],seed_metric_rows=receipt.get('seed_metric_rows'),
            arm_case_rows=receipt.get('arm_case_rows'),GPU_hours=0)),flush=True)
    if error is not None:raise SystemExit(1)


if __name__=='__main__':main()
