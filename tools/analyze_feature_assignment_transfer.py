"""POST HOC saved-JSON transfer description; never a new method/selection test."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

ARMS=('baseline','relation_assignment','gcn_assignment','matched_random_permutation')
SEEDS=(3,4,5)
CHECKS=0
TEXT_COUNTS=collections.Counter()


def check(value,message):
    global CHECKS
    CHECKS+=1
    if not value:raise AssertionError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    value=json.loads(Path(path).read_text(encoding='utf-8'));TEXT_COUNTS['JSON_files_decoded']+=1;return value
def lines(path):
    values=[json.loads(v) for v in Path(path).read_text(encoding='utf-8').splitlines() if v.strip()]
    TEXT_COUNTS['JSONL_files_decoded']+=1;TEXT_COUNTS['JSONL_rows_decoded']+=len(values);return values
def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def classes(values,n):
    check(len(values)==n,'Incomplete binary-logit rows.')
    check(all(len(v)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in v) for v in values),'Invalid finite logits.')
    return [0 if row[0]>=row[1] else 1 for row in values]
def loss(values):
    margins=[float(v[1])-float(v[0]) for v in values]
    return math.fsum(v+math.log1p(math.exp(-v)) if v>=0 else math.log1p(math.exp(v)) for v in margins)/len(margins)
def transitions(labels,before,after):
    return dict(occurrences=len(labels),clean_wrong=sum(c!=y for c,y in zip(before,labels)),
        attacked_wrong=sum(z!=y for z,y in zip(after,labels)),new_wrong=sum(c==y and z!=y for c,z,y in zip(before,after,labels)),
        recoveries=sum(c!=y and z==y for c,z,y in zip(before,after,labels)))
def flips(before,after):
    return ['new_wrong' if b==1 and z==0 else 'recovery' if b==0 and z==1 else 'unchanged_wrong' if z==0 else 'unchanged_correct' for b,z in zip(before,after)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','stage','audit','input-manifest','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    wall,cpu=time.monotonic(),time.process_time();result=dict(status='failed',post_hoc=True,producer_author_reporting=True,human_verification=False)
    try:
        manifest=load(args.input_manifest);stage=Path(args.stage)
        expected={'cases.jsonl','evaluation/clean.jsonl'}|{'evaluation/%s_seed%d.jsonl'%(a,s) for a in ARMS for s in SEEDS}
        check(manifest['schema']=='feature_assignment_transfer_inputs_v1' and set(manifest['stage_files'])==expected,'Wrong frozen input inventory.')
        check(sha(args.protocol)==manifest['protocol_sha256'] and sha(args.audit)==manifest['audit_sha256'],'Protocol/audit bytes differ before parsing.')
        for relative,wanted in manifest['stage_files'].items():check(sha(stage/relative)==wanted,'Frozen stage bytes differ: '+relative)
        p=load(args.protocol);audit=load(args.audit)
        provenance=dict(study_source_commit=p['source_manifest']['source_commit'],runtime_commit=p['runtime_commit'],protocol_sha256=manifest['protocol_sha256'])
        check(audit['status']=='passed' and audit['provenance']==provenance and audit['neural_model_replay'] is False,'Matching passed independent audit required.')
        check(tuple(p['arms'])==ARMS and tuple(p['seeds'])==SEEDS and p['final_TEST_allowed'] is False,'Frozen development scope differs.')
        groups=p['valid_groups'];members={int(k):v for k,v in p['cohort']['members'].items()}
        check(len(groups)==len(set(groups))==128 and set(members)==set(groups),'Incomplete fixed group membership.')
        cases=lines(stage/'cases.jsonl');case_map={(v['seed'],v['group']):v for v in cases}
        check(len(case_map)==len(cases)==384 and set(case_map)=={(s,g) for s in SEEDS for g in groups},'Retain all384cases.')
        clean_rows=lines(stage/'evaluation/clean.jsonl');clean={v['group']:v for v in clean_rows}
        check(len(clean)==len(clean_rows)==128 and set(clean)==set(groups),'Incomplete original clean channel.')
        evaluated={}
        for seed in SEEDS:
            for arm in ARMS:
                rows=lines(stage/('evaluation/%s_seed%d.jsonl'%(arm,seed)));mapping={v['group']:v for v in rows}
                check(len(mapping)==len(rows)==128 and set(mapping)==set(groups),'Incomplete final evaluation stream.')
                for group,row in mapping.items():
                    source=case_map[seed,group];n=len(members[group])
                    check(row['seed']==seed and row['arm']==arm and row['node_ids']==source['node_ids']==members[group]
                        and row['labels']==source['labels']==clean[group]['labels'] and clean[group]['node_ids']==members[group]
                        and all(y==1 for y in row['labels']),'Target order/truth join differs.')
                    check(row['artifact_sha256']==source['artifact']['sha256']
                        and row['canonical_sequence_sha256']==source['final_sequence_hashes'][arm]
                        and row['feature_sha256']==source['final_invariants'][arm]['feature_sha256'],'Sealed native input binding differs.')
                    check(row['gcn_logits']==source['own_logits_final'][arm]['gcn'],'Cached original-GCN channel differs.')
                    prediction=classes(row['gaga_logits'],n);original=classes(clean[group]['gaga_logits'],n)
                    check(row['gaga']==transitions(row['labels'],original,prediction),'Official clean-to-attack transitions differ.')
                    evaluated[seed,group,arm]=prediction
        aggregates=collections.defaultdict(collections.Counter);outputs=[];private_map=[]
        for seed in SEEDS:
            for group in groups:
                source=case_map[seed,group];n=len(members[group]);key=hashlib.sha256(json.dumps([manifest['protocol_sha256'],seed,group,
                    'feature-assignment-publiccase-v1'],separators=(',',':')).encode()).hexdigest()[:20]
                own_base=classes(source['own_logits_final']['baseline']['gaga'],n);off_base=evaluated[seed,group,'baseline']
                gcn_base=classes(source['own_logits_final']['baseline']['gcn'],n);base_loss=loss(source['own_logits_final']['baseline']['gaga'])
                row=dict(opaque_case_key=key,seed=seed,targets=n,node_budget=source['node_budget'],edge_budget=source['edge_budget'],arms={})
                private_map.append(dict(opaque_case_key=key,seed=seed,group=group))
                for arm in ARMS:
                    own=classes(source['own_logits_final'][arm]['gaga'],n);official=evaluated[seed,group,arm]
                    own_flip,off_flip=flips(own_base,own),flips(off_base,official)
                    current_loss=loss(source['own_logits_final'][arm]['gaga']);delta=current_loss-base_loss
                    direction='improved' if delta < -1e-7 else 'worsened' if delta > 1e-7 else 'within_fixed_tolerance'
                    accepted=False if arm=='baseline' else source['acceptance'][arm]['accepted']
                    count=collections.Counter(occurrences=n,group_cases=1,accepted_group_cases=int(accepted),
                        own_baseline_wrong=own_base.count(0),own_final_wrong=own.count(0),official_baseline_wrong=off_base.count(0),
                        official_final_wrong=official.count(0),baseline_own_official_disagreement=sum(a!=b for a,b in zip(own_base,off_base)),
                        final_own_official_disagreement=sum(a!=b for a,b in zip(own,official)))
                    count['own_loss_'+direction+'_groups']+=1
                    for own_state,official_state in zip(own_flip,off_flip):
                        count['own_'+own_state]+=1;count['official_'+official_state]+=1
                        count['joint_own_'+own_state+'__official_'+official_state]+=1
                        if arm=='relation_assignment' and accepted and official_state in ('new_wrong','recovery'):
                            count['accepted_primary_official_'+('gain' if official_state=='new_wrong' else 'loss')+'__own_'+own_state]+=1
                    count['official_net_wrong']=official.count(0)-off_base.count(0)
                    count['own_net_wrong']=own.count(0)-own_base.count(0)
                    if accepted:
                        own_gcn=classes(source['own_logits_final'][arm]['gcn'],n)
                        check(all(a!=0 or b==0 for a,b in zip(gcn_base,own_gcn)),'Accepted own-GCN pointwise guard failed.')
                    aggregates[seed,arm].update(count)
                    row['arms'][arm]=dict(counts=dict(count),own_teacher_within_model_softplus_delta=delta,own_loss_direction=direction,accepted=accepted)
                outputs.append(row)
        metrics=[];N=p['cohort']['target_occurrences']
        check(sum(len(v) for v in members.values())==N and 3*N==audit['cohort']['correlated_occurrences_all_seeds'],'Dynamic full denominator differs.')
        for (seed,arm),count in sorted(aggregates.items()):
            check(count['occurrences']==N and count['group_cases']==128,'An arm/seed case was filtered.')
            check(count['official_net_wrong']==count['official_new_wrong']-count['official_recovery']
                and count['own_net_wrong']==count['own_new_wrong']-count['own_recovery'],'Baseline-attack transition decomposition differs.')
            metrics.append(dict(seed=seed,arm=arm,counts=dict(count)))
        for arm in ARMS[1:]:
            check(sum(aggregates[s,arm]['accepted_group_cases'] for s in SEEDS)==audit['acceptance_counts'][arm+'_accepted'],'Audit acceptance total differs.')
        ordered_gain=sorted(outputs,key=lambda r:(-r['arms']['relation_assignment']['counts']['official_net_wrong'],r['seed'],r['opaque_case_key']))
        ordered_loss=sorted(outputs,key=lambda r:(r['arms']['relation_assignment']['counts']['official_net_wrong'],r['seed'],r['opaque_case_key']))
        write(out/'all_case_counts.json',outputs);write(out/'private_case_key_map.json',private_map)
        write(out/'illustrative_cases.json',dict(selection='POST HOC maximum/minimum official net-wrong delta; deterministic seed/opaque-key ties, never method/cohort selection',
            largest_net_gain=ordered_gain[0],largest_net_loss=ordered_loss[0]))
        result.update(status='passed',provenance=provenance,metrics=metrics,all_group_seed_cases=384,
            target_occurrences_per_seed=N,correlated_occurrences_per_arm_all_seeds=3*N,unique_targets=p['cohort']['unique_targets'],
            input_manifest_sha256=sha(args.input_manifest),audit_result_sha256=sha(args.audit),source_sha256=sha(__file__),
            stage_input_hashes=manifest['stage_files'],output_hashes={name:sha(out/name) for name in ('all_case_counts.json','illustrative_cases.json')},
            interpretation=['Own-teacher and official transitions use the SAME sealed inputs; saved-record agreement is not a new independent model replay.',
                'New-wrong/recovery here is baseline-ATTACK to final arm, not original-clean to attack. Only official clean-to-attack counts were separately checked.',
                'Own-teacher original-clean predictions were not saved; no new clean inference was made.',
                'Own loss comparisons stay within one teacher. No cross-model logit-scale subtraction, calibration, sole-cause attribution or IID-news uncertainty.',
                'The common filter enforces own-loss improvement by construction; observed transfer associations do not establish a new method or selected winner.'])
    except BaseException as error:result.update(error=repr(error),traceback=traceback.format_exc())
    finally:
        result.update(checks=CHECKS,text_read_counters=dict(TEXT_COUNTS),process_cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-wall,
            dataset_reads=0,tensor_loads=0,model_calls=0,autograd_calls=0,optimizer_updates=0,GPU_hours=0,API_calls=0,TEST_queries=0)
        write(out/'result.json',result);print(json.dumps({k:result.get(k) for k in ('status','checks','process_cpu_seconds','wall_seconds','error')}))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
