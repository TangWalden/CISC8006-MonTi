"""Register fresh ID-only VALID contexts and explicit source/runtime roots."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys

from runtime import (SOURCE_ROOT,RUNTIME_COMMIT,ARMS,SEEDS,TRAIN_PREFLIGHT,BASE_PATH,BASE_SHA,PREVIOUS_PATH,PREVIOUS_SHA,
    TEACHER_MANIFEST,TEACHER_SHA,TEACHER_CHECKPOINT_SHA,UUIDS,sha,load_json,write_json,relative_file,verify_source_manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root',required=True);parser.add_argument('--asset-root',required=True)
    parser.add_argument('--source-manifest',required=True);parser.add_argument('--synthetic-receipt',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    runtime=pathlib.Path(args.runtime_root).resolve();assets=pathlib.Path(args.asset_root).resolve();out=pathlib.Path(args.output).resolve()
    if runtime!=assets or (assets/'runs').resolve() not in out.parents:raise RuntimeError('Bind historical runtime/assets and a new private runs directory.')
    source_manifest=load_json(args.source_manifest);verify_source_manifest(SOURCE_ROOT,source_manifest)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()!=RUNTIME_COMMIT:
        raise RuntimeError('Historical canonical runtime must remainca093, not curated main.')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip():raise RuntimeError('Historical source tracked files dirty.')
    refs={}
    def retain(name,wanted=None,root_key='asset'):
        root=assets if root_key=='asset' else runtime;path=relative_file(root,name);actual=sha(path)
        if wanted is not None and wanted!=actual:raise RuntimeError('Frozen old input changed: '+name)
        key=(root_key,name)
        if refs.setdefault(key,actual)!=actual:raise RuntimeError('Conflicting input hash.')
        return dict(root=root_key,path=name,sha256=actual)
    retain(BASE_PATH,BASE_SHA);base=load_json(assets/BASE_PATH)
    ledger_ref=retain('evidence/research_cost_ledger.json');ledger=load_json(assets/ledger_ref['path'])
    remaining=ledger['limit']-ledger['measured_or_conservatively_charged']-ledger['unmetered_setup_reserve']
    if abs(remaining-ledger['remaining_after_reserve'])>1e-10 or remaining<2:
        raise RuntimeError('The additional two-GPU-hour phase must fit the actual remaining research ledger.')
    retain(PREVIOUS_PATH,PREVIOUS_SHA);previous=load_json(assets/PREVIOUS_PATH)
    for v in previous['inputs']+previous.get('retained_inputs',[]):retain(v['path'],v['sha256'])
    for name,wanted in (dict(previous['source_files'],**previous['excluded_sharpen_source_files'])).items():retain(name,wanted,'runtime')
    for name in ('course/configs/baseline.json','course/feature_polish/teacher.py','course/feature_polish/affine.py',
        'course/source_role_fix/saved_teacher_diagnostic.py','course/dual_surrogate/runtime.py','course/dual_surrogate/experiment.py',
        'course/dual_surrogate/common.py','course/dual_surrogate/verify.py','course/dual_surrogate/sequence_adapter.py',
        'course/feature_polish/common.py','course/feature_polish/refine.py','course/source_role_fix/common.py'):
        retain(name,root_key='runtime')
    retain(TEACHER_MANIFEST,TEACHER_SHA);teacher=load_json(assets/TEACHER_MANIFEST)
    checkpoint=(pathlib.PurePosixPath(TEACHER_MANIFEST).parent/teacher['checkpoint']).as_posix()
    retain(checkpoint,TEACHER_CHECKPOINT_SHA)
    if teacher['status']!='completed' or teacher['settings']['epochs']!=38 or teacher['full_train_nodes']!=6595:
        raise RuntimeError('Original full-TRAIN teacher identity differs.')
    retain('models/pretrained_models/victim_models/GAGA-gossipcop_0.05-0.pickle',
        '729b8624baac7429bd60e7bc2d3741a9d9ee7c4c7df921d618b53ceb0435c383')
    synthetic=load_json(args.synthetic_receipt)
    if (synthetic['status']!='passed' or synthetic['failures'] or synthetic['errors'] or synthetic.get('model_calls',0)
            or synthetic['tests_run']!=12 or synthetic['skipped']!=0 or synthetic['source_immutable'] is not True):
        raise RuntimeError('Passed source-bound synthetic checks required.')
    for name in ('methods/feature_assignment.py','tests/test_feature_assignment.py'):
        if (synthetic['source_hashes_before'].get(name)!=source_manifest['files'][name] or
                synthetic['source_hashes_after'].get(name)!=source_manifest['files'][name]):raise RuntimeError('Synthetic/core source differs.')
    target_sha='1f824f842988d84edcdfdfaa9df18f2b83d722c124b8f18fa9a022e5c910bce2'
    if source_manifest['files']['methods/target_reallocation.py']!=target_sha:
        raise RuntimeError('Exact previously tested and published target-reallocation core required.')
    retain('course/candidate_target_reallocation/core.py',target_sha,'runtime')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'metadata_scope.json',dict(source_manifest=source_manifest,runtime_commit=RUNTIME_COMMIT,
        input_refs=[dict(root=k[0],path=k[1],sha256=v) for k,v in refs.items()],numeric_check_receipt_sha256=sha(args.synthetic_receipt),
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_ID_array_reads_started=False,
        scope='Original VALID group-ID and target-membership metadata only; no labels/outcomes/checkpoint deserialization/models. Exclude three earlier128lists before fixedID-only draw.'))
    import numpy as np
    import scipy.sparse as sp
    population=np.load(assets/'Datasets/GossipCop/data_split/valid_target_mask_0.npy',allow_pickle=False)
    ids=population.astype(np.int64)
    if population.ndim!=1 or not np.array_equal(population,ids) or len(ids)!=592 or len(set(ids.tolist()))!=592:
        raise RuntimeError('Expected exact original592VALIDgroup IDs.')
    exclusions=[previous['excluded_original_valid_groups'],previous['excluded_sharpen_valid_groups'],previous['valid_groups']]
    excluded=set(v for row in exclusions for v in row)
    if any(len(row)!=len(set(row)) or len(row)!=128 for row in exclusions) or len(excluded)!=384:
        raise RuntimeError('Three disjoint earlier cohorts are required.')
    eligible=sorted(set(map(int,ids))-excluded)
    if len(eligible)!=208:raise RuntimeError('Expected208remaining original contexts; no replacement draw.')
    chosen=random.Random(20261111).sample(eligible,128)
    assignment=sp.load_npz(assets/'Datasets/GossipCop/data_split/target_assign_matrix_0.npz').tocsr()
    def members(g):return list(map(int,np.flatnonzero(assignment.getrow(g).toarray().reshape(-1))))
    chosen_members={str(g):members(g) for g in chosen};new_union={v for values in chosen_members.values() for v in values}
    old_union={v for g in excluded for v in members(g)}
    if not all(chosen_members.values()):raise RuntimeError('Empty selected context; preserve failure rather than replace.')
    device=dict(kind='cpu',cuda_visible_devices='',uuid=None,selection='No allowed GPU had8192MiBfree; fixed CPU fallback before Torch import')
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        available={parts[0].strip():(parts[1].strip(),int(parts[2].strip())) for parts in (line.split(',') for line in raw.splitlines())}
        for ordinal in ('0','1'):
            if ordinal in available and available[ordinal][0]==UUIDS[ordinal] and available[ordinal][1]>=8192:
                device=dict(kind='gpu',cuda_visible_devices=ordinal,uuid=UUIDS[ordinal],free_MiB_at_selection=available[ordinal][1]);break
    except (OSError,subprocess.CalledProcessError,ValueError):pass
    p=dict(schema='monti_feature_assignment_validation_v1',status='frozen',owner_execution_authorized=True,post_hoc=True,final_TEST_allowed=False,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_root=str(SOURCE_ROOT),runtime_root=str(runtime),asset_root=str(assets),source_manifest=source_manifest,
        runtime_commit=RUNTIME_COMMIT,runtime_tree=subprocess.check_output(['git','rev-parse',RUNTIME_COMMIT+'^{tree}'],cwd=runtime,text=True).strip(),
        input_refs=[dict(root=k[0],path=k[1],sha256=v) for k,v in refs.items()],
        valid_groups=chosen,seeds=list(SEEDS),arms=list(ARMS),train_groups=base['train_groups'],
        selection=dict(seed=20261111,original_population=list(map(int,ids)),excluded_ordered_cohorts=exclusions,eligible_sorted=eligible,
            selected_ordered=chosen,excluded_union_target_ids=sorted(old_union),no_size_label_outcome_selection=True,original_lists_unchanged=True),
        cohort=dict(members=chosen_members,groups=128,target_occurrences=sum(map(len,chosen_members.values())),unique_targets=len(new_union),
            overlap_with_previous_three_cohorts=len(new_union&old_union),new_unique_relative_to_previous_contexts=len(new_union-old_union),
            interpretation='Fresh contexts, not independent unseen news; adaptive previous development history remains.'),
        own_teacher_manifest=dict(path=TEACHER_MANIFEST,sha256=TEACHER_SHA,checkpoint_sha256=TEACHER_CHECKPOINT_SHA),
        budget=dict(ledger=ledger_ref,charged_GPU_hours=ledger['measured_or_conservatively_charged'],reserve_GPU_hours=ledger['unmetered_setup_reserve'],
            remaining_GPU_hours=remaining,new_GPU_cap_hours=2,no_ledger_mutation_during_job=True),
        settings=dict(acceptance_improvement=1e-7,iterations=1,loss='uniform_mean_softplus_fraud_minus_benign',gradient_point='baseline_x0'),
        device=device,resource=dict(GPU_phase_seconds=7200,CPU_phase_seconds=21600,main_inner_seconds=6000 if device['kind']=='gpu' else 18000,
            preflight_inner_seconds=900 if device['kind']=='gpu' else 1800,outer_extra_seconds=60,threads=8,automatic_retry=False),
        preflight=dict(groups=list(TRAIN_PREFLIGHT),sequence_atol=1e-5,logit_atol=1e-4,FD_epsilon=1e-4,FD_atol=1e-6,FD_rtol=5e-3),
        qualification=dict(primary='relation_assignment',mean_gain_at_least_pp=1,all_three_seed_deltas_positive=True,
            paired_t95_lower_positive=True,mean_above_both_controls=True,controls_not_candidates=True),
        logical_main_caps=dict(generations=384,own_forward_requests=3072,autograd_calls=768,official_GAGA_forwards=1664,clean_GCN_forwards=128,
            final_canonical_sequences=1536,optimizer_updates=0,API_calls=0,TEST_queries=0),
        synthetic_receipt_sha256=sha(args.synthetic_receipt),synthetic_receipt=synthetic,metadata_scope_sha256=sha(out/'metadata_scope.json'))
    write_json(out/'protocol.json',p)
    print(json.dumps(dict(status='frozen_no_model_calls',protocol_sha256=sha(out/'protocol.json'),device=device,
        target_occurrences=p['cohort']['target_occurrences'],unique_targets=p['cohort']['unique_targets'])))


if __name__=='__main__':main()
