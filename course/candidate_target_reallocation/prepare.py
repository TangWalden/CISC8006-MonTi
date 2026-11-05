"""Freeze a third ID-only VALID-context draw before structural model calls."""
import argparse
import datetime
import hashlib
import json
import pathlib
import random
import subprocess
import sys

from course.feature_polish.common import ROOT,private_output,sha256,write_json
from course.candidate_target_reallocation.run import EXPECTED_COUNTS


BASE='runs/feature_polish_20261104_attempt4/experiment_protocol_v1.json'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
BASE_SOURCE='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1'
SHARPEN_SHA='d40fc7d8c01b1eac1f8afac72e215161897fe8296206bdc194e4821d0a050447'
SHARPEN_SOURCE='43e46e37621f203a51bbdfdb1667f5466e73ad68'
SELECTION_SEED=20261106


def compact_hash(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('ascii')).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sharpen-protocol',required=True)
    parser.add_argument('--sharpen-protocol-sha256',required=True)
    parser.add_argument('--synthetic-receipt',required=True)
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit all intended source before cohort metadata extraction.')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();inputs={}
    def retain(path,wanted=None):
        path=pathlib.Path(path);path=(ROOT/path).resolve() if not path.is_absolute() else path.resolve()
        if ROOT not in path.parents:raise RuntimeError('Frozen input outside project.')
        actual=sha256(path)
        if wanted is not None and actual!=wanted:raise RuntimeError('Frozen input changed: '+str(path))
        name=path.relative_to(ROOT).as_posix()
        if inputs.setdefault(name,actual)!=actual:raise RuntimeError('Conflicting frozen input hashes.')
        return dict(path=name,sha256=actual)
    base_ref=retain(BASE,BASE_SHA);old=json.loads((ROOT/BASE).read_text(encoding='utf-8'))
    if old['source_commit']!=BASE_SOURCE or len(old['train_groups'])!=256:
        raise RuntimeError('The exact source-role legacy terminal-eight family is required.')
    sources=dict(old['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Reused scientific implementation changed: '+name)
    sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    for row in old['inputs']+old.get('retained_inputs',[]):retain(row['path'],row['sha256'])
    for seed in (3,4,5):
        entry=old['baseline_runs'][str(seed)];directory=pathlib.Path(entry['directory'])
        retain(directory/'result.json',entry['result_sha256']);retain(directory/'final_model.pt',entry['final_model_sha256'])
    if args.sharpen_protocol_sha256!=SHARPEN_SHA:raise RuntimeError('The exact previous context protocol SHA is required.')
    sharpen_ref=retain(args.sharpen_protocol,SHARPEN_SHA)
    sharpen=json.loads((ROOT/sharpen_ref['path']).read_text(encoding='utf-8'))
    if (sharpen['schema']!='monti_feature_logit_sharpen_v1' or sharpen['source_commit']!=SHARPEN_SOURCE or
            sharpen['base_experiment_ref']!=base_ref or sharpen['excluded_valid_groups']!=old['valid_groups']):
        raise RuntimeError('The completed second-context registration does not match the baseline family.')
    sharpen_sources=dict(sharpen['source_files'])
    for name,wanted in sharpen_sources.items():
        retain(name,wanted)
        committed=subprocess.check_output(['git','show',SHARPEN_SOURCE+':'+name],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=wanted:raise RuntimeError('Prior context source commit does not resolve its frozen bytes.')
    synthetic_ref=retain(args.synthetic_receipt)
    synthetic=json.loads((ROOT/synthetic_ref['path']).read_text(encoding='utf-8'))
    if (synthetic['status']!='passed' or synthetic['tests_run']!=11 or synthetic['failures'] or synthetic['errors'] or
            synthetic['real_data_reads'] or synthetic['model_calls'] or synthetic['GPU_hours'] or synthetic.get('API_calls',0)):
        raise RuntimeError('Frozen meaningful synthetic checks must pass before fresh metadata extraction.')
    for name in ('core.py','synthetic_tests.py'):
        key='course/candidate_target_reallocation/'+name
        if synthetic['source_files'][key]!=sources[key]:raise RuntimeError('The synthetic checks used different new core/test bytes.')
    names=dict(graph=old['data']['graph'],valid_group_ids='Datasets/GossipCop/data_split/valid_target_mask_0.npy',
        target_assignment='Datasets/GossipCop/data_split/target_assign_matrix_0.npz')
    reference_data={}
    for role,name in names.items():
        if name not in inputs:raise RuntimeError('Metadata missing from the original frozen input inventory.')
        reference_data[role]=retain(name,inputs[name])
    ledger_path=ROOT/'evidence/research_cost_ledger.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger['measured_or_conservatively_charged']+ledger['unmetered_setup_reserve']>ledger['limit']:
        raise RuntimeError('Research allowance is exhausted.')
    out=private_output(args.output,create=True)
    write_json(out/'metadata_scope.json',dict(source_commit=source,source_files=sources,
        registered_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_data=reference_data,
        base_experiment_ref=base_ref,excluded_sharpen_protocol_ref=sharpen_ref,excluded_sharpen_source_files=sharpen_sources,
        selection_seed=SELECTION_SEED,draw_count=128,new_group_ID_arrays_opened=False,
        scope='Read only original VALID group-ID and original membership CSR metadata after committed source. Exclude both earlier lists, sample128IDs before examining sizes, then record complete unchanged memberships. No graph labels, checkpoints, models or prediction arrays are deserialized.'),exclusive=True)
    import numpy as np
    import scipy.sparse as sp
    raw=np.load(ROOT/reference_data['valid_group_ids']['path'],allow_pickle=False)
    if raw.ndim!=1 or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise RuntimeError('Original VALID IDs must be a finite numeric vector.')
    integer=raw.astype(np.int64)
    if (not np.array_equal(raw.astype(np.float64),integer.astype(np.float64)) or
            len(integer)!=len(set(map(int,integer))) or (integer<0).any()):
        raise RuntimeError('Original group IDs must be exact, unique and nonnegative.')
    population=set(map(int,integer));excluded_original=list(map(int,old['valid_groups']))
    excluded_sharpen=list(map(int,sharpen['valid_groups']))
    if (len(population)!=592 or len(excluded_original)!=128 or len(set(excluded_original))!=128 or
            len(excluded_sharpen)!=128 or len(set(excluded_sharpen))!=128 or
            set(excluded_original)&set(excluded_sharpen) or not (set(excluded_original)|set(excluded_sharpen))<=population):
        raise RuntimeError('Preserve exactly the original592VALIDcontexts and two disjoint128context exclusions.')
    excluded=sorted(set(excluded_original)|set(excluded_sharpen));eligible=sorted(population-set(excluded))
    if len(excluded)!=256 or len(eligible)!=336:raise RuntimeError('The registered third-draw population differs.')
    selected=random.Random(SELECTION_SEED).sample(eligible,128)
    assignment=sp.load_npz(ROOT/reference_data['target_assignment']['path']).tocsr()
    if max(population)>=assignment.shape[0]:raise RuntimeError('Group IDs exceed the original membership matrix.')
    def members(group):
        values=np.flatnonzero(assignment.getrow(group).toarray().reshape(-1))
        if not len(values):raise RuntimeError('A selected original group is empty; preserve failure without substitution.')
        return set(map(int,values))
    previous_members={group:members(group) for group in excluded};previous_union=set().union(*previous_members.values())
    selected_members={group:members(group) for group in selected};union=set().union(*selected_members.values())
    selection=dict(algorithm='python_random_sample_sorted_eligible_v1',rng_seed=SELECTION_SEED,draw_count=128,
        original_valid_group_count=len(population),eligible_count=len(eligible),excluded_group_count=len(excluded),
        ordered_groups_sha256=compact_hash(selected),runtime=sys.version,no_group_size_or_outcome_selection=True,
        no_target_list_regrouping=True)
    cohort=dict(target_occurrences_per_seed=sum(map(len,selected_members.values())),unique_targets=len(union),
        original_node_count=assignment.shape[1],overlap_unique_targets_with_either_previous_context_list=len(union&previous_union),
        new_unique_targets_relative_to_previous_context_lists=len(union-previous_union),
        group_target_counts={str(group):len(selected_members[group]) for group in selected},
        target_ids_sha256=compact_hash(sorted(union)),excluded_target_ids_sha256=compact_hash(sorted(previous_union)),
        interpretation='Third ID-only context draw, not independent unseen news or a confirmatory untouched validation population. All original target lists and memberships are unchanged; overlap is reported.')
    p=dict(schema='monti_candidate_target_reallocation_v1',scientific_version='candidate-target-reallocation-v1',
        status='frozen',post_hoc=True,owner_execution_authorized=True,source_commit=source,source_files=sources,
        inputs=[dict(path=name,sha256=value) for name,value in inputs.items()],base_experiment_ref=base_ref,
        excluded_sharpen_protocol_ref=sharpen_ref,excluded_sharpen_source_files=sharpen_sources,
        excluded_original_valid_groups=excluded_original,excluded_sharpen_valid_groups=excluded_sharpen,
        excluded_valid_groups=excluded,synthetic_receipt_ref=synthetic_ref,reference_data=reference_data,
        seeds=[3,4,5],valid_groups=selected,selection=selection,cohort_metadata=cohort,
        arms=['baseline','balanced_target_fill','random_target_fill','candidate_rewire'],
        execution_device='cpu',cuda_visible_devices='',cpu_threads=8,deadline_seconds=5400,outer_deadline_seconds=5460,
        expected_if_evaluated=dict(EXPECTED_COUNTS),
        baseline_family='Exact base8222/eedad source-role legacy terminal8 seeds3/4/5; unchanged generator/candidate/scorer/scheduler/h_GCN. Deep-clone old protocol and change only valid_groups. Three preparatoryGCN loads with zero forward calls,3generator loads and384generator calls are spent.',
        intervention='Per row k=min(original candidate edges,original target nonedges,original candidate nonedges). The same SHA-keyed original candidate-edge donors are removed in every nonbaseline arm. Primary adds missing target edges ordered by ONCE-computed BASELINE total injected-attachment column count then frozen SHA ties; it does not use full original graph degrees. Random target-fill samples baseline target nonedges; candidate-rewire samples baseline candidate nonedges. All use the same row k.',
        preserved='Every x0 feature byte, all original-original edges/features, target/candidate membership/order, injected-injected block, every original target edge, each row edge count, every physical injected degree, total injected nodes and nonself undirected edges. Target-vs-candidate role counts intentionally change only in target-fill arms.',
        canonical_gate='All384fresh baseline captures and1536topologies/full canonical input sequences seal before any dedicated evaluator or official-victim loading. Only original SELF token0 is required identical; known/unknown relation and second-hop tokens may change and are measured.',
        headroom_stop_rule='After all384generations/1536canonical variants seal, if any seed has zero nonidentity primary topologies, retain all work and stop before dedicated GCN/GAGA loading or queries. This still spends384generator calls and3preparatoryGCN loads.',
        qualification=dict(primary='balanced_target_fill',min_mean_gain_pp=1.0,require_all_seed_deltas_positive=True,
            require_paired_t95_lower_above_zero=True,require_mean_above_candidate_rewire=True,
            random_target_comparator_need_not_be_worse=True,controls_not_candidates=True,
            uncertainty='Nominal conditional paired t95 over3generator seeds/df2; repeated news and adaptively inspected development history are correlated. Not confirmatory population, teacher or generalization uncertainty.'),
        pre_outcome_design_amendment=dict(from_design='source-only draft v0',to_design='frozen v1',
            new_context_metadata_opened=False,new_model_calls=0,
            change='Primary targets candidate-to-target ROLE allocation, not superior target ranking. Drop a required win over random_target_fill; retain it as a full comparator, never a promoted primary. Require a win over matched candidate_rewire instead.',
            reason='When common-feasibility k equals all missing target slots, both target-fill choices are algebraically identical. This observation is about the rule, not measured new-cohort headroom or outcomes.'),
        budget_snapshot=dict(unit=ledger['unit'],limit=ledger['limit'],charged=ledger['measured_or_conservatively_charged'],
            reserve=ledger['unmetered_setup_reserve'],remaining=ledger['remaining_after_reserve'],
            ledger_sha256_at_registered_source=sha256(ledger_path),new_phase_GPU_hours=0),
        context='A separate bounded structural development hypothesis after completed negative/insufficient earlier attempts, not a claim of novel balancing, camouflage, cost savings, article realizability or guaranteed improvement.',
        metadata_scope_sha256=sha256(out/'metadata_scope.json'),automation_recreated=False,final_TEST_allowed=False,
        failure_rule='Retain every partial case/file/counter/time. No automatic retries, outcome-dependent cohort/seed/cap expansion, dropped cases or promotion of a favorable control.')
    path=out/'protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=source,protocol_sha256=sha256(path),
        original_VALID_groups=len(population),eligible_groups=len(eligible),selected_groups=len(selected),
        target_occurrences_per_seed=cohort['target_occurrences_per_seed'],unique_targets=cohort['unique_targets'],
        inputs=len(inputs),output=str(path))))


if __name__=='__main__':main()
