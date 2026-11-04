"""Freeze an outcome-blind structural transformation after negative studies."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT,private_output,sha256,write_json

BASE='runs/feature_polish_20261104_attempt4'
BASE_SHA='eedad10194edbe96c0496aa04f514b9b2ec8ecd49e11979aa523f01d29caadc6'
PARITY='runs/feature_parity_20261104_attempt1/result_attempt1/result.json'
PARITY_SHA='caf3d16548e80336d758213ede04bce9827d44e7bf23558e34c7e8ad123c4bdf'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit all intended source before registration.')
    base_path=ROOT/BASE/'experiment_protocol_v1.json'
    if sha256(base_path)!=BASE_SHA:raise RuntimeError('The complete original paired experiment changed.')
    old=json.loads(base_path.read_text(encoding='utf-8'))
    if old['source_commit']!='8222dfa7bea7ba8115f31420e2a2420a93e5b5b1':
        raise RuntimeError('This study requires the exact completed original captures.')
    sources=dict(old['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Reused scientific code changed: '+name)
    sources.update({p.relative_to(ROOT).as_posix():sha256(p) for p in pathlib.Path(__file__).parent.glob('*.py')})
    inputs={}
    def retain(path,wanted=None):
        path=path.resolve()
        if ROOT not in path.parents:raise RuntimeError('Input outside the project.')
        value=sha256(path)
        if wanted is not None and value!=wanted:raise RuntimeError('A frozen input changed: '+str(path))
        name=path.relative_to(ROOT).as_posix();inputs[name]=value
        return dict(path=name,sha256=value)
    base_ref=retain(base_path,BASE_SHA)
    for row in old['inputs']+old.get('retained_inputs',[]):retain(ROOT/row['path'],row['sha256'])
    complete=ROOT/BASE/'queue_complete.json'
    if json.loads(complete.read_text(encoding='utf-8'))['status']!='completed':
        raise RuntimeError('The previous study did not complete.')
    retain(complete)
    parity_ref=retain(ROOT/PARITY,PARITY_SHA)
    parity=json.loads((ROOT/PARITY).read_text(encoding='utf-8'))
    parity_registration_path=(ROOT/PARITY).parent/'registered.json'
    retain(parity_registration_path,parity['output_hashes']['registered.json'])
    parity_registration=json.loads(parity_registration_path.read_text(encoding='utf-8'))
    if parity_registration['base_experiment_ref']!=base_ref:
        raise RuntimeError('The passed parity diagnostic belongs to a different source experiment.')
    if (parity['status']!='completed' or not parity['parity_passed'] or
            not parity['sequence_numeric_parity_passed'] or not parity['teacher_numeric_logit_parity_passed'] or
            not parity['teacher_argmax_identity'] or parity['counters']['official_victim_forwards']):
        raise RuntimeError('Resolve the registered canonical consistency question before structural efficacy calls.')
    synthetic=retain(ROOT/'evidence/target_degree_balance_v1/synthetic_receipt.json')
    retain(ROOT/'evidence/target_degree_balance_v1/synthetic_registration.json')
    retain(ROOT/'evidence/target_degree_balance_v1/synthetic_test.log')
    runs={}
    for seed in (3,4,5):
        directory=ROOT/BASE/'queue_attempt1/refinements'/('seed%d'%seed)
        result=json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if (result['status']!='completed' or result['seed']!=seed or result['groups']!=128 or
                result['protocol_sha256']!=BASE_SHA or result['source_commit']!=old['source_commit']):
            raise RuntimeError('A complete common capture matrix is required.')
        retain(directory/'registered.json')
        captures=[dict(group=row['group'],**retain(directory/row['path'],row['sha256'])) for row in result['exports']]
        if [row['group'] for row in captures]!=old['valid_groups']:
            raise RuntimeError('All128 original captured cases must remain in order.')
        runs[str(seed)]=dict(result_ref=retain(directory/'result.json'),
            groups_ref=retain(directory/'groups.jsonl',result['groups_jsonl_sha256']),captures=captures)
    ledger_path=ROOT/'evidence/research_cost_ledger.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    if ledger['measured_or_conservatively_charged']+ledger['unmetered_setup_reserve']>ledger['limit']:
        raise RuntimeError('The research budget is exhausted.')
    out=private_output(args.output,create=True)
    p=dict(schema='monti_target_degree_balance_v1',status='frozen',post_hoc=True,owner_execution_authorized=True,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_files=sources,
        base_experiment_ref=base_ref,parity_result_ref=parity_ref,
        inputs=[dict(path=name,sha256=value) for name,value in inputs.items()],runs=runs,
        seeds=[3,4,5],valid_groups=old['valid_groups'],arms=['baseline','balanced_target','matched_random'],
        deadline_seconds=5400,cpu_threads=8,execution_device='cpu',cuda_visible_devices='',
        synthetic_receipt_ref=synthetic,
        context='Prospective bounded development comparison after known negative feature refinements and complete canonical parity. Not preregistration of the project, a final TEST, or a claimed novel method.',
        preserved='All x0/original features, target membership/order, candidate order, full non-target adjacency block, each row target/total count, each physical injected degree and all edge-role totals.',
        transformation='Only reallocate captured target-prefix ones. Strictly decrease the sum of squared target incoming-injected degrees until its fixed-total floor/ceiling optimum. No model output, label or feature selects a swap.',
        random_control='Match final per-row removed/added target edge count; use SHA256-seeded Python Random.sample on original ones/zeros. Locked server Python3.8.20 replay, no cross-runtime identity claim.',
        geometry_gate='Seal all384captures and1152canonical input sequences with self/known1hop tokens0/1/2 identical before any classifier construction. If any seed has zero nonidentity balanced cases, stop the whole comparison without classification. Never filter cases.',
        qualification=dict(primary='balanced_target',min_mean_gain_pp=1.0,require_all_seed_deltas_positive=True,
            require_mean_above_matched_random=True,control_not_candidate=True,
            uncertainty='Report conditional paired t95 across3attack-generator seeds; repeated contexts are correlated and development was repeatedly inspected.'),
        expected_if_evaluated=dict(source_captures=384,geometry_graph_reconstructions=1152,
            canonical_sequence_updates=1152,evaluation_graph_reconstructions=1152,
            official_GAGA_forwards=1280,original_GCN_forwards=1280,clean_forwards_each=128,attacked_forwards_each=1152,
            fresh_teacher_or_generator_forwards=0,optimizer_steps=0,autograd_calls=0,API_calls=0,TEST_queries=0),
        budget_snapshot=dict(unit=ledger['unit'],limit=ledger['limit'],charged=ledger['measured_or_conservatively_charged'],
            reserve=ledger['unmetered_setup_reserve'],remaining=ledger['remaining_after_reserve'],
            ledger_sha256_at_registered_source=sha256(ledger_path),new_phase_GPU_hours=0),
        no_first_novelty_or_efficacy_guarantee=True,final_TEST_allowed=False,
        failure_rule='Retain every partial count/file/time; no automatic model retry or outcome-dependent seed/step changes.',
        source_scope='Geometry first; original GCN and sealed official GAGA load only for fresh paired development evaluation after the geometry marker. The independent full-TRAIN teacher and LLM are not used.')
    path=out/'protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),inputs=len(inputs),output=str(path))))


if __name__=='__main__':main()
