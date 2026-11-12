"""Freeze a new development study from existing cohort metadata; no ML imports."""
import argparse,datetime,hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    with pathlib.Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);ap.add_argument('--manifest',required=True);a=ap.parse_args()
    stage=pathlib.Path(a.stage).resolve();runtime=stage.parent.parent
    assert stage.parent.name=='runs'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()=='ca093d8b12ea153a076b02bbdbc4ede58508560c'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip()
    parent=ROOT/'cohort_parent.json';assert sha(parent)=='5174bd650628b3300fcf6c091101d176cda304e4136a6801845c421ed451df13'
    q=json.loads(parent.read_text());manifest=json.loads(pathlib.Path(a.manifest).read_text())
    for n,h in manifest['files'].items():assert sha(ROOT/n)==h,n
    refs=[r for r in q['input_refs'] if r['path'].startswith(('Datasets/','models/pretrained_models/'))]
    for r in refs:assert sha(runtime/r['path'])==r['sha256'],r['path']
    author={}
    for role,n,wanted in [('checkpoint_ref','GCN-GAGA-gossipcop_0.05-best_model.pickle','9f5dc7dd594a8e8927449665ef3a07b777199c2436d6fd1cc1f7adf0fdb6c23d'),
                          ('config_ref','GCN-GAGA-gossipcop_0.05-best_config.json','e27fd6959f35ab1786f35e4e70c89bec5156da1a6577f4481287310ebf552c2a'),
                          ('retrieval_ref','official_retrieval.json',None)]:
        f=stage/'assets'/n;h=sha(f);assert wanted is None or h==wanted
        r=dict(root='asset',path=f.relative_to(runtime).as_posix(),sha256=h);refs.append(r);author[role]=r
    ledger=runtime/'evidence/research_cost_ledger.json';cost=json.loads(ledger.read_text())
    assert cost['remaining_after_reserve']>=0
    refs.append(dict(root='asset',path='evidence/research_cost_ledger.json',sha256=sha(ledger)))
    p=dict(schema='monti_local_swap_v1',status='frozen',owner_execution_authorized=True,final_TEST_allowed=False,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),runtime_root=str(runtime),
        runtime_commit='ca093d8b12ea153a076b02bbdbc4ede58508560c',source_manifest=manifest,input_refs=refs,author_upstream=author,
        train_groups=q['train_groups'],valid_groups=q['valid_groups'],train_members=q['train_members'],valid_members=q['valid_members'],
        seeds=[0,1,2],arms=['native','local_rank','matched_random','always_random'],cohort=q['cohort'],
        exposure='Reused development group contexts and target identities; not untouched confirmation. Prior final TEST outcomes known, but no TEST access or tuning in this run.',
        checkpoint_scope='One official author state; three capture-noise repetitions and separately fitted downstream models, not three independent author models.',
        settings=dict(pool_size=8,max_exchanges=1,training_target='native_softplus_minus_swapped_softplus_on_original_GCN',
            ridge_regularization=.1,feature_count=32,fit='one exact float64 zero-intercept ridge solve per seed',
            threshold='max(1e-6,0.1*TRAIN_gain_RMS)',policy='best pool prediction above threshold; otherwise identity',
            random='uniform fixed-seed draw from the same proposal pool; matched arm uses learned activation, always arm ignores activation',
            preserve='all original node features, target and injected-to-injected blocks, candidate identities/order, every injected physical degree and total budgets',
            training_importance='One row per sampled legal swap; proposal sampling is uniform over legal triples within a case, not uniform across all graph edges.',
            no_gradient_through_graph=True),
        resources=dict(device='cpu',threads=4,GPU_hours=0,API_calls=0,preflight_seconds=300,main_seconds=7200,outer_extra=30,automatic_retry=False),
        caps=dict(preflight_generations=2,preflight_GCN=6,main_generations=576,train_GCN=3456,evaluation_GCN=832,official_GAGA_VALID=832,ridge_fits=3,upstream_updates=0,TEST_queries=0),
        qualification=dict(primary='local_rank-minus-matched_random official GAGA occurrence MR',
            secondary='local_rank-minus-native',rule='Both contrasts mean > 0, all three differences > 0, and nominal paired t95 lower > 0; otherwise no reliable improvement established.',
            no_choice_cases='Identity retained in every denominator; no replacement or favorable case filtering.',
            uncertainty='Three conditional repeats, correlated target occurrences and prior adaptive development; nominal seed interval only.'),
        ledger_snapshot=dict(sha256=sha(ledger),charged_GPU_hours=cost['measured_or_conservatively_charged'],remaining_GPU_hours=cost['remaining_after_reserve']))
    assert len(p['train_groups'])==128 and len(p['valid_groups'])==64
    assert not set(n for ns in p['train_members'].values() for n in ns)&set(n for ns in p['valid_members'].values() for n in ns)
    reg=stage/'registration';reg.mkdir(exist_ok=False);write(reg/'protocol.json',p)
    print(json.dumps(dict(status='frozen',protocol_sha256=sha(reg/'protocol.json'),model_calls=0,checkpoint_deserializations=0,source_commit=manifest['source_commit'])))
if __name__=='__main__':main()
