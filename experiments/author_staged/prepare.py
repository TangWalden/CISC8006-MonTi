"""Derive an isolated author-upstream registration from frozen split metadata.

Standard library only: no array, prediction, checkpoint-object, graph or model
read. Scientific assets are opaque-byte hashed; only existing registration,
official configuration and acquisition-receipt JSON are decoded.
"""
import argparse
import copy
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

SOURCE_ROOT=pathlib.Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:sys.path.insert(0,str(SOURCE_ROOT))
from experiments.author_staged.loader import AUTHOR_CHECKPOINT_SHA,AUTHOR_CONFIG_SHA,CONFIG

RUNTIME_COMMIT='ca093d8b12ea153a076b02bbdbc4ede58508560c'
COHORT_PROTOCOL_SHA='2716a20b1b8c251466e35f7e93b750a3ab72b668c38d0199f9026b9c884f675f'
COHORT_SOURCE='32d563f49e19cea3b74bc74f048ddd9a5a934074'
ASSET_PREFIX='runs/author_staged_20261112_attempt1/assets/'


def require(condition,message):
    if not condition:raise RuntimeError(message)
def sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def read_json(path):return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def write_json(path,value):
    with pathlib.Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
def relative_file(root,name):
    root=pathlib.Path(root).resolve();relative=pathlib.PurePosixPath(name)
    require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe relative input.')
    path=(root/str(relative)).resolve(strict=True)
    require(root in path.parents and path.is_file(),'Frozen input escapes historical root.')
    return path


def verify_source(manifest):
    commit=manifest.get('source_commit','')
    require(len(commit)==40 and all(v in '0123456789abcdef' for v in commit),'Committed source identity required.')
    required={'methods/downstream_quota.py','experiments/downstream_cooperation/common.py',
        'experiments/downstream_cooperation/model.py','experiments/downstream_cooperation/budget_audit.py',
        'tests/test_downstream_budget_audit.py','tests/test_downstream_quota.py','tests/test_downstream_model.py',
        'tests/test_author_staged_profile.py'}|{'experiments/final_staged/'+v for v in ('common.py','driver.py','run.py','preflight.py')}
    required|={'experiments/author_staged/'+v for v in ('__init__.py','loader.py','prepare.py','audit_saved.py','audit_tensors.py')}
    files=manifest['files'];require(required<=set(files),'Source manifest misses required reused/author-profile/audit files.')
    for name,wanted in files.items():
        relative=pathlib.PurePosixPath(name)
        allowed=(name in ('methods/__init__.py','methods/downstream_quota.py','tests/__init__.py')
            or (len(relative.parts)==3 and relative.parts[:2] in (('experiments','final_staged'),('experiments','author_staged'),('experiments','downstream_cooperation')) and relative.suffix=='.py')
            or (len(relative.parts)==2 and relative.parts[0]=='tests' and relative.suffix=='.py'))
        require(allowed and sha(relative_file(SOURCE_ROOT,name))==wanted,'Unexpected or changed isolated source: '+name)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runtime-root','source-manifest','cohort-protocol','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--author-checkpoint',default=ASSET_PREFIX+'GCN-GAGA-gossipcop_0.05-best_model.pickle')
    parser.add_argument('--author-config',default=ASSET_PREFIX+'GCN-GAGA-gossipcop_0.05-best_config.json')
    parser.add_argument('--retrieval-receipt',default=ASSET_PREFIX+'official_retrieval.json')
    args=parser.parse_args();runtime=pathlib.Path(args.runtime_root).resolve(strict=True);output=pathlib.Path(args.output).resolve()
    require((runtime/'runs') in output.parents and not output.exists(),'Use a fresh ignored registration directory.')
    manifest=read_json(args.source_manifest);verify_source(manifest)
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()==RUNTIME_COMMIT
        and not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=runtime,text=True).strip(),
        'Historical runtime must remain clean ca093; active formal jobs are never changed.')
    parent_path=pathlib.Path(args.cohort_protocol).resolve(strict=True)
    require((runtime/'runs') in parent_path.parents and sha(parent_path)==COHORT_PROTOCOL_SHA,'Exact frozen full-cohort metadata parent required.')
    parent=read_json(parent_path)
    require(parent['schema']=='monti_final_staged_v1' and parent['status']=='frozen'
        and parent['source_manifest']['source_commit']==COHORT_SOURCE and parent['runtime_commit']==RUNTIME_COMMIT
        and parent['owner_execution_authorized'] is True and parent['final_TEST_allowed'] is True,
        'Original registered full-cohort parent identity differs.')
    require(len(parent['train_groups'])==937 and len(parent['test_groups'])==909
        and sum(map(len,parent['train_members'].values()))==11211
        and sum(map(len,parent['test_members'].values()))==10515,'Full split metadata changed.')
    unused={ref['path'] for baseline in parent['formal_baselines'].values() for ref in baseline.values()
        if isinstance(ref,dict) and 'path' in ref}
    if parent.get('registry_ref'):unused.add(parent['registry_ref']['path'])
    refs=[copy.deepcopy(ref) for ref in parent['input_refs'] if ref['path'] not in unused]
    def retain(name,wanted=None):
        path=relative_file(runtime,name);actual=sha(path)
        require(wanted is None or actual==wanted,'Frozen official asset bytes changed: '+name)
        ref=dict(root='asset',path=name,sha256=actual)
        require(not any(v['path']==name and v['sha256']!=actual for v in refs),'Conflicting frozen reference.')
        if not any(v['path']==name for v in refs):refs.append(ref)
        return ref
    for ref in refs:require(sha(relative_file(runtime,ref['path']))==ref['sha256'],'Inherited scientific bytes changed.')
    for name in (args.author_checkpoint,args.author_config,args.retrieval_receipt):
        require(name.startswith(ASSET_PREFIX),'Official assets must stay in the separate ignored acquisition stage.')
    checkpoint_ref=retain(args.author_checkpoint,AUTHOR_CHECKPOINT_SHA);config_ref=retain(args.author_config,AUTHOR_CONFIG_SHA)
    retrieval_ref=retain(args.retrieval_receipt);retrieval=read_json(runtime/args.retrieval_receipt)
    require(read_json(runtime/args.author_config)==CONFIG,'Official737-byte configuration differs.')
    require(retrieval['status']=='completed' and retrieval['extracted_bytes']==7765119
        and retrieval['extracted_sha256']==AUTHOR_CHECKPOINT_SHA and retrieval['member']['crc32']==3103861045
        and retrieval['model_deserializations']==retrieval['model_calls']==0,'Official acquisition receipt differs.')
    parent_ref=retain(parent_path.relative_to(runtime).as_posix(),COHORT_PROTOCOL_SHA)
    protocol=copy.deepcopy(parent)
    for name in ('formal_baselines','registry_ref'):protocol.pop(name,None)
    protocol.update(schema='monti_author_staged_v1',execution_profile='author_upstream',guardian_lock='author_staged_exclusive.lock',
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(SOURCE_ROOT),
        source_manifest=manifest,input_refs=sorted(refs,key=lambda r:(r['root'],r['path'])),
        cohort_metadata_parent=dict(reference=parent_ref,source_commit=COHORT_SOURCE,metadata_only=True,no_new_array_or_prediction_reads=True),
        author_upstream=dict(checkpoint_ref=checkpoint_ref,config_ref=config_ref,retrieval_ref=retrieval_ref,
            distinct_upstream_checkpoints=1,author_configuration_seed=3,capture_noise_repetition_seeds=[0,1,2],
            downstream_training_seeds=[0,1,2],checkpoint_format='Strict plain model.state_dict mapping, verified only by registered TRAIN preflight',
            compatibility_status='unverified_until_TRAIN_preflight',source_archive='Official bdi-lab/MonTi linked best_models.zip',
            interpretation='One official frozen generator; three conditional downstream/capture-noise repetitions, not three author models or paper-table reproduction'),
        supplemental_study=True,historical_baseline_limit='Only freshly paired author-native/random/staged outputs are controls. Own-trained reproduction and paper tables are separate context.')
    protocol['settings'].update(scheduler_restore='fresh_official_inference_initial_schedulers_no_steps',
        capture_rng_epoch='synthetic_epoch0_seeded_per_case_no_historical_bitwise_claim',
        upstream_configuration_seed=3,distinct_author_upstream_states=1,
        capture_noise_seeds=[0,1,2],capture_rng_phase_labels='Unchanged formal_frozen_downstream_train/test and registered preflight label',
        interpretation='No upstream refit. Eval still samples auxiliary latent noise when node_budget>1; per-case RNG is frozen separately from author config seed3.')
    protocol['cohort']['interpretation']='Complete original split inherited from pinned metadata. Repeated targets and repeated TEST use are not untouched IID confirmation. Three repetitions are conditional on one official generator.'
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'metadata_scope.json',dict(schema='monti_author_staged_metadata_scope_v1',
        source_manifest=manifest,cohort_metadata_parent=protocol['cohort_metadata_parent'],
        author_upstream=protocol['author_upstream'],new_array_reads=0,prediction_reads=0,
        checkpoint_deserializations=0,model_calls=0,upstream_training_steps=0,GPU_hours=0,
        scope='Existing full-cohort registration/config/retrieval JSON plus opaque scientific-input byte hashes only. Active formal execution untouched.'))
    protocol['metadata_scope_sha256']=sha(output/'metadata_scope.json')
    write_json(output/'protocol.json',protocol)
    print(json.dumps(dict(status='frozen_author_profile_no_model_calls',protocol_sha256=sha(output/'protocol.json'),
        distinct_author_generators=1,downstream_capture_repetitions=[0,1,2],train_groups=937,test_groups=909,
        main_algorithm='Unchanged final_staged run/preflight/scorer/quota',GPU_hours=0)))


if __name__=='__main__':main()
