"""Bind inherited scientific bytes and all retained baseline models, no ML."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT,MASTER,sha256,write_json,private_output


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--continue-protocol');parser.add_argument('--reuse-teacher');parser.add_argument('--prior-progress')
    args=parser.parse_args()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,universal_newlines=True).strip():
        raise RuntimeError('Commit before freezing execution.')
    p=json.loads((ROOT/MASTER).read_text(encoding='utf-8'))
    ancestor=ROOT/p['ancestor_protocol']
    old=json.loads(ancestor.read_text(encoding='utf-8'))
    sources=dict(old['source_files'])
    for name,wanted in sources.items():
        if sha256(ROOT/name)!=wanted:raise RuntimeError('Inherited scientific source changed: '+name)
    sources.update({f.relative_to(ROOT).as_posix():sha256(f) for f in pathlib.Path(__file__).parent.glob('*.py')})
    sources[MASTER]=sha256(ROOT/MASTER)
    for name in sources:
        subprocess.check_output(['git','ls-files','--error-unmatch','--',name],cwd=ROOT,stderr=subprocess.STDOUT)
    for row in old['inputs']:
        if sha256(ROOT/row['path'])!=row['sha256']:raise RuntimeError('Scientific input changed: '+row['path'])
    retained=[dict(path=ancestor.relative_to(ROOT).as_posix(),sha256=sha256(ancestor))]
    baselines={}
    for seed in p['seeds']:
        directory=ROOT/p['baseline_directory']/('legacy_seed%d'%seed)
        result_path=directory/'result.json'
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if (result['status']!='completed' or result['seed']!=seed or result['arm']!='legacy' or
                result['source_mode']!='legacy' or result['optimizer_steps']!=2048 or result['epochs']!=8 or
                sha256(directory/'final_model.pt')!=result['final_model_sha256']):
            raise RuntimeError('Retained original-GCN baseline is not the registered terminal model.')
        baselines[str(seed)]=dict(directory=directory.relative_to(ROOT).as_posix(),
            final_model_sha256=result['final_model_sha256'],result_sha256=sha256(result_path))
        for name in ('result.json','registered.json','initial_state.json','final_model.pt'):
            f=directory/name;retained.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)))
    initial_ref=ROOT/p['teacher_initial_reference']
    initial=json.loads(initial_ref.read_text(encoding='utf-8'))
    # The exact old teacher initialization is known before any new fit, whereas
    # its fitted weights are never reused or deserialized here.
    teacher=initial['teachers']['gaga']
    expected=teacher.get('initial_state_sha256',teacher.get('initial_model_state_sha256'))
    if not expected:raise RuntimeError('Missing original independent GAGA initialization hash.')
    if expected!=p['teacher_settings']['reference_initial_state_sha256']:
        raise RuntimeError('Original independent GAGA initialization differs from the committed reference.')
    retained.append(dict(path=initial_ref.relative_to(ROOT).as_posix(),sha256=sha256(initial_ref)))
    continuation={}
    if args.continue_protocol:
        if not args.reuse_teacher or not args.prior_progress:
            raise RuntimeError('A continuation must explicitly retain teacher and all prior accounting.')
        previous=private_output(args.continue_protocol)
        previous_protocol=json.loads(previous.read_text(encoding='utf-8'))
        allowed={'course/feature_polish/refine.py','course/feature_polish/prepare_protocol.py','course/feature_polish/queue.py'}
        for name,wanted in previous_protocol['source_files'].items():
            if name not in allowed and sha256(ROOT/name)!=wanted:
                raise RuntimeError('Scientific settings/source changed during CPU-loader continuation: '+name)
        teacher=private_output(args.reuse_teacher)
        result=json.loads((teacher/'result.json').read_text(encoding='utf-8'))
        manifest=json.loads((teacher/'teacher_artifacts.json').read_text(encoding='utf-8'))
        manifest_ref=dict(path=(teacher/'teacher_artifacts.json').relative_to(ROOT).as_posix(),sha256=sha256(teacher/'teacher_artifacts.json'))
        if (manifest_ref!=previous_protocol['teacher_ref'] or
                manifest['provenance']['source_commit']!=previous_protocol['teacher_training_source'] or
                manifest['provenance']['protocol_sha256']!=previous_protocol['teacher_preparation_protocol_sha256']):
            raise RuntimeError('Reused teacher is not the exact one bound before the failed preflight.')
        if (result['status']!='completed' or result['counters']!=dict(optimizer_steps=494,training_forwards=494,final_train_forwards=13) or
                sha256(teacher/'teacher_artifacts.json')!=result['teacher_artifacts_sha256'] or
                sha256(teacher/manifest['checkpoint'])!=manifest['checkpoint_sha256'] or
                manifest['initial_state_sha256']!=expected or manifest['official_victim_loaded'] or manifest['TEST_used']):
            raise RuntimeError('The successfully completed full-TRAIN teacher changed.')
        progress=private_output(args.prior_progress)
        prior=json.loads(progress.read_text(encoding='utf-8'))
        if any(row['name'].startswith('refine_seed') or row['name']=='development_evaluation' for row in prior['jobs']):
            raise RuntimeError('This loader amendment only applies before all refinement/victim work.')
        old_stage=pathlib.Path(previous_protocol['stage_root'])
        if progress.parent.parent!=old_stage:
            raise RuntimeError('Prior accounting belongs to another phase stage.')
        failed_preflight=old_stage/'preflight_attempt1/result.json'
        failure=json.loads(failed_preflight.read_text(encoding='utf-8'))
        if (failure['status']!='failed' or failure['official_victim_loaded'] or
                any(value!=0 for value in failure['counts'].values())):
            raise RuntimeError('This CPU-loader-only continuation requires the preserved zero-forward failure.')
        launch=old_stage/'launch_receipt.json';launch_record=json.loads(launch.read_text(encoding='utf-8'))
        synthetic=old_stage/'queue_attempt1/synthetic_receipt.json'
        synthetic_record=json.loads(synthetic.read_text(encoding='utf-8'))
        synthetic_log=synthetic.parent/'synthetic_checks.log'
        if (synthetic_record['status']!='passed' or synthetic_record['exit_code'] or
                sha256(synthetic_log)!=synthetic_record['stdout_sha256']):
            raise RuntimeError('The retained unchanged affine CPU checks did not pass.')
        for row in prior['jobs']:
            guardian=progress.parent/(row['name']+'_guardian.json')
            if sha256(guardian)!=row['guardian_sha256']:
                raise RuntimeError('A prior process guardian changed.')
            retained.append(dict(path=guardian.relative_to(ROOT).as_posix(),sha256=sha256(guardian)))
        for f in (previous,progress,launch,synthetic,synthetic_log,failed_preflight,teacher/'result.json',teacher/'registered.json',teacher/'teacher_artifacts.json',teacher/manifest['checkpoint']):
            retained.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)))
        continuation=dict(reuse_teacher_directory=teacher.relative_to(ROOT).as_posix(),
            teacher_training_source=manifest['provenance']['source_commit'],
            teacher_preparation_protocol_sha256=manifest['provenance']['protocol_sha256'],
            prior_progress_ref=dict(path=progress.relative_to(ROOT).as_posix(),sha256=sha256(progress)),
            reuse_synthetic_ref=dict(path=synthetic.relative_to(ROOT).as_posix(),sha256=sha256(synthetic)),
            reuse_execution_resource={key:previous_protocol[key] for key in ('execution_device','cuda_visible_devices','physical_gpu_uuid','resource_selection')},
            phase_origin_started_utc=launch_record['started_utc'],
            construction_amendment=dict(previous_source=previous_protocol['source_commit'],previous_protocol_sha256=sha256(previous),
                reason='CPU-safe original-GCN checkpoint deserialization only; retain completed teacher and unchanged passed affine tests. Failed preflight had zero generations/classifier forwards/autograd/updates/victim calls. Method/settings unchanged.'))
    stage=private_output(args.stage,create=True)
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=old['inputs'],retained_inputs=retained,baseline_runs=baselines,
        teacher_reference_initial_state_sha256=expected,stage_root=str(stage),
        execution_device=None,cuda_visible_devices='',physical_gpu_uuid=None)
    p.update(continuation)
    path=stage/'preparation_protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
