"""Bind inherited scientific bytes and all retained baseline models, no ML."""
import argparse
import json
import pathlib
import subprocess

from course.feature_polish.common import ROOT,MASTER,sha256,write_json,private_output


def timer_retry(p,previous,teacher,progress,retained,expected_initial):
    """A bounded CPU retry of an incomplete process, never outcome selection."""
    old=json.loads(previous.read_text(encoding='utf-8'))
    if (old['source_commit']!='9a42793ccbff5d3783e001619c60449a56c00bdf' or
            sha256(previous)!='86377a95a6acfecd46b1e71440a1d749a1b619f9cd1ecf1815c9e748ed38be9a'):
        raise RuntimeError('Timer retry must bind the exact pre-outcome failed experiment.')
    previous_master=json.loads(subprocess.check_output(['git','show',old['source_commit']+':'+MASTER],cwd=ROOT).decode())
    timers=dict(refine_job_wall_seconds=5400,evaluation_wall_seconds=5400,phase_total_wall_hours=6)
    amended=dict(previous_master,**timers)
    if p!=amended or (old['execution_device'],old['cuda_visible_devices'],old['physical_gpu_uuid'])!=('cpu','',None):
        raise RuntimeError('Only the three registered CPU wall timers may change.')
    allowed={'course/feature_polish/prepare_protocol.py','course/feature_polish/queue.py',MASTER,
             'course/feature_polish/refine.py','course/feature_polish/guard.py'}
    for name,wanted in old['source_files'].items():
        if name not in allowed and sha256(ROOT/name)!=wanted:
            raise RuntimeError('Timer retry changed frozen scientific source: '+name)
    replacements={
        'course/feature_polish/refine.py':("gate['provenance']['source_commit']!=provenance['source_commit']",
            "gate['provenance']['source_commit']!=protocol.get('preflight_execution_source',provenance['source_commit'])"),
        'course/feature_polish/guard.py':('1<=args.limit_seconds<=1800','1<=args.limit_seconds<=5400')}
    import hashlib
    for name,(before,after) in replacements.items():
        original=subprocess.check_output(['git','show',old['source_commit']+':'+name],cwd=ROOT)
        if (hashlib.sha256(original).hexdigest()!=old['source_files'][name] or
                original.count(before.encode())!=1 or (ROOT/name).read_bytes()!=original.replace(before.encode(),after.encode())):
            raise RuntimeError('Timer retry changed numerical/guardian logic beyond its exact allowed replacement: '+name)
    old_stage=pathlib.Path(old['stage_root'])
    if progress.parent.parent!=old_stage:
        raise RuntimeError('The cumulative progress belongs to another stage.')
    prior=json.loads(progress.read_text(encoding='utf-8'))
    if ([row['name'] for row in prior['jobs']]!=['full_train_teacher','real_train_preflight','real_train_preflight','real_train_preflight','refine_seed3'] or
            [row['exit_code']==0 for row in prior['jobs']]!=[True,False,False,True,False] or
            (old_stage/'evaluation_attempt1').exists()):
        raise RuntimeError('Timer retry requires only the first incomplete seed and no victim evaluation.')
    failed_directory=old_stage/'queue_attempt1/refinements/seed3'
    failed_path=failed_directory/'result.json';failure=json.loads(failed_path.read_text(encoding='utf-8'))
    counts=failure['counters']
    if (failure['status']!='failed' or failure['seed']!=3 or 'TimeoutError' not in failure.get('error','') or
            any(counts[key]!=0 for key in ('generator_optimizer_steps','teacher_optimizer_steps','official_victim_forwards','API_calls','TEST_queries'))):
        raise RuntimeError('The failed refinement must be an unselected no-update/no-victim timeout.')
    guardian=ROOT/prior['jobs'][-1]['guardian_path'];g=json.loads(guardian.read_text(encoding='utf-8'))
    if (sha256(guardian)!=prior['jobs'][-1]['guardian_sha256'] or g['status']!='failed' or
            g['module']!='course.feature_polish.refine' or g['limit_seconds']!=1800 or
            g['execution_device']!='cpu' or g.get('exit_code')!=1 or g['elapsed_seconds']<1799):
        raise RuntimeError('The original CPU deadline receipt does not match the failed module.')
    tree=ROOT/'evidence/feature_polish_v1/build_attempt3_failed_tree.json'
    tree_rows=json.loads(tree.read_text(encoding='utf-8'))
    actual={path.relative_to(ROOT).as_posix() for path in failed_directory.rglob('*') if path.is_file()}
    if actual!={row['path'] for row in tree_rows}:
        raise RuntimeError('The preserved failed refinement tree differs.')
    for row in tree_rows:
        f=ROOT/row['path']
        if sha256(f)!=row['sha256'] or f.stat().st_size!=row['bytes']:
            raise RuntimeError('A failed partial record changed: '+row['path'])
        retained.append(dict(path=row['path'],sha256=row['sha256']))
    complete_groups=sum(bool(line.strip()) for line in (failed_directory/'groups.jsonl').read_bytes().splitlines())
    if complete_groups!=118:
        raise RuntimeError('Unexpected preserved whole-group count; do not silently reuse or discard partial work.')
    gate_ref=old['preflight_ref'];gate_path=ROOT/gate_ref['path'];gate=json.loads(gate_path.read_text(encoding='utf-8'))
    preprotocol=old_stage/'refinement_protocol_v1.json'
    if (sha256(gate_path)!=gate_ref['sha256'] or gate['status']!='passed' or gate['official_victim_loaded'] or
            gate['counts']['optimizer_steps'] or gate['counts']['official_victim_forwards'] or
            gate['provenance']!=dict(source_commit=old['source_commit'],protocol_sha256=sha256(preprotocol)) or
            sha256(ROOT/'course/feature_polish/preflight.py')!=old['source_files']['course/feature_polish/preflight.py']):
        raise RuntimeError('Reuse only the exact passed unchanged functional check, with its original source.')
    manifest_path=teacher/'teacher_artifacts.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    result=json.loads((teacher/'result.json').read_text(encoding='utf-8'))
    manifest_ref=dict(path=manifest_path.relative_to(ROOT).as_posix(),sha256=sha256(manifest_path))
    if (manifest_ref!=old['teacher_ref'] or result['status']!='completed' or
            manifest['provenance']!=dict(source_commit=old['teacher_training_source'],protocol_sha256=old['teacher_preparation_protocol_sha256']) or
            manifest['initial_state_sha256']!=expected_initial or
            result['teacher_artifacts_sha256']!=manifest_ref['sha256'] or manifest['official_victim_loaded'] or manifest['TEST_used'] or
            sha256(teacher/manifest['checkpoint'])!=manifest['checkpoint_sha256'] or
            result['counters']!=dict(optimizer_steps=494,training_forwards=494,final_train_forwards=13)):
        raise RuntimeError('Do not repeat or substitute the completed original teacher.')
    synthetic_ref=old['reuse_synthetic_ref'];synthetic=ROOT/synthetic_ref['path'];s=json.loads(synthetic.read_text(encoding='utf-8'))
    if (sha256(synthetic)!=synthetic_ref['sha256'] or s['status']!='passed' or s['exit_code'] or
            sha256(synthetic.parent/'synthetic_checks.log')!=s['stdout_sha256']):
        raise RuntimeError('The unchanged passed synthetic checks are not bound.')
    retained.extend(old['retained_inputs'])
    for row in prior['jobs']:
        f=ROOT/row['guardian_path']
        if sha256(f)!=row['guardian_sha256']:raise RuntimeError('A prior process receipt changed.')
        retained.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)))
    for f in (previous,progress,old_stage/'launch_receipt.json',old_stage/'preparation_protocol_v1.json',
              old_stage/'queue_attempt1/queue_failed.json',old_stage/'queue_attempt1/refine_seed3.log',preprotocol,gate_path,tree,
              teacher/'result.json',teacher/'registered.json',manifest_path,teacher/manifest['checkpoint']):
        retained.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=sha256(f)))
    return dict(reuse_teacher_directory=teacher.relative_to(ROOT).as_posix(),
        teacher_training_source=old['teacher_training_source'],teacher_preparation_protocol_sha256=old['teacher_preparation_protocol_sha256'],
        prior_progress_ref=dict(path=progress.relative_to(ROOT).as_posix(),sha256=sha256(progress)),
        reuse_synthetic_ref=synthetic_ref,reuse_passed_preflight_ref=gate_ref,
        preflight_execution_source=old['source_commit'],
        preflight_refinement_protocol_ref=dict(path=preprotocol.relative_to(ROOT).as_posix(),sha256=sha256(preprotocol)),
        failed_refinement_tree_ref=dict(path=tree.relative_to(ROOT).as_posix(),sha256=sha256(tree)),
        failed_refinement_ref=dict(path=failed_path.relative_to(ROOT).as_posix(),sha256=sha256(failed_path)),
        reuse_execution_resource={key:old[key] for key in ('execution_device','cuda_visible_devices','physical_gpu_uuid','resource_selection')},
        phase_origin_started_utc=old['phase_origin_started_utc'],
        timer_retry=dict(previous_source=old['source_commit'],previous_experiment_protocol_sha256=sha256(previous),
            failed_directory=failed_directory.relative_to(ROOT).as_posix(),failed_complete_groups=118,
            reuse_failed_cases=False,old_timers=dict(refine_job_wall_seconds=1800,evaluation_wall_seconds=1800,phase_total_wall_hours=3),
            new_timers=timers,bitwise_uninterrupted_trajectory_identity_claimed=False,
            reason='CPU throughput exceeded the original per-process limit before any victim evaluation. Rerun only the incomplete seed job; all partial dispatches, applied feature steps and CPU/outer time remain extra failed work. No scientific settings or outcome criteria change.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    parser.add_argument('--continue-protocol');parser.add_argument('--reuse-teacher');parser.add_argument('--prior-progress')
    parser.add_argument('--timer-retry',action='store_true')
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
    if args.timer_retry:
        if not all((args.continue_protocol,args.reuse_teacher,args.prior_progress)):
            raise RuntimeError('Timer retry requires all original lineage/accounting paths.')
        continuation=timer_retry(p,private_output(args.continue_protocol),private_output(args.reuse_teacher),
            private_output(args.prior_progress),retained,expected)
    if args.continue_protocol and not args.timer_retry:
        if not args.reuse_teacher or not args.prior_progress:
            raise RuntimeError('A continuation must explicitly retain teacher and all prior accounting.')
        previous=private_output(args.continue_protocol)
        previous_protocol=json.loads(previous.read_text(encoding='utf-8'))
        allowed={'course/feature_polish/refine.py','course/feature_polish/prepare_protocol.py','course/feature_polish/queue.py',
                 'course/feature_polish/preflight.py',MASTER}
        for name,wanted in previous_protocol['source_files'].items():
            if name not in allowed and sha256(ROOT/name)!=wanted:
                raise RuntimeError('Scientific source changed during construction-only continuation: '+name)
        for key in ('arms','seeds','train_groups','valid_groups','polish_settings','teacher_settings','method','evaluation',
                    'phase_gpu_hours','phase_total_wall_hours','final_test_allowed','teacher_device','teacher_wall_seconds',
                    'preflight_wall_seconds','refine_job_wall_seconds','evaluation_wall_seconds','allowed_gpus',
                    'minimum_free_gpu_mib','cpu_threads'):
            if previous_protocol[key]!=p[key]:
                raise RuntimeError('Construction amendment changed scientific/resource settings: '+key)
        if previous_protocol['preflight']['train_groups']!=p['preflight']['train_groups']:
            raise RuntimeError('Construction amendment changed TRAIN preflight cases.')
        retained.extend(previous_protocol.get('retained_inputs',[]))
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
        counts=failure['counts']
        if (failure['status']!='failed' or failure['official_victim_loaded'] or
                any(counts.get(key,0)!=0 for key in ('optimizer_steps','official_victim_forwards','API_calls','TEST_queries'))):
            raise RuntimeError('Construction continuation requires a preserved no-update/no-victim TRAIN failure.')
        nonzero=any(value!=0 for value in counts.values())
        if nonzero and not (failure['error_type']=='RuntimeError' and
                'Fixed feature finite-difference check failed for gcn:' in failure['error'] and
                [row['group'] for row in failure['cases']]==[263,238] and
                counts['generator_forwards']==2 and counts['gaga_completed_forwards']==30 and
                counts['gcn_completed_forwards']==20 and counts['autograd_calls']==8):
            raise RuntimeError('Only the recorded GCN finite-difference construction failure is eligible for this amendment.')
        if nonzero:
            for name in ('course/feature_polish/refine.py','course/feature_polish/queue.py'):
                if sha256(ROOT/name)!=previous_protocol['source_files'][name]:
                    raise RuntimeError('Derivative-check amendment changed executed method/orchestration: '+name)
        launch=old_stage/'launch_receipt.json';launch_record=json.loads(launch.read_text(encoding='utf-8'))
        synthetic=(ROOT/previous_protocol['reuse_synthetic_ref']['path'] if 'reuse_synthetic_ref' in previous_protocol
                   else old_stage/'queue_attempt1/synthetic_receipt.json')
        synthetic_record=json.loads(synthetic.read_text(encoding='utf-8'))
        synthetic_log=synthetic.parent/'synthetic_checks.log'
        if (synthetic_record['status']!='passed' or synthetic_record['exit_code'] or
                sha256(synthetic_log)!=synthetic_record['stdout_sha256']):
            raise RuntimeError('The retained unchanged affine CPU checks did not pass.')
        for row in prior['jobs']:
            guardian=(ROOT/row['guardian_path'] if row.get('guardian_path')
                      else progress.parent/(row['name']+'_guardian.json'))
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
            phase_origin_started_utc=previous_protocol.get('phase_origin_started_utc',launch_record['started_utc']),
            construction_amendment=dict(previous_source=previous_protocol['source_commit'],previous_protocol_sha256=sha256(previous),
                reason=('Fixed sampled GCN block order plus isolated float64 finite differences with a fixed five-step convergence grid and unchanged tolerance. Retain both prior failures, completed teacher, unchanged passed synthetic tests and all counters. No VALID refinement, victim calls or optimizer updates before amendment.'
                        if nonzero else 'CPU-safe original-GCN checkpoint deserialization only; retain completed teacher and unchanged passed affine tests. Failed preflight had zero generations/classifier forwards/autograd/updates/victim calls. Method/settings unchanged.')))
    stage=private_output(args.stage,create=True)
    p.update(status='frozen',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,universal_newlines=True).strip(),
        source_files=sources,inputs=old['inputs'],retained_inputs=retained,baseline_runs=baselines,
        teacher_reference_initial_state_sha256=expected,stage_root=str(stage),
        execution_device=None,cuda_visible_devices='',physical_gpu_uuid=None)
    p.update(continuation)
    unique={}
    for row in p['retained_inputs']:
        if row['path'] in unique and unique[row['path']]!=row['sha256']:
            raise RuntimeError('Conflicting retained bytes: '+row['path'])
        unique[row['path']]=row['sha256']
    p['retained_inputs']=[dict(path=name,sha256=value) for name,value in unique.items()]
    path=stage/'preparation_protocol_v1.json';write_json(path,p,exclusive=True)
    print(json.dumps(dict(status='frozen',source_commit=p['source_commit'],protocol_sha256=sha256(path),output=str(path))))


if __name__=='__main__':main()
