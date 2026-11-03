"""Private saved-output audit; stdlib plus the adjacent private audit helper.

Never imports project/ML modules or deserializes tensors. Git cat-file reads
historical source bytes only; it does not run project code or access a network.
Run only after root explicitly authorizes saved-output auditing.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import subprocess
import sys
import time

from audit_teacher_saved import audit_teachers, canonical_sha, load, near, require, sha256


ARMS = ('gcn_only','gaga_only','two_gcn','gcn_gaga')
ROLES = {'gcn_only':('gcn_base',),'gaga_only':('gaga_new',),
         'two_gcn':('gcn_base','gcn_new'),'gcn_gaga':('gcn_base','gaga_new')}


def jsonl(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for line in stream if line.strip()]


def predictions(logits, count):
    require(len(logits) == count, 'Logit denominator differs')
    result = []
    for row in logits:
        require(isinstance(row,list) and len(row) == 2 and all(type(x) in (int,float) and math.isfinite(x) for x in row),
                'Saved logit shape/values invalid')
        result.append(0 if row[0] >= row[1] else 1)
    return result


def transition(labels, clean, attacked):
    require(len(labels) == len(clean) == len(attacked) and labels, 'Transition denominator differs')
    cw = [a != b for a,b in zip(labels,clean)]
    aw = [a != b for a,b in zip(labels,attacked)]
    result = dict(occurrences=len(labels),clean_wrong=sum(cw),attacked_wrong=sum(aw),
        new_wrong=sum(not a and b for a,b in zip(cw,aw)),recoveries=sum(a and not b for a,b in zip(cw,aw)))
    require(result['attacked_wrong'] == result['clean_wrong']+result['new_wrong']-result['recoveries'], 'Transition identity failed')
    return result


def generation_seed(seed, phase, epoch, group):
    return int(hashlib.sha256(('%d|%s|%d|%d|dual-surrogate-v1'%(seed,phase,epoch,group)).encode()).hexdigest()[:8],16)


def verify_sources(repo, protocol, historical=False):
    verified = []
    for name,wanted in protocol['source_files'].items():
        if historical:
            data = subprocess.check_output(['git','cat-file','blob',protocol['source_commit']+':'+name],cwd=repo)
            actual = hashlib.sha256(data).hexdigest()
        else:
            actual = sha256(repo/name)
        require(actual == wanted,'Frozen source bytes differ: '+name)
        verified.append(dict(source_commit=protocol['source_commit'],path=name,sha256=actual,historical_git_blob=historical))
    return verified


def audit(args):
    repo,stage,matrix,evaluation,teacher = map(pathlib.Path,
        (args.repo_root,args.stage_root,args.matrix_root,args.evaluation_directory,args.teacher_directory))
    p,original = load(args.experiment_protocol),load(args.original_preparation_protocol)
    protocol_sha = sha256(args.experiment_protocol)
    if args.expected_source:
        require(p['source_commit'] == args.expected_source,'Unexpected continuation source')
    if args.expected_protocol_sha256:
        require(protocol_sha == args.expected_protocol_sha256,'Unexpected experiment protocol bytes')
    require(p['arms'] == list(ARMS) and p['seeds'] == [3,4,5] and p['epochs'] == 8 and p['phase_gpu_hours'] == 2,
            'Frozen matrix/epoch/resource settings differ')
    require(p['train_groups'] == original['train_groups'] and p['valid_groups'] == original['valid_groups'] and
            p['teachers'] == original['teachers'] and p['attack_training'] == original['attack_training'],
            'Continuation changed scientific cohorts/settings')
    require(p['final_test_allowed'] is False and p['LLM_API_calls'] == 0,'Query boundary differs')
    verified_sources = verify_sources(repo,p)+verify_sources(repo,original,historical=True)
    protocol_files={sha256(path):(path,load(path)) for path in stage.glob('*protocol_v*.json')}
    retained_runs=p.get('reused_training_runs',{})
    if retained_runs:
        require(p['reuse_passed_preflight'] is True,'Retained TRAIN continuation repeated preflight')
        core_source=p['scientific_core_source']
        checked_originals=set()
        for record in retained_runs.values():
            provenance=record['provenance'];old_sha=provenance['protocol_sha256']
            require(old_sha in protocol_files and provenance['source_commit']==core_source,'Retained job original provenance unavailable')
            old_path,old_protocol=protocol_files[old_sha]
            require(old_protocol['source_commit']==core_source,'Retained job protocol/source mismatch')
            if old_sha not in checked_originals:
                verified_sources+=verify_sources(repo,old_protocol,historical=True)
                checked_originals.add(old_sha)
            for name,wanted in old_protocol['source_files'].items():
                if name not in {'course/dual_surrogate/prepare_protocol.py','course/dual_surrogate/queue.py'}:
                    require(p['source_files'][name]==wanted,'Scientific source changed for retained TRAIN continuation: '+name)
    verified = []
    for row in p['inputs']+p.get('reused_teacher_receipts',[]):
        require(sha256(repo/row['path']) == row['sha256'],'Frozen input/retained receipt changed: '+row['path'])
        verified.append(dict(path=row['path'],sha256=row['sha256']))
    require(sha256(repo/p['teacher_artifacts_ref']['path']) == p['teacher_artifacts_ref']['sha256'], 'Reused artifact manifest changed')
    teacher_audit = audit_teachers(teacher,args.original_preparation_protocol,repo)
    old_artifacts = load(teacher/'teacher_artifacts.json')
    base_teacher_content,initials,orders,train_budgets = set(),collections.defaultdict(set),{},{}
    trained = {}; totals = collections.Counter()
    for seed in p['seeds']:
        for arm in ARMS:
            directory = matrix/('%s_seed%d'%(arm,seed))
            r,registered,initial = (load(directory/name) for name in ('result.json','registered.json','initial_state.json'))
            require(r['status'] == 'completed' and r['arm'] == arm and r['seed'] == seed and r['epochs'] == 8,'Training identity/status differs')
            name='%s_seed%d'%(arm,seed)
            if name in retained_runs:
                retained=retained_runs[name]
                require(directory.resolve()==(repo/retained['directory']).resolve(),'Retained job symlink/directory differs')
                require(registered['provenance']==retained['provenance'],'Retained original registration was rewritten')
                actual={f.relative_to(directory).as_posix() for f in directory.rglob('*') if f.is_file()}
                require(actual==set(retained['files']),'Retained output file inventory changed')
                for relative,wanted in retained['files'].items():
                    require(sha256(directory/relative)==wanted,'Retained output bytes changed: '+name+'/'+relative)
            else:
                require(registered['provenance']['source_commit'] == p['source_commit'] and registered['provenance']['protocol_sha256'] == protocol_sha,
                        'Fresh rerun training provenance differs')
            require(registered['evaluation_performed'] is False and registered['sealed_victim_deserialized'] is False and
                    r['validation_performed'] is False and r['victim_loaded'] is False and r['test_queries'] == 0,'Training crossed evaluation boundary')
            require(initial['arm'] == arm and initial['seed'] == seed and initial['captured_before_training'] is True and initial['optimizer_steps'] == 0,
                    'Initial attacker capture differs')
            require(initial['initial_state_sha256'] == r['initial_model_state_sha256'],'Initial attacker hash receipts disagree')
            initials[seed].add(r['initial_model_state_sha256'])
            require(r['teacher_initial_hashes'] == r['teacher_final_hashes'],'A frozen teacher changed during attacker training')
            base_teacher_content.add(r['teacher_initial_hashes']['gcn_base'])
            for role,name in [('gcn_new','gcn2'),('gaga_new','gaga')]:
                if role in r['teacher_initial_hashes']:
                    require(r['teacher_initial_hashes'][role] == old_artifacts['teachers'][name]['final_state_sha256'],'Loaded teacher content differs from selected checkpoint')
            expected = {role:(2048 if role in ROLES[arm] else 0) for role in ('gcn_base','gcn_new','gaga_new','gaga_victim')}
            require(r['optimizer_steps'] == r['generator_forwards'] == 2048 and r['classifier_forwards'] == expected,'Training physical counters differ')
            for name,key in [('final_model.pt','final_model_sha256'),('final_optimizer.pt','final_optimizer_sha256')]:
                require(sha256(directory/name) == r[key],'Terminal checkpoint bytes differ')
                verified.append(dict(path=str((directory/name).relative_to(repo)),sha256=r[key]))
            for epoch in range(1,9):
                rows = jsonl(directory/'raw'/('train_epoch_%03d.jsonl'%epoch))
                summary = load(directory/('epoch_%03d.json'%epoch))
                order = [row['group'] for row in rows]
                require(len(order) == 256 and len(set(order)) == 256 and set(order) == set(p['train_groups']),'TRAIN epoch cohort differs')
                if (seed,epoch) not in orders:orders[(seed,epoch)] = order
                require(orders[(seed,epoch)] == order,'Same-seed TRAIN order differs between arms')
                for row in rows:
                    group,nb,eb = row['group'],row['node_budget'],row['edge_budget']
                    identity = (row['targets'],nb,eb)
                    require(group not in train_budgets or train_budgets[group] == identity,'TRAIN per-group budgets differ')
                    train_budgets[group] = identity
                    require(nb >= 1 and eb >= nb and row['actual_injected_nodes'] == nb and row['actual_injected_edges'] == eb,'TRAIN physical injection counters differ')
                    require(row['generation_seed'] == generation_seed(seed,'train',epoch,group),'TRAIN generation RNG key differs')
                    require(set(row['losses']) == set(ROLES[arm]),'Teacher loss roles differ')
                    losses = []
                    for role,value in row['losses'].items():
                        require(math.isfinite(value['loss']) and value['loss'] >= 0 and math.isfinite(value['mean_attack_margin']) and
                                0 <= value['wrong'] <= row['targets'],'Invalid saved teacher-loss statistics')
                        losses.append(value['loss'])
                    near(row['total_loss'],statistics.mean(losses),'Uniform raw-margin mixture loss',tolerance=2e-6)
                require(summary['epoch'] == epoch and summary['groups'] == 256 and
                        summary['optimizer_steps'] == summary['generator_forwards'] == 256*epoch and
                        summary['classifier_forwards'] == {role:(256*epoch if role in ROLES[arm] else 0) for role in expected} and
                        summary['validation_performed'] is False and summary['victim_loaded'] is False,'TRAIN epoch counters/boundary differ')
                near(summary['mean_loss'],statistics.mean(row['total_loss'] for row in rows),'TRAIN epoch mean loss',tolerance=2e-6)
            require(not list(directory.glob('*validation*')),'Unexpected interim validation file')
            totals.update(r['classifier_forwards']);trained[(arm,seed)] = r
    require(all(len(x) == 1 for x in initials.values()) and len(base_teacher_content) == 1,'Paired initial attacker/base-teacher hashes differ')
    require(dict(totals) == dict(gcn_base=18432,gcn_new=6144,gaga_new=12288,gaga_victim=0),'Aggregate attacker TRAIN teacher counters differ')
    pref = p['preflight_ref'];preflight = load(repo/pref['path'])
    require(sha256(repo/pref['path']) == pref['sha256'] and preflight['status'] == 'passed' and preflight['optimizer_steps'] == 0 and
            preflight['sealed_victim_forwards'] == 0 and preflight['sealed_victim_loaded'] is False and
            preflight['generator_forwards'] == 4 and preflight['autograd_calls'] == 8 and
            preflight['classifier_forwards'] == dict(gcn_base=4,gcn_new=4,gaga_new=8,gaga_victim=0),'Preflight result/counters differ')
    require({r['group'] for r in preflight['cases']} == set(p['preflight']['train_groups']),'Preflight TRAIN groups differ')
    for field in ('feature_gradient_norm','edge_gradient_norm','feature_head_gradient_norm','edge_head_gradient_norm'):
        require(all(math.isfinite(row[field]) and row[field] >= 0 for row in preflight['cases']) and any(row[field] > 0 for row in preflight['cases']),
                'Preflight gradient path invalid: '+field)
    require(preflight['partition_check']['fit_ids_sha256'] == teacher_audit['partition']['fit_ids_sha256'] and
            preflight['mask_check']['fit_ids_sha256'] == teacher_audit['partition']['fit_ids_sha256'] and
            preflight['mask_check']['label_scope'] == 'internal_fit_only','Passed FIT mask membership receipt differs')
    e,er = load(evaluation/'result.json'),load(evaluation/'registered.json')
    require(e['status'] == 'completed' and e['test_queries'] == 0 and e['source_commit'] == p['source_commit'] and
            e['protocol_sha256'] == protocol_sha and e['counts'] == dict(gcn_base=1664,gaga_victim=1664,generations=1536), 'Evaluation boundary/counters differ')
    require(er['all12_trainings_complete_before_victim_load'] is True and er['final_test_used'] is False and
            er['provenance']['source_commit'] == p['source_commit'] and er['provenance']['protocol_sha256'] == protocol_sha,
            'Delayed-victim registration differs')
    clean_rows = jsonl(evaluation/'clean_predictions.jsonl')
    require([r['group'] for r in clean_rows] == p['valid_groups'] and sum(len(r['node_ids']) for r in clean_rows) == 1091,
            'Shared clean baseline denominator/order differs')
    clean = {}
    for row in clean_rows:
        n = len(row['node_ids'])
        require(n == len(row['labels']) and n == len(set(row['node_ids'])) and set(row['labels']) == {1},'Clean target identities/labels differ')
        clean[row['group']] = dict(row,predictions={name:predictions(row[name+'_logits'],n) for name in ('gcn','gaga')})
    metadata = {}
    with (repo/'evidence/results_first_diagnosis_v1/group_outcomes.csv').open(encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            item = tuple(int(row[key]) for key in ('group_size','node_budget','edge_budget'))
            group = int(row['target_group'])
            require(group not in metadata or metadata[group] == item,'Saved VALID budget metadata inconsistent')
            metadata[group] = item
    summaries = {(r['arm'],r['seed']):r for r in e['summaries']}
    require(set(summaries) == set(trained),'Evaluation matrix differs from completed trainings')
    audited = []
    for (arm,seed),r in trained.items():
        summary = summaries[(arm,seed)]
        rows = jsonl(evaluation/('%s_seed%d_predictions.jsonl'%(arm,seed)))
        public = load(evaluation/('%s_seed%d_groups.json'%(arm,seed)))
        require(len(rows) == len(public) == 128 and [x['group'] for x in rows] == p['valid_groups'],'Attacked VALID cohort/order differs')
        cumulative = {name:collections.Counter() for name in ('gcn','gaga')}
        for row,pub in zip(rows,public):
            require(all(row[key] == value for key,value in pub.items()),'Private/public group rows differ')
            group,n = row['group'],len(row['node_ids'])
            require(row['node_ids'] == clean[group]['node_ids'] and row['labels'] == clean[group]['labels'],'Attacked/clean identities differ')
            require((n,row['node_budget'],row['edge_budget']) == metadata[group] and row['actual_injected_nodes'] == row['node_budget'] and
                    row['actual_injected_edges'] == row['edge_budget'],'VALID physical budgets differ from frozen metadata')
            for name in ('gcn','gaga'):
                pred = predictions(row[name+'_logits'],n)
                tr = transition(row['labels'],clean[group]['predictions'][name],pred)
                require(tr == row[name+'_transitions'],'Saved per-group transition counts differ')
                cumulative[name].update(tr)
        for name in ('gcn','gaga'):
            require(dict(cumulative[name]) == summary['totals'][name] and cumulative[name]['occurrences'] == 1091,'Pooled prediction/transition denominator differs')
            near(summary[name+'_mr_percent'],100*cumulative[name]['attacked_wrong']/1091,'Pooled '+name+' MR')
        key=arm+'_s'+str(seed)
        require(summary['checkpoint_sha256'] == r['final_model_sha256'] == er['checkpoint_sha256s'][key], 'Evaluated checkpoint digest differs')
        require(summary['injected_nodes'] == sum(x['actual_injected_nodes'] for x in rows) and
                summary['injected_edges'] == sum(x['actual_injected_edges'] for x in rows),'Pooled physical budgets differ')
        require({x['group'] for x in summary['exports']} == set(p['export_valid_groups']) and len(summary['exports']) == 8,'Fixed export cohort differs')
        for export in summary['exports']:
            require(sha256(evaluation/export['path']) == export['sha256'],'Raw attack export byte hash differs')
            verified.append(dict(path=str((evaluation/export['path']).relative_to(repo)),sha256=export['sha256']))
        audited.append(dict(arm=arm,seed=seed,training_source_commit=load(matrix/('%s_seed%d'%(arm,seed))/'registered.json')['provenance']['source_commit'],
                            retained_terminal_model=('%s_seed%d'%(arm,seed)) in retained_runs,
                            gcn_mr_percent=summary['gcn_mr_percent'],gaga_mr_percent=summary['gaga_mr_percent'],
                            totals=summary['totals'],injected_nodes=summary['injected_nodes'],injected_edges=summary['injected_edges']))
    queue,progress = load(matrix/'queue_complete.json'),load(matrix/'queue_progress.json')
    prior_ref=p['prior_progress_ref'];prior_path=repo/prior_ref['path'];prior=load(prior_path)
    require(sha256(prior_path) == prior_ref['sha256'] and queue['jobs'][:len(prior['jobs'])] == prior['jobs'],'Retained failed-prefix ledger changed')
    require(queue['jobs'] == progress['jobs'] and queue['status'] == 'completed' and queue['test_queries'] == queue['LLM_API_calls'] == 0,
            'Final queue receipts disagree')
    require(sum(row['name'] == 'teacher_preparation' for row in queue['jobs']) == 1,'Completed teachers were replayed/omitted')
    training_names = ['%s_seed%d'%(arm,seed) for seed in p['seeds'] for arm in ARMS]
    require([row['name'] for row in queue['jobs'] if row['name'] in training_names and row['exit_code']==0] == training_names,
            'Successful terminal attacker job replay/order mismatch')
    # The prefix can span multiple explicit continuations. Resolve their
    # retained queue directories from frozen protocol references, not job names.
    queue_roots = {matrix,prior_path.parent}
    for protocol_file in stage.glob('*protocol_v*.json'):
        historical = load(protocol_file)
        historical_ref = historical.get('prior_progress_ref')
        if historical_ref:
            path = repo/historical_ref['path']
            require(sha256(path) == historical_ref['sha256'],'Historical accounting prefix changed')
            queue_roots.add(path.parent)
    registered_uuids = {original['physical_gpu_uuid'],p['physical_gpu_uuid']}
    if p.get('resource_amendment'):
        registered_uuids.add(p['resource_amendment']['previous_uuid'])
    total,charges,native_gaps,excluded_training = 0.0,[],[],[]
    for index,row in enumerate(queue['jobs']):
        candidates = [root/(row['name']+'_guardian.json') for root in queue_roots]
        matches = [path for path in candidates if path.is_file() and sha256(path) == row['guardian_sha256']]
        if row['guardian_sha256'] is None:
            require(index < len(prior['jobs']) and row['exit_code'] != 0,
                    'A successful/new process has no guardian receipt')
            originating = []
            for root in queue_roots:
                progress_file=root/'queue_progress.json'
                if progress_file.is_file():
                    items=load(progress_file)['jobs']
                    if items and items[-1] == row and (root/(row['name']+'.log')).is_file():
                        originating.append(root/(row['name']+'.log'))
            require(originating,'Retained native-failure outer record/log unavailable')
            cost=row['outer_seconds']/3600
            native_gaps.append(dict(name=row['name'],exit_code=row['exit_code'],
                outer_seconds=row['outer_seconds'],log_sha256=sha256(originating[0]),
                python_guardian='Absent after native termination; not fabricated',
                model_forward_and_generation_counters='Unknown; not inferred as zero'))
        else:
            require(matches,'Guardian bytes unavailable/different for ledger entry '+str(index))
            guardian=load(matches[0]);cost=max(row['outer_seconds']/3600,guardian['charged_gpu_wall_hours'])
            allowed=registered_uuids if index < len(prior['jobs']) else {p['physical_gpu_uuid']}
            require(len(guardian['own_physical_uuids']) == 1 and set(guardian['own_physical_uuids']) <= allowed,
                    'Guardian physical GPU differs from registered device history')
            require((row['exit_code'] == 0) == (guardian['status'] == 'completed'),'Guardian/exit status differs')
            if row['name'] in training_names and row['exit_code']!=0:
                failed_directory=matches[0].parent/row['name']
                failed_result=load(failed_directory/'result.json')
                require(failed_result['status']=='failed' and failed_result['optimizer_steps']<2048 and
                        failed_result['victim_loaded'] is False and failed_result['test_queries']==0,
                        'Excluded incomplete TRAIN attempt is not a zero-victim failed build')
                raw_files=sorted((failed_directory/'raw').glob('train_epoch_*.jsonl'))
                completed_rows=sum(len(jsonl(path)) for path in raw_files)
                require(completed_rows==failed_result['optimizer_steps'],'Failed TRAIN complete-row/update counters differ')
                excluded_training.append(dict(name=row['name'],exit_code=row['exit_code'],
                    optimizer_steps=failed_result['optimizer_steps'],generator_forwards=failed_result['generator_forwards'],
                    attempted_classifier_dispatches=failed_result['classifier_forwards'],
                    completed_classifier_forwards='Not inferred from dispatch counters; timeout can occur inside sequence preparation.',
                    completed_training_rows=completed_rows,result_sha256=sha256(failed_directory/'result.json'),
                    raw_record_sha256s={path.name:sha256(path) for path in raw_files},
                    excluded_from_primary_twelve=True))
        near(row['charged_gpu_wall_hours'],cost,'Conservative whole-process charge')
        if index >= len(prior['jobs']):require(row['exit_code'] == 0,'New continuation process failed')
        total += cost
        charges.append(dict(name=row['name'],exit_code=row['exit_code'],charged_gpu_wall_hours=cost,guardian_sha256=row['guardian_sha256']))
    near(prior['charged_gpu_hours'],sum(x['charged_gpu_wall_hours'] for x in charges[:len(prior['jobs'])]),'Retained-prefix charge counted once')
    near(queue['charged_gpu_hours'],total,'Final phase cost');near(progress['charged_gpu_hours'],total,'Progress phase cost')
    require(total <= 2 and queue['scientific_source'] == p['source_commit'] and queue['experiment_protocol_sha256'] == protocol_sha and
            queue['evaluation_result_sha256'] == sha256(evaluation/'result.json'),'Final phase cap/provenance differs')
    marker=load(stage/'all12_trainings_complete.json')
    require(marker['status'] == 'completed' and marker['jobs'] == queue['jobs'][:-1] and queue['jobs'][-1]['name'] == 'development_evaluation',
            'All-twelve completion gate not before shared evaluation')
    return dict(schema='monti_dual_surrogate_independent_saved_audit_v2',status='ALL_SAVED_PHASE_CHECKS_PASS',
        preparation_source=original['source_commit'],experiment_source=p['source_commit'],experiment_protocol_sha256=protocol_sha,
        teacher_audit=teacher_audit,attacker_trainings=12,attacker_updates=24576,attacker_training_generations=24576,
        attacker_training_teacher_forwards=dict(totals),preflight_forwards=preflight['classifier_forwards'],
        evaluation_counts=e['counts'],total_train_and_evaluation_generations=26112,jobs=audited,
        same_seed_training_orders_checked=24,same_seed_initial_hashes_checked=True,
        verified_source_records=verified_sources,verified_artifacts=verified,
        cost=dict(charged_gpu_hours=total,retained_prefix_gpu_hours=prior['charged_gpu_hours'],jobs=charges),
        retained_native_failure_counter_gaps=native_gaps,
        excluded_incomplete_training_attempts=excluded_training,
        primary_model_training_sources=sorted({row['training_source_commit'] for row in audited}),
        scope=dict(new_model_calls=0,new_API_calls=0,tensor_deserialization=0,network_calls=0),
        limitations=['Counters/budgets are reconciled from saved records; raw graph tensors are hashed, not reconstructed.',
                     'Recorded model content hashes are reconciled; checkpoint tensor states are not independently decoded.',
                     'Passed FIT-mask membership/source checks and artifact bytes are audited; raw mask tensors are not decoded.',
                     'Victim sealing is supported by frozen source, terminal result/digest gates and the saved serial queue, not external tracing.',
                     'Training loss values/counters are checked; full training logits were not saved for independent argmax reconstruction.',
                     'Native failed preflights without Python receipts retain charged outer time and explicit unknown forward/generation counters; no grand total silently treats them as zero.',
                     'The primary twelve can retain earlier completed models plus one explicit fresh timer rerun under identical scientific bytes; original registrations are preserved. Failed attempted-dispatch counters are not completed classifier-call counts.',
                     'Original validation is development data; this audit does not make it a fresh final test or establish novelty.'])


def main():
    started_wall,started_cpu=time.monotonic(),time.process_time()
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root','stage-root','matrix-root','teacher-directory','evaluation-directory','original-preparation-protocol','experiment-protocol','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--expected-source');parser.add_argument('--expected-protocol-sha256')
    args=parser.parse_args();out=pathlib.Path(args.output);require(not out.exists(),'Preserve earlier audit receipts')
    try:report=audit(args)
    except (ValueError,KeyError,TypeError,OSError,IndexError,ZeroDivisionError,subprocess.CalledProcessError) as error:
        report=dict(schema='monti_dual_surrogate_independent_saved_audit_v2',status='SAVED_PHASE_CHECKS_FAILED',error=repr(error))
    report.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_script_sha256=sha256(__file__),
                  teacher_audit_helper_sha256=sha256(pathlib.Path(__file__).with_name('audit_teacher_saved.py')),
                  auditor='Codex rubric_review agent; not human verification or instructor approval',
                  execution=dict(command=[sys.executable,str(pathlib.Path(__file__).resolve())]+sys.argv[1:],
                                 python=sys.version,cpu_seconds=time.process_time()-started_cpu,
                                 wall_seconds=time.monotonic()-started_wall,
                                 timing_scope='Audit process work through receipt serialization start',
                                 model_calls=0,API_calls=0,tensor_deserialization=0))
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],output=str(out),sha256=sha256(out))))
    return 0 if report['status']=='ALL_SAVED_PHASE_CHECKS_PASS' else 1


if __name__=='__main__':sys.exit(main())
