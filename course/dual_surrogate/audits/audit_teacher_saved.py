"""Private independent stdlib audit of saved TRAIN-only teacher preparation.

Not executed on import. No ML/project imports, network, tensor deserialization,
training or inference. The optional original TRAIN .npy is decoded only as a
primitive one-dimensional integer identity array, never as pickled objects.
"""
import argparse
import ast
import collections
import datetime
import hashlib
import json
import math
import pathlib
import struct
import sys
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load(path):
    with pathlib.Path(path).open(encoding='utf-8') as stream:
        return json.load(stream, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def near(actual, expected, label, tolerance=1e-10):
    require(type(actual) in (int, float) and math.isfinite(actual) and
            math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance),
            '%s differs: %r versus %r' % (label, actual, expected))


def integer_npy(path):
    """Read primitive identity metadata, accepting floats only when exact IDs."""
    with pathlib.Path(path).open('rb') as stream:
        require(stream.read(6) == b'\x93NUMPY', 'Invalid integer-array magic')
        version = tuple(stream.read(2))
        require(version in ((1, 0), (2, 0), (3, 0)), 'Unsupported array version')
        header_bytes = 2 if version == (1, 0) else 4
        length = int.from_bytes(stream.read(header_bytes), 'little')
        require(0 < length < 65536, 'Unbounded array header')
        header = ast.literal_eval(stream.read(length).decode('utf-8' if version == (3, 0) else 'latin1').strip())
        require(header['fortran_order'] is False and len(header['shape']) == 1, 'Only one-dimensional identities are accepted')
        descriptor = header['descr']
        formats = {'<i8':'<q', '<i4':'<i', '<u8':'<Q', '<u4':'<I',
                   '>i8':'>q', '>i4':'>i', '>u8':'>Q', '>u4':'>I',
                   '<f8':'<d', '<f4':'<f', '>f8':'>d', '>f4':'>f'}
        require(descriptor in formats, 'Only primitive numeric identity metadata is accepted')
        raw = stream.read()
        require(len(raw) == header['shape'][0]*struct.calcsize(formats[descriptor]), 'Identity-array byte count mismatch')
        values = [value[0] for value in struct.iter_unpack(formats[descriptor], raw)]
        require(all(math.isfinite(value) and value >= 0 and int(value) == value for value in values),
                'Identity metadata contains a nonintegral/invalid value')
        return [int(value) for value in values]


def binary_metrics(labels, logits):
    require(len(labels) == len(logits) and set(labels) == {0, 1}, 'Invalid internal-STOP denominator')
    predictions, scores = [], []
    for row in logits:
        require(len(row) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in row), 'Invalid saved teacher logits')
        predictions.append(0 if row[0] >= row[1] else 1)
        shift = max(row)
        a, b = math.exp(row[0]-shift), math.exp(row[1]-shift)
        scores.append(b/(a+b))
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = len(labels)-tp-tn-fp
    # Float32 stored probabilities can collapse distinct logits into ties.
    # The caller passes those original saved probabilities for the AUC below.
    f1 = 0.5*((2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0) +
              (2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0))
    return dict(cases=len(labels), accuracy=(tp+tn)/len(labels), macro_f1=f1,
                benign_recall=tn/(tn+fp), fraud_recall=tp/(tp+fn),
                tp=tp, tn=tn, fp=fp, fn=fn, class_counts=[tn+fp, tp+fn]), predictions, scores


def auc_from_saved_probabilities(labels, scores):
    require(len(labels) == len(scores) and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in scores),
            'Invalid saved AUC probabilities')
    ordered = sorted(zip(scores, labels))
    total, begin = 0.0, 0
    while begin < len(ordered):
        end = begin+1
        while end < len(ordered) and ordered[end][0] == ordered[begin][0]:
            end += 1
        total += (begin+1+end)/2 * sum(y for _, y in ordered[begin:end])
        begin = end
    positives = sum(labels)
    return (total-positives*(positives+1)/2)/(positives*(len(labels)-positives))


def audit_teachers(directory, protocol_path, repo_root):
    directory, repo_root = pathlib.Path(directory), pathlib.Path(repo_root)
    protocol, registered = load(protocol_path), load(directory/'registered.json')
    result, artifacts = load(directory/'result.json'), load(directory/'teacher_artifacts.json')
    require(result['status'] == 'completed' and artifacts['both_quality_gates_passed'] is True, 'Teacher preparation did not pass')
    require(registered['source_commit'] == protocol['source_commit'] and
            registered['protocol_sha256'] == sha256(protocol_path), 'Teacher preparation provenance differs')
    require(result['teacher_artifacts_sha256'] == sha256(directory/'teacher_artifacts.json'), 'Teacher artifact manifest bytes changed')
    require(result['original_victim_loaded'] is False and result['original_victim_forwards'] == 0 and
            result['test_used'] is False and artifacts['original_victim_loaded'] is False and
            artifacts['labels_outside_original_train_used'] is False, 'Teacher boundary declarations differ')
    split = load(directory/'train_split_private.json')
    original, fit, stop = (split[k] for k in ('original_train_nodes','fit_nodes','stop_nodes'))
    for values in (original, fit, stop):
        require(values and all(type(n) is int and n >= 0 for n in values) and len(values) == len(set(values)), 'Invalid teacher partition identities')
    require(not set(fit)&set(stop) and set(fit)|set(stop) == set(original), 'Teacher FIT/STOP partition is not a disjoint exact cover')
    original_file = repo_root/protocol['data']['train_indices']
    require(set(integer_npy(original_file)) == set(original), 'Teacher identities differ from original TRAIN indices')
    require(sha256(directory/'train_split_private.json') == artifacts['train_split_sha256'], 'Private split byte hash changed')
    partition = dict(original_train=len(original), fit=len(fit), stop=len(stop),
                     fit_ids_sha256=canonical_sha(sorted(fit)), tune_ids_sha256=canonical_sha(sorted(stop)))
    cache = load(directory/artifacts['cache_manifest']['path'])
    require(sha256(directory/artifacts['cache_manifest']['path']) == artifacts['cache_manifest']['sha256'], 'Cache manifest byte hash changed')
    require(cache == artifacts['cache'] and cache['fit_count'] == len(fit) and cache['stop_count'] == len(stop), 'Cache partition manifest differs')
    require(cache['cached_full_train_sequences_used'] is False, 'GAGA teacher used original full-TRAIN sequences')
    verified = []
    for row in cache['files'].values():
        path = directory/row['path']
        require(sha256(path) == row['sha256'], 'FIT cache bytes changed: '+row['path'])
        verified.append(dict(artifact=row['path'], sha256=row['sha256']))
    require(artifacts['fit_masks_sha256'] == cache['fit_group_masks_sha256'], 'FIT mask digest chain differs')
    teachers, expected_forwards, expected_updates = [], 0, 0
    for name in ('gaga', 'gcn2'):
        item = load(directory/(name+'_teacher.json'))
        require(item == artifacts['teachers'][name], 'Teacher receipt differs from artifact manifest')
        require(item['quality_gate'] == protocol['teachers']['quality_gate'] and item['quality_passed'] is True, 'Teacher quality gate changed')
        checkpoint = directory/item['checkpoint']
        require(sha256(checkpoint) == item['checkpoint_sha256'] and item['checkpoint_sha256'] != protocol['sealed_victim_sha256'], 'Teacher checkpoint bytes/role differ')
        verified.append(dict(artifact=item['checkpoint'], sha256=item['checkpoint_sha256']))
        receipt = item['training_receipt']
        require(receipt['original_valid_used'] is False and receipt['test_used'] is False and receipt['sealed_victim_loaded'] is False and
                receipt['checkpoint_selection'] == 'internal_tune_only' and receipt['initial_captured_before_optimizer'] is True,
                'Teacher training receipt boundary changed')
        require(receipt['fit_ids_sha256'] == partition['fit_ids_sha256'] and receipt['tune_ids_sha256'] == partition['tune_ids_sha256'], 'Teacher normalized split digests differ')
        require(receipt['initial_state_sha256'] == item['initial_state_sha256'] and receipt['final_state_sha256'] == item['final_state_sha256'] and
                receipt['initial_state_sha256'] != receipt['final_state_sha256'], 'Teacher initial/selected content hash chain differs')
        if name == 'gaga':
            require(receipt['aggregation_label_scope'] == 'internal_fit_only' and
                    receipt['aggregation_label_ids_sha256'] == partition['fit_ids_sha256'], 'Teacher aggregation label scope differs')
        epochs, best_epoch, best_score, best_metrics = item['epochs'], None, None, None
        for epoch in range(1, epochs+1):
            raw = load(directory/'raw'/('%s_internal_stop_epoch_%03d.json'%(name,epoch)))
            summary = load(directory/('%s_epoch_%03d.json'%(name,epoch)))
            require(raw['node_ids'] == stop and len(raw['labels']) == len(stop), 'Internal STOP cohort/order differs')
            require(collections.Counter(raw['labels']) == collections.Counter({0:split['stop_label_counts'][0],1:split['stop_label_counts'][1]}), 'Internal STOP class counts differ')
            metrics, predictions, precise_scores = binary_metrics(raw['labels'],raw['logits'])
            require(predictions == raw['predictions'], 'Teacher saved argmax differs')
            for a, b in zip(raw['scores'],precise_scores):
                near(a,b,'Stored float32 softmax probability',tolerance=2e-6)
            metrics['auc'] = auc_from_saved_probabilities(raw['labels'],raw['scores'])
            for key,value in metrics.items():
                if isinstance(value,list): require(summary['internal_stop'][key] == value,'Teacher class counts differ')
                else: near(summary['internal_stop'][key],value,'Teacher '+key)
            improved = best_score is None or metrics['macro_f1'] > best_score
            if improved:
                best_epoch, best_score, best_metrics = epoch, metrics['macro_f1'], metrics
            require(summary['epoch'] == epoch and summary['selected_if_improved'] == improved and summary['best_epoch'] == best_epoch,
                    'Teacher earliest-best checkpoint selection differs')
        require(item['selected_epoch'] == best_epoch and best_metrics == item['internal_stop'], 'Selected teacher metrics/checkpoint epoch differ')
        fit_batches = 1 if name == 'gcn2' else math.ceil(len(fit)/512)
        stop_batches = 1 if name == 'gcn2' else math.ceil(len(stop)/512)
        updates = epochs*fit_batches
        forwards = epochs*(fit_batches+stop_batches)
        require(item['optimizer_steps'] == receipt['optimizer_steps'] == updates and
                item['forward_counts'] == dict(fit=epochs*fit_batches,internal_stop=epochs*stop_batches), 'Teacher physical forward/update counters differ')
        for field, threshold in protocol['teachers']['quality_gate'].items():
            require(best_metrics[field[:-4]] >= threshold, 'Teacher selected checkpoint fails frozen quality gate')
        expected_updates += updates; expected_forwards += forwards
        teachers.append(dict(role=name,epochs=epochs,selected_epoch=best_epoch,optimizer_steps=updates,
                             forwards=forwards,internal_stop=best_metrics,checkpoint_sha256=item['checkpoint_sha256']))
    require(result['optimizer_steps'] == expected_updates and result['forwards'] == expected_forwards and
            artifacts['counters']['optimizer_steps'] == expected_updates and artifacts['counters']['forwards'] == expected_forwards,
            'Aggregate teacher counters differ')
    return dict(schema='monti_independent_saved_teacher_audit_v1',status='SAVED_TEACHER_CHECKS_PASS',
                preparation_source_commit=protocol['source_commit'],preparation_protocol_sha256=sha256(protocol_path),
                partition=partition,teachers=teachers,optimizer_steps=expected_updates,forwards=expected_forwards,
                verified_artifacts=verified,teacher_artifact_manifest_sha256=sha256(directory/'teacher_artifacts.json'),
                no_new_model_calls=True,no_tensor_deserialization=True,
                limitations=['Mask tensor bytes and FIT-only cache provenance are checked; raw tensor mask values are not independently decoded by this stdlib audit.',
                             'Recorded initial/selected content hashes are reconciled, not independently reconstructed from checkpoint tensors.',
                             'Only original TRAIN identity metadata and saved internal STOP labels/predictions are read; no victim/test predictions.'])


def main():
    started_wall,started_cpu=time.monotonic(),time.process_time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-directory',required=True)
    parser.add_argument('--preparation-protocol',required=True)
    parser.add_argument('--repo-root',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    require(not output.exists(),'Preserve previous receipts; select a new audit output')
    try:
        report = audit_teachers(args.teacher_directory,args.preparation_protocol,args.repo_root)
    except (ValueError,KeyError,TypeError,OSError,IndexError,ZeroDivisionError) as error:
        report = dict(schema='monti_independent_saved_teacher_audit_v1',status='SAVED_TEACHER_CHECKS_FAILED',error=repr(error))
    report.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  auditor='Codex rubric_review agent; not human verification or instructor approval',audit_script_sha256=sha256(__file__),
                  execution=dict(command=[sys.executable,str(pathlib.Path(__file__).resolve())]+sys.argv[1:],
                                 python=sys.version,cpu_seconds=time.process_time()-started_cpu,
                                 wall_seconds=time.monotonic()-started_wall,
                                 timing_scope='Audit process work through receipt serialization start',
                                 model_calls=0,API_calls=0,tensor_deserialization=0))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'],output=str(output),sha256=sha256(output))))
    return 0 if report['status']=='SAVED_TEACHER_CHECKS_PASS' else 1


if __name__=='__main__':
    sys.exit(main())
