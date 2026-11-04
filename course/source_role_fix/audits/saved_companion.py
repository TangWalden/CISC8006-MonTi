"""Independent saved-JSON/byte-only audit of the frozen 96-graph companion.

No ML, NumPy, graph, tensor or checkpoint deserialization; export .pt files are
only streamed as opaque bytes. Counts of physical calls are checked against
receipts and remain recorded claims rather than external execution telemetry.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import time
import traceback


OLD_ARMS = ('gcn_only', 'gaga_only', 'two_gcn', 'gcn_gaga')
SEEDS = (3, 4, 5)


def digest(path):
    value = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key.')
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--companion', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    root = pathlib.Path(args.root).resolve()
    output = pathlib.Path(args.receipt)
    if output.exists():
        raise RuntimeError('Audit receipts are exclusive; retain any prior failure.')
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu, wall = time.process_time(), time.monotonic()
    checks = 0
    hashed = {}
    receipt = dict(status='failed', auditor_source_sha256=digest(__file__),
                   audit_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   model_queries=0, tensor_deserializations=0, dataset_loads=0,
                   checkpoint_deserializations=0, API_calls=0, GPU_hours=0,
                   independent_of_companion_implementation=True,
                   independent_of_entire_project_implementation=False,
                   human_verification=False)

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    def safe(name):
        path = (root/name).resolve()
        require(root in path.parents and path.is_file(), 'Missing or unsafe input path.')
        return path

    def checked(name, wanted=None):
        path = safe(name)
        actual = digest(path)
        if wanted is not None:
            require(actual == wanted, 'A frozen input byte hash differs.')
        hashed[str(path.relative_to(root))] = actual
        return path

    def load(path):
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)

    def lines(path):
        return [json.loads(line, object_pairs_hook=unique_object)
                for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def index(values, key):
        result = {}
        for value in values:
            identity = key(value)
            require(identity not in result, 'Duplicate row identity.')
            result[identity] = value
        return result

    def prediction(logits, count):
        require(type(logits) is list and len(logits) == count, 'Logit denominator differs.')
        result = []
        for pair in logits:
            require(len(pair) == 2 and all(type(x) in (int, float) and math.isfinite(x) for x in pair),
                    'Nonfinite or malformed binary logits.')
            result.append(0 if pair[0] >= pair[1] else 1)
        return result

    def transitions(labels, before, after):
        require(len(labels) == len(before) == len(after), 'Transition length differs.')
        result = dict(occurrences=len(labels), clean_wrong=0, attacked_wrong=0, new_wrong=0, recoveries=0)
        for truth, clean, attacked in zip(labels, before, after):
            result['clean_wrong'] += int(clean != truth)
            result['attacked_wrong'] += int(attacked != truth)
            result['new_wrong'] += int(clean == truth and attacked != truth)
            result['recoveries'] += int(clean != truth and attacked == truth)
        require(result['attacked_wrong'] == result['clean_wrong']+result['new_wrong']-result['recoveries'],
                'Within-channel transition identity failed.')
        return result

    try:
        protocol_path = checked(args.protocol)
        protocol = load(protocol_path)
        spec = protocol['saved_teacher_diagnostic']
        require(protocol['status'] == 'frozen', 'Protocol is not frozen.')
        require(spec['max_forwards'] == 104, 'Forward cap differs.')
        old_protocol = load(checked(spec['old_protocol_path'], spec['old_protocol_sha256']))
        require(tuple(old_protocol['arms']) == OLD_ARMS and tuple(old_protocol['seeds']) == SEEDS,
                'Old matrix identity differs.')
        groups = set(old_protocol['export_valid_groups'])
        require(len(groups) == 8 and groups <= set(old_protocol['valid_groups']), 'Predetermined group list differs.')
        old_result = load(checked(spec['evaluation_path']+'/result.json', spec['evaluation_result_sha256']))
        require(old_result['status'] == 'completed' and old_result['test_queries'] == 0, 'Old evaluation is not complete VALID-only.')
        old_clean = index(lines(checked(spec['evaluation_path']+'/clean_predictions.jsonl', spec['clean_predictions_sha256'])),
                          lambda value: value['group'])
        require(set(old_clean) == set(old_protocol['valid_groups']) and len(old_clean) == 128,
                'Old clean cohort differs.')
        expected_runs = {(arm, seed) for arm in OLD_ARMS for seed in SEEDS}
        old_summaries = index(old_result['summaries'], lambda value: (value['arm'], value['seed']))
        require(set(old_summaries) == expected_runs, 'Old matrix does not contain twelve unique runs.')
        exports = {}
        old_predictions = {}
        for arm, seed in sorted(expected_runs):
            run = arm+'_seed'+str(seed)
            reference = spec['prediction_files'][run]
            predictions = index(lines(checked(reference['path'], reference['sha256'])), lambda value: value['group'])
            require(set(predictions) == set(old_protocol['valid_groups']), 'Old attack prediction cohort differs.')
            old_predictions[arm, seed] = predictions
            rows = old_summaries[arm, seed]['exports']
            require(len(rows) == 8 and {row['group'] for row in rows} == groups, 'Old exports do not retain all eight groups.')
            for row in rows:
                path = checked(spec['evaluation_path']+'/'+row['path'], row['sha256'])
                exports[arm, seed, row['group']] = dict(path=str(path.relative_to(root)), sha256=row['sha256'])
        require(len(exports) == 96 and len({row['path'] for row in exports.values()}) == 96,
                'Saved export count or distinctness differs.')
        explicit = {(row['arm'], row['seed'], row['group']):dict(path=row['path'], sha256=row['sha256'])
                    for row in spec['export_files']}
        require(len(spec['export_files']) == 96 and explicit == exports, 'Protocol export freeze differs.')

        companion = pathlib.Path(args.companion)
        if companion.is_absolute():
            companion = companion.relative_to(root)
        result_path = checked(str(companion/'result.json'))
        registered_path = checked(str(companion/'registered.json'))
        result, registered = load(result_path), load(registered_path)
        require(result['status'] == 'completed', 'Companion result is incomplete.')
        provenance = dict(source_commit=protocol['source_commit'], protocol_sha256=digest(protocol_path))
        require(result['provenance'] == provenance and registered['provenance'] == provenance, 'Companion provenance differs.')
        own_source = 'course/source_role_fix/saved_teacher_diagnostic.py'
        checked(own_source, registered['own_source_sha256'])
        require(registered['own_source_sha256'] == protocol['source_files'][own_source], 'Companion source differs from execution freeze.')
        require(registered['saved_graphs'] == 96 and registered['original_exported_groups'] == 8
                and registered['max_fresh_teacher_forwards'] == 104, 'Registered matrix count differs.')
        require(registered['training_adaptation_allowed'] is False and registered['crossed_mask_inference'] is False,
                'Companion scope differs.')
        for row in registered['retained_input_hashes']:
            checked(row['path'], row['sha256'])
        clean_path = checked(str(companion/'clean_predictions.jsonl'), result['output_hashes']['clean_predictions.jsonl'])
        attack_path = checked(str(companion/'attacked_predictions.jsonl'), result['output_hashes']['attacked_predictions.jsonl'])
        clean = index(lines(clean_path), lambda value: value['group'])
        attacked = index(lines(attack_path), lambda value: (value['arm'], value['seed'], value['group']))
        expected_keys = {(arm, seed, group) for arm, seed in expected_runs for group in groups}
        require(set(clean) == groups and set(attacked) == expected_keys, 'Completed companion row cohort differs.')
        require(len(clean) == 8 and len(attacked) == 96, 'Completed row count differs.')
        clean_predictions = {}
        for group, row in clean.items():
            old = old_clean[group]
            for key in ('node_ids', 'labels'):
                require(row[key] == old[key], 'Clean target order or labels changed.')
            require(row['saved_victim_logits'] == old['gaga_logits'], 'Saved clean victim channel changed.')
            require(len(set(row['node_ids'])) == len(row['node_ids']) and all(label == 1 for label in row['labels']),
                    'Clean target uniqueness or fraud labels differ.')
            clean_predictions[group] = (
                prediction(row['fresh_teacher_logits'], len(row['labels'])),
                prediction(row['saved_victim_logits'], len(row['labels'])))

        recounted = {}
        all_teacher, all_victim = collections.Counter(), collections.Counter()
        agreement_total = both_wrong_total = occurrences_total = 0
        for (arm, seed, group), row in sorted(attacked.items()):
            old, baseline = old_predictions[arm, seed][group], clean[group]
            for key in ('node_ids', 'labels', 'node_budget', 'edge_budget'):
                require(row[key] == old[key], 'Saved attack identity/order/label/budget changed.')
            require(row['node_ids'] == baseline['node_ids'] and row['labels'] == baseline['labels'],
                    'Clean and attacked targets differ.')
            require(row['saved_victim_logits'] == old['gaga_logits'], 'Saved attacked victim logits changed.')
            require(row['export_sha256'] == exports[arm, seed, group]['sha256'], 'Attack output points to another export.')
            labels = row['labels']
            teacher_pred = prediction(row['fresh_teacher_logits'], len(labels))
            victim_pred = prediction(row['saved_victim_logits'], len(labels))
            teacher_transition = transitions(labels, clean_predictions[group][0], teacher_pred)
            victim_transition = transitions(labels, clean_predictions[group][1], victim_pred)
            require(teacher_transition == row['teacher_transitions'], 'Fresh teacher transitions do not recount.')
            require(victim_transition == row['saved_victim_transitions'] == old['gaga_transitions'], 'Saved victim transitions do not recount.')
            agreements = sum(int(a == b) for a, b in zip(teacher_pred, victim_pred))
            both_wrong = sum(int(a != label and b != label) for label, a, b in zip(labels, teacher_pred, victim_pred))
            require(agreements == row['prediction_agreements'], 'Prediction agreements do not recount.')
            require(both_wrong == row['teacher_and_victim_wrong'], 'Wrong intersections do not recount.')
            require(agreements == len(labels)-teacher_transition['attacked_wrong']-victim_transition['attacked_wrong']+2*both_wrong,
                    'Binary disagreement/intersection identity failed.')
            summary = recounted.setdefault((arm, seed), dict(graphs=0, teacher=collections.Counter(),
                        saved_victim=collections.Counter(), prediction_agreements=0, teacher_and_victim_wrong=0))
            summary['graphs'] += 1
            summary['teacher'].update(teacher_transition)
            summary['saved_victim'].update(victim_transition)
            summary['prediction_agreements'] += agreements
            summary['teacher_and_victim_wrong'] += both_wrong
            all_teacher.update(teacher_transition);all_victim.update(victim_transition)
            agreement_total += agreements;both_wrong_total += both_wrong;occurrences_total += len(labels)

        summaries = index(result['summaries'], lambda value: (value['arm'], value['seed']))
        require(set(summaries) == expected_runs, 'Companion summaries omit or duplicate runs.')
        for key, counts in recounted.items():
            summary = summaries[key]
            require(counts['graphs'] == summary['graphs'] == 8, 'Summary graph count differs.')
            for channel in ('teacher', 'saved_victim'):
                for name, value in counts[channel].items():
                    require(summary[channel][name] == value, 'Summary transition total differs.')
                for field, numerator in (('clean_mr_percent', 'clean_wrong'), ('attacked_mr_percent', 'attacked_wrong')):
                    measured = 100*counts[channel][numerator]/counts[channel]['occurrences']
                    require(math.isclose(summary[channel][field], measured, rel_tol=0, abs_tol=1e-12), 'Summary rate differs.')
            for name in ('prediction_agreements', 'teacher_and_victim_wrong'):
                require(summary[name] == counts[name], 'Summary agreement or wrong intersection differs.')

        expected_counters = dict(gaga_teacher_dispatches=104, gaga_teacher_completed_forwards=104,
                                 attack_graphs_reconstructed=96, generator_forwards=0,
                                 victim_forwards=0, optimizer_steps=0, API_calls=0, TEST_queries=0)
        require(result['counters'] == expected_counters and result['official_victim_loaded'] is False,
                'Recorded physical operation counters differ from bounded design.')
        teacher_ref = old_protocol['teacher_artifacts_ref']
        manifest = load(checked(teacher_ref['path'], teacher_ref['sha256']))
        require(result['teacher_state_sha256'] == manifest['teachers']['gaga']['final_state_sha256'],
                'Recorded frozen teacher identity differs from retained selected checkpoint identity.')
        require(result['saved_attack_graphs'] == 96 and result['original_exported_groups'] == 8,
                'Result cohort summary differs.')
        receipt.update(status='passed', checks_passed=checks, unique_files_hashed=len(hashed),
                       opaque_tensor_exports_hashed=96, old_runs=12, groups_per_run=8,
                       fresh_clean_rows=8, fresh_attacked_rows=96,
                       recounted_occurrences=occurrences_total,
                       recorded_physical_counters=expected_counters,
                       teacher_transition_totals=dict(all_teacher), saved_victim_transition_totals=dict(all_victim),
                       prediction_agreements=agreement_total, teacher_and_victim_wrong=both_wrong_total,
                       protocol_sha256=digest(protocol_path), companion_result_sha256=digest(result_path),
                       registered_sha256=digest(registered_path),
                       clean_predictions_sha256=digest(clean_path), attacked_predictions_sha256=digest(attack_path),
                       input_hashes=hashed,
                       limitations=['Saved-record and byte audit, not a fresh model replay or human verification.',
                                    'No tensor contents were deserialized, so graph reconstruction internals are checked by execution/source review, not reconstructed here.',
                                    'Physical zero-victim/generator/update claims reconcile recorded counters and scope; they are not independently instrumented telemetry.',
                                    'Auditor authored source-role comparison model code, but did not author companion diagnostic.'])
    except BaseException as error:
        receipt.update(status='failed', checks_passed_before_failure=checks,
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        receipt.update(process_cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-wall)
        with output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False, ensure_ascii=True)
            stream.write('\n')
        # All input paths are relative; no target IDs, labels, logits or secrets
        # are copied into this shareable receipt.
        print(json.dumps({key:value for key,value in receipt.items() if key != 'input_hashes'}, allow_nan=False))
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
