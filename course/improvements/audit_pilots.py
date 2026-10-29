"""Post-run recount and cost reconciliation; never generates new predictions."""
import datetime as dt
import json
import math
import re
import statistics
import subprocess
import time
import traceback

import numpy as np

from course.runner import ROOT, sha256, write_json
from course.improvements.run import validate
from modules.data_handler import DataHandler
from utils.utils import get_n_ids


def read(path):
    return json.loads(path.read_text())


def main():
    started = time.monotonic()
    try:
        protocol_path = ROOT / 'course/improvements/pilot_protocol_v1.json'
        protocol = read(protocol_path)
        validate(protocol)
        events_path = ROOT / 'evidence/improvement_pilot_events.jsonl'
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        assert events[-1]['event'] == 'all_complete', 'Queue is not complete; do not issue a passing audit.'
        starts = [x for x in events if x['event'] == 'run_start']
        ends = [x for x in events if x['event'] == 'run_end']
        assert len(starts) == len(ends) == 9
        assert all(x['exit_code'] == 0 for x in ends)
        assert not [x for x in events if x['event'] in ('failed_stop', 'budget_stop', 'own_child_terminated')]
        ending = {(x['variant'], x['seed']): x for x in ends}
        assert len(ending) == 9
        base_config = read(ROOT / 'course/configs/baseline.json')
        cpu_config = dict(base_config, cuda_id='cpu')
        data = DataHandler(cpu_config)
        records, cells = [], {}
        public = ROOT / 'evidence/published_improvement_pilots'
        public.mkdir(exist_ok=True)
        for registration_path in sorted((ROOT / 'evidence/improvements').glob('*/registered.json')):
            folder, reg = registration_path.parent, read(registration_path)
            result, config, groups = [read(folder / name) for name in ('result.json', 'config.json', 'target_groups.json')]
            key = reg['variant'], reg['seed']
            assert key in ending and key not in cells
            assert result['status'] == 'completed' and result['result'] == {'test_skipped': 1.0}
            assert reg['phase'] == 'improvement_pilot' and reg['protocol_sha256'] == sha256(protocol_path)
            assert sha256(folder / 'config.json') == reg['config_sha256']
            expected = dict(base_config)
            for name, default in dict(shared_parameters=False, without_candidates=False, pos_encoding=True, ada_budget=True).items():
                expected.setdefault(name, default)
            expected.update(seed=reg['seed'], cuda_id=reg['gpu'], random_candidates=False, exp_name=reg['run_id'],
                            candidate_policy='coverage' if reg['variant'] == 'coverage' else 'baseline',
                            epochs=protocol['epochs'], patience=protocol['patience'],
                            config_path=str((folder / 'config.json').relative_to(ROOT)))
            assert config == expected and config['valid_epoch'] == 1
            assert groups == dict(train=protocol['train_groups'], valid=protocol['valid_groups'], test_used=False)
            assert not list(folder.glob('*_test_*'))
            assert sha256(ROOT / result['checkpoint']) == result['checkpoint_sha256']
            # Verify the registered commit contains the exact code and protocol used.
            for item in protocol['code']:
                content = subprocess.check_output(['git', 'show', reg['commit']+':'+item['path']], cwd=ROOT)
                import hashlib
                assert hashlib.sha256(content).hexdigest() == item['sha256']
            validation = sorted(folder.glob('*_surrogate_valid_attack.json'))
            assert len(validation) == result['completed_epochs']
            best_count = None
            for metadata_path in validation:
                metadata, raw_path = read(metadata_path), metadata_path.with_suffix('.jsonl')
                assert sha256(raw_path) == metadata['prediction_sha256']
                seen, wrong, count, counts = set(), 0, 0, []
                for line in raw_path.read_text().splitlines():
                    row = json.loads(line)
                    group = row['target_group']
                    assert group in protocol['valid_groups'] and group not in seen
                    seen.add(group)
                    members = get_n_ids(data.target_assignment, group).tolist()
                    assert row['node_ids'] == members
                    truth = data.labels[members].tolist()
                    assert row['labels'] == truth and len(row['predictions']) == len(truth)
                    assert row['node_budget'] == int(data.node_budgets[group])
                    assert row['edge_budget'] == int(data.edge_budgets[group])
                    errors = sum(a != b for a, b in zip(truth, row['predictions']))
                    assert errors == row['incorrect']
                    wrong += errors
                    count += len(truth)
                    counts.append(dict(target_group=group, target_occurrences=len(truth), incorrect=errors))
                assert seen == set(protocol['valid_groups'])
                assert len(seen) == metadata['groups'] and count == metadata['target_occurrences'] and wrong == metadata['incorrect']
                assert math.isclose(wrong/count, metadata['result']['misclf_rate'], abs_tol=1e-14)
                epoch = int(metadata_path.name.split('_')[0])
                if epoch == result['best_epoch']:
                    assert math.isclose(wrong/count, result['best_validation']['misclf_rate'], abs_tol=1e-14)
                    public_path = public / (reg['run_id']+'_best_counts.jsonl')
                    public_path.write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in counts))
                    best_count = dict(errors=wrong, target_occurrences=count, groups=len(seen), mr_percent=100*wrong/count,
                                      raw_prediction_sha256=sha256(raw_path), public_counts=str(public_path.relative_to(ROOT)),
                                      public_counts_sha256=sha256(public_path))
            assert best_count is not None
            traces = sorted(folder.glob('sampling_epoch_*.json'))
            assert len(traces) == result['completed_epochs']
            processed = 0
            for trace_path in traces:
                trace = read(trace_path)
                assert trace['complete'] and len(trace['groups']) == trace['group_updates'] == len(protocol['train_groups'])
                assert set(trace['groups']).issubset(protocol['train_groups'])
                occurrences = sum(len(get_n_ids(data.target_assignment, group)) for group in trace['groups'])
                assert occurrences == trace['target_occurrences']
                if reg['variant'] != 'size_sampling':
                    assert sorted(trace['groups']) == protocol['train_groups']
                processed += occurrences
            log = ROOT / 'runs' / ('improvement_pilot_%s_seed%d_gpu%d_attempt1.log' % (reg['variant'], reg['seed'], reg['gpu']))
            records.append(dict(run_id=reg['run_id'], variant=reg['variant'], seed=reg['seed'],
                                registration_sha256=sha256(registration_path), configuration_sha256=sha256(folder/'config.json'),
                                result_sha256=sha256(folder/'result.json'), checkpoint_sha256=result['checkpoint_sha256'],
                                source_commit=reg['commit'], best_epoch=result['best_epoch'], completed_epochs=result['completed_epochs'],
                                validated_epochs=len(validation), best_validation=best_count, training_group_updates=len(traces)*len(protocol['train_groups']),
                                training_target_occurrences=processed, terminal_log_sha256=sha256(log),
                                gpu_wall_hours_charged=max(result['elapsed_seconds'], ending[key]['elapsed_seconds'])/3600))
            cells[key] = best_count['mr_percent']
        assert len(records) == 9
        effects = {variant: statistics.mean(cells[variant, seed]-cells['baseline', seed] for seed in protocol['seeds'])
                   for variant in ['coverage', 'size_sampling']}
        selected = 'coverage' if effects['coverage'] >= effects['size_sampling']-.25 else 'size_sampling'
        summary = read(ROOT / 'evidence/improvement_pilot_summary.json')
        assert summary['selected'] == selected and not summary['test_evaluated']
        assert all(math.isclose(effects[name], summary['effects'][name]['mean_gain_pp'], abs_tol=1e-12) for name in effects)
        diagnostic = read(ROOT / 'evidence/training_diagnostic_v1/result.json')
        checks = []
        for path in sorted((ROOT/'evidence').glob('improvement_smoke_attempt*.log')):
            match = re.search(r'Ran \d+ tests in ([0-9.]+)s', path.read_text())
            assert match
            checks.append(dict(path=str(path.relative_to(ROOT)), wall_seconds=float(match[1]),
                               conservative_gpu_hours=float(match[1])/3600))
        pilot_cost = sum(row['gpu_wall_hours_charged'] for row in records)
        measured = pilot_cost+diagnostic['gpu_wall_hours']+sum(x['conservative_gpu_hours'] for x in checks)
        ledger = dict(unit='GPU wall-clock hours, summed across concurrent devices; price unavailable',
                      limit=12, pilot_queue_limit=2, pilots_charged=pilot_cost, diagnosis=diagnostic['gpu_wall_hours'],
                      checks=checks, measured_or_conservatively_charged=measured,
                      unmetered_setup_reserve=0.02, remaining_after_reserve=12-measured-.02,
                      failed_or_interrupted_incidents=[],
                      reconciliation='All nine queue starts have successful exits and one complete registration/result; no missing attempt.',
                      methodology='Charge max registered elapsed and queue elapsed per pilot; charge full smoke wall time as one GPU; reserve is not measured usage.')
        assert pilot_cost < 2 and ledger['remaining_after_reserve'] > 0
        write_json(ROOT/'evidence/research_cost_ledger.json', ledger)
        report = dict(status='passed', audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      audit_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      kind='Post-run evidence audit using saved predictions; no new test or validation inference',
                      pilot_protocol_sha256=sha256(protocol_path), queue_events_sha256=sha256(events_path),
                      selected=selected, mean_validation_gains_pp=effects, runs=records,
                      failures_or_interruptions=[], cost_ledger='evidence/research_cost_ledger.json',
                      cpu_only_audit_elapsed_seconds=time.monotonic()-started,
                      human_verification='pending; this is an agent-run evidence audit')
        write_json(ROOT/'evidence/improvement_pilot_audit.json', report)
        print(json.dumps(dict(status='passed', selected=selected, gains=effects, cost_ledger=ledger), indent=2), flush=True)
    except Exception:
        failure = dict(status='failed', error=traceback.format_exc(), recorded_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        path = ROOT/'evidence'/('pilot_audit_failure_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')
        write_json(path, failure)
        raise


if __name__ == '__main__':
    main()
