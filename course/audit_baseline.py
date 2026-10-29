"""Independently recount saved baseline predictions without rerunning a test."""
import hashlib
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frozen = json.loads((ROOT / 'course/frozen_protocol.json').read_text())
    for item in frozen['scientific_code'] + frozen['inputs']:
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    summary = json.loads((ROOT / 'evidence/reproduction_summary.json').read_text())
    registry = json.loads((ROOT / 'evidence/run_registry.json').read_text())
    runs = [r for r in registry if r['registration']['phase'] == 'reproduction']
    assert len(runs) == 6
    output = ROOT / 'evidence/published_baseline'
    output.mkdir(exist_ok=True)
    cells, audits = {}, []
    for run in runs:
        registration, result = run['registration'], run['status']
        assert result['status'] == 'completed'
        assert registration['commit'] == '9f782dfbb98581e8bbe3c10f5ea5de91237e09ab'
        assert not registration['tracked_changes']
        folder = ROOT / run['evidence']
        final = list(folder.glob('*_victim_test_attack.json'))
        assert len(final) == 1
        metadata = json.loads(final[0].read_text())
        raw = final[0].with_suffix('.jsonl')
        assert digest(raw) == metadata['prediction_sha256']
        published, wrong, count = [], 0, 0
        for line in raw.read_text().splitlines():
            row = json.loads(line)
            errors = sum(a != b for a, b in zip(row['labels'], row['predictions']))
            size = len(row['labels'])
            assert size == len(row['predictions']) == len(row['node_ids'])
            assert errors == row['incorrect']
            wrong += errors
            count += size
            published.append(dict(target_group=row['target_group'], target_occurrences=size,
                                  incorrect=errors, node_budget=row['node_budget'], edge_budget=row['edge_budget']))
        assert len(published) == 909 and count == 10515
        assert metadata['incorrect'] == wrong and metadata['target_occurrences'] == count
        assert math.isclose(wrong / count, result['result']['misclf_rate'], abs_tol=1e-14)
        key = (registration['mode'], registration['seed'])
        assert key not in cells
        cells[key] = 100 * wrong / count
        public_path = output / (registration['run_id'] + '_counts.jsonl')
        public_path.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in published))
        audits.append(dict(run_id=registration['run_id'], groups=len(published),
                           target_occurrences=count, incorrect=wrong, mr_percent=cells[key],
                           raw_prediction_sha256=digest(raw),
                           public_counts=str(public_path.relative_to(ROOT)),
                           public_counts_sha256=digest(public_path)))
    differences = [cells['learned', seed] - cells['random', seed] for seed in (0, 1, 2)]
    mean = statistics.mean(differences)
    sd = statistics.stdev(differences)
    half = 4.302652729911275 * sd / math.sqrt(3)
    assert math.isclose(mean, summary['paired_mean_difference_pp'], abs_tol=1e-10)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip([mean-half, mean+half], summary['paired_95ci_pp']))
    assert summary['conclusion'] == 'inconclusive' and mean-half < 0 < mean+half
    report = dict(status='passed', kind='post-run evidence recount; no new test inference',
                  runs=audits, paired_differences_pp=differences,
                  paired_mean_pp=mean, paired_sample_sd_pp=sd, paired_95ci_pp=[mean-half, mean+half],
                  ci_method='Student paired t interval, df=2; three runs, not individual target occurrences',
                  frozen_inputs_and_scientific_code='all SHA-256 values matched',
                  public_output_policy='counts omit node IDs, labels, features and model weights; full predictions retained separately')
    (ROOT / 'evidence/baseline_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
