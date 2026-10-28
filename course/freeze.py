import datetime as dt
import json

from course.runner import ROOT, sha256, write_json


def main():
    path = ROOT / 'course/frozen_protocol.json'
    if path.exists():
        raise RuntimeError('Existing freeze must not be silently replaced.')
    pilots = []
    for result_path in (ROOT / 'evidence/runs').glob('*/result.json'):
        result = json.loads(result_path.read_text())
        registration = json.loads(result_path.with_name('registered.json').read_text())
        if registration['phase'] == 'pilot' and result['status'] == 'completed':
            pilots.append(dict(mode=registration['mode'], result=result, evidence=str(result_path.relative_to(ROOT))))
    if set(x['mode'] for x in pilots) != {'learned', 'random'}:
        raise RuntimeError('Both selection modes must pass feasibility pilots.')
    manifest = json.loads((ROOT / 'evidence/data/input_manifest.json').read_text())
    scientific_files = list(ROOT.glob('models/*.py')) + list(ROOT.glob('modules/*.py')) + list(ROOT.glob('utils/*.py'))
    scientific_files += [ROOT / x for x in ['course/runner.py', 'course/run.sh', 'course/analyze.py']]
    cached_files = [p for p in (ROOT / 'Datasets/GossipCop').rglob('*') if p.is_file() and (p.name.startswith('h_') or '_gaga_' in p.name)]
    frozen = dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  claim='learned candidate selector produces higher size-weighted target MR than the defined random selector',
                  level='Component; course approval not yet recorded',
                  configuration='course/configs/baseline.json',
                  source_config_sha256=sha256(ROOT / 'course/configs/baseline.json'),
                  seeds=[0, 1, 2], modes=['learned', 'random'], gpu_hour_limit=48,
                  training='official maximum epochs, validation cadence and early stopping; no test-set selection',
                  random_control='uniform without replacement every forward in training and evaluation, same exploration pool and K',
                  metric='size-weighted target-occurrence MR percent',
                  decision='paired seed t(2) 95% CI: lower>0 supported; upper<=0 not supported; otherwise inconclusive',
                  inputs=manifest,
                  scientific_code=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in scientific_files],
                  caches=[dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in cached_files],
                  environment_sha256=sha256(ROOT / 'environment/pip-freeze.txt'),
                  pilots=pilots,
                  differences=['candidate flag mapping corrected', 'upstream incidence-based budget retained',
                               'historical table5 random-control semantics unconfirmed', 'fixed split/checkpoints; three paired seeds',
                               'Table4/Table5 numerical discrepancy unresolved'])
    write_json(path, frozen)
    print('FROZEN', sha256(path), flush=True)


if __name__ == '__main__':
    main()
