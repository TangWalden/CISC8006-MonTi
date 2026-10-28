import hashlib
import json
import pathlib
import time
import zipfile
import gdown
import requests

root = pathlib.Path(__file__).resolve().parent
items = [('best_models', '1A5d3l4-4_N9jUzok9J6TFd_DHALDBDgV'),
         ('Datasets', '1sLcalw491WFjZ3rzgD32DldsSU6MNjHx')]
manifest = []
for name, file_id in items:
    path = root / 'downloads' / (name + '.zip')
    print('START', name, time.time(), flush=True)
    if not path.exists() or not zipfile.is_zipfile(path):
        result = gdown.download(id=file_id, output=str(path), quiet=True, resume=True)
        if result is None:
            raise RuntimeError('Download failed: ' + name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        (root / 'evidence/bootstrap' / (name + '_members.json')).write_text(json.dumps(members, indent=2))
        selected = members if name == 'best_models' else [x for x in members if 'GossipCop/' in x]
        for member in selected:
            target = (root / member).resolve()
            if root not in target.parents:
                raise RuntimeError('Unsafe archive path: ' + member)
        archive.extractall(root, members=selected)
    manifest.append(dict(name=name, google_drive_id=file_id, bytes=path.stat().st_size,
                         sha256=digest.hexdigest(), extracted=len(selected)))
    (root / 'evidence/bootstrap/assets.json').write_text(json.dumps(manifest, indent=2))
    print('COMPLETE', manifest[-1], flush=True)
print('ALL_COMPLETE', flush=True)
