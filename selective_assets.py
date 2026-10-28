"""Verified HTTP-range extraction of selected public official ZIP members."""
import concurrent.futures
import hashlib
import importlib
import io
import json
import pathlib
import struct
import time
import threading
import zipfile
import zlib

import requests

ROOT = pathlib.Path(__file__).resolve().parent
CHUNK = 512 * 1024
manifest = []
thread_state = threading.local()
confirm = importlib.import_module('gdown.download').get_url_from_gdrive_confirmation


def connect(name, file_id):
    session = requests.Session()
    session.trust_env = False
    routing = ROOT / 'downloads/drive-download-urls.json'
    if routing.exists():
        cached = json.loads(routing.read_text(encoding='utf-8-sig'))[name]
        return session, cached['url'], cached['bytes']
    url = 'https://drive.google.com/uc?id=' + file_id
    for attempt in range(4):
        response = session.get(url, stream=True, timeout=(90, 90))
        response.raise_for_status()
        if 'text/html' not in response.headers.get('Content-Type', ''):
            total = int(response.headers['Content-Length'])
            url = response.url
            response.close()
            return session, url, total
        url = confirm(response.text)
        response.close()
    raise RuntimeError('No public archive response')


def fetch(session, url, start, stop):
    for attempt in range(5):
        try:
            response = session.get(url, headers={'Range': 'bytes=%d-%d' % (start, stop)}, timeout=(90, 90))
            expected = 'bytes %d-%d/' % (start, stop)
            if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(expected):
                raise RuntimeError('Server did not honor exact byte range')
            data = response.content
            if len(data) != stop-start+1:
                raise RuntimeError('Truncated HTTP range')
            return data
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1+attempt)


for name, file_id in [('best_models', '1A5d3l4-4_N9jUzok9J6TFd_DHALDBDgV'), ('Datasets', '1sLcalw491WFjZ3rzgD32DldsSU6MNjHx')]:
    print('CONNECT', name, flush=True)
    session, url, total = connect(name, file_id)
    tail_path = ROOT / 'downloads' / (name + '_tail.bin')
    tail = tail_path.read_bytes() if tail_path.exists() else fetch(session, url, total-65536, total-1)
    archive = zipfile.ZipFile(io.BytesIO(tail))
    offset_adjustment = total-len(tail)
    entries = archive.infolist()
    listing = [dict(name=e.filename, size=e.file_size, compressed=e.compress_size, offset=e.header_offset+offset_adjustment, crc32=e.CRC) for e in entries]
    (ROOT / 'evidence/bootstrap' / (name + '_range_members.json')).write_text(json.dumps(listing, indent=2))
    wanted = ['configs/GCN-GAGA-gossipcop_0.05-best_config.json'] if name == 'best_models' else [
        'GossipCop/GossipCop_0.05.bin',
        'GossipCop/data_split/train_idx_surro_0.npy', 'GossipCop/data_split/valid_idx_surro_0.npy', 'GossipCop/data_split/test_idx_surro_0.npy',
        'GossipCop/data_split/target_assign_matrix_0.npz',
        'GossipCop/data_split/train_target_mask_0.npy', 'GossipCop/data_split/valid_target_mask_0.npy', 'GossipCop/data_split/test_target_mask_0.npy']
    selected = [e for e in entries if any(e.filename == w or e.filename.endswith('/' + w) for w in wanted)]
    if len(selected) != len(wanted):
        raise RuntimeError('Archive layout differs: matched %s of %s' % ([x.filename for x in selected], wanted))
    for entry in selected:
        offset = entry.header_offset + offset_adjustment
        relative = entry.filename
        if relative.startswith(name + '/'):
            relative = relative[len(name)+1:]
        output = ROOT / name / relative
        if ROOT not in output.resolve().parents:
            raise RuntimeError('Unsafe member path')
        output.parent.mkdir(parents=True, exist_ok=True)
        print('MEMBER', entry.filename, entry.file_size, entry.compress_size, flush=True)
        if output.exists() and output.stat().st_size == entry.file_size:
            existing = output.read_bytes()
            if zlib.crc32(existing) & 0xffffffff == entry.CRC:
                manifest.append(dict(archive=name, google_drive_id=file_id, archive_bytes=total, member=entry.filename,
                                     bytes=len(existing), crc32=entry.CRC, sha256=hashlib.sha256(existing).hexdigest(),
                                     local_path=str(output.relative_to(ROOT)), method='cached verified exact HTTP ranges + ZIP CRC32'))
                (ROOT / 'evidence/bootstrap/selected_assets.json').write_text(json.dumps(manifest, indent=2))
                print('VERIFIED_CACHED', str(output.relative_to(ROOT)), flush=True)
                continue
        header = fetch(session, url, offset, offset+1023)
        fields = struct.unpack_from('<4s5H3I2H', header)
        if fields[0] != b'PK\x03\x04' or fields[2] & 8:
            raise RuntimeError('Unexpected local ZIP header')
        data_start = offset + 30 + fields[9] + fields[10]
        pieces = list(range(0, entry.compress_size, CHUNK))
        piece_dir = ROOT / 'downloads' / 'range_chunks' / hashlib.sha256((name + '/' + relative).encode()).hexdigest()[:16]
        piece_dir.mkdir(parents=True, exist_ok=True)

        def get_piece(position):
            size = min(CHUNK, entry.compress_size-position)
            piece = piece_dir / str(position)
            if not piece.exists() or piece.stat().st_size != size:
                if not hasattr(thread_state, 'session'):
                    thread_state.session = requests.Session()
                    thread_state.session.trust_env = False
                    thread_state.session.cookies.update(session.cookies)
                worker = thread_state.session
                data = fetch(worker, url, data_start+position, data_start+position+size-1)
                piece.write_bytes(data)
            return position

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            for completed, _ in enumerate(pool.map(get_piece, pieces), 1):
                if completed % 8 == 0 or completed == len(pieces):
                    print('CHUNKS', completed, '/', len(pieces), flush=True)
        payload = b''.join((piece_dir / str(position)).read_bytes() for position in pieces)
        raw = zlib.decompress(payload, -15) if entry.compress_type == 8 else payload
        if len(raw) != entry.file_size or zlib.crc32(raw) & 0xffffffff != entry.CRC:
            raise RuntimeError('Extracted member failed size or CRC32 verification')
        output.write_bytes(raw)
        manifest.append(dict(archive=name, google_drive_id=file_id, archive_bytes=total, member=entry.filename,
                             bytes=len(raw), crc32=entry.CRC, sha256=hashlib.sha256(raw).hexdigest(),
                             local_path=str(output.relative_to(ROOT)), method='verified exact HTTP ranges + ZIP CRC32'))
        (ROOT / 'evidence/bootstrap/selected_assets.json').write_text(json.dumps(manifest, indent=2))
        print('VERIFIED', str(output.relative_to(ROOT)), flush=True)
print('ALL_COMPLETE', flush=True)
