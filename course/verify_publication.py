"""Check public history and publication files without ML dependencies."""
import argparse
import datetime as dt
import json
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
CJK = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')


def git(*args, data=None):
    return subprocess.check_output(['git']+list(args), cwd=ROOT, input=data)


def contains_chinese(data):
    try:
        value = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        return False
    if CJK.search(value):
        return True
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return False
    return bool(CJK.search(json.dumps(decoded, ensure_ascii=False)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    mapping = json.loads((ROOT/'evidence/publication_history_map.json').read_text())
    refs = ['HEAD']+[row['public_commit'] for row in mapping['tag_map'].values()]
    rows = git('rev-list','--objects',*refs).decode().splitlines()
    objects = [(row.split(' ',1)+[''])[:2] for row in rows]
    ids = {oid for oid,_ in objects}
    assert not [old for old,new in mapping['commit_map'].items() if old != new and old in ids], 'Private original execution history entered public ancestry.'
    batch = git('cat-file','--batch',data=''.join(oid+'\n' for oid,_ in objects).encode())
    offset, findings = 0, []
    for oid,path in objects:
        end=batch.index(b'\n',offset)
        fields=batch[offset:end].split()
        size=int(fields[2])
        content=batch[end+1:end+1+size]
        offset=end+size+2
        if contains_chinese(content):
            findings.append(dict(scope='published_history',object=oid,path=path))
    files=git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0')
    checked=0
    for relative in files:
        path=ROOT/relative
        if not relative or not path.is_file():
            continue
        checked+=1
        if contains_chinese(path.read_bytes()):
            findings.append(dict(scope='working_publication_files',path=relative))
    result=dict(status='passed' if not findings else 'failed',
                checked_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                public_head=git('rev-parse','HEAD').decode().strip(),
                public_history_objects=len(objects),publication_files=checked,findings=findings,
                scope='Current public HEAD, mapped public freeze snapshots, and tracked/unignored publication files; private execution refs are excluded.')
    if args.output:
        destination=ROOT/args.output
        assert str(destination.resolve()).startswith(str(ROOT.resolve())+os.sep)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
