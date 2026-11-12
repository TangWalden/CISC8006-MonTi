import hashlib,json,pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
def sha(p):
    h=hashlib.sha256()
    with pathlib.Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def write(p,v):
    with pathlib.Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def verify(path):
    p=read(path);assert p['schema']=='monti_block_swap_v1' and not p['final_TEST_allowed']
    assert pathlib.Path(p['source_root']).resolve()==ROOT
    rt=pathlib.Path(p['runtime_root'])
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=rt,text=True).strip()==p['runtime_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=rt,text=True).strip()
    for n,h in p['source_manifest']['files'].items():assert sha(ROOT/n)==h,n
    for r in p['input_refs']:assert sha(rt/r['path'])==r['sha256'],r['path']
    for root in (str(rt),str(ROOT)):
        if root not in sys.path:sys.path.insert(0,root)
    return p,dict(source_commit=p['source_manifest']['source_commit'],protocol_sha256=sha(path),runtime_commit=p['runtime_commit'])
