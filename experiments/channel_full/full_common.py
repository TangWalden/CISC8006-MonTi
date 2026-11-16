"""Full TEST orchestration around the unchanged development-selected method."""
import pathlib,json,hashlib,sys
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def protocol(stage):
 s=pathlib.Path(stage);p=read(s/'protocol.json');assert p['schema']=='channel_full_v1' and p['TEST_allowed']
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h,n
 for n,h in p['inputs'].items():assert sha(n)==h,n
 return s,p
def setup(stage,ds):
 s,p=protocol(stage);sys.path.insert(0,str(pathlib.Path(p['method_parent'])/'source'))
 import cr_common
 loaded=cr_common.setup(p['method_parent'],ds)
 return s,p,loaded
def counts():return dict(physical_builds=0,teacher_forwards=0,teacher_contexts=0,backward_calls=0,algebraic_pool_scores=0,official_GAGA_TEST=0,upstream_generations=0,native_canonical_builds=0,updates=0,API_calls=0,GPU_hours=0)
