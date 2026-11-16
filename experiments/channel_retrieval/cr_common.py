"""Frozen parent, source and dataset loading for the next registered round."""
import pathlib,sys,json,hashlib
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def setup(stage,ds):
 s=pathlib.Path(stage);p=read(s/'protocol.json');assert p['schema']=='channel_retrieval_v1' and not p['TEST_allowed']
 for n,h in p['sources'].items():assert sha(s/'source'/n)==h
 for n,h in p['inputs'].items():assert sha(n)==h,n
 parent=pathlib.Path(p['parent']);sys.path.insert(0,str(parent/'source'))
 import gr_common
 _,old,v,ancestor,ancestor_protocol,q,data,view,teacher,th,benign,common=gr_common.setup(str(parent),ds)
 return s,p,parent,old,v,ancestor,ancestor_protocol,q,data,view,teacher,th,benign,common
def counts():return dict(physical_builds=0,teacher_forwards=0,teacher_contexts=0,backward_calls=0,algebraic_pool_scores=0,official_GAGA_VALID=0,upstream_generations=0,updates=0,TEST_queries=0,API_calls=0)
