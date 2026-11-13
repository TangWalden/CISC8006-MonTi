"""Fetch exact official asset bytes and build input caches, never experiments."""
import pathlib,json,hashlib,time,urllib.request,struct,zlib,subprocess,os
S=pathlib.Path(__file__).resolve().parent;R=S.parent.parent
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,v):
 with pathlib.Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
start=time.monotonic();manifest=json.loads((S/'asset_manifest.json').read_text());opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));downloads=[]
result={'status':'failed','GPU_hours':0,'training':False,'efficacy_evaluation':False}
try:
 for item in manifest:
  target=S/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
  assert not target.exists()
  if 'member' in item:
   total=84226498 if item['official_file_id']=='1sLcalw491WFjZ3rzgD32DldsSU6MNjHx' else 52174348
   url='https://drive.usercontent.google.com/download?id=%s&export=download&confirm=t'%item['official_file_id'];ent=item['member']
   def fetch(offset,size):
    req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(offset,offset+size-1)})
    with opener.open(req,timeout=35) as r:
     b=r.read(size+1);assert r.status==206 and r.headers['Content-Range']=='bytes %d-%d/%d'%(offset,offset+size-1,total) and len(b)==size
    return b
   hdr=fetch(ent['offset'],1024);f=struct.unpack_from('<4s5H3I2H',hdr)
   assert f[0]==b'PK\x03\x04' and not f[2]&8 and f[3]==8 and hdr[30:30+f[9]].decode()==ent['name']
   begin=ent['offset']+30+f[9]+f[10];payload=bytearray()
   for off in range(0,ent['compressed'],1048576):payload.extend(fetch(begin+off,min(1048576,ent['compressed']-off)))
   raw=zlib.decompress(payload,-15);assert zlib.crc32(raw)&0xffffffff==ent['crc32']
  else:
   url='https://media.githubusercontent.com/media/bdi-lab/MonTi/10f233e953679ec3715125dc058ef6d28bee4691/'+item['path']
   with opener.open(url,timeout=35) as r:raw=r.read(item['bytes']+1)
  assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
  target.write_bytes(raw);downloads.append({'path':item['path'],'sha256':sha(target),'bytes':len(raw)})
  print('ASSET_VERIFIED',item['path'],flush=True)
 files={x['path']:x['sha256'] for x in manifest};files['cache_prepare.py']=sha(S/'cache_prepare.py')
 sources={str(p.relative_to(R)):sha(p) for folder in ['modules','models','utils'] for p in (R/folder).rglob('*.py')}
 registration={'purpose':'Owner requests YelpChi data/cache preparation only; no training or efficacy evaluation.','files':files,'runtime_sources':sources,'runtime_commit':'ca093d8b12ea153a076b02bbdbc4ede58508560c','CPU_threads':4,'nice':10,'GPU_hours':0,'timeout_seconds':1800,'new_dependencies':0}
 write(S/'cache_registration.json',registration)
 result['download_wall_seconds']=time.monotonic()-start
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DGLBACKEND='pytorch',LD_LIBRARY_PATH='/home/tjy/miniconda3/envs/monti/lib',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
 with (S/'cache.stdout').open('xb') as out,(S/'cache.stderr').open('xb') as err:
  child=subprocess.run(['/home/tjy/miniconda3/envs/monti/bin/python','-u',str(S/'cache_prepare.py')],cwd=S,env=env,stdout=out,stderr=err,timeout=1830)
 result.update(status='completed' if child.returncode==0 else 'failed',cache_exit_code=child.returncode)
except Exception as e:result.update(error_type=type(e).__name__,error=str(e))
finally:
 result.update(downloads=downloads,outer_wall_seconds=time.monotonic()-start);write(S/'asset_cache_receipt.json',result)
