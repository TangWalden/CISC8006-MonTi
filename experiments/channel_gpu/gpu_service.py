"""One bounded GPU teacher service; CPU graph construction stays unchanged."""
import argparse,pathlib,json,time,os,signal,subprocess,importlib.util
from multiprocessing.connection import Listener
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);p=json.loads((s/'protocol.json').read_text());start=time.monotonic();cpu=time.process_time();calls=[];status='failed';listener=None
 import torch
 assert os.environ['CUDA_VISIBLE_DEVICES']==str(p['gpu_ordinal'])
 assert torch.cuda.device_count()==torch._C._cuda_getDeviceCount()==1
 assert torch.__version__=='1.13.1+cu117';torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines();row=[r.split(',') for r in rows if int(r.split(',')[0])==p['gpu_ordinal']][0];assert row[1].strip()==p['gpu_uuid'] and int(row[2])>=p['minimum_free_MiB']
 torch.cuda.set_per_process_memory_fraction(p['allocator_fraction'],0);torch.cuda.init();torch.zeros(1,device='cuda:0');torch.cuda.synchronize()
 own=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True).splitlines();assert any(int(r.split(',')[0])==os.getpid() and r.split(',')[1].strip()==p['gpu_uuid'] for r in own)
 models={};states={}
 import hashlib
 for ds in p['datasets']:
  f=s/(ds+'_teacher.pt');meta=json.loads((s/(ds+'_export.json')).read_text());assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'];mf=pathlib.Path(meta['model_source']);assert hashlib.sha256(mf.read_bytes()).hexdigest()==meta['model_source_sha256']
  spec=importlib.util.spec_from_file_location('fixed_gaga_'+ds,mf);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);m=mod.GAGA(meta['feature_dim'],64,2);state=torch.load(f,map_location='cpu');m.load_state_dict(state,strict=True);m.eval()
  for v in m.parameters():v.requires_grad_(False)
  models[ds]=m.to('cuda:0');states[ds]=state
 key=(s/'rpc.key').read_bytes();socket=str(s/'teacher.sock');listener=Listener(socket,family='AF_UNIX',authkey=key);os.chmod(socket,0o600)
 (s/'gpu_ready.json').write_text(json.dumps(dict(pid=os.getpid(),uuid=p['gpu_uuid'],both_visible_counts=1,allocated_bytes=torch.cuda.memory_allocated(),TF32=False)))
 signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Bounded GPU service deadline')));signal.alarm(p['GPU_service_seconds'])
 try:
  while True:
   conn=listener.accept()
   try:
    req=conn.recv()
    if req['action']=='stop':conn.send(dict(status='stopped'));status='completed';break
    assert req['action']=='predict' and req['dataset'] in models and req['mode']=='boundary'
    x=torch.from_numpy(req['x']).to('cuda:0').detach().requires_grad_(req['grad']);k,t,_,d=x.shape;assert k<=48 and d in [32,768] and t<=p['maximum_targets'];began=time.monotonic()
    with torch.set_grad_enabled(req['grad']):
     z=models[req['dataset']](x.reshape(-1,7,d)).reshape(k,t,2);loss=torch.sigmoid(z[...,1]-z[...,0]).mean(-1);grad=torch.autograd.grad(loss.sum(),x)[0] if req['grad'] else None
    assert torch.isfinite(z).all() and (grad is None or torch.isfinite(grad).all())
    result=dict(status='ok',z=z.detach().cpu().numpy(),loss=loss.detach().cpu().numpy(),grad=None if grad is None else grad.detach().cpu().numpy());torch.cuda.synchronize();calls.append(dict(dataset=req['dataset'],contexts=k,targets=t,gradient=req['grad'],wall_seconds=time.monotonic()-began));conn.send(result)
    del x,z,loss,grad,result
   finally:conn.close()
  for ds,m in models.items():
   for k,v in m.state_dict().items():assert torch.equal(v.cpu(),states[ds][k]),(ds,k)
 finally:
  signal.alarm(0)
  if listener:listener.close()
  (s/'gpu_receipt.json').write_text(json.dumps(dict(status=status,pid=os.getpid(),wall_seconds=time.monotonic()-start,process_CPU_seconds=time.process_time()-cpu,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),calls=calls,updates=0,official_TEST_calls=0),indent=2)+'\n')
if __name__=='__main__':main()
