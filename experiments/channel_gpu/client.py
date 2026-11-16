"""Explicit CPU input/gradient boundary; GPU service never sees labels or IDs."""
import pathlib,time
from multiprocessing.connection import Client
def predictor(stage,dataset):
 s=pathlib.Path(stage);key=(s/'rpc.key').read_bytes()
 def predict(model,seqs,c,mode,grad=False):
  import torch
  assert mode=='boundary';x=torch.stack(seqs).detach().contiguous();began=time.monotonic()
  conn=Client(str(s/'teacher.sock'),family='AF_UNIX',authkey=key)
  try:
   conn.send(dict(action='predict',dataset=dataset,x=x.numpy(),grad=bool(grad),mode=mode));r=conn.recv()
  finally:conn.close()
  assert r['status']=='ok';c['teacher_forwards']+=1;c['teacher_contexts']+=len(seqs);c['backward_calls']+=int(grad);c['GPU_RPC_calls']=c.get('GPU_RPC_calls',0)+1;c['GPU_RPC_wait_seconds']=c.get('GPU_RPC_wait_seconds',0)+time.monotonic()-began
  return torch.from_numpy(r['z']),torch.from_numpy(r['loss']),None if r['grad'] is None else torch.from_numpy(r['grad'])
 return predict
