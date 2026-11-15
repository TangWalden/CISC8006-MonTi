"""Same independent full-TRAIN GAGA teacher fit as the GossipCop method."""
import pathlib,argparse,time,signal,random
from common import setup,sha,write
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);s,p,q=setup(ap.parse_args().stage);out=s/'teacher';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time();signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('teacher ceiling')));signal.alarm(1800)
 import torch,numpy as np,dgl
 from models.victim_model import GAGA
 from experiments.downstream_cooperation.common import check_device
 check_device();torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113);root=pathlib.Path(p['asset_root'])/'Datasets/YelpChi_RTR'
 graph=dgl.load_graphs(str(root/'YelpChi.bin'))[0][0];ids=torch.from_numpy(np.load(root/'data_split/train_idx_surro_0.npy',allow_pickle=False)).long();labels=graph.ndata['y'][ids].long();del graph
 seq=np.load(root/'data_split/yelpchi_rtr_gaga_sequence_0.npy',mmap_mode='r',allow_pickle=False);X=torch.from_numpy(seq[ids.numpy()].copy());model=GAGA(X.shape[-1],64,2);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);weights=len(ids)/(2*torch.bincount(labels,minlength=2).float());history=[];steps=forwards=0
 for epoch in range(1,39):
  model.train();total=0
  for batch in torch.randperm(len(ids)).split(512):
   opt.zero_grad();z=model(X[batch]).reshape(len(batch),2);forwards+=1;loss=torch.nn.functional.cross_entropy(z,labels[batch],weight=weights);assert torch.isfinite(loss);loss.backward();opt.step();steps+=1;total+=float(loss)*len(batch)
  history.append(dict(epoch=epoch,train_loss=total/len(ids)));print('TEACHER_EPOCH',epoch,flush=True)
 model.eval()
 with torch.no_grad():
  zz=[]
  for batch in torch.arange(len(ids)).split(512):zz.append(model(X[batch]).reshape(len(batch),2));forwards+=1
  z=torch.cat(zz);pred=z.argmax(1)
 accuracy=float((pred==labels).float().mean());recalls=[float((pred[labels==k]==k).float().mean()) for k in [0,1]];f1=[]
 for k in [0,1]:
  tp=int(((pred==k)&(labels==k)).sum());fp=int(((pred==k)&(labels!=k)).sum());fn=int(((pred!=k)&(labels==k)).sum());f1.append(2*tp/max(1,2*tp+fp+fn))
 torch.save(model.state_dict(),out/'model.pt');quality=accuracy>=.6 and min(recalls)>=.2 and sum(f1)/2>=.6
 write(out/'result.json',dict(status='completed',quality_passed=quality,quality_scope='Original fixed in-sample TRAIN sanity gate,notcalibration/OOF/heldoutselection',seed=20261113,epochs=38,full_TRAIN_count=len(ids),training_steps=steps,forwards=forwards,TRAIN_accuracy=accuracy,TRAIN_recalls=recalls,TRAIN_macro_F1=sum(f1)/2,history=history,checkpoint_sha256=sha(out/'model.pt'),official_victim_loaded=False,TEST_queries=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu));signal.alarm(0);assert quality
if __name__=='__main__':main()
