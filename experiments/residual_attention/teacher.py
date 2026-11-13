"""Independent full-TRAIN GAGA refit; never reads a victim checkpoint."""
import argparse,json,pathlib,time,hashlib,os,sys,signal
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);a=ap.parse_args();s=pathlib.Path(a.registration).parent;reg=json.loads(pathlib.Path(a.registration).read_text());assert sha(__file__)==reg['sources']['teacher.py']
    rt=pathlib.Path(reg['runtime_root']);sys.path.insert(0,str(rt));assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    for name,h in reg['inputs'].items():assert sha(rt/name)==h,name
    out=s/'teacher';out.mkdir(exist_ok=False);start=time.monotonic();cpu=time.process_time()
    signal.signal(signal.SIGALRM,lambda *x:(_ for _ in ()).throw(TimeoutError('teacher deadline')));signal.alarm(1800)
    import torch,numpy as np,dgl,random
    torch.set_num_threads(4);torch.manual_seed(20261113);np.random.seed(20261113);random.seed(20261113)
    from models.victim_model import GAGA
    assert torch.cuda.device_count()==0
    graph=dgl.load_graphs(str(rt/'Datasets/GossipCop/GossipCop_0.05.bin'))[0][0]
    ids=np.load(rt/'Datasets/GossipCop/data_split/train_idx_surro_0.npy',allow_pickle=False);ids=torch.from_numpy(ids).long()
    all_labels=graph.ndata.pop('y');labels=all_labels[ids].long();del all_labels
    seq=np.load(rt/'Datasets/GossipCop/data_split/gossipcop_0.05_gaga_sequence_0.npy',mmap_mode='r',allow_pickle=False)
    X=torch.from_numpy(seq[ids.numpy()].copy());width=X.shape[-1];model=GAGA(width,64,2);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    weights=len(labels)/(2*torch.bincount(labels,minlength=2).float());steps=forwards=0;history=[]
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    for epoch in range(1,39):
        model.train();total=0
        for batch in torch.randperm(len(ids)).split(512):
            optimizer.zero_grad();logits=model(X[batch]);forwards+=1;loss=torch.nn.functional.cross_entropy(logits,labels[batch],weight=weights);assert bool(torch.isfinite(loss));loss.backward();optimizer.step();steps+=1;total+=float(loss)*len(batch)
        history.append(dict(epoch=epoch,train_loss=total/len(ids)));print('TEACHER_EPOCH',epoch,flush=True)
    model.eval()
    with torch.no_grad():
        logits=[]
        for batch in torch.arange(len(ids)).split(512):logits.append(model(X[batch]));forwards+=1
        logits=torch.cat(logits);pred=logits.argmax(1);accuracy=float((pred==labels).float().mean())
    recalls=[float((pred[labels==k]==k).float().mean()) for k in (0,1)]
    f1=[]
    for k in (0,1):
        tp=int(((pred==k)&(labels==k)).sum());fp=int(((pred==k)&(labels!=k)).sum());fn=int(((pred!=k)&(labels==k)).sum());f1.append(2*tp/max(1,2*tp+fp+fn))
    assert any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    torch.save(model.state_dict(),out/'model.pt')
    result=dict(status='completed',seed=20261113,epochs=38,epoch_rule='Fixed38 from earlier TRAIN-only internal study, no new VALID/TEST selection.',full_TRAIN_count=len(ids),training_steps=steps,forwards=forwards,TRAIN_accuracy=accuracy,TRAIN_recalls=recalls,TRAIN_macro_F1=sum(f1)/2,quality_scope='In-sample training check, not fresh held-out calibration/generalization.',quality_passed=accuracy>=.6 and min(recalls)>=.2 and sum(f1)/2>=.6,history=history,checkpoint_sha256=sha(out/'model.pt'),original_victim_loaded=False,TEST_queries=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':main()
