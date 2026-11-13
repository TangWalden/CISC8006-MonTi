"""Independent canonical forward and feature-gradient check on two TRAIN cases."""
import pathlib,json,sys,argparse,time,hashlib,os
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True);a=ap.parse_args();s=pathlib.Path(a.stage);reg=json.loads((s/'preflight_registration.json').read_text())
    assert sha(__file__)==reg['sources']['preflight.py']
    for n,h in reg['sources'].items():assert sha(s/'source'/n)==h
    result=json.loads((s/'teacher/result.json').read_text());assert result['status']=='completed' and result['quality_passed'];assert sha(s/'teacher/model.pt')==result['checkpoint_sha256']
    rt=reg['runtime_root'];support=reg['support_source'];sys.path[:0]=[str(s/'source'),support,rt]
    import torch,numpy as np
    from models.victim_model import GAGA
    from modules.data_handler import DataHandler
    from experiments.local_swap.loader import CONFIG,DEFAULTS
    from experiments.downstream_cooperation.common import check_device,make_graph,canonical
    from course.dual_surrogate.runtime import load_original_view
    from model import Residual,feature_update
    from coefficients import coefficients
    check_device();start=time.monotonic();cpu=time.process_time();torch.manual_seed(20261113)
    config=dict(CONFIG,**DEFAULTS);config.update(victim_model='GCN');data=DataHandler(config);view=load_original_view(data)
    model=GAGA(data.feat_dim,64,2);model.load_state_dict(torch.load(s/'teacher/model.pt',map_location='cpu'));model.eval()
    for p in model.parameters():p.requires_grad_(False)
    train=data.features[torch.as_tensor(data.idx_train).long()];std=train.std(0,unbiased=False);low=data.features.min(0).values;high=data.features.max(0).values
    parent=pathlib.Path('/home/tjy/Project/MonTi-course/runs/local_swap_20261113_attempt1/main');up=json.loads((parent/'upstream_seal.json').read_text());refs=[r for r in up['cases'] if r['seed']==0 and r['phase']=='train'][:2]
    rows=[];forwards=0
    for ref in refs:
        path=parent/ref['path'];assert sha(path)==ref['sha256'];v=torch.load(path,map_location='cpu');L1,L2=coefficients(view,v);base=make_graph(data,v,v['native_adjacency'])
        with torch.no_grad():seq=canonical(data,view,base,v)
        radius=torch.minimum(1.0*std,torch.minimum(v['x_aux']-low,high-v['x_aux']).clamp_min(0));d=(.2*radius*torch.randn_like(radius).tanh()).requires_grad_();direct=d.detach().clone().requires_grad_()
        changed=make_graph(data,v,v['native_adjacency']);changed.ndata['x']=torch.cat((data.features,v['x_aux']+direct))
        actual=canonical(data,view,changed,v);linear=feature_update(seq,L1,L2,d);error=float((actual-linear).abs().max());assert error<2e-5,error
        za=model(actual);zb=model(linear);forwards+=2;logit_error=float((za-zb).abs().max());assert logit_error<2e-5,logit_error
        ga=torch.autograd.grad(za.sum(),direct)[0];gb=torch.autograd.grad(zb.sum(),d)[0];grad_error=float((ga-gb).abs().max());assert grad_error<2e-5 and float(ga.norm())>0
        net=Residual(data.feat_dim,True);initial=net(v,std,low,high);assert torch.equal(initial,torch.zeros_like(initial));opt=torch.optim.Adam(net.parameters(),lr=.001)
        nonzero=[]
        for _ in range(2):
            opt.zero_grad();delta=net(v,std,low,high);z=model(feature_update(seq,L1,L2,delta));forwards+=1;loss=torch.nn.functional.softplus(z[:,1]-z[:,0]).mean();loss.backward();nonzero.append(sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None));opt.step()
        assert min(nonzero)>0;delta=net(v,std,low,high).detach();assert float(delta.abs().max())>0
        assert bool(((v['x_aux']+delta>=low-1e-6)&(v['x_aux']+delta<=high+1e-6)).all());assert bool((delta.abs()<=radius+1e-7).all())
        rows.append(dict(group=ref['group'],sequence_error=error,logit_error=logit_error,feature_gradient_error=grad_error,feature_gradient_norm=float(ga.norm()),network_gradient_sums=nonzero,initial_identity=True,bounds_passed=True))
    out=dict(status='passed',cases=rows,own_teacher_forwards=forwards,preflight_downstream_updates=4,original_victim_forwards=0,TEST_queries=0,GPU_hours=0,wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu)
    with (s/'preflight_result.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out))
if __name__=='__main__':main()
