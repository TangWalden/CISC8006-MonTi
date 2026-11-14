"""Synthetic model invariants; no efficacy evidence."""
def synthetic():
 import torch,itertools
 from network import Rewriter
 from kernel import options,randomize,dose
 torch.manual_seed(23);net=Rewriter(5).eval();targets=torch.randn(3,16);actions=torch.randn(8,13);coverage=torch.randint(0,2,(8,3)).float();regions=[0]*4+[2]*4
 p=net(targets,actions,coverage,regions);assert torch.isfinite(p).all() and abs(float(p.exp().sum())-1)<1e-6
 perm=torch.tensor([2,0,1]);q=net(targets[perm],actions,coverage[:,perm],regions);assert torch.allclose(p,q,atol=1e-6,rtol=1e-6)
 ap=torch.tensor([4,1,6,3,0,5,2,7]);q=net(targets,actions[ap],coverage[ap],[regions[i] for i in ap]);assert torch.allclose(q,torch.cat((p[:1],p[ap+1])),atol=1e-6,rtol=1e-6)
 one=net(targets,actions[:4],coverage[:4],[1]*4);assert abs(float(one.exp().sum())-1)<1e-6
 empty=net(targets,actions[:0],coverage[:0],[]);assert empty.shape==(1,) and float(empty)==0
 steps=[dict(donor=i,bank=list(range(10+i*4,14+i*4))) for i in range(3)];a=torch.tensor([[1.,1.,1.,0.],[1.,0.,0.,1.]]);base=dict(seed=0,group=1,ids=torch.tensor([1]),native_adjacency=a)
 checks=5
 for k in range(3):
  for rs in itertools.combinations(range(3),k):
   for bs in itertools.product(range(4),repeat=k):
    chosen=list(zip(rs,bs));assert len(options(steps,rs))==(3-k)*4
    for kind in ['joint','endpoint']:
     random=randomize(base,steps,chosen,kind);assert len(random)==k and len({r for r,b in random})==k and dose(base,steps,random)==dose(base,steps,chosen);assert random==randomize(base,steps,chosen,kind);checks+=1
 return dict(status='passed',checks=checks,scope='Synthetic target/action equivariance, normalization, empty STOP and exact random dose/count invariants')
if __name__=='__main__':print(synthetic())
