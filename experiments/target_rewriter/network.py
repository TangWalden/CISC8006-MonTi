"""Per-target cross attention with separate region and conditional rule heads."""
import torch
from torch import nn

class Rewriter(nn.Module):
 def __init__(self,d):
  super().__init__();self.target=nn.Sequential(nn.Linear(3*d+1,64),nn.GELU(),nn.LayerNorm(64))
  self.action=nn.Sequential(nn.Linear(2*d+3,64),nn.GELU(),nn.LayerNorm(64))
  self.attn=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
  self.rule=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
  self.region=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
  self.stop=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,targets,actions,coverage,regions):
  t=self.target(targets);a=self.action(actions)
  if not len(a):return self.stop(t.mean(0)).reshape(1)*0
  context,_=self.attn(a.unsqueeze(0),t.unsqueeze(0),t.unsqueeze(0),attn_mask=-2*(1-coverage))
  h=torch.cat((a,context.squeeze(0)),1);rules=self.rule(h).flatten()
  ids=sorted(set(regions));region_logits=torch.cat((self.stop(t.mean(0)).reshape(1),torch.stack([self.region(h[[i for i,r in enumerate(regions) if r==g]].mean(0)).reshape(()) for g in ids])))
  lp=torch.log_softmax(region_logits,0);out=[lp[0]]+[None]*len(regions)
  for k,g in enumerate(ids,1):
   ix=[i for i,r in enumerate(regions) if r==g];conditional=torch.log_softmax(rules[ix],0)
   for j,i in enumerate(ix):out[i+1]=lp[k]+conditional[j]
  return torch.stack(out)
