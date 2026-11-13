"""Two local context streams and a bounded feature residual, no graph edits."""
import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self,width,attention=True):
        super().__init__();self.attention=attention
        self.project=nn.Linear(width,64);self.targets=nn.Linear(width,64);self.context=nn.Linear(width,64)
        if attention:
            self.att_t=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
            self.att_c=nn.MultiheadAttention(64,4,dropout=0,batch_first=True)
        self.gate=nn.Linear(192,64);self.decode=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,width))
        nn.init.zeros_(self.decode[-1].weight);nn.init.zeros_(self.decode[-1].bias)
    def forward(self,v,std,low,high):
        x=v['x_aux'];t=len(v['ids']);c=len(v['candidate_ids']);a=v['native_adjacency'];q=self.project(x)
        def stream(values,mask,projection,att=None):
            if not len(values):return torch.zeros_like(q)
            # A zero null token prevents empty-row attention NaNs.
            keys=torch.cat((projection(values),torch.zeros(1,64)),0)
            valid=torch.cat((mask.bool(),(~mask.bool().any(1)).unsqueeze(1)),1)
            if self.attention:
                return att(q.unsqueeze(1),keys.unsqueeze(0).expand(len(q),-1,-1),keys.unsqueeze(0).expand(len(q),-1,-1),key_padding_mask=~valid,need_weights=False)[0][:,0]
            return valid.to(keys.dtype)@keys/valid.sum(1).clamp_min(1).unsqueeze(1)
        zt=stream(v['x_targets'],a[:,:t],self.targets,getattr(self,'att_t',None))
        zc=stream(v['x_candidates'],a[:,t:t+c],self.context,getattr(self,'att_c',None))
        gate=torch.sigmoid(self.gate(torch.cat((q,zt,zc),1)));fused=gate*zt+(1-gate)*zc
        radius=torch.minimum(.1*std,torch.minimum(x-low,high-x).clamp_min(0))
        return radius*torch.tanh(self.decode(torch.cat((q,fused),1)))

def feature_update(sequence,L1,L2,delta):
    out=sequence.clone();out[:,3]=out[:,3]+L1@delta;out[:,6]=out[:,6]+L2@delta
    return out
