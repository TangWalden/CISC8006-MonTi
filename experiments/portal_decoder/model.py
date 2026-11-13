"""Contextual dual-encoder ranks counterfactual two-hop benign evidence."""
import torch
from torch import nn
class PortalRanker(nn.Module):
    def __init__(self,width):
        super().__init__();self.query=nn.Sequential(nn.Linear(3*width+3,64),nn.GELU())
        self.key=nn.Linear(width,64);self.change=nn.Linear(width,64)
        self.score=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,query,keys,changes):
        q=self.query(query).expand(len(keys),-1);k=self.key(keys);d=self.change(changes)
        raw=self.score(torch.cat((q,k,q*k,d),1)).flatten()
        # Row0 is the native identity action. Scores are relative to identity.
        return raw-raw[0]
