import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
import Config as C

class AcrobotNet (nn.Module) :
    """
    Simple multi-layer feed forward network
    with ReLU non linearity.
    """
    def __init__ (self, s=C.OBSERVATION_SPACE, 
            a=C.ACTION_SPACE, hdims=C.HDIMS) : 
        super(AcrobotNet, self).__init__()
        self.pi = self.buildff([s, *hdims, a])
        self.v  = self.buildff([s, *hdims, 1])
        self.dropout = nn.Dropout(p=0.5)

    def buildff (self, lst) : 
        layerDims = zip(lst, lst[1:])
        return nn.ModuleList([
            nn.Linear(a, b) for a, b in layerDims
        ])

    def withReluDropout (self, y, f) : 
        return F.relu(self.dropout(f(y)))

    def forward (self, x, previous_action, previous_reward) :
        pi = reduce(self.withReluDropout, self.pi[:-1], x)
        pi = self.pi[-1](pi)
        pi = F.softmax(pi, dim=-1)
        v  = reduce(self.withReluDropout, self.v[:-1], x)
        v  = self.v[-1](v)
        v  = v.squeeze(-1)
        return pi, v

