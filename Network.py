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
    def __init__ (self, s=6, a=3, hdims=C.HDIMS) : 
        super(AcrobotNet, self).__init__()
        self.shared = self.buildff([s, *hdims])
        self.pi = nn.Linear(hdims[-1], a)
        self.v = nn.Linear(hdims[-1], 1)
        self.dropout = nn.Dropout(p=0.5)

    def buildff (self, lst) : 
        layerDims = zip(lst, lst[1:])
        return nn.ModuleList([
            nn.Linear(a, b) for a, b in layerDims
        ])

    def withReluDropout (self, y, f) : 
        return F.relu(self.dropout(f(y)))

    def forward (self, x, previous_action, previous_reward) :
        feature = reduce(self.withReluDropout, self.shared, x)
        pi = F.softmax(self.pi(feature), dim=-1)
        v = self.v(feature).squeeze(-1)
        return pi, v

