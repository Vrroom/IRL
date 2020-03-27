import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce

class FeedForwardNetwork (nn.Module) :
    """
    Simple multi-layer feed forward network
    with ReLU non linearity.
    """
    def __init__ (self, dims, bias=True) : 
        """
        Constructor.
        
        Parameters
        ----------
        dims : list
            A list of integers indicating the 
            input/output dimensions of each 
            layer.
        bias : bool
            Whether to have bias in the last
            layer.
        """
        super(FeedForwardNetwork, self).__init__()

        layerDims = zip(dims[:-1], dims[1:-1])
        self.layers = nn.ModuleList([nn.Linear(a,b) for a,b in layerDims])

        # We don't apply ReLU after the last
        # layer which is why we have instantiated
        # a seperate linear layer for it.
        self.last = nn.Linear(dims[-2], dims[-1], bias=bias)
        self.dropout = nn.Dropout(p=0.5)

    def forward (self, x) :
        x = reduce(lambda y, f : F.relu(self.dropout(f(y))), self.layers, x)
        x = self.last(x)
        return x

