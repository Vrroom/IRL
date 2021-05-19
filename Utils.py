""" Utility functions """
import torch
import numpy as np

def computeReturns(R, gamma, normalize=False) : 
    """ Compute discounted returns """
    g = 0
    G = []
    for r in R[::-1] : 
        g = g * gamma + r
        G.insert(0, g)
    G = np.array(G)
    if normalize : 
        G = (G - G.mean()) / (G.std() + 1e-7)
    return G

def inRange(a, interval) : 
    """ 
    Check whether a number is in the given interval.
    """
    lo, hi = interval
    return a >= lo and a < hi

