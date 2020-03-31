from Utils import *
from functools import partial, reduce
from itertools import product
import numpy as np
import scipy.integrate
import gym

def mountainCarRewardBases (delX) :
    xs = np.arange(-1.2, 0.6, delX) 
    bases = []
    for interval in zip(xs, xs[1:]) :
        fn = lambda t : 0 if inRange(t[0], interval) else -1
        bases.append(fn)
    return bases

