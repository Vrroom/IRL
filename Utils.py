import torch
import numpy as np

def toTensor (s) : 
    return torch.tensor(s).float()

def getTrajectory (env, agent) : 
    """
    Get a trajectory by playing out
    the agent's policy. 

    Parameters
    ----------
    env : object
        OpenAI gym environment, or any
        other environment which supports
        the same API.
    agent : lambda
        Any object which can output a
        policy for the MDP given a 
        particular state. 
    """
    s = env.reset()
    done = False
    trajectory = []
    while not done : 
        a = agent(s) 
        s_, r, done, _ = env.step(a)
        trajectory.append((s, a, r))
        s = s_
    return trajectory

def computeReturns(R, gamma, normalize=True) : 
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
    Check whether a number
    is in the given interval.
    """
    lo, hi = interval
    return a >= lo and a < hi

