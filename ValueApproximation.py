from Utils import * 
from more_itertools import *

def monteCarlo(valFn, rewardFn,  env, agent, gamma, lr):
    """
    Gradient Monte Carlo Algorithm for 
    estimating V^{\pi}.

    It is assumed that the approximation
    of the value function is represented
    by a computational graph. 

    Parameters
    ----------
    valFn : nn.Module
        Computational graph parametrizing
        the approx. to value function.
    rewardFn : lambda
        Reward function for which the 
        value function has to be computed.
    env : object
        Environment. 
    agent : object
        The policy function. Mapping
        states to actions.
    gamma : float
        Discount factor of the MDP.
    lr : float
        Learning rate.
    """
    def updateWts () : 
        for wt in valFn.parameters() : 
            if wt.grad is not None : 
                # Update rule from Sutton and Barto
                wt.data.add_(lr * (g - v) * wt.grad)

    episodes = 100
    for episode in range(episodes) : 
        trajectory = getTrajectory(env, agent)
        states = unzip(trajectory)[0]
        R = list(map(rewardFn, states))
        G = computeReturns(R, gamma, normalize=False)

        for step, g in zip(trajectory, G) : 
            s, _, _ = step

            valFn.zero_grad()
            v = valFn(toTensor(s))
            v.backward()

            updateWts() 
            print(list(valFn.parameters()))

    import pdb
    pdb.set_trace()

def td0(valFn, rewardFn, env, agent, gamma, lr):
    """
    Semi-Gradient TD(0) Algorithm for 
    estimating V^{\pi}.

    It is assumed that the approximation
    of the value function is represented
    by a computational graph. 

    Parameters
    ----------
    valFn : nn.Module
        Computational graph parametrizing
        the approx. to value function.
    rewardFn : lambda
        Reward function for which we 
        are calculating the value function.
    env : object
        Environment. 
    agent : object
        The policy function. Mapping
        states to actions.
    gamma : float
        Discount factor of the MDP.
    lr : float
        Learning rate.
    """
    def updateWts () : 
        for wt in valFn.parameters() : 
            if wt.grad is not None : 
                # Update rule from Sutton and Barto
                wt.data.add_(lr * (r + gamma * vs_ - vs) * wt.grad)

    episodes = 100
    for episode in range(episodes) : 
        trajectory = getTrajectory(env, agent)

        for step1, step2 in zip(trajectory, trajectory[1:]) : 
            s, _, _ = step1
            s_, _, _ = step2

            r = rewardFn(s)
            valFn.zero_grad()
            vs = valFn(toTensor(s))
            vs_ = valFn(toTensor(s_))
            vs.backward()
            updateWts() 


