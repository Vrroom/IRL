from Network import *
import Agents
import gym
from Utils import * 
import functional
from AcrobotUtils import *
from more_itertools import *
from PlotUtils import *

def monteCarlo(valFn, rewardFn, env, agent, featureExtractor, gamma, lr):
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
    featureExtractor : function
        Computes a succint representation
        of the state.
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

    episodes = 20
    for episode in range(episodes) : 
        trajectory = getTrajectory(env, agent)
        states = unzip(trajectory)[0]
        R = list(map(compose(rewardFn, featureExtractor), states))
        G = computeReturns(R, gamma, normalize=False)

        for step, g in zip(trajectory, G) : 
            s, _, _ = step

            valFn.zero_grad()
            v = valFn(toTensor(s))
            v.backward()

            updateWts() 

def td0(valFn, rewardFn, env, agent, featureExtractor, gamma, lr):
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
    featureExtractor : function
        Computes a succint representation
        of the state.
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

    episodes = 5
    for episode in range(episodes) : 
        trajectory = getTrajectory(env, agent)

        for step1, step2 in zip(trajectory, trajectory[1:]) : 
            s, _, _ = step1
            s_, _, _ = step2

            r = rewardFn(featureExtractor(s))
            valFn.zero_grad()

            vs  = valFn(toTensor(s ))
            vs_ = valFn(toTensor(s_))

            vs.backward()
            updateWts() 

def td1 (valFn, rewardFn, env, agent, featureExtractor, gamma, lr) :
    episodes = 10
    trajectories = [[t[0] for t in getTrajectory(env, agent)] for _ in range(episodes)]
    rewards = [[rewardFn(featureExtractor(s)) for s in t] for t in trajectories]
    returns = [computeReturns(r, gamma, normalize=False) for r in rewards]

    trajectories = [toTensor(t) for t in trajectories]
    returns = [toTensor(r) for r in returns]

    optimizer = torch.optim.Adam(valFn.parameters(), lr=lr)
    epochs = 200
    for epoch in range(epochs) : 
        for t, r in zip(trajectories, returns) : 
            optimizer.zero_grad()
            estimatedReturn = valFn(t) 
            diff = estimatedReturn - r
            loss = torch.sum(diff * diff)
            loss.backward()
            optimizer.step()

if __name__ == "__main__" : 
    env = gym.make('Acrobot-v1')
    agent = Agents.REINFORCE('./Models/acrobotMimicer.pkl')
    rFn = lambda x : -1
    vFn = FeedForwardNetwork([6, 1])
    featureExtractor = lambda s : toInternalStateRep(s)[:2]
    td1(vFn, rFn, env, agent, featureExtractor, 0.99, 1e-1)
    hi = env.observation_space.high
    lo = env.observation_space.low
    X = np.arange(-np.pi, np.pi, 0.1)
    Y = np.arange(-np.pi, np.pi, 0.1)
    plotFunction(lambda x, y: vFn(toTensor([x, y, 0, 0, 0, 0])).item(), X, Y, 'theta1', 'theta2', 'value')
