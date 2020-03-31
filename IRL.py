from Utils import * 
import gym
import numpy as np
from ValueApproximation import *
from Network import FeedForwardNetwork
from MountainCarUtils import *
from pulp import *
import Agents
from PlotUtils import *

def inverseRL (env, agent, gamma, valueEstimator, 
    rewardBases, valueBases) :
    """
    Applying section 4 of: 
        
        Algorithms for Inverse Reinforcement Learning
        - Ng, Russell ('00)

    to the Acrobot environment. Inverse RL seems like
    a fun thing because it retrieves the incentive 
    mechanism underlying a particular behaviour.

    Should find out how to do this on Humans.

    Examples
    --------
    >>> env = gym.make('Acrobot-v1')
    >>> agent = Agents.REINFORCE('./Models/acrobotMimicer.pkl')  
    >>> gamma = 0.99
    >>> bases = acrobotRewardBases(0.4, 0.4)
    >>> valBases = [FeedForwardNetwork([6, 3]) for _ in range(len(bases))]
    >>> R = inverseRL(env, agent, gamma, monteCarlo, bases, valBases)

    Parameters
    ----------
    env : object
        An OpenAI environment. 
    agent : object
        A policy i.e. a mapping from 
        states to actions.
    gamma : float
        The discount factor.
    valueEstimator : function
        Algorithm which estimates value
        function. Examples are monteCarlo
        and td0 present in 
        ValueApproximation.py.
    rewardBases : list
        List of basis functions 
        characterizing the reward.
    valueBases : 
        List of value function 
        approximators that will be 
        tuned to fit the reward basis
        and policy.
    """

    def setupObjective() : 
        """
        Encode the LP objective from the paper.
        """
        nonlocal problem
        alphas = [LpVariable(f'a{i}', -1, 1) for i in range(len(valueBases))]
        actions = set(range(env.action_space.n))
        bs = []
        for i, s in enumerate(stateSamples) : 
            b = LpVariable(f'b{i}')
            bs.append(b)
            a = agent(s)
            s1 = toTensor(mountainSampleNextState(env, s, a))
            for ai in actions - {a} : 
                si = toTensor(mountainSampleNextState(env, s, ai))
                coeffs = [(vFn(s1)-vFn(si)).item() for vFn in valueBases]
                terms = [c * a for c, a in zip(coeffs, alphas)]
                constraint1 = b <= 2 * lpSum(terms)
                constraint2 = b <= lpSum(terms)
                problem += constraint1
                problem += constraint2
        problem += lpSum(bs)
            
    def rewardFunction (s) : 
        rTotal = 0
        for a, fn in zip(alphas, rewardBases) : 
            rTotal += (a * fn(s))
        return rTotal

    stateSamples = [env.observation_space.sample() for _ in range(1)]

    # Tweak the value function approximator's 
    # parameters to fit to the value function
    # under given policy and for each reward
    # basis.
    for vFn, rFn in zip(valueBases, rewardBases) :
        valueEstimator(vFn, rFn, env, agent, gamma, 1e-1)

    problem = LpProblem('Inverse RL Problem', LpMaximize)
    setupObjective()
    problem.solve()
    alphas = [a.varValue for a in problem.variables() if a.name.startswith('a')]

    return rewardFunction

if __name__ == "__main__" :
    env = gym.make('MountainCar-v0')
    agent = Agents.REINFORCE('./Models/mountainCarMimicer.pkl')  
    s = env.reset()
    gamma = 0.99
    bases = mountainCarRewardBases(0.2)
    print(len(bases))
    valBases = [FeedForwardNetwork([2, 1]) for _ in range(len(bases))]
    R = inverseRL(env, agent, gamma, td0, bases, valBases)
    xRange = np.arange(-1.2, 0.6, 0.1)
    yRange = np.arange(-0.07, 0.07, 0.01)
    plotFunction(lambda x, y : R([x, y]), xRange, yRange, 'pos', 'vel', 'reward')
