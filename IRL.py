from Utils import * 
import gym
import numpy as np
from ValueApproximation import *
from Network import FeedForwardNetwork
from AcrobotUtils import *
from pulp import *

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
        alphas = [LpVariable(f'a{i}', -1, 1) for i in range(len(valueBasis))]
        actions = set(range(env.action_space.n))
        for i, s in enumerate(stateSamples) : 
            b = LpVariable(f'b{i}')
            problem += b
            a = agent(s)
            s1 = sampleNextState(env, s, a)
            for ai in actions - {a} : 
                si = sampleNextState(env, s, ai)
                terms = [(vFn(s1) - vFn(si)) * alpha for vFn, alpha in zip(valueBasis, alphas)]
                constraint = b <= lpSum(terms)
                problem += constraint
            
    def rewardFunction (s) : 
        rTotal = 0
        for a, fn in zip(alphas, rewardBases) : 
            rTotal += (a * fn(s))
        return rTotal

    stateSamples = [env.observation_space.sample() for _ in range(n)]

    # Tweak the value function approximator's 
    # parameters to fit to the value function
    # under given policy and for each reward
    # basis.
    for vFn, rFn in zip(valueBases, rewardBases) :
        valueEstimator(vFn, rFn, env, agent, gamma, 1e-1)
    
    problem = LpProblem('Inverse RL Problem', LpMaximize)
    setupObjective()
    problem.solve()
    alphas = [a.varValue for a in problem.variables()]

    return rewardFunction

if __name__ == "__main__" :
    env = gym.make('Acrobot-v1')
    agent = Agents.REINFORCE('./Models/acrobotMimicer.pkl')  
    gamma = 0.99
    bases = acrobotRewardBases(0.4, 0.4)
    valBases = [FeedForwardNetwork([6, 3]) for _ in range(len(bases))]
    R = inverseRL(env, agent, gamma, monteCarlo, bases, valBases)
