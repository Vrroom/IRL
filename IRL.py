import numpy as np
np.random.seed(0)
import random
random.seed(0)
import pickle
import gym
import more_itertools
from AcrobotUtils import *
from pulp import *
from scipy.spatial.distance import pdist, squareform
import os
import os.path as osp
import Config as C
from A2C import *

def findSamplesInTrajs (stateSamples, trajs) : 
    """ 
    For each state sample, find all indices (i, j) such that
    the jth state in ith trajectory is approximately the state
    sample
    """
    nSamples = stateSamples.shape[0]
    stateOccurenceIndices = [[] for _ in range(nSamples)]
    allStates = [np.stack([s for s, _, _ in t]) for t in trajs]
    for i, traj in enumerate(trajs) : 
        trajLen = len(traj)
        D = squareform(pdist(np.concatenate((stateSamples, allStates[i]), axis=0)))
        D = D[:nSamples, nSamples:]
        indices = np.where(D < C.STATE_SIMILARITY_THRESH)
        for j, k  in zip(*indices) : 
            stateOccurenceIndices[j].append((i, k))
    return stateOccurenceIndices

def generateStateSamples (trajs, nSamples) : 
    """ get the distribution of start states """
    allStates = [[s for s, _, _ in t] for t in trajs]
    allStates = list(more_itertools.flatten(allStates))
    states = random.sample(allStates, k=nSamples)
    states = np.array(states)
    return states

def estimateValueFromTrajs (stateIndices, trajs, rewardFn) :
    """ 
    Estimate the value for each state from expert 
    trajectories.
    """
    def computeReturnOnTraj (traj) : 
        R = [rewardFn(s) for s, _, _ in traj]
        return computeReturns(R, C.DISCOUNT)[0]

    values = []
    for i, indices in enumerate(stateIndices) : 
        truncatedTrajs = [trajs[i][j:] for i, j in indices] 
        vhat = np.mean([computeReturnOnTraj(t) for t in truncatedTrajs])
        values.append(vhat)
    return values

def estimateValueFromAgent (stateSamples, agent) : 
    """
    Use the learnt value function network through
    A2C to estimate value for states.
    """
    value = lambda s : agent.model.v[-1](
        reduce(agent.model.withReluDropout, 
            agent.model.v[:-1], s))
    return list(map(
        reduce(compose, [float, 
                         value, 
                         lambda t : t.float(), 
                         torch.tensor]),
        stateSamples))

def getAllTraj () : 
    """ get all trajectories from C.TRAJ_DIR """
    def loadPickle (f) : 
        with open(osp.join(C.TRAJ_DIR, f), 'rb') as fd : 
            return pickle.load(fd)
    return list(map(loadPickle, os.listdir(C.TRAJ_DIR)))

def irl (rewardFnSpace) :
    """
    Find the explanatory reward function for expert's 
    policy in the space of reward functions.
    """
    trajs = getAllTraj()
    stateSamples = generateStateSamples(trajs, C.IRL_STATE_SAMPLES)
    indices = findSamplesInTrajs(stateSamples, trajs) 
    for i in range(C.IRL_ITR) : 
        rewardFn = rewardFnSpace.current()
        agent = findOptimalAgent(rewardFn)
        expertValues = [estimateValueFromTrajs(indices, trajs, _) 
                        for _ in rewardFnSpace.rewardBases]
        inferiorValues = estimateValueFromAgent(stateSamples, agent) 
        rewardFnSpace.refine(expertValues, inferiorValues)
    return pi, rewardFn

if __name__ == "__main__" :
    irl (RewardFnSpace(acrobotRewardBases(np.pi / 4, np.pi / 4)))
