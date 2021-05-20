""" Main reward optimization loop """
import Config as C
import numpy as np
np.random.seed(C.SEED)
import random
random.seed(C.SEED)
from RewardFnSpace import *
import pickle
import more_itertools
from AcrobotUtils import *
from scipy.spatial.distance import pdist, squareform
import os
import os.path as osp
from A2C import *
from PlotUtils import *
from Eval import *
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

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
    allStates = [[s for s, _, _ in t] for t in trajs[:10]]
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

def estimateValueFromAgent (stateSamples, agent, rewardFn) : 
    """
    Use the learnt value function network through
    A2C to estimate value for states.
    """
    def estimateForState (s) : 
        cpus = list(range(C.N_PARALLEL))
        affinity = dict(cuda_idx=C.CUDA_IDX, workers_cpus=cpus)
        agent_ = CategoricalPgAgent(
            AcrobotNet, 
            initial_model_state_dict=agent.state_dict())
        sampler = SerialSampler(
            EnvCls=rlpyt_make,
            env_kwargs=dict(
                id=C.ENV, 
                reward=rewardFn, 
                internalStateFn=C.INTERNAL_STATE_FN, 
                s0=s),
            batch_T=C.HORIZON,
            batch_B=C.BATCH_B,
            max_decorrelation_steps=0,
        )
        sampler.initialize(
            agent=agent_,
            affinity=affinity,
            seed=C.SEED
        )
        _, traj_info = sampler.obtain_samples(0)
        returns = [t['DiscountedReturn'] for t in traj_info]
        return np.mean(returns)
    estimates = list(map(estimateForState, stateSamples))
    return estimates

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
        env = rlpyt_make('Acrobot-v1', internalStateFn=C.INTERNAL_STATE_FN)
        expertValues = [estimateValueFromTrajs(indices, trajs, _) 
                        for _ in rewardFnSpace.rewardBases]
        inferiorValues = [estimateValueFromAgent(stateSamples, agent, _)
                          for _ in rewardFnSpace.rewardBases]
        rewardFnSpace.refine(expertValues, inferiorValues)
    return agent, rewardFn

if __name__ == "__main__" :
    agent, rewardFn = irl(RewardFnSpace(acrobotRewardBases(np.pi / 2, np.pi / 2)))
    xRange = np.arange(-np.pi, np.pi, 0.1)
    yRange = np.arange(-np.pi, np.pi, 0.1)
    toExternal = lambda x, y : toExternalStateRep([x, y, 0, 0])
    RFn = compose(rewardFn, toExternal)
    plotFunction(RFn, xRange, yRange, 'theta1', 'theta2', 'R')
    plt.savefig('recovered.png')
    plt.show()
    simulateAgent(agent, render=True)
