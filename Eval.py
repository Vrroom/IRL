""" 
Evaluate an agent based on average number of
steps to finish an environment
"""
from tqdm import tqdm
import os
import os.path as osp
import pickle
import numpy as np
import gym
import sys
import time
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from Network import *
from collections import namedtuple

def simulateAgentFile (agentFile, render=False) :
    """ Load rlpyt agent from file and simulate  """
    state_dict = torch.load(
        agentFile, 
        map_location=torch.device('cpu')) 
    agent = CategoricalPgAgent(AcrobotNet)
    env = gym.make('Acrobot-v1')
    EnvSpace = namedtuple('EnvSpace', ['action', 'observation'])
    agent.initialize(EnvSpace(env.action_space, env.observation_space))
    agent.load_state_dict(state_dict)
    simulateAgent(agent, render)

def simulateAgent (agent, render=False) : 
    """ 
    Simulate agent on environment till the task
    is over and return the number of steps taken
    """
    env = gym.make('Acrobot-v1')
    done = False
    trajectory = []
    s = torch.tensor(env.reset()).float()
    a = torch.tensor(0)
    r = torch.tensor(0).float()
    i = 0
    while not done : 
        i += 1
        if render: 
            env.render()
            time.sleep(0.05)
        a = agent.step(s, a, r).action
        s_, r, done, info = env.step(a.item())
        s_ = torch.tensor(s_).float()
        r = torch.tensor(r).float()
        s = s_
    if render: 
        env.render()
        time.sleep(0.05)
    env.close()
    return i
