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

def simulateAgent (agentFile) : 
    env = gym.make('Acrobot-v1')
    EnvSpace = namedtuple('EnvSpace', ['action', 'observation'])
    state_dict = torch.load(agentFile, map_location=torch.device('cpu'))["agent_state_dict"]
    agent = CategoricalPgAgent(AcrobotNet)
    agent.initialize(EnvSpace(env.action_space, env.observation_space))
    agent.load_state_dict(state_dict)
    done = False
    trajectory = []
    s = torch.tensor(env.reset()).float()
    a = torch.tensor(0)
    r = torch.tensor(0).float()
    i = 0
    while not done : 
        i += 1
        a = agent.step(s, a, r).action
        s_, r, done, info = env.step(a.item())
        s_ = torch.tensor(s_).float()
        r = torch.tensor(r).float()
        s = s_
    env.close()
    return i

def main () : 
    agentFiles = os.listdir('a2c_acrobot/run_0/')
    agentFiles = [osp.join('a2c_acrobot/run_0/', f) for f in agentFiles if f.endswith('pkl')]
    scores = [np.mean([simulateAgent(f) for _ in range(10)]) for f in tqdm(agentFiles)]
    bestFile = agentFiles[np.argmin(scores)]
    print(bestFile, min(scores))

if __name__ == "__main__" : 
    main ()
