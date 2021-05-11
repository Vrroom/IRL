# python3 Play.py <nEpisodes>
import os
import pickle
import os.path as osp
import gym
import sys
import time
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from Network import *
from collections import namedtuple

def main () : 

    env = gym.make('Acrobot-v1')
    EnvSpace = namedtuple('EnvSpace', ['action', 'observation'])
    state_dict = torch.load('Models/acrobot-a2c.pth', map_location=torch.device('cpu'))
    agent = CategoricalPgAgent(AcrobotNet)
    agent.initialize(EnvSpace(env.action_space, env.observation_space))
    agent.load_state_dict(state_dict)
    # agent = Agents.Human(env)
    done = False
    trajectory = []
    s = torch.tensor(env.reset()).float()
    a = torch.tensor(0)
    r = torch.tensor(0).float()
    i = 0
    while not done : 
        i += 1
        env.render()
        a = agent.step(s, a, r).action
        s_, r, done, info = env.step(a.item())
        s_ = torch.tensor(s_).float()
        r = torch.tensor(r).float()
        s = s_
        time.sleep(0.1)
    
    env.close()

if __name__ == "__main__" : 
    main ()
