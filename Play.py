""" 
Run this to play an environment and record trajectories. 
Trajectories are saved in C.TRAJ_DIR as pickle files.
"""
import os
import os.path as osp
import pickle
import numpy as np
import gym
import sys
import time
from HumanAgent import Human
from Network import *
from collections import namedtuple
import Config as C

def main () : 
    env = gym.make(C.ENV)
    agent = Human(env)
    done = False
    trajectory = []
    s = env.reset()
    a = 0
    r = 0
    trajectory.append((s, a, r))
    while not done : 
        env.render()
        a = agent(s)
        s_, r, done, info = env.step(a)
        s = s_
        trajectory.append((s, a, r))
        time.sleep(0.1)
    env.close()
    # Save trajectory
    files = os.listdir(C.TRAJ_DIR)
    n = len(files)
    with open(osp.join(C.TRAJ_DIR, f'traj_{n}.pkl'), 'wb') as fd : 
        pickle.dump(trajectory, fd)
    
if __name__ == "__main__" : 
    main ()
