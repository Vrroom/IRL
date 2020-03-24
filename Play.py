# python3 Play.py <nEpisodes>
import os
import pickle
import os.path as osp
import gym
import sys
import time
from PolicyGradient import PolicyNet
import PolicyGradient
import Agents

def main () : 

    env = gym.make('Acrobot-v1')

    observation = env.reset()
    done = False
    trajectory = []

    agent = Agents.REINFORCE('./Models/acrobotMimicer.pkl')
    # agent = Agents.Human(env)
    t = 0
    while not done : 
        env.render()
        action = agent(observation)
        observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward))
        time.sleep(0.1)
        t += 1
    print(t)
    fileName = osp.join('./Trajectories', 'acrobot-trajectory.pkl')
    with open(fileName, 'wb') as fd : 
        pickle.dump(trajectory, fd)

    env.close()

if __name__ == "__main__" : 
    main ()
