# python3 Play.py <nEpisodes>
import os
import pickle
import os.path as osp
import gym
import sys
import time
import PolicyGradient
import Agents

def main () : 

    env = gym.make('Acrobot-v1')

#    agent = Agents.REINFORCE('./Models/acrobotMimicer.pkl')
    agent = Agents.Human(env)
    for i in range(10) :
        observation = env.reset()
        done = False
        while not done : 
            env.render()
            action = agent(observation)
            action = 0
            observation, reward, done, info = env.step(action)
            time.sleep(0.1)

    # fileName = osp.join('./Trajectories', 'acrobot-trajectory.pkl')
    # with open(fileName, 'wb') as fd : 
    #     pickle.dump(trajectory, fd)

    env.close()

if __name__ == "__main__" : 
    main ()
