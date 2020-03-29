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

    env = gym.make('MountainCar-v0')

    agent = Agents.REINFORCE('./Models/mountainCarMimicer.pkl')
    s = env.reset()
    #agent = Agents.Human(env)
    done = False
    trajectory = []
    i = 0
    while not done : 
        i += 1
        env.render()
        a = agent(s)
        s_, r, done, info = env.step(a)
        trajectory.append((s, a, r))
        s = s_
        time.sleep(0.1)
    
    print(i)
#     fileName = osp.join('./Trajectories', 'mountainCar-trajectory.pkl')
#     with open(fileName, 'wb') as fd : 
#         pickle.dump(trajectory, fd)

    env.close()

if __name__ == "__main__" : 
    main ()
