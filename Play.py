# python3 Play.py <nEpisodes>
import os
import os.path as osp
import pickle
import keyboard
import gym
import sys
import time

def main () : 

    def moveLeft (e) :
        nonlocal action
        action = 0

    def moveRight (e) : 
        nonlocal action
        action = 0

    def save (tNum, obj) : 
        fileName = osp.join('./Trajectories', f'trajectory{tNum}')
        with open(fileName, 'wb') as fd : 
            pickle.dump(obj, fd)

    keyboard.on_press_key("left", moveLeft)
    keyboard.on_press_key("right", moveRight)

    episodes = int(sys.argv[1])

    env = gym.make('CartPole-v0')

    for i in range(episodes) : 

        env.reset()
        done = False
        action = 0
        trajectory = []

        while not done : 
            env.render()
            observation, reward, done, info = env.step(action)
            trajectory.append(action)
            time.sleep(0.1)

        save(i, trajectory)

if __name__ == "__main__" : 
    main ()
