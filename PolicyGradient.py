import pickle
import math
import torch
from functools import partial, reduce
from itertools import product
from sklearn.manifold import TSNE
import torch.nn as nn
import numpy as np
import gym
import torch.optim as optim
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Utils import *
from Network import FeedForwardNetwork

class TrajectoryDataset (Dataset) :

    def __init__ (self, pickleFilePath, transform=None) : 
        """
        Obtain the human made (or otherwise)
        trajectory for the task.

        Parameters
        ----------
        pickleFilePath : str
            Path to the pickle file
            containing a list of (s, a, r)
            tuples.
        """
        with open(pickleFilePath, 'rb') as fd :
            self.trajectory = pickle.load(fd)

    def __len__ (self) :
        return len(self.trajectory)

    def __getitem__ (self, idx) : 
        s, a, _ = self.trajectory[idx]
        s = torch.from_numpy(s).float()
        a = torch.tensor(a, dtype=torch.long)
        return s, a

def teachToMimic (model, trajectoryFile, lr, weight_decay, batch_size, showPlot=False) : 
    """
    Regress model to a trajectory 
    which is a list :
        [(s_1, a_1, r_1),(s_2, a_2, r_2),...]

    Parameters
    ----------
    model : nn.Module
        Policy Network.
    trajectoryFile : str
        Path to pickle file containing trajectory.
    showPlot : bool
        Whether to show training curve.
    saveModel : bool
        Whether to save the model
    """
    epochs = list(range(100))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dataset = TrajectoryDataset(trajectoryFile)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in epochs: 
        for data in dataloader : 
            states, actions = data
            optimizer.zero_grad()

            outputs = model(states)
            loss = criterion(outputs, actions)
            loss.backward()

            optimizer.step()

        losses.append(loss.item())

    if showPlot : 
        plt.plot(epochs, losses)
        plt.show()

    return losses[-1]

def getAction (model, s) : 
    pmf = model(toTensor(s))
    action = Categorical(pmf).sample()
    return action.item()

def getBestAction(model, s) :
    pmf = model(toTensor(s))
    return pmf.detach().numpy().argmax()

def train (model, env, gamma, lr, weight_decay, save=True) :

    def logStep () : 
        S.append(s)
        A.append(a)
        R.append(r)

    def gradientDescent () : 
        optimizer.zero_grad()
        G = computeReturns()
        loss = 0
        for s, a, g in zip(S, A, G) : 
            loss -= g * torch.log(model(toTensor(s))[a])
        loss.backward()
        optimizer.step()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    episode = 0
    
    while episode < 1000 :
        done = False
        s = env.reset()
        S, A, R = [], [], []

        while not done : 
            a = getBestAction(model, s)
            s, r, done, info = env.step(a)
            logStep()

        # gradientDescent()
        episode += 1
        print(episode, sum(R))

def main () : 
    env = gym.make('MountainCar-v0')
    print(env.observation_space)
    dims = [ [2, 32, 3] ]
    lrs = [1e-1]
    batches = [128]
    wts = [0]
    for d, l, b, w in product(dims, lrs, batches, wts) :
        model = FeedForwardNetwork(d)
        loss = teachToMimic(model,  './Trajectories/mountainCar-trajectory.pkl', l, w, b)
        print(d, l, b, w, loss)
        torch.save(model, './Models/mountainCarMimicer.pkl')
    # model = torch.load('./Models/acrobotMimicer.pkl')
    # train(model, env, gamma=0.99, lr=0, weight_decay=0.)
    # with open( './Trajectories/acrobot-trajectory.pkl', 'rb') as fd :
    #     trajectory = pickle.load(fd)
    # trajectory = trajectory[100:]
    # obs = [t[0] for t in trajectory] 
    # actions = [t[1] for t in trajectory]

    # actions = np.array(actions)
    # obs = np.vstack(obs)
    # y = TSNE ().fit_transform(obs)
    # print(len(y))
    # plt.scatter(y[:, 0][actions == 1], y[:, 1][actions == 1], c='r')
    # plt.scatter(y[:, 0][actions == 0], y[:, 1][actions == 0], c='b')
    # plt.scatter(y[:, 0][actions == 2], y[:, 1][actions == 2], c='g')
    # plt.show()


if __name__ == "__main__" :
    main() 
