import pickle
import math
import torch
from functools import partial, reduce
from itertools import product
from sklearn.manifold import TSNE
import numpy as np
import gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Utils import *

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
        self.trajectory = self.trajectory[:200]

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
    criterion = nn.NLLLoss()
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

class PolicyNet (nn.Module) :
    
    def __init__ (self, dims) : 
        super(PolicyNet, self).__init__()
        layerDims = zip(dims[:-1], dims[1:-1])
        self.layers = nn.ModuleList([nn.Linear(a,b) for a, b in layerDims])
        self.last = nn.Linear(dims[-2], dims[-1])
        self.dropout = nn.Dropout(p=0.5)
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def forward (self, x) :
        x = reduce(lambda y, f : F.relu(self.dropout(f(y))), self.layers, x)
        x = self.last(x)
        return self.logSoftmax(x)

def getAction (model, s) : 
    pmf = model(toTensor(s))
    action = Categorical(pmf).sample()
    return action.item()

def getBestAction(model, s) :
    pmf = model(toTensor(s))
    return pmf.detach().numpy().argmax()

def train (env, gamma, lr, weight_decay, save=True) :

    def logStep () : 
        S.append(s)
        A.append(a)
        R.append(r)

    def computeReturns () :
        g = 0
        G = []
        for r in R[::-1] : 
            g = g * gamma + r
            G.insert(0, g)
        G = np.array(G)
        G = (G - G.mean()) / (G.std() + 1e-7)
        return G

    def gradientDescent () : 
        optimizer.zero_grad()
        G = computeReturns()
        loss = 0
        for s, a, g in zip(S, A, G) : 
            loss -= g * torch.log(model(toTensor(s))[a])
        total_norm = 0
        print('loss', loss)
        loss.backward()
        for p in model.parameters():
            if p.grad is not None : 
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print('Grad norm', total_norm)

        optimizer.step()

    model = PolicyNet(6, 2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    episode = 0
    
    while episode < 1000 :
        done = False
        s = env.reset()
        S, A, R = [], [], []

        while not done : 
            a = getAction(model, s)
            s, r, done, info = env.step(a)
            logStep()

        gradientDescent()

        episode += 1
        print(episode, sum(R))

    if save : 
        torch.save(model, './Models/acrobotREINFORCE.pkl')

def main () : 
    # env = gym.make('Acrobot-v1')
    # train(env, gamma=0.99, lr=1e-2, weight_decay=0)
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
    model = PolicyNet([6, 128, 3])
    loss = teachToMimic(model, './Trajectories/acrobot-trajectory.pkl', lr=8e-3, batch_size=16, weight_decay=1e-3)
    torch.save(model, './Models/acrobotMimicer.pkl')
    print(loss)


if __name__ == "__main__" :
    main() 
