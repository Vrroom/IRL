"""
Convenience functions for plotting 2D scalar functions
and histograms.
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plotFunction (f, xRange, yRange, xLabel, yLabel, zLabel) : 
    """
    Create a 3D plot of the function 
    over xRange x yRange.
    """

    F = np.vectorize(f)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(xRange, yRange)
    Z = F(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.show()

def plotHist (samples, xRange, yRange, xLabel, yLabel, zLabel) : 
    """
    Visualize a distribution over two random variables.
    Stolen from :
        https://matplotlib.org/stable/gallery/mplot3d/hist3d.html
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x, y = samples[:, 0], samples[:, 1]
    xm, xM = xRange.min(), xRange.max()
    ym, yM = yRange.min(), yRange.max()
    bins = max(xRange.size, yRange.size)
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[xm, xM], [ym, yM]])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    plt.show()

