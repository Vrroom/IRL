from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def plotFunction (f, xRange, yRange, xLabel, yLabel, zLabel) : 
    """
    Create a 3D plot of the function 
    over xRange x yRange.
    
    Examples
    --------
    >>> xRange = np.arange(-1, 1, 0.1)
    >>> yRange = np.arange(-1, 1, 0.1)
    >>> plotFunction(lambda x, y : x + y, xRange, yRange)

    Parameters
    ----------
    f : function
        Function from R^2 to R.
    xRange : array-like
    yRange : array-like
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
