from Utils import *
from RewardFnSpace import *
from functional import compose
from functools import partial, reduce, lru_cache
from itertools import product
import numpy as np
import scipy.integrate
import gym

@lru_cache(maxsize=128)
def findTheta (sin, cos) :
    """
    Calculate theta in radians
    from sin and cosine values.

    The output is in the range 
    [-pi, pi]

    Parameters
    ----------
    sin : float
    cos : float
    """
    if sin == 1 and cos == 0 : 
        return np.pi / 2
    elif sin == -1 and cos == 0 : 
        return -np.pi / 2
    elif cos > 0 : 
        return np.arctan(sin / cos)
    elif sin > 0 : 
        return np.pi + np.arctan(sin / cos)
    else : 
        return -np.pi + np.arctan(sin / cos)
        
@lru_cache(maxsize=128)
def toInternalStateRep (s) : 
    """
    The acrobot environment maintains an
    internal representation of the state
    as a 4-tuple : 

        [theta1, theta2, dtheta1, dtheta2]

    where theta1 is the angle made by the 
    first link with the vertical and theta2
    is the angle of the second link with 
    respect to the vertical. dtheta1 and dtheta2
    are the angular velocities.

    This function converts the representation
    that is visible to the agent to the 
    internal representation.

    Parameters
    ----------
    s : array-like
        External representation of the state.
    """
    theta1 = findTheta(s[1], s[0])
    theta2 = findTheta(s[3], s[2])
    return (theta1, theta2, s[4], s[5])

def toExternalStateRep (s) : 
    """
    Inverse function of toInternalStateRep.

    Parameters
    ----------
    s : array-like
        Internal representation of the state.
    """
    cos1 = np.cos(s[0])
    sin1 = np.sin(s[0])

    cos2 = np.cos(s[1])
    sin2 = np.sin(s[1])

    return np.array([cos1, sin1, cos2, sin2, s[2], s[3]])

def bound(x, m, M):
    """
    Bound x between m and M.

    Parameters
    ----------
    x : float
        Value.
    m : float
        Lower bound.
    M : float
        Upper bound.
    """
    return min(max(x, m), M)

def wrap(x, m, M):
    """
    Rotate x to fit in range (m, M). 
    Useful while dealing with angles.

    Parameters
    ----------
    x : float
        Value.
    m : float
        Lower bound.
    M : float
        Upper bound.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x

def stepFunction (s, xRange, yRange) :
    """
    A step function in R^2. Given
    a rectangle A in R^2 :

    f(x, y) =  0  ; (x, y) \in A
               -1 ; else
               
    Parameters
    ----------
    s : np.ndarray
        4-D state vector of which
        we are interested in the first
        2 dimensions.
    xRange : tuple
    yRange : tuple
        Together, these ranges define a
        rectangle in R^2 over which the
        reward will be -1. Everywhere 
        else, reward will be 0. 
    """
    x, y = s[0], s[1]
    inRectangle = inRange(x, xRange) and inRange(y, yRange)
    return -1 if inRectangle else 0

def acrobotRewardBases (delX, delY) : 
    xs = np.arange(-np.pi, np.pi, delX)
    ys = np.arange(-np.pi, np.pi, delY)
    bases = []
    for x, y in product(xs, ys) : 
        x_, y_ = x + delX, y + delY
        fn = reduce(compose, 
            [partial(stepFunction, 
                xRange=(x, x_), 
                yRange=(y, y_)),
             toInternalStateRep,
             tuple])
        bases.append(Reward(fn, (-1, 0)))
    return bases
