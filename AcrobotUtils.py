from Utils import *
from functools import partial, reduce
import numpy as np
import scipy.integrate
import gym

def findTheta (sin, cos) :
    """
    Calculate theta in radians
    from sin and cosine values.

    Parameters
    ----------
    sin : float
    cos : float
    """
    if sin == 1 and cos == 0 : 
        return np.pi / 2
    elif sin == -1 and cos == 0 : 
        return -np.pi / 2
    else :
        return np.arctan(sin / cos)

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
    return np.array([theta1, theta2, s[4], s[5]])

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

def sampleNextState (env, s, a) : 
    """
    The OpenAI Acrobot environment doesn't 
    let us sample states according to P(s, a)
    for a particular state s and action a. It
    always updates to the next state.

    To avoid this difficulty, I pulled out code
    from acrobot.py file in the gym repository
    so that we can do the next state computation
    without actually transitioning to it.

    Apply torque based on the given
    action and integrate to get the new position
    of the system.
    
    Examples 
    --------
    >>> env = gym.make('Acrobot-v1')
    >>> s = env.reset()
    >>> s1 = sampleNextState(env, s, 2)
    >>> s2, _, _, _ = env.step(2)
    >>> print(np.linalg.norm(s1 - s2))

    Parameters
    ----------
    env : object
        OpenAI gym environment.
    s : array-like
        State.
    a : int
        Action.
    """
    s = toInternalStateRep(s)
    torque = env.AVAIL_TORQUE[a]

    s_augmented = np.append(s, torque)

    dsdt = lambda t, y : env.env._dsdt(y, t)

    integrator = scipy.integrate.RK45(dsdt, 0, s_augmented, env.dt)
    while integrator.t < integrator.t_bound : 
        integrator.step()

    ns = integrator.y
    ns = ns[:4]  

    ns[0] = wrap(ns[0], -np.pi, np.pi)
    ns[1] = wrap(ns[1], -np.pi, np.pi)
    ns[2] = bound(ns[2], -env.MAX_VEL_1, env.MAX_VEL_1)
    ns[3] = bound(ns[3], -env.MAX_VEL_2, env.MAX_VEL_2)

    return toExternalStateRep(ns)

def stepFunction (s, xRange, yRange) :
    """
    A step function in R^2. Given
    a rectangle A in R^2 :

    f(x, y) =  0  ; (x, y) \in A
               -1 ; else
               
    Parameters
    ----------
    s : np.ndarray
        6-D state vector of which
        we are interested in the first
        2 dimensions.
    xRange : tuple
    yRange : tuple
        Together, these ranges define a
        rectangle in R^2 over which the
        reward will be 0. Everywhere 
        else, reward will be -1. 
    """
    x, y = s[0], s[1]
    inRectangle = inRange(x, xRange) and inRange(y, yRange)
    return 0 if inRectangle else -1

def acrobotRewardBases (delX, delY) : 
    """
    The reward bases are a collection of step
    functions which cover the interval
    [-1, 1] x [-1, 1] in R^2. 

    The reward bases look at the first two 
    components of the state. This the cos and 
    sin value of the angle made by the first
    link with the vertical.
    
    Intuitively, it seems that the angle
    of the first link is a good enough 
    indication of having reached the goal
    state.

    Parameters
    ----------
    delX : float
        The size of the rectangle
        along x-axis.
    delY : float
        The size of the rectangle
        along y-axis.
    """
    xs = np.arange(-1, 1, delX)
    ys = np.arange(-1, 1, delY)
    bases = []
    for x, y in product(xs, ys) : 
        x_, y_ = x + delX, y + delY
        fn = partial(stepFunction, xRange=(x, x_), yRange=(y, y_))
        bases.append(fn)
    return bases

if __name__ == "__main__" :
    env = gym.make('Acrobot-v1')
    s = env.reset()
    s1 = sampleNextState(env, s, 2)
    s2, _, _, _ = env.step(2)
    print(np.linalg.norm(s1 - s2))

