""" Space of reward functions """
from pulp import *
import Config as C
from Reward import Reward
from more_itertools import unzip
import numpy as np
import logging
logging.basicConfig(filename='irl.log', level=logging.INFO)

class RewardFnSpace () : 
    """
    This class exposes two methods: current and refine.
    current : return the current best reward function
    that explains the observed policy.
    refine  : shrink the space of candidate reward 
    functions that can explain the observed policy.
    """
    def __init__ (self, rewardBases) : 
        self.rewardBases = rewardBases
        self.rng = np.random.RandomState(C.SEED)
        self._initializeLP ()

    def _initializeLP (self) : 
        """
        Initialize the Linear Program by creating 
        variables and setting constraints on them.

        The objective of the LP is split into two parts. 
        The first one is the Value Function objective 
        and the second is the Sparsity objective.

        The Value Function objective asserts that the 
        valid reward functions are those for which
        the value estimates for the observed policy is
        greater than the value estimates of any other policy.

        The Sparsity objective asserts that most of the alpha
        values should be 1. This is because in most places in 
        the state space the agents accrue a -1 reward. Only
        in a sparse set of states, is the reward 0.

        alphas are the coefficients of the rewardBases
        y1s, y2s and bs are auxillary variables.
        y1s and y2s are used for the Sparsity objective 
            = -gamma * ||alpha - 1||_1

        y1 and y2 are defined as: 
            y1 = max(alpha - 1, 0) (>= 0)
            y2 = max(1 - alpha, 0) (>= 0)
        
        Hence,
            alpha = 1 - (y1 - y2) and 
            ||alpha - 1||_1 = y1 + y2
        ensuring that the Sparsity objective is linear 
        in y1 and y2.

        The Value Function objective is discussed 
        in the refine method.
        """
        self.lp = LpProblem('IRL', LpMaximize)
        self.y1s, self.y2s, self.alphas, self.bs = [], [], [], []
        for i, _ in enumerate(self.rewardBases):  
            y1 = LpVariable(f'y1_{i}')
            y2 = LpVariable(f'y2_{i}')
            self.y1s.append(y1)
            self.y2s.append(y2)
            self.alphas.append(1 - (y1 - y2))
        for y1 in self.y1s : 
            self.lp += y1 >= 0 
        for y2 in self.y2s : 
            self.lp += y2 >= 0
        for alpha in self.alphas : 
            self.lp += alpha >= 0
            self.lp += alpha <= 1
        self.l1Term = lpSum([-C.L1_NORM_GAMMA * (y1 + y2) 
                             for y1, y2 in zip(self.y1s, self.y2s)])
        self.lp += self.l1Term
        self.coeffs = [self.rng.rand() for _ in self.rewardBases]

    def _estimatedValueExpressions (self, stateValuesForBases) :
        svfb = np.array(stateValuesForBases)
        alphas = np.array(self.alphas)
        estimates = (svfb.T * alphas).sum(axis=1).tolist()
        return estimates

    def _setCoeffs (self) : 
        self.coeffs = [value(a) for a in self.alphas]

    def current (self) : 
        """
        Obtain the reward function which currently maximizes
        the objective = Value Function objective + Sparsity objective.
        """
        pairs = list(zip(self.coeffs, self.rewardBases))
        fn = lambda s : sum([c * rfn(s) for c, rfn in pairs])
        ranges = [rfn.reward_range for rfn in self.rewardBases]
        mins, maxs = list(map(list, unzip(ranges)))
        rMin = min(c * m for c, m in zip(self.coeffs, mins))
        rMax = max(c * M for c, M in zip(self.coeffs, maxs))
        return Reward(fn, (rMin, rMax))

    def refine (self, expertValues, inferiorValues) :
        """
        In this function, we shrink the space of 
        candidate functions using the inferiorValues 
        for the new policy.

        expertValues and inferiorValues are value estimates
        for each state, each fn in the reward bases.
            
        expertValues[i, j] = value estimate for a sampled
        state i while following the expert trajectory on an
        MDP with the jth reward basis. Similarly for inferiorValues.

        Since any reward function in the family is a linear
        combination of these basis, 

        expertEstimates[i]   = \sum_k alpha_k * expertValues  [i, k]
        inferiorEstimates[i] = \sum_k alpha_k * inferiorValues[i, k]

        In terms, of these, the Value Function objective is:

        Let x_i = expertEstimate[i] - inferiorEstimates[i],
            \sum_i min(x_i, 2 * x_i)

        Since we can't solve this objective exactly, we penalize those
        cases where the inferiorEstimate is greater than expertEstimate.
        
        b_i is an auxillary variable defined as min(x_i, 2 * x_i)
        with the constrains that: 
            b_i <= x_i
            b_i <= 2 * x_i

        Note that for a fixed x_i, the maximum value of b_i
        is min(x_i, 2 * x_i) as desired.
        """
        n = len(self.bs)
        expertEstimates = self._estimatedValueExpressions(expertValues)
        inferiorEstimates = self._estimatedValueExpressions(inferiorValues)
        for i, (exp, inf) in enumerate(
                zip(expertEstimates, inferiorEstimates)) : 
            b = LpVariable(f'b_{n + i}')
            self.lp += b <= 2 * (exp - inf)
            self.lp += b <=     (exp - inf)
            self.bs.append(b)
        self.lp += lpSum(self.bs) + self.l1Term
        self.lp.solve()
        self._setCoeffs()
