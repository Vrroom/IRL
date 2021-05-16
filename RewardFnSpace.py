from pulp import *
import Config as C
import numpy as np

class RewardFnSpace () : 

    def __init__ (self, rewardBases) : 
        self.rewardBases = rewardBases
        self.rng = np.random.RandomState(0)
        self._initializeLP ()

    def _initializeLP (self) : 
        self.lp = LpProblem('IRL', LpMaximize)
        self.y1s, self.y2s, self.alphas, self.bs = [], [], [], []
        for i, _ in enumerate(self.rewardBases):  
            y1 = LpVariable(f'y1{i}')
            y2 = LpVariable(f'y2{i}')
            b  = LpVariable(f'b{i}')
            self.y1s.append(y1)
            self.y2s.append(y2)
            self.bs.append(b)
            self.alphas.append(1 - (y1 - y2))
        for y1 in self.y1s : 
            self.lp += y1 >= 0 
        for y2 in self.y2s : 
            self.lp += y2 >= 0
        for alpha in self.alphas : 
            self.lp += alpha >= 0
            self.lp += alpha <= 1
        l1NormExprs = [-C.L1_NORM_GAMMA * (y1 + y2) for y1, y2 in zip(self.y1s, self.y2s)]
        self.lp += lpSum(self.bs) + lpSum(l1NormExprs)
        self.coeffs = [self.rng.rand() for _ in self.rewardBases]

    def _estimatedValueExpressions (self, stateValuesForBases) :
        svfb = np.array(stateValuesForBases)
        alphas = np.array(self.alphas)
        estimates = (svfb.T @ alphas).tolist()
        return estimates

    def _setCoeffs (self) : 
        y1s = [y.varValue for y in self.y1s]
        y2s = [y.varValue for y in self.y2s]
        self.coeffs = [1 - (y1 - y2) for y1, y2 in zip(y1s, y2s)]

    def current (self) : 
        return lambda s : sum(
            [c * rfn(s) 
             for c, rfn 
             in zip(self.coeffs, self.rewardBases)])

    def refine (self, expertValues, inferiorEstimates) :
        expertEstimates = self._estimatedValueExpressions(expertValues)
        for i, (exp, inf) in enumerate(
                zip(expertEstimates, inferiorEstimates)) : 
            b = self.bs[i]
            self.lp += b <= 2 * (exp - inf)
            self.lp += b <= (exp - inf)
        self.lp.solve()
        self._setCoeffs()

class Reward () :
    def __init__ (self, rewardFn, reward_range): 
        self.rewardFn = rewardFn
        self.reward_range = reward_range

    def __call__ (self, s) :
        r = self.rewardFn(s)
        lo, hi = self.reward_range
        assert(lo <= r <= hi)
        return r

