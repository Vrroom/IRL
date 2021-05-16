from RewardFnSpace import *
from PlotUtils import *
from AcrobotUtils import *
from A2C import *
from IRL import *

def test_l1norm () : 
    n = 10
    rfs = RewardFnSpace(list(range(n)))
    for i in range(10): 
        b = rfs.bs[i]
        rfs.lp += b == 0
    rfs.lp.solve()
    rfs._setCoeffs()
    coeffs = np.array(rfs.coeffs)
    assert(np.linalg.norm(coeffs - np.ones(n)) < 1e-4)

def test_acrobotbases (): 
    xRange = np.arange(-np.pi, np.pi, 0.1)
    yRange = np.arange(-np.pi, np.pi, 0.1)
    acrobotBases = acrobotRewardBases(np.pi, np.pi)
    toExternal = lambda x, y : toExternalStateRep([x, y, 0, 0])
    for basis in acrobotBases : 
        f = compose(basis, toExternal)
        plotFunction(f, xRange, yRange, 'theta1', 'theta2', 'R')

def test_optimalpolicyfinder () :
    def valNetwork (s) : 
        s = s.float()
        v = reduce(model.withReluDropout, model.v[:-1], s)
        v = model.v[-1](v)
        return v
    acrobotBases = acrobotRewardBases(np.pi, np.pi)
    for fn in acrobotBases : 
        policy = findOptimalPolicy(fn)
        model = policy.model
        toExternal = lambda x, y : toExternalStateRep([x, y, 0, 0])
        valFn = reduce(compose, [float, valNetwork, torch.tensor, toExternal])
        RFn = compose(fn, toExternal)
        xRange = np.arange(-np.pi, np.pi, 0.1)
        yRange = np.arange(-np.pi, np.pi, 0.1)
        plotFunction(RFn, xRange, yRange, 'theta1', 'theta2', 'R')
        plotFunction(valFn, xRange, yRange, 'theta1', 'theta2', 'V')

def test_piecewiselinearlp1 () :
    # maximize min(x, 2 * x) ; -10 <= x <= 10
    # expected answer: x = 10.
    lp = LpProblem('test', LpMaximize)
    x = LpVariable('x', -10, 10)
    t = LpVariable('t')
    lp += t
    lp += t <= x
    lp += t <= 2 * x
    lp.solve()
    optimum = value(lp.objective)
    assert(abs(optimum - 10) < 1e-3)

def test_piecewiselinearlp2 () : 
    # maximize (min(x, 2 * x) + min(3 * x, 4 * x)) 
    # -10 <= x <= 10
    # expected answer: x = 40
    lp = LpProblem('test', LpMaximize)
    x = LpVariable('x', -10, 10)
    t1 = LpVariable('t1')
    t2 = LpVariable('t2')
    lp += t1 + t2
    lp += t1 <= x
    lp += t1 <= 2 * x
    lp += t2 <= 3 * x
    lp += t2 <= 4 * x
    lp.solve()
    optimum = value(lp.objective)
    assert(abs(optimum - 40) < 1e-3)

if __name__ == "__main__" :
    test_piecewiselinearlp1()
    test_piecewiselinearlp2()
    # test_l1norm()
    # test_acrobotbases()
    # test_optimalpolicyfinder()
