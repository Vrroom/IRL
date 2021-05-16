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

if __name__ == "__main__" :
    test_l1norm()
    test_acrobotbases()
    test_optimalpolicyfinder()
