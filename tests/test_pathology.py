import cvxpy as cp
import numpy as np
from dsp import MinimizeMaximize, saddle_inner
from dsp.problem import SaddlePointProblem


def test_ubounded_domains_exp():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    f = -saddle_inner(cp.exp(x), cp.log(y))

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f), [cp.sum(y) >= 4, x<=1]) #, x >= 1, cp.sum(x) == 2])

    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, 0.0)

def test_unbounded_domains_inv_pos():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # f_cvx = cp.inv_pos(x)
    # f_ccv = -cp.inv_pos(y)+2

    a = 4
    f_cvx = cp.power(x, -a)
    f_ccv = -cp.power(y,-a) + 2

    f = saddle_inner(10*f_cvx, 10*f_ccv)

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f), [x >= 1, y >= 1])

    saddle_problem.solve(solver=cp.MOSEK)
    x.value, y.value

    # assert np.isclose(saddle_problem.value, 0.0)