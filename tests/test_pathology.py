import cvxpy as cp
import numpy as np
import pytest

from dsp import MinimizeMaximize, inner, saddle_inner, saddle_max
from dsp.local import LocalVariable
from dsp.problem import SaddlePointProblem


def test_ubounded_domains_exp():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    with pytest.warns(UserWarning, match="Gy is non-positive"):
        f = -saddle_inner(cp.exp(x), cp.log(y))

    saddle_problem = SaddlePointProblem(
        MinimizeMaximize(f), [cp.sum(y) >= 4, x <= 1]
    )  # , x >= 1, cp.sum(x) == 2])

    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, 0.0, atol=1e-6)


def test_unbounded_domains_inv_pos():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # f_cvx = cp.inv_pos(x)
    # f_ccv = -cp.inv_pos(y)+2

    a = 1.2
    f_cvx = cp.power(x, -a)
    f_ccv = -cp.power(y, -a) + 2

    with pytest.warns(UserWarning, match="Gy is non-positive"):
        f = saddle_inner(f_cvx, f_ccv)

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f), [x >= 1, y >= 1])

    # saddle_problem.solve(solver=cp.MOSEK)
    # x.value, y.value

    y_local = LocalVariable(name="y")
    f_cvx = cp.inv_pos(x) + 0.00001
    f_ccv = cp.power(y_local, 0.01)
    f = saddle_inner(f_cvx, f_ccv)

    F = saddle_max(f, [y_local >= 1])
    prob = cp.Problem(cp.Minimize(F), [x >= 1])
    prob.solve()
    prob.value
    x.value, y.value

    # for x_val in [1, 1.1,1.3,1.5,2]:
    #     x.value = x_val
    #     print(f"{x_val=}, {F.value=}")

    # print()


def test_lagrangian():
    x = cp.Variable(name="x")
    nu = cp.Variable(name="y")

    f = cp.exp(x) + inner(x - 1, nu)

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f), [x >= -10, 0 * nu == 0])
    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, np.e)
