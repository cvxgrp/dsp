import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import LocalVariable, saddle_inner, saddle_min


def test_robust_bond():

    C = np.loadtxt("tests/example_data/robust_bond_portfolio/C.csv")
    p = np.loadtxt("tests/example_data/robust_bond_portfolio/p.csv")
    w_bar = np.loadtxt("tests/example_data/robust_bond_portfolio/target_weights.csv")
    y_nominal = np.loadtxt("tests/example_data/robust_bond_portfolio/y_nominal.csv")

    # Constants and parameters
    n, T = C.shape
    delta_max, kappa, omega = 0.05, 0.5, 0.1
    B = 100
    V_limit = 80

    # Creating variables
    h = cp.Variable(n, nonneg=True)
    w = cp.multiply(h, p) / B

    delta = LocalVariable(T)
    y = y_nominal + delta

    # Objective
    phi = 0.5 * cp.norm1(w - w_bar)

    # Creating saddle min function
    V = 0
    for i in range(n):
        t_plus_1 = np.arange(T) + 1  # Account for zero-indexing
        V += saddle_inner(cp.exp(cp.multiply(-t_plus_1, y)), h[i] * C[i])

    Y = [
        cp.norm_inf(delta) <= delta_max,
        cp.norm1(delta) <= kappa,
        cp.norm2(delta[1:] - delta[:-1]) <= omega,
    ]

    V_wc = saddle_min(V, Y)

    # Creating and solving the problem
    problem = cp.Problem(cp.Minimize(phi), [cp.sum(w) == 1, V_wc >= V_limit])
    problem.solve()

    assert problem.status == cp.OPTIMAL


def test_arbitrage():

    p = np.ones(2)

    V = np.array([[1.5, 0.5], [0.5, 1.5]])

    x = cp.Variable(2, name="x")

    # No arbitrage
    obj = cp.Minimize(p @ x)
    constraints = [V @ x >= 0]
    no_arb_problem = cp.Problem(obj, constraints)
    no_arb_problem.solve()
    assert np.isclose(no_arb_problem.value, 0)

    # Arbitrage
    p[0] = 0.1  # arb possible for p[0] < 1/3
    obj = cp.Minimize(p @ x)
    constraints = [V @ x >= 0]
    arb_problem = cp.Problem(obj, constraints)
    arb_problem.solve()
    assert arb_problem.status == cp.UNBOUNDED

    # Robust arbitrage - small perturbation
    p_tilde = cp.Variable(2, name="p_tilde")
    obj = dsp.MinimizeMaximize(dsp.inner(x, p_tilde))
    constraints = [V @ x >= 0, cp.abs(p_tilde - p) <= 0.1]
    saddle_problem = dsp.SaddlePointProblem(obj, constraints)
    with pytest.raises(AssertionError):
        saddle_problem.solve()

    # Robust arbitrage - large perturbation
    constraints = [V @ x >= 0, cp.abs(p_tilde - p) <= 1]
    saddle_problem = dsp.SaddlePointProblem(obj, constraints)
    saddle_problem.solve()
    assert saddle_problem.status == cp.OPTIMAL
    assert np.isclose(saddle_problem.value, 0)
