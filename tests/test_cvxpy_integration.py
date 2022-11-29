import cvxpy as cp
import numpy as np
import pytest

from dspp.atoms import inner, saddle_max, saddle_min, weighted_log_sum_exp
from dspp.cvxpy_integration import extend_cone_canon_methods
from dspp.dummy import Dummy
from dspp.problem import MinimizeMaximize, SaddleProblem

extend_cone_canon_methods()


def test_semi_infinite_matrix():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    x_d = Dummy(2, name="x_d", nonneg=True)
    y_d = Dummy(2, name="y_d", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y)

    # Saddle problem
    saddle_obj = MinimizeMaximize(inner_expr)
    saddle_problem = SaddleProblem(saddle_obj, [cp.sum(x) == 1, cp.sum(y) == 1])
    saddle_problem.solve()

    assert np.isclose(saddle_problem.value, 2.0, atol=1e-4)
    assert np.allclose(x.value, [1, 0], atol=1e-4)
    assert np.allclose(y.value, [0, 1], atol=1e-4)

    # Saddle max problem

    inner_expr = inner(x, A @ y_d)

    obj = cp.Minimize(saddle_max(inner_expr, [y_d], [cp.sum(y_d) == 1]))
    constraints = [cp.sum(x) == 1]

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)

    assert np.isclose(problem.value, 2.0, atol=1e-4)

    # Saddle min problem

    inner_expr = inner(x_d, A @ y)

    f = saddle_min(inner_expr, [x_d], [cp.sum(x_d) == 1])
    obj = cp.Maximize(f)
    constraints = [cp.sum(y) == 1]

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 2.0, atol=1e-4)
    assert np.allclose(x.value, [1, 0], atol=1e-4)
    assert np.allclose(y.value, [0, 1], atol=1e-4)


def test_dcp_concave_max_and_dummy():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y)

    with pytest.raises(AssertionError, match="vars must be"):
        f_max = saddle_max(inner_expr, [y], [cp.sum(y) == 1])

    y2 = Dummy(2, name="y", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y2)

    f_max = saddle_max(inner_expr, [y2], [cp.sum(y2) == 1])

    obj = cp.Maximize(f_max)
    assert not obj.is_dcp()


def test_semi_infinite_expr():
    x = cp.Variable(2, name="x", nonneg=True)
    y = Dummy(2, name="y", nonneg=True)

    wlse = weighted_log_sum_exp(x, y)

    sup_y_f = saddle_max(2 * wlse + y[1] + cp.exp(x[1]), [y], [y <= 1])

    with pytest.raises(AssertionError, match="x must have a value"):
        sup_y_f.numeric(values=np.ones(1))

    with pytest.raises(AssertionError, match="x must have a value"):
        y.value

    x.value = np.ones(2)
    val = sup_y_f.numeric(values=np.ones(1))
    assert np.isclose(val, 2 * np.log(2 * np.e) + 1 + np.e)
