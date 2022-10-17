import cvxpy as cp
import numpy as np

from dspp.problem import SaddlePointProblem


def test_matrix_game():
    n = 2

    A = np.eye(2)

    x = cp.Variable(n, nonneg=True)
    y = cp.Variable(n, nonneg=True)

    obj = x @ A @ y

    constraints = [
        cp.sum(x) == 1,
        cp.sum(y) == 1
    ]

    minimization_problem = cp.Problem(cp.Minimize(obj), constraints)
    maximization_problem = cp.Problem(cp.Maximize(obj), constraints)

    assert not minimization_problem.is_dcp()
    assert not maximization_problem.is_dcp()

    saddle_point_problem = SaddlePointProblem(obj, constraints, [x], [y])
    assert saddle_point_problem.is_dspp()

    saddle_point_problem.solve('DR', max_iters=100)
    assert np.allclose(x.value, np.array([0.5, 0.5]))
    assert np.allclose(y.value, np.array([0.5, 0.5]))


def test_scalar_bilinear():
    x = cp.Variable()
    y = cp.Variable()

    obj = x * y

    constraints = [
        -1 <= x, x <= 1,
        -1 <= y, y <= 1
    ]

    minimization_problem = cp.Problem(cp.Minimize(obj), constraints)
    maximization_problem = cp.Problem(cp.Maximize(obj), constraints)

    assert not minimization_problem.is_dcp()
    assert not maximization_problem.is_dcp()

    saddle_point_problem = SaddlePointProblem(obj, constraints, [x], [y])
    assert saddle_point_problem.is_dspp()

    saddle_point_problem.solve('DR', max_iters=100, eps=1e-6)
    assert np.isclose(x.value, 0, atol=1e-5)
    assert np.isclose(y.value, 0, atol=1e-5)
