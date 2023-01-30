import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import MinimizeMaximize, SaddlePointProblem, weighted_log_sum_exp
from dsp.parser import DSPError, Parser


def test_sum():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    parser = Parser({x}, {y})
    expr = x + y
    parser.parse_expr_variables(expr, switched=False)

    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_negative_sum():
    x = cp.Variable()
    y = cp.Variable()
    parser = Parser({x}, {y})
    expr = -(x + y)
    parser.parse_expr_variables(expr, switched=False)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_mul_and_add():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    parser = Parser({x}, {y})
    wlse = weighted_log_sum_exp(x, y)

    expr = 1 + (-2) * (2 * (-wlse + 1))
    parser.parse_expr_variables(expr, switched=False)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_vars():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    z = cp.Variable(name="z")
    obj = dsp.MinimizeMaximize(weighted_log_sum_exp(x, y) + cp.exp(z))
    prob = SaddlePointProblem(obj, [y == 1, x == 1, z == 1])
    assert set(prob.convex_variables()) == {x, z}
    assert set(prob.concave_variables()) == {y}


def test_curvature_lumping():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    obj_ccv = MinimizeMaximize(cp.log(y) + x)
    prob = SaddlePointProblem(obj_ccv, [y == 1, x == 1])
    with pytest.raises(DSPError, match="specify"):
        prob.solve()

    prob = SaddlePointProblem(obj_ccv, [y == 1, x == 1], minimization_vars=[x])
    prob.solve()
    assert np.isclose(prob.x_prob.value, 1)


def test_just_affine():
    a = np.array([1, 2, 3])
    x = cp.Variable(3, name="x")
    y = cp.Variable(name="y")

    f = a @ x + cp.log(y) + cp.exp(x[0])
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [x >= 0, y >= 0])
    assert prob.is_dsp()

    f = np.ones(4) @ cp.hstack((x, y)) + cp.log(y) + cp.exp(x[0])
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [x >= 0, y >= 0])
    assert prob.is_dsp()  # Appears we can lump the variables despite different curvatures.

    f = cp.log(y) + cp.exp(x[0]) + np.ones(4) @ cp.hstack((x, y))
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [x >= 0, y >= 0])
    assert prob.is_dsp()  # Works independent of order.

    f = np.ones(4) @ cp.hstack((x, y))
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [x >= 0, y >= 0])
    assert not prob.is_dsp()  # Should fail if no implied curvatures.


@pytest.mark.parametrize("divisor", [-2, 2])
def test_divide(divisor):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    f = weighted_log_sum_exp(x, y) / divisor
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [y == 1, x == 1])
    assert prob.is_dsp()

    prob.solve()
    assert np.isclose(prob.value, 1 / divisor)


def test_overlapping_vars():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    f = x + y
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(
        obj, [y == 1, x == 1], minimization_vars=[x], maximization_vars=[x, y]
    )
    assert not prob.is_dsp()

def test_unknown_curvature():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    f = cp.square(cp.log(x))
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [y == 1, x == 1], minimization_vars=[x])
    assert not prob.is_dsp()