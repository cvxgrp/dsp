import cvxpy as cp
import numpy as np
import pytest

from dsp import MinimizeMaximize, SaddlePointProblem
from dsp.atoms import weighted_log_sum_exp

# from dsp.problem import Parser
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
    obj = weighted_log_sum_exp(x, y) + cp.exp(z)
    print()


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
