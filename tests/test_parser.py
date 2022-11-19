import cvxpy as cp

from dspp.atoms import weighted_log_sum_exp
from dspp.problem import Parser


def test_sum():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    parser = Parser({x}, {y})
    expr = x + y
    parser.parse_expr(expr, switched=False, repr_parse=False)

    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_negative_sum():
    x = cp.Variable()
    y = cp.Variable()
    parser = Parser({x}, {y})
    expr = -(x + y)
    parser.parse_expr(expr, switched=False, repr_parse=False)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_mul_and_add():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y", nonneg=True)
    parser = Parser({x}, {y})
    wlse = weighted_log_sum_exp(x, y)

    expr = 1 + (-2) * (2 * (-wlse + 1))
    parser.parse_expr(expr, switched=False, repr_parse=False)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}
