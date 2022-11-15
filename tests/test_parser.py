import cvxpy as cp
from dspp.problem import Parser


def test_sum():
    x = cp.Variable()
    y = cp.Variable()
    parser = Parser({x}, {y})
    expr = x + y
    parser.split_up_variables(expr)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}


def test_negative_sum():
    x = cp.Variable()
    y = cp.Variable()
    parser = Parser({x}, {y})
    expr = -(x + y)
    parser.split_up_variables(expr)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}

