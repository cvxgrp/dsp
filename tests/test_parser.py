import cvxpy as cp
from dspp.atoms import WeightedLogSumExp
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

def test_mul_and_add():
    x = cp.Variable(name = 'x')
    y = cp.Variable(name = 'y', nonneg = True)
    parser = Parser({x}, {y})
    wlse = WeightedLogSumExp(x, y)

    obj = 1 + (-2) * (2 * (-wlse + 1))
    parser.split_up_variables(obj)
    assert parser.convex_vars == {x}
    assert parser.concave_vars == {y}
