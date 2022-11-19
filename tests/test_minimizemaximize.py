import cvxpy as cp

from dspp.problem import MinimizeMaximize


def test_float_obj():
    obj = MinimizeMaximize(0)
    assert isinstance(obj.expr, cp.Expression)
