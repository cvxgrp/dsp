import cvxpy as cp
import pytest

from dsp.problem import MinimizeMaximize


def test_float_obj():
    obj = MinimizeMaximize(0)
    assert isinstance(obj.expr, cp.Expression)


def test_value():
    obj = MinimizeMaximize(cp.Constant(5))
    assert obj.value == 5


def test_invalid_argument():
    with pytest.raises(TypeError):
        MinimizeMaximize("hello")
