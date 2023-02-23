import cvxpy as cp
import pytest

import dsp
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


def test_properties():
    x = cp.Variable()
    y = cp.Variable()
    f = dsp.saddle_inner(x, y)

    obj = MinimizeMaximize(f)
    assert obj.is_dsp()
    assert obj.value is None
    assert obj.variables() == [x, y]
    assert obj.parameters() == []
    assert obj.atoms() == [dsp.saddle_inner]

    x.value = 1
    y.value = 2
    assert obj.value == 2
