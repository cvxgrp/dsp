import cvxpy as cp
import numpy as np
import pytest

from dsp import saddle_quad_form
from dsp.problem import MinimizeMaximize, SaddlePointProblem


@pytest.mark.parametrize(
    "x_val,P_val", [([1, 0], [[1, 0], [0, 1]]), ([1, 1], [[1, 0.2], [0.2, 1]])]
)
def test_quad_form_equality(x_val, P_val):
    x = cp.Variable(2, name="x")
    P = cp.Variable((2, 2), name="P", PSD=True)

    x_val = np.array(x_val)
    P_val = np.array(P_val)

    f = saddle_quad_form(x, P)

    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [P == P_val, x == x_val])
    prob.solve()

    assert np.isclose(prob.value, x_val.T @ P_val @ x_val)


def test_quad_form_inequality():
    n = 2
    x = cp.Variable(n, name="x")
    P = cp.Variable((n, n), name="P", PSD=True)

    f = saddle_quad_form(x, P)

    obj = MinimizeMaximize(f)
    P_val = np.eye(n)
    prob = SaddlePointProblem(obj, [P << P_val, cp.sum(x) == 1, x >= 0])
    prob.solve()

    p_val = P.value
    x_val = x.value

    # validate
    P_obj = x_val.T @ P @ x_val
    P_prob = cp.Problem(cp.Maximize(P_obj), [P << P_val])
    P_prob.solve()

    x_obj = cp.quad_form(x, p_val)
    x_prob = cp.Problem(cp.Minimize(x_obj), [cp.sum(x) == 1, x >= 0])
    x_prob.solve()

    assert np.isclose(P_prob.value, x_prob.value)


def test_value():
    n = 2
    x = cp.Variable(n)
    P = cp.Variable((n, n), PSD=True)

    saddle_quad = saddle_quad_form(x, P)
    x.value = np.arange(n)
    P.value = np.eye(n)

    assert saddle_quad.value == x.value.T @ P.value @ x.value
