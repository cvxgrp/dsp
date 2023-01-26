import cvxpy as cp
import numpy as np
import pytest

from dsp.problem import MinimizeMaximize, SaddlePointProblem
from dsp.saddle_atoms import inner, quasidef_quad_form


def test_quasidef_quad_form():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    P = np.eye(2)
    Q = -np.eye(2)
    S = np.eye(2)

    f_explicit = cp.quad_form(x, P) - cp.quad_form(y, -Q) + 2 * inner(x, S @ y)

    saddle_problem = SaddlePointProblem(
        MinimizeMaximize(f_explicit), [cp.sum(x) <= 1, cp.sum(y) <= 1, x >= 0, y >= 0]
    )
    saddle_problem.solve()

    ref_value = saddle_problem.value
    ref_x = x.value
    ref_y = y.value

    f = quasidef_quad_form(x, y, P, Q, S)
    saddle_problem = SaddlePointProblem(
        MinimizeMaximize(f), [cp.sum(x) <= 1, cp.sum(y) <= 1, x >= 0, y >= 0]
    )
    saddle_problem.solve()

    assert np.isclose(saddle_problem.value, ref_value, atol=1e-4)
    assert np.allclose(x.value, ref_x, atol=1e-4)
    assert np.allclose(y.value, ref_y, atol=1e-4)  # unclear why these arent exactly equal
