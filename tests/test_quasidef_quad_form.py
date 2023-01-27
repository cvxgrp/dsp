import cvxpy as cp
import numpy as np

from dsp.problem import MinimizeMaximize, SaddlePointProblem
from dsp.saddle_atoms import inner, quasidef_quad_form


def test_quasidef_quad_form():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    P = np.eye(2)
    Q = -np.eye(2)
    S = np.eye(2)

    f_explicit = cp.quad_form(x, P) - cp.quad_form(y, -Q) + 2 * inner(x, S @ y)

    constraints = [cp.sum(x) <= 1, cp.sum(y) <= 1, x >= 0, y >= 0]

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f_explicit), constraints)
    saddle_problem.solve()

    ref_value = saddle_problem.value
    ref_x = x.value
    ref_y = y.value

    f = quasidef_quad_form(x, y, P, Q, S)
    saddle_problem = SaddlePointProblem(MinimizeMaximize(f), constraints)
    saddle_problem.solve()

    assert np.isclose(saddle_problem.value, ref_value, atol=1e-4)
    assert np.allclose(x.value, ref_x, atol=1e-4)
    assert np.allclose(y.value, ref_y, atol=1e-4)  # unclear why these arent exactly equal
    assert np.isclose(f.value, ref_value, atol=1e-4)
    assert f.name().startswith("quasidef_quad_form(x, y")

    x.value = ref_x
    y.value = ref_y

    min_val = cp.Problem(cp.Minimize(f.get_convex_expression()), constraints).solve()
    max_val = cp.Problem(cp.Maximize(f.get_concave_expression()), constraints).solve()

    assert np.isclose(min_val, ref_value, atol=1e-4)
    assert np.isclose(max_val, ref_value, atol=1e-4)
