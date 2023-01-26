import cvxpy as cp
import numpy as np
import pytest

from dsp.saddle_atoms import quasidef_quad_form, inner
from dsp.problem import MinimizeMaximize, SaddlePointProblem, 

def test_quasidef_quad_form():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    P = np.eye(2)
    Q = -np.eye(2)
    S = np.eye(2)

    f_explicit = cp.quad_form(x, P) - cp.quad_form(y, -Q) + 2 * inner(x, S@y)

    saddle_problem = SaddlePointProblem(MinimizeMaximize(f_explicit), [cp.sum(x) <= 1, cp.sum(y) <= 1, x >= 0, y >= 0])
    saddle_problem.solve()


