import cvxpy as cp
import numpy as np

import dsp
from dsp.atoms import weighted_norm2


def test_weighted_norm2():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    norm = weighted_norm2(x, y)

    obj = dsp.MinimizeMaximize(norm)
    prob = dsp.SaddlePointProblem(obj, [y <= 2, x >= 1])
    prob.solve()

    assert prob.status == "optimal"
    assert np.isclose(prob.value, 2.0)
    assert np.allclose(x.value, 1)
    assert np.allclose(y.value, 2)
