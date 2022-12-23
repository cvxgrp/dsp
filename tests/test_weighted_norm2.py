import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp.atoms import weighted_norm2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_weighted_norm2(n):
    x = cp.Variable(n, name="x", nonneg=True)
    y = cp.Variable(n, name="y", nonneg=True)

    norm = weighted_norm2(x, y)

    obj = dsp.MinimizeMaximize(norm)
    prob = dsp.SaddlePointProblem(obj, [y <= 2, x >= 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1

    assert prob.status == "optimal"
    assert np.isclose(prob.value, np.sqrt(y_expected @ x_expected**2))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)
