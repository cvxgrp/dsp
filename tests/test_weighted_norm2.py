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
    assert norm.is_dsp()

    obj = dsp.MinimizeMaximize(norm)
    prob = dsp.SaddlePointProblem(obj, [y <= 2, x >= 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1

    assert prob.status == "optimal"
    assert np.isclose(prob.value, np.sqrt(y_expected @ x_expected**2))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_neg_weighted_norm2(n):
    x = cp.Variable(n, name="x", nonneg=True)
    y = cp.Variable(n, name="y", nonneg=True)

    neg_norm = -weighted_norm2(y, x)
    assert neg_norm.is_dsp()

    obj = dsp.MinimizeMaximize(neg_norm)
    prob = dsp.SaddlePointProblem(obj, [y >= 2, x <= 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1

    assert prob.status == "optimal"
    assert np.isclose(prob.value, -np.sqrt(x_expected @ y_expected**2))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_affine_precompositions(n):
    x = cp.Variable(n, name="x", nonneg=True)
    y = cp.Variable(n, name="y", nonneg=True)

    norm = weighted_norm2(2 * x + 1, 2 * y + 1)
    assert norm.is_dsp()

    obj = dsp.MinimizeMaximize(norm)
    prob = dsp.SaddlePointProblem(obj, [y <= 2, x >= 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1

    assert prob.status == "optimal"
    assert np.isclose(prob.value, np.sqrt((2 * y_expected + 1) @ (2 * x_expected + 1) ** 2))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_nonaffine_precompositions(n):

    x = cp.Variable(n, name="x", nonneg=True)
    y = cp.Variable(n, name="y", nonneg=True)

    with pytest.warns(UserWarning, match="Weights are non-positive."):
        norm = weighted_norm2(cp.exp(x), cp.log(y))
    assert norm.is_dsp()

    obj = dsp.MinimizeMaximize(norm)
    prob = dsp.SaddlePointProblem(obj, [y <= 2, x >= 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1

    assert prob.status == "optimal"
    assert np.isclose(prob.value, np.sqrt(np.log(y_expected) @ np.exp(x_expected) ** 2))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)
