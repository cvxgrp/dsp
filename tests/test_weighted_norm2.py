import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import weighted_norm2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_weighted_norm2(n):
    x = cp.Variable(n, name="x", nonneg=True)
    y = cp.Variable(n, name="y", nonneg=True)

    norm = weighted_norm2(x, y)
    assert norm.value is None
    assert not norm.is_decr(0)
    assert norm.is_dsp()

    obj = dsp.MinimizeMaximize(norm)
    constraints = [y <= 2, x >= 1]
    prob = dsp.SaddlePointProblem(obj, constraints)
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n) * 1
    expected_prob_val = np.sqrt(y_expected @ x_expected**2)

    assert prob.status == "optimal"
    assert np.isclose(prob.value, expected_prob_val)
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected)

    y.value = y_expected
    min_val = cp.Problem(cp.Minimize(norm.get_convex_expression()), constraints).solve()

    x.value = x_expected
    max_val = cp.Problem(cp.Maximize(norm.get_concave_expression()), constraints).solve()

    assert np.isclose(min_val, expected_prob_val)
    assert np.isclose(max_val, expected_prob_val)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_neg_weighted_norm2(n: int) -> None:
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
def test_affine_precompositions(n: int) -> None:
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


@pytest.mark.parametrize("x_val", [-1, 0.1, 1])
@pytest.mark.parametrize(
    "x_expr", [lambda x: cp.square(x), lambda x: cp.square(x) - 1, lambda x: cp.exp(x)]
)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_nonaffine_precompositions(x_val, x_expr, n: int) -> None:

    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y")

    cvx_expr = x_expr(x)

    with pytest.warns(UserWarning, match="Weights are non-positive."):
        norm = weighted_norm2(cvx_expr, y)

    if cvx_expr.is_affine() or cvx_expr.is_nonneg():
        assert norm.is_dsp()

        obj = dsp.MinimizeMaximize(norm)

        x_range_constr = x >= x_val if x_val > 0 or cvx_expr.is_incr(0) else x <= x_val
        prob = dsp.SaddlePointProblem(obj, [y <= 2, x_range_constr])
        prob.solve()

        y_expected = np.ones(n) * 2
        x_expected = np.ones(n) * x_val

        assert prob.status == "optimal"
        assert np.isclose(prob.value, np.sqrt(y_expected @ (x_expr(x_expected).value ** 2)))
        assert np.allclose(x.value, x_expected)
        assert np.allclose(y.value, y_expected, atol=1e-5)
    else:
        assert not norm.is_dsp()


def test_concave_composition() -> None:
    n = 2
    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y", nonneg=True)

    Gy = cp.log1p(y)
    norm = weighted_norm2(x, Gy)

    obj = dsp.MinimizeMaximize(norm)

    prob = dsp.SaddlePointProblem(obj, [y <= 2, x == 1])
    prob.solve()

    y_expected = np.ones(n) * 2
    x_expected = np.ones(n)

    assert prob.status == "optimal"
    assert np.isclose(prob.value, np.sqrt(np.log(1 + y_expected) @ (x_expected**2)))
    assert np.allclose(x.value, x_expected)
    assert np.allclose(y.value, y_expected, atol=1e-5)


def test_value() -> None:
    n = 2
    x = cp.Variable(n)
    y = cp.Variable(n)

    with pytest.warns(UserWarning, match="Weights are non-positive"):
        norm2 = weighted_norm2(x, y)

    assert norm2.value is None
    assert norm2.is_nonneg()
    assert not norm2.is_incr(0)
    assert not norm2.is_incr(1)

    x.value = np.arange(n)
    y.value = np.arange(n)

    assert norm2.value == np.sqrt(y.value @ np.square(x.value))
