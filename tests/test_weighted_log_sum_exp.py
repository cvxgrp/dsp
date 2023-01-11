import cvxpy as cp
import numpy as np
import pytest

from dsp import weighted_log_sum_exp
from dsp.cone_transforms import K_repr_x_Gy, LocalToGlob, minimax_to_min
from dsp.problem import MinimizeMaximize, SaddlePointProblem


@pytest.mark.parametrize(
    "x_val,y_val,n", [(1, 1, 2), (2, 2, 2), (0.5, 3, 5), (2, 6, 20), (6, 0.1, 1)]
)
def test_eval_weighted_log_sum_exp(x_val, y_val, n):
    x = cp.Variable(n)

    y = cp.Variable(n, nonneg=True)
    y_value = np.ones(n) * y_val
    x_value = np.ones(n) * x_val

    wlse = weighted_log_sum_exp(x, y)

    lgt = LocalToGlob([x], [y])
    K = wlse.get_K_repr(lgt)

    prob = cp.Problem(
        cp.Minimize(K.f @ y_value + K.t),
        [
            *K.constraints,
            x == x_value,
        ],
    )
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log(np.exp(x_value) @ y_value), atol=1e-4)


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (2, 1.23, 2), (3, 2, 3)])
def test_wlse_composition_switch(x_val, y_val, c):
    y = cp.Variable(name="y")
    x1 = cp.Variable(name="x1", nonneg=True)
    x2 = cp.Variable(name="x2", nonneg=True)

    wlse = weighted_log_sum_exp(cp.exp(y), x1 + x2)  # these variables are global scope

    obj = MinimizeMaximize(-wlse)

    x_constraints = [x1 + x2 <= x_val]
    y_constraints = [y >= y_val]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)
    opt_val = -np.log(x_val * np.exp(np.exp(y_val)))

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val)

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, opt_val)


def test_wlse_non_dsp():
    y = cp.Variable(name="y")
    x = cp.Variable(name="x", nonneg=True)

    with pytest.warns(UserWarning, match="Weights are non-positive."):
        assert weighted_log_sum_exp(x, y).is_dsp()
    with pytest.warns(UserWarning, match="Weights are non-positive."):
        assert not weighted_log_sum_exp(cp.log(x), y).is_dsp()
    assert not weighted_log_sum_exp(x, cp.exp(y)).is_dsp()


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (2.1, 0.5, 2), (1.1, 2, 3)])
def test_wlse_composition(x_val, y_val, c):
    y = cp.Variable(name="y", nonneg=True)
    x1 = cp.Variable(name="x1", nonneg=True)
    x2 = cp.Variable(name="x2", nonneg=True)

    f = cp.square(cp.exp(x1) + x2)
    f_val = (np.exp(x_val) + x_val) ** 2

    # f = cp.square(x1)
    # f_val = (x_val)**2

    # f = cp.exp(x1)
    # f_val = np.exp(x_val)

    wlse = weighted_log_sum_exp(f, y)

    obj = MinimizeMaximize(wlse)

    x_constraints = [x1 >= x_val, x2 >= x_val]
    y_constraints = [y <= y_val]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)
    assert set(prob.convex_variables()) == {x1, x2}
    assert set(prob.concave_variables()) == {y}
    opt_val = np.log(y_val * np.exp(f_val))

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, opt_val)


def test_eval_weighted_log_sum_exp_affine_sum():
    n = 1
    y_val = 1
    x_val = 1

    x = cp.Variable(n)
    y = cp.Variable(n + 1, nonneg=True)
    y_value = np.ones(n + 1) * y_val
    x_value = np.ones(n) * x_val

    a = np.array([2, 1])
    wlse = weighted_log_sum_exp(x, a @ y + 1)

    ltg = LocalToGlob([x], [y])

    K = wlse.get_K_repr(ltg)

    prob = cp.Problem(
        cp.Minimize(K.f @ y_value + K.t),
        [
            *K.constraints,
            x == x_value,
        ],
    )
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log(4 * np.exp(1)))

    x_val = 1
    X_constraints = [x >= x_val]
    y_val = 1.1
    Y_constraints = [y <= y_val]
    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints, [y], ltg))
    min_prob.solve(solver=cp.SCS)
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, np.log((sum(a) * y_val + 1) * np.exp(x_val)))


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (1, 1, 1), (0.31, 1.04, 3)])
def test_wlse_multi_var(x_val, y_val, c):
    x1 = cp.Variable(nonneg=True, name="x1")
    x2 = cp.Variable(name="x2")

    y = cp.Variable(3, nonneg=True, name="y")

    a = np.array([2, 1])
    wlse = weighted_log_sum_exp(x1, a @ y[:2] + c)

    obj = MinimizeMaximize(wlse + cp.exp(x2) + cp.log(y[2]))
    x_constraints = [x1 >= x_val, x2 >= x_val]
    y_constraints = [y <= y_val]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)

    opt_val = np.log((sum(a) * y_val + c) * np.exp(x_val)) + np.exp(x_val) + np.log(y_val)

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val, atol=1e-3)

    prob.y_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.y_prob.value, -opt_val)

    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, opt_val, atol=1e-3)
    assert np.allclose(y.value, y_val)


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (5, 5, 2), (3, 2, 3)])
def test_wlse_switching(x_val, y_val, c):
    y = cp.Variable(name="y")
    x1 = cp.Variable(name="x1", nonneg=True)
    x2 = cp.Variable(name="x2", nonneg=True)

    wlse = weighted_log_sum_exp(y, x1 + x2)  # these variables are global scope

    obj = MinimizeMaximize(-wlse)

    x_constraints = [x1 + x2 <= x_val]
    y_constraints = [y >= y_val]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)
    opt_val = -np.log(x_val * np.exp(y_val))

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val)

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, opt_val)


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (3.01, 1.4, 2), (3, 2, 3)])
def test_wlse_multi_var_switching(x_val, y_val, c):
    y1 = cp.Variable()
    y2 = cp.Variable()

    x = cp.Variable(3, nonneg=True)

    a = 2
    wlse = weighted_log_sum_exp(a * y1 + a * y2 + c, cp.sum(x[1:]))

    obj = MinimizeMaximize(-wlse + cp.exp(x[0]) + cp.log(y1))
    x_constraints = [x == x_val]
    y_constraints = [y1 == y_val, y2 == y_val]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)

    opt_val = -np.log(2 * x_val * np.exp(2 * a * y_val + c)) + np.exp(x_val) + np.log(y_val)

    prob.y_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.y_prob.value, -opt_val, atol=1e-2)

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val, atol=1e-3)

    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, opt_val, atol=1e-4)


@pytest.mark.parametrize("x_val,y_val,c", [(1, 1, 1)])
def test_neg_wlse(x_val, y_val, c):
    x1 = cp.Variable(nonneg=True, name="x1")
    # x2 = cp.Variable()

    # y = cp.Variable(3, nonneg=True)
    y1 = cp.Variable(nonneg=True, name="y1")

    # a = np.array([2, 1])
    # wlse = WeightedLogSumExp(x1, a@y[:2]+c)
    wlse = weighted_log_sum_exp(y1, x1)

    # obj = MinimizeMaximize(wlse + cp.exp(x2) + cp.log(y[2]))
    obj = MinimizeMaximize(-1 * wlse)
    x_constraints = [x_val >= x1]  # , x2 >= x_val]
    y_constraints = [y_val <= y1]

    prob = SaddlePointProblem(obj, x_constraints + y_constraints)
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, -np.log(x_val * np.exp(y_val)))


def test_weighted_log_sum_exp_with_switching():
    n = 2
    x = cp.Variable(n, name="x_outer", nonneg=True)
    y = cp.Variable(n, name="y_outer", nonneg=True)

    a = np.ones(n)

    wsle = weighted_log_sum_exp(a @ y + 1, a @ x + 1)

    x_val = 1
    y_val = 3

    X_constraints = [x <= x_val]
    Y_constraints = [y >= y_val]

    ltg = LocalToGlob([x], [y])

    K_switched = wsle.get_K_repr(ltg, switched=True)  # -log(x exp(y))
    min_prob = cp.Problem(*minimax_to_min(K_switched, X_constraints, Y_constraints, [y], ltg))

    min_prob.solve(solver=cp.SCS)
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, -np.log((n * x_val + 1) * np.exp(n * y_val + 1)))


@pytest.mark.parametrize("n", [1, 2, 3, 4, 20])
def test_weighted_sum_exp_with_switching(n):
    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y", nonneg=True)

    X_constraints = [x == 1]
    Y_constraints = [
        y == 1,
    ]

    # Using x_Gy instead of switched y_Fx
    ltg = LocalToGlob([x], [y])
    K_x_Gy = K_repr_x_Gy(-cp.exp(y), x, ltg)
    min_prob_x_Gy = cp.Problem(*minimax_to_min(K_x_Gy, X_constraints, Y_constraints, [y], ltg))
    min_prob_x_Gy.solve(solver=cp.SCS)
    assert min_prob_x_Gy.status == cp.OPTIMAL
    assert np.isclose(min_prob_x_Gy.value, -n * np.e)


@pytest.mark.parametrize("x_val,y_val", [(1, 2 + np.e + 0.01), (2, 13.01), (1.3, 11.1)])
def test_wlse_comp(x_val, y_val):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    with pytest.warns(UserWarning, match="are non-positive"):
        wlse = weighted_log_sum_exp(cp.square(x), cp.log(cp.log(y)))

    obj = MinimizeMaximize(wlse)

    constraints = [x >= x_val, y <= y_val]

    problem = SaddlePointProblem(obj, constraints)
    opt_val = np.log(np.log(np.log(y_val)) * np.exp(x_val**2))

    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, opt_val, atol=1e-4)


def test_wsle_with_external_affine_constraints():
    x = cp.Variable(4, name="x")
    y = cp.Variable(4, name="y", nonneg=True)
    z = cp.Variable(2, name="z")

    wlse = weighted_log_sum_exp(x, y)

    obj = MinimizeMaximize(wlse + z[0] + z[1])

    constraints = [x >= 1, y <= 2, y == np.ones((4, 2)) @ z]

    problem = SaddlePointProblem(obj, constraints)
    opt_val = np.log(8 * np.exp(1)) + 2

    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, opt_val, atol=1e-4)


def test_wsle_value():
    n = 2
    x = cp.Variable(n)
    y = cp.Variable(n)

    wlse = weighted_log_sum_exp(x, y)

    x.value = np.arange(n)
    y.value = np.arange(n)

    assert wlse.value == np.log(np.sum(y.value * np.exp(x.value)))
