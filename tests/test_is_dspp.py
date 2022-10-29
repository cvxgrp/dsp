import dmcp
import numpy as np
import pytest

import cvxpy as cp
from dspp.nemirovski import minimax_to_min, KRepresentation, switch_convex_concave, \
    log_sum_exp_K_repr, K_repr_generalized_bilinear
from dspp.problem import SaddlePointProblem, MinimizeMaximize, SaddleProblem


# def test_matrix_game():
#     n = 2
#
#     A = np.eye(n)
#
#     x = cp.Variable(n, nonneg=True)
#     y = cp.Variable(n, nonneg=True)
#
#     obj = x @ A @ y
#
#     constraints = [
#         cp.sum(x) == 1,
#         cp.sum(y) == 1
#     ]
#
#     minimization_problem = cp.Problem(cp.Minimize(obj), constraints)
#     maximization_problem = cp.Problem(cp.Maximize(obj), constraints)
#
#     assert not minimization_problem.is_dcp()
#     assert not maximization_problem.is_dcp()
#
#     saddle_point_problem = SaddlePointProblem(obj, constraints, [x], [y])
#     assert saddle_point_problem.is_dspp()
#
#     saddle_point_problem.solve('DR', max_iters=50)
#     assert np.allclose(x.value, np.array([0.5, 0.5]))
#     assert np.allclose(y.value, np.array([0.5, 0.5]))


def test_matrix_game_nemirovski():
    n = 2

    x = cp.Variable(n, nonneg=True, name='x')
    y = cp.Variable(n, nonneg=True, name='y')

    X_constraints = [
        -1 <= x, x <= 1,
        # x == z,
    ]
    Y_constraints = [
        -1 <= y, y <= 1,
        # y == z
    ]

    K = K_repr_generalized_bilinear(x, y)
    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    print(min_prob.status, min_prob.value)
    for v in min_prob.variables():
        print(v, v.name(), v.value)


def test_matrix_game_nemirovski_new():
    n = 2

    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y')

    F_x = x + 0.5
    F_y = y + 0.5

    objective = MinimizeMaximize(F_x @ F_y)
    constraints = [
        -1 <= x, x <= 1,
        -1 <= y, y <= 1
    ]
    prob = SaddleProblem(objective, constraints)
    prob.solve()

    print(prob.status)
    for v in prob.variables():
        print(prob.value)
        print(v, v.name(), v.value)


def test_minimax_to_min():
    n = 2

    x = cp.Variable(n)
    xx = cp.Variable(n)
    y = cp.Variable(n)
    yy = cp.Variable(n)

    X_constraints = [
        cp.sum(xx) == 1,
        xx >= 0,
        x == xx
    ]

    Y_constraints = [
        cp.sum(yy) == 1,
        yy >= 0,
        y == yy
    ]

    f = cp.Variable(x.size)
    t = cp.Variable()
    u = cp.Variable()
    constraints = [
        f == x,
        u == 0,
        t == 0
    ]
    K = KRepresentation(
        f=f,
        t=t,
        u_or_Q=u,
        x=x,
        y=y,
        constraints=constraints
    )

    switched_K = switch_convex_concave(K)
    min_prob = cp.Problem(*minimax_to_min(switched_K, Y_constraints, X_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    # assert np.allclose(x.value, np.array([0.5, 0.5]))
    for v in min_prob.variables():
        print(v, v.value)
    # assert np.allclose(y.value, np.array([0.5, 0.5]))


def test_minimax_to_min_weighted_log_sum_exp():
    n = 2
    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y')

    K = log_sum_exp_K_repr(x, y)

    X_constraints = [
        x == 1
    ]
    Y_constraints = [
        y == 0.5
    ]

    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL


@pytest.mark.parametrize('x_val,y_val,n',
                         [(1, 1, 2), (2, 2, 2), (0.5, 3, 5), (2, 6, 20), (6, .1, 1)])
def test_eval_weighted_log_sum_exp(x_val, y_val, n):
    x = cp.Variable(n)

    y = cp.Variable(n)
    y_value = np.ones(n) * y_val
    x_value = np.ones(n) * x_val

    K = log_sum_exp_K_repr(x, y)

    prob = cp.Problem(cp.Minimize(K.f @ y_value + K.t),
                      [
                          *K.constraints,
                          x == x_value,
                      ])
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log(np.exp(x_value) @ y_value))


def test_weighted_log_sum_exp_with_switching():
    n = 2
    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y')

    K = log_sum_exp_K_repr(x, y)

    X_constraints = [
        x == 1
    ]
    Y_constraints = [
        y == 1
    ]

    K_switched = switch_convex_concave(K)
    min_prob = cp.Problem(*minimax_to_min(K_switched, Y_constraints, X_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, -np.log(n * np.e))


@pytest.mark.parametrize('n', [1, 2, 3, 4, 20])
def test_weighted_sum_exp(n):
    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y', nonneg=True)

    F = cp.exp(x)

    K = K_repr_generalized_bilinear(F, y)

    X_constraints = [
        x == 1
    ]
    Y_constraints = [
        y == 1
    ]

    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, n * np.e)


def test_weighted_sum_exp_with_switching():
    n = 2
    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y', nonneg=True)

    F = cp.exp(x)

    K = K_repr_generalized_bilinear(F, y)

    X_constraints = [
        x == 1
    ]
    Y_constraints = [
        y == 1,
    ]

    K = switch_convex_concave(K)
    min_prob = cp.Problem(*minimax_to_min(K, Y_constraints, X_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, -n * np.e)


# def test_scalar_bilinear():
#     x = cp.Variable()
#     y = cp.Variable()
#
#     obj = x * y
#
#     constraints = [
#         -1 <= x, x <= 0.5,
#         -1 <= y, y <= 2
#     ]
#
#     minimization_problem = cp.Problem(cp.Minimize(obj), constraints)
#     maximization_problem = cp.Problem(cp.Maximize(obj), constraints)
#
#     assert not minimization_problem.is_dcp()
#     assert not maximization_problem.is_dcp()
#
#     saddle_point_problem = SaddlePointProblem(obj, constraints, [x], [y])
#     assert saddle_point_problem.is_dspp()
#
#     saddle_point_problem.solve('DR', max_iters=200, eps=1e-6)
#     assert np.isclose(x.value, 0, atol=1e-5)
#     assert np.isclose(y.value, 0, atol=1e-5)


def test_robust_ols():
    np.random.seed(0)
    N = 100
    alpha = 0.05
    k = int(alpha * N)
    theta = np.array([0.5, 2])
    eps = np.random.normal(0, 0.05, N)
    A = np.concatenate([np.ones((N, 1)), np.random.normal(0, 1, (N, 1))], axis=1)
    observations = A @ theta + eps
    largest_inds = np.argpartition(observations, -k)[-k:]
    observations[largest_inds] -= 10

    wgts = cp.Variable(N, nonneg=True)
    theta_hat = cp.Variable(2)

    loss = cp.square(observations - A @ theta_hat)

    # OLS
    cp.Problem(cp.Minimize(cp.sum(loss))).solve()
    ols_vals = theta_hat.value

    # DMCP approach
    constraints_dmcp = [cp.sum(wgts) == N - k, wgts <= 1]
    prob_dmcp = cp.Problem(cp.Minimize(cp.sum(cp.multiply(wgts, loss))), constraints_dmcp)
    assert dmcp.is_dmcp(prob_dmcp)
    prob_dmcp.solve(method='bcd')
    cooperative_vals = theta_hat.value

    # DSPP approach
    X_constraints = []
    Y_constraints = [cp.sum(wgts) == N, wgts <= 2, wgts >= 0.5]

    K = K_repr_generalized_bilinear(loss, wgts)
    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    adversarial_vals = theta_hat.value

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(A[:, 1], observations)
    # plt.plot(A[:, 1], A @ adversarial_vals, label='Adversarial', c='red')
    # plt.plot(A[:, 1], A @ cooperative_vals, label='Cooperative', c='green')
    # plt.plot(A[:, 1], A @ ols_vals, label='OLS', color='orange')
    # plt.legend()
    # plt.show()
