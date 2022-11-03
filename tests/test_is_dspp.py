import dmcp
import numpy as np
import pytest

import cvxpy as cp
from dspp.nemirovski import minimax_to_min, KRepresentation, switch_convex_concave, \
    log_sum_exp_K_repr, K_repr_y_Fx, K_repr_x_Gy, K_repr_ax, K_repr_by, add_cone_constraints, \
    get_cone_repr, SwitchableKRepresentation
from dspp.problem import MinimizeMaximize, SaddleProblem, AffineVariableError


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


def test_matrix_game_y_Fx():
    n = 2

    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y')
    xx = cp.Variable(n, name='xx')

    X_constraints = [
        -1 <= xx, xx <= 1,
        xx <= x, xx >= x,
    ]
    Y_constraints = [
        -1 <= y, y <= 1,
    ]

    F = x + 0.5

    K = K_repr_y_Fx(F, y)
    prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, 0)


def test_matrix_game_x_Gy():
    n = 2

    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y')
    yy = cp.Variable(n, name='yy')

    X_constraints = [
        -1 <= x, x <= 1,
    ]
    Y_constraints = [
        -1 <= yy, yy <= 1,
        yy <= y, y <= yy
    ]

    F = y + 0.5

    K = K_repr_x_Gy(F, x)
    prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    prob.solve()
    assert np.isclose(prob.value, 0)
    print(prob.status, prob.value)
    for v in prob.variables():
        print(v, v.name(), v.value)


@pytest.mark.parametrize('a,expected', [(cp.exp, np.exp(2)), (cp.square, 4)])
def test_ax(a, expected):
    x = cp.Variable()
    y = cp.Variable()
    ax = a(x)

    X_const = [2 <= x, x <= 4]
    Y_const = [y == 0]

    K = K_repr_ax(ax)
    prob = cp.Problem(*minimax_to_min(K, X_const, Y_const))
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, expected)


@pytest.mark.parametrize('b_neg', [lambda y: cp.exp(y), lambda y:cp.inv_pos(y),
                                            (lambda y: cp.square(y)),
                                            (lambda y:cp.abs(y)),
                                            (lambda y:-cp.log(y))])
@pytest.mark.parametrize('y_val', range(-2, 3))
def test_by(b_neg, y_val):

    expected = -b_neg(y_val).value

    x = cp.Variable(name='x')
    y = cp.Variable(name='y')

    by = -b_neg(y)

    if by.domain and y_val <= 0:
        return  # skip test

    X_const = [x == 0]
    Y_const = [y == y_val]

    K = K_repr_by(by)
    prob = cp.Problem(*minimax_to_min(K, X_const, Y_const))
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, expected, atol=1e-6)


@pytest.mark.parametrize('y_val', range(-3, 3))
def test_epigraph_exp(y_val):
    y = cp.Variable(name='y')
    b = cp.square(cp.square(y))

    t_primal = cp.Variable(name='t_by_primal')

    constraints = [
        t_primal >= b,
    ]

    var_to_mat_mapping, s_bar, cone_dims, = get_cone_repr(constraints, [y, t_primal])

    R_bar = var_to_mat_mapping[y.id]
    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping['eta']

    s_bar = s_bar.reshape(-1, 1)
    u_temp = cp.Variable((Q_bar.shape[1], 1))

    Ax_b = s_bar - (R_bar * y + t_primal * p_bar + Q_bar @ u_temp)

    prob = cp.Problem(cp.Minimize(t_primal),
                      [*add_cone_constraints(Ax_b, cone_dims, dual=False), y == y_val])
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, y_val**4)

    u = cp.Variable((Q_bar.shape[0], 1))
    max_prob = cp.Problem(cp.Maximize((R_bar.T @ u) * y_val - s_bar.T @ u),
                          [-u.T @ p_bar == 1, Q_bar.T @ u == 0] + add_cone_constraints(u, cone_dims,
                                                                                       dual=True))
    max_prob.solve()
    assert max_prob.status == cp.OPTIMAL
    assert np.isclose(max_prob.value, y_val**4, atol=1e-7)


def test_matrix_game_nemirovski_Fx_Gy():
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
    assert np.isclose(prob.value, 0)

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
    K = SwitchableKRepresentation(
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
    # assert np.allclose(y.value, np.array([0.5, 0.5]))


def test_saddle_composition():
    x = cp.Variable()
    y = cp.Variable()

    objective = MinimizeMaximize(x + x*y)
    constraints = [
        -1 <= x, x <= 1,
        -1 <= y, y <= 1
    ]
    prob = SaddleProblem(objective, constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, 0)


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

    K = K_repr_y_Fx(F, y)

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


@pytest.mark.parametrize('n', [1, 2, 3, 4, 20])
def test_weighted_sum_exp_with_switching(n):

    x = cp.Variable(n, name='x')
    y = cp.Variable(n, name='y', nonneg=True)

    K = K_repr_y_Fx(cp.exp(y), x)  # We want the final problem to be concave in y

    X_constraints = [
        x == 1
    ]
    Y_constraints = [
        y == 1,
    ]

    # We do not need this case in practice, as we rather construct x_Gy instead, see below

    var_to_mat_mapping, _, _, = get_cone_repr(K.constraints, K.constraints[0].variables())
    Q = var_to_mat_mapping['eta']
    K = SwitchableKRepresentation(K.f, K.t, K.x, K.y, K.constraints, Q)

    K_switched = switch_convex_concave(K)

    min_prob = cp.Problem(*minimax_to_min(K_switched, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, -n * np.e)

    # Using x_Gy instead of switched y_Fx
    K_x_Gy = K_repr_x_Gy(-cp.exp(y), x)
    min_prob_x_Gy = cp.Problem(*minimax_to_min(K_x_Gy, X_constraints, Y_constraints))
    min_prob_x_Gy.solve()
    assert min_prob_x_Gy.status == cp.OPTIMAL
    assert np.isclose(min_prob_x_Gy.value, -n * np.e)


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

    K = K_repr_y_Fx(loss, wgts)
    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    adversarial_vals = theta_hat.value

    assert ols_vals[0] < adversarial_vals[0]
    assert ols_vals[1] > adversarial_vals[1]

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(A[:, 1], observations)
    # plt.plot(A[:, 1], A @ adversarial_vals, label='Adversarial', c='red')
    # plt.plot(A[:, 1], A @ cooperative_vals, label='Cooperative', c='green')
    # plt.plot(A[:, 1], A @ ols_vals, label='OLS', color='orange')
    # plt.legend()
    # plt.show()


def test_constant():
    obj = MinimizeMaximize(10)
    problem = SaddleProblem(obj)
    problem.solve()
    assert problem.value == 10


def test_variable():
    k = cp.Variable(name='k')
    with pytest.raises(AffineVariableError, match='Specify curvature'):
        MinimizeMaximize(k)

    constraints = [-10 <= k, k <= 10]

    obj = MinimizeMaximize(k, minimization_vars={k})
    problem = SaddleProblem(obj, constraints)
    problem.solve()
    assert problem.value == -10

    obj = MinimizeMaximize(k, maximization_vars={k})
    problem = SaddleProblem(obj, constraints)
    problem.solve()
    assert problem.value == 10
