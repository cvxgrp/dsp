import numpy as np
import pytest

import cvxpy as cp
from dspp.atoms import ConvexConcaveAtom, GeneralizedInnerProduct, switch_convex_concave, WeightedLogSumExp
from dspp.cone_transforms import LocalToGlob, minimax_to_min, K_repr_y_Fx, K_repr_x_Gy, K_repr_ax, K_repr_by, \
    add_cone_constraints, get_cone_repr, SwitchableKRepresentation
from dspp.problem import MinimizeMaximize, SaddleProblem, AffineVariableError


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

    ltg = LocalToGlob([y])

    K = K_repr_x_Gy(F, x, ltg)
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
@pytest.mark.parametrize('y_val', range(1, 4))
def test_by(b_neg, y_val):

    expected = -b_neg(y_val).value

    x = cp.Variable(name='x')
    y = cp.Variable(name='y')

    by = -b_neg(y)

    if by.domain and y_val <= 0:
        return  # skip test

    X_const = [x == 0]
    Y_const = [y == y_val]

    prob = SaddleProblem(MinimizeMaximize(by), X_const + Y_const, minimization_vars={x})
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, expected, atol=1e-6)
    assert np.isclose(y.value, y_val)


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

    F_x = x
    G_y = y

    FxGy = GeneralizedInnerProduct(F_x, G_y)

    objective = MinimizeMaximize(FxGy)
    constraints = [
        -1 <=x, x <= 1,
        -1 <= y, y <= 1
    ]
    prob = SaddleProblem(objective, constraints)
    prob.solve()
    # assert np.isclose(prob.value, 0)

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


@pytest.mark.parametrize('x_val,y_val,n',
                         [(1, 1, 2), (2, 2, 2), (0.5, 3, 5), (2, 6, 20), (6, .1, 1)])
def test_eval_weighted_log_sum_exp(x_val, y_val, n):
    x = cp.Variable(n)

    y = cp.Variable(n, nonneg=True)
    y_value = np.ones(n) * y_val
    x_value = np.ones(n) * x_val

    wlse = WeightedLogSumExp(x, y)

    lgt = LocalToGlob([y])
    K = wlse.get_K_repr(lgt)

    prob = cp.Problem(cp.Minimize(K.f @ y_value + K.t),
                      [
                          *K.constraints,
                          x == x_value,
    ])
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log(np.exp(x_value) @ y_value))


def test_eval_weighted_log_sum_exp_affine_sum():
    n = 1
    y_val = 1
    x_val = 1

    x = cp.Variable(n)
    y = cp.Variable(n+1, nonneg=True)
    y_value = np.ones(n+1) * y_val
    x_value = np.ones(n) * x_val

    a = np.array([2, 1])
    wlse = WeightedLogSumExp(x, a@y+1)

    ltg = LocalToGlob([y])

    K = wlse.get_K_repr(ltg)

    prob = cp.Problem(cp.Minimize(K.f @ y_value + K.t),
                      [
                          *K.constraints,
                          x == x_value,
    ])
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log(4*np.exp(1)))

    x_val = 1
    X_constraints = [x >= x_val]
    y_val = 1.1
    Y_constraints = [y <= y_val]
    min_prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, np.log((sum(a)*y_val+1)*np.exp(x_val)))


@pytest.mark.parametrize('x_val,y_val,c', [(1, 1, 1), (1, 0.5, 1), (1, 1, 1), (5, 5, 2), (3, 2, 3)])
def test_wlse_multi_var(x_val, y_val, c):
    x1 = cp.Variable(nonneg=True)
    x2 = cp.Variable()

    y = cp.Variable(3, nonneg=True)

    a = np.array([2, 1])
    wlse = WeightedLogSumExp(x1, a@y[:2]+c)

    obj = MinimizeMaximize(wlse + cp.exp(x2) + cp.log(y[2]))
    x_constraints = [x1 >= x_val, x2 >= x_val]
    y_constraints = [y <= y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, np.log((sum(a)*y_val+c) *
                      np.exp(x_val))+np.exp(x_val)+np.log(y_val))
    assert np.allclose(y.value, y_val)


@pytest.mark.parametrize('x_val,y_val,c', [(1, 1, 1), (1, 0.5, 1), (5, 5, 2), (3, 2, 3)])
def test_wlse_multi_var_switching(x_val, y_val, c):
    y1 = cp.Variable()
    y2 = cp.Variable()

    x = cp.Variable(3, nonneg=True)

    a = 2
    wlse = WeightedLogSumExp(a*y1+a*y2+c, cp.sum(x[1:]))

    obj = MinimizeMaximize(-wlse + cp.exp(x[0]) + cp.log(y1))
    x_constraints = [x == x_val]
    y_constraints = [y1 == y_val, y2 == y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, -np.log(2*x_val*np.exp(2*a*y_val+c))+np.exp(x_val)+np.log(y_val))


@pytest.mark.parametrize('x_val,y_val,c', [(1, 1, 1)])
def test_neg_wlse(x_val, y_val, c):
    x1 = cp.Variable(nonneg=True, name="x1")
    # x2 = cp.Variable()

    # y = cp.Variable(3, nonneg=True)
    y1 = cp.Variable(nonneg=True, name="y1")

    # a = np.array([2, 1])
    # wlse = WeightedLogSumExp(x1, a@y[:2]+c)
    wlse = WeightedLogSumExp(y1, x1)

    # obj = MinimizeMaximize(wlse + cp.exp(x2) + cp.log(y[2]))
    obj = MinimizeMaximize(-1*wlse)
    x_constraints = [x_val >= x1]  # , x2 >= x_val]
    y_constraints = [y_val <= y1]

    prob = SaddleProblem(obj, x_constraints + y_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, -np.log(x_val*np.exp(y_val)))


def test_weighted_log_sum_exp_with_switching():
    n = 2
    x = cp.Variable(n, name='x_outer', nonneg=True)
    y = cp.Variable(n, name='y_outer', nonneg=True)

    a = np.ones(n)

    wsle = WeightedLogSumExp(a@y+1, a@x+1)

    x_val = 1
    y_val = 3

    X_constraints = [
        x <= x_val
    ]
    Y_constraints = [
        y >= y_val
    ]

    ltg = LocalToGlob([y])

    K_switched = wsle.get_K_repr(ltg, switched=True)  # -log(x exp(y))
    min_prob = cp.Problem(*minimax_to_min(K_switched, X_constraints, Y_constraints))
    min_prob.solve()
    assert min_prob.status == cp.OPTIMAL
    assert np.isclose(min_prob.value, -np.log((n*x_val+1) * np.exp(n*y_val+1)))


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
    ltg = LocalToGlob([y])
    K_x_Gy = K_repr_x_Gy(-cp.exp(y), x, ltg)
    min_prob_x_Gy = cp.Problem(*minimax_to_min(K_x_Gy, X_constraints, Y_constraints))
    min_prob_x_Gy.solve()
    assert min_prob_x_Gy.status == cp.OPTIMAL
    assert np.isclose(min_prob_x_Gy.value, -n * np.e)


def test_constant():
    obj = MinimizeMaximize(10)
    problem = SaddleProblem(obj)
    problem.solve()
    assert problem.value == 10


def test_variable():
    k = cp.Variable(name='k')
    constraints = [-10 <= k, k <= 10]

    obj = MinimizeMaximize(k)
    with pytest.raises(ValueError, match="Cannot split"):
        SaddleProblem(obj, constraints)

    with pytest.raises(AssertionError, match="Cannot resolve"):
        SaddleProblem(obj, [])

    problem = SaddleProblem(obj, constraints, minimization_vars={k})
    problem.solve()
    assert problem.value == -10

    problem = SaddleProblem(obj, constraints, maximization_vars={k})
    problem.solve()
    assert problem.value == 10

    # No need to specify when convex/concave terms are present
    obj = MinimizeMaximize(k + 1e-10 * cp.pos(k))
    problem = SaddleProblem(obj, constraints)
    problem.solve()
    assert problem.value == -10

    obj = MinimizeMaximize(k - 1e-10 * cp.pos(k))
    problem = SaddleProblem(obj, constraints)
    problem.solve()
    assert np.isclose(problem.value, 10)


def test_sum():
    x = cp.Variable(name='x')
    y = cp.Variable(name='y')

    obj = MinimizeMaximize(x + y)
    constraints = [
        -1 <= x, x <= 1,
        -2 <= y, y <= 2
    ]
    problem = SaddleProblem(obj, constraints, minimization_vars={x}, maximization_vars={y})
    problem.solve()
    assert np.isclose(problem.value, 1.0)


def test_mixed_curvature_affine():

    x = cp.Variable()
    y = cp.Variable()

    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + np.array([1, 2]) @ cp.vstack([x, y]))

    with pytest.raises(AssertionError, match="convex and concave"):
        SaddleProblem(obj)


def test_indeterminate_problem():
    x = cp.Variable(name='x')
    y = cp.Variable(name='y')
    z = cp.Variable(name='z')
    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + z)
    with pytest.raises(AssertionError, match="Cannot resolve"):
        SaddleProblem(obj)

    prob = SaddleProblem(obj, minimization_vars={z}, constraints= [y >= 0, x >= 0, z >= 0])
    assert set(prob.x_prob.variables()) & set(prob.variables()) == {x, z}
    assert set(prob.y_prob.variables()) & set(prob.variables()) == {y}


def test_stacked_variables():
    x1 = cp.Variable(2)
    x2 = cp.Variable()
    y1 = cp.Variable()
    y2 = cp.Variable(2)

    obj = MinimizeMaximize(cp.sum(cp.exp(x1)) + cp.square(x2) + cp.log(y1) + cp.sum(cp.sqrt(y2)))
    constraints = [
        -1 <= x1, x1 <= 1,
        -2 <= x2, x2 <= 2,
        1 <= y1, y1 <= 3,
        1 <= y2, y2 <= 3
    ]
    problem = SaddleProblem(obj, constraints)
    problem.solve()
    assert np.isclose(problem.value, 2*np.exp(-1) + np.log(3) + 2*np.sqrt(3))
