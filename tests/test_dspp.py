import cvxpy as cp
import numpy as np
import pytest

from dspp.atoms import convex_concave_inner, inner, weighted_log_sum_exp
from dspp.cone_transforms import (
    K_repr_ax,
    K_repr_x_Gy,
    LocalToGlob,
    add_cone_constraints,
    get_cone_repr,
    minimax_to_min,
)
from dspp.problem import MinimizeMaximize, RobustConstraints, SaddleProblem


def test_matrix_game_x_Gy():
    n = 2

    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y")
    yy = cp.Variable(n, name="yy")

    X_constraints = [
        -1 <= x,
        x <= 1,
    ]
    Y_constraints = [-1 <= yy, yy <= 1, yy <= y, y <= yy]

    F = y + 0.5

    ltg = LocalToGlob([x], [y, yy])

    K = K_repr_x_Gy(F, x, ltg)
    prob = cp.Problem(*minimax_to_min(K, X_constraints, Y_constraints, [y, yy], ltg))
    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, 0, atol=1e-4)
    print(prob.status, prob.value)
    for v in prob.variables():
        print(v, v.name(), v.value)


@pytest.mark.parametrize("a,expected", [(cp.exp, np.exp(2)), (cp.square, 4)])
def test_ax(a, expected):
    x = cp.Variable()
    y = cp.Variable()
    ax = a(x)

    X_const = [2 <= x, x <= 4]
    Y_const = [y == 0]

    K = K_repr_ax(ax)
    local_to_glob = LocalToGlob([x], [y])
    prob = cp.Problem(*minimax_to_min(K, X_const, Y_const, [y], local_to_glob))
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, expected)


@pytest.mark.parametrize(
    "b_neg",
    [
        lambda y: cp.exp(y),
        lambda y: cp.inv_pos(y),
        (lambda y: cp.square(y)),
        (lambda y: cp.abs(y)),
        (lambda y: -cp.log(y)),
    ],
)
@pytest.mark.parametrize("y_val", range(1, 4))
def test_by(b_neg, y_val):
    expected = -b_neg(y_val).value

    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    by = -b_neg(y)

    if by.domain and y_val <= 0:
        return  # skip test

    X_const = [x == 0]
    Y_const = [y == y_val]

    prob = SaddleProblem(MinimizeMaximize(by), X_const + Y_const, minimization_vars={x})
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, expected, atol=1e-3)
    assert np.isclose(y.value, y_val)


@pytest.mark.parametrize("y_val", range(-3, 3))
def test_epigraph_exp(y_val):
    y = cp.Variable(name="y")
    b = cp.square(cp.square(y))

    t_primal = cp.Variable(name="t_by_primal")

    constraints = [
        t_primal >= b,
    ]

    (
        var_to_mat_mapping,
        s_bar,
        cone_dims,
    ) = get_cone_repr(constraints, [y, t_primal])

    R_bar = var_to_mat_mapping[y.id]
    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping["eta"]

    s_bar = s_bar.reshape(-1, 1)
    u_temp = cp.Variable((Q_bar.shape[1], 1))

    Ax_b = s_bar - (R_bar * y + t_primal * p_bar + Q_bar @ u_temp)

    prob = cp.Problem(
        cp.Minimize(t_primal),
        [*add_cone_constraints(Ax_b, cone_dims, dual=False), y == y_val],
    )
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, y_val**4, atol=1e-3)

    u = cp.Variable((Q_bar.shape[0], 1))
    max_prob = cp.Problem(
        cp.Maximize((R_bar.T @ u) * y_val - s_bar.T @ u),
        [-u.T @ p_bar == 1, Q_bar.T @ u == 0]
        + add_cone_constraints(u, cone_dims, dual=True),
    )
    max_prob.solve(solver=cp.SCS)
    assert max_prob.status == cp.OPTIMAL
    assert np.isclose(max_prob.value, y_val**4, atol=1e-7)


def test_matrix_game_nemirovski_Fx_Gy():
    n = 2

    x = cp.Variable(n, name="x")
    y = cp.Variable(n, name="y")

    F_x = x + 0.5
    G_y = y + 0.5

    FxGy = convex_concave_inner(F_x, G_y)

    objective = MinimizeMaximize(FxGy)
    constraints = [-1 <= x, x <= 1, -1 <= y, y <= 1]
    prob = SaddleProblem(objective, constraints)
    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, 0, atol=1e-4)


@pytest.mark.parametrize(
    "obj",
    [
        lambda x, y: inner(x, 1 + y),
        lambda x, y: x + inner(x, y),
        lambda x, y: x * (1 + y),
    ],
)
def test_saddle_composition(obj):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # objective = MinimizeMaximize(x + x * y)
    # objective = MinimizeMaximize(x * (1+y))
    # objective = MinimizeMaximize(x + Bilinear(x, y))
    # objective = MinimizeMaximize(Bilinear(x, 1+y))

    objective = MinimizeMaximize(obj(x, y))

    # TODO: why are these different? Optimal y only correct in second formulation

    constraints = [-1 <= x, x <= 1, -1.2 <= y, y <= -0.8]
    prob = SaddleProblem(
        objective, constraints, minimization_vars={x}, maximization_vars={y}
    )

    print(prob.y_prob)

    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, 0)
    assert np.isclose(y.value, -1)


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


@pytest.mark.parametrize(
    "x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (2, 1.23, 2), (3, 2, 3)]
)
def test_wlse_composition_switch(x_val, y_val, c):
    y = cp.Variable(name="y")
    x1 = cp.Variable(name="x1", nonneg=True)
    x2 = cp.Variable(name="x2", nonneg=True)

    wlse = weighted_log_sum_exp(cp.exp(y), x1 + x2)  # these variables are global scope

    obj = MinimizeMaximize(-wlse)

    x_constraints = [x1 + x2 <= x_val]
    y_constraints = [y >= y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)
    opt_val = -np.log(x_val * np.exp(np.exp(y_val)))

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val)

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, opt_val)


@pytest.mark.parametrize(
    "x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (2.1, 0.5, 2), (1.1, 2, 3)]
)
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

    prob = SaddleProblem(obj, x_constraints + y_constraints)
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


@pytest.mark.parametrize(
    "x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (1, 1, 1), (0.31, 1.04, 3)]
)
def test_wlse_multi_var(x_val, y_val, c):
    x1 = cp.Variable(nonneg=True, name="x1")
    x2 = cp.Variable(name="x2")

    y = cp.Variable(3, nonneg=True, name="y")

    a = np.array([2, 1])
    wlse = weighted_log_sum_exp(x1, a @ y[:2] + c)

    obj = MinimizeMaximize(wlse + cp.exp(x2) + cp.log(y[2]))
    x_constraints = [x1 >= x_val, x2 >= x_val]
    y_constraints = [y <= y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)

    opt_val = (
        np.log((sum(a) * y_val + c) * np.exp(x_val)) + np.exp(x_val) + np.log(y_val)
    )

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val, atol=1e-3)

    prob.y_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.y_prob.value, -opt_val)

    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, opt_val, atol=1e-3)
    assert np.allclose(y.value, y_val)


@pytest.mark.parametrize(
    "x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (5, 5, 2), (3, 2, 3)]
)
def test_wlse_switching(x_val, y_val, c):
    y = cp.Variable(name="y")
    x1 = cp.Variable(name="x1", nonneg=True)
    x2 = cp.Variable(name="x2", nonneg=True)

    wlse = weighted_log_sum_exp(y, x1 + x2)  # these variables are global scope

    obj = MinimizeMaximize(-wlse)

    x_constraints = [x1 + x2 <= x_val]
    y_constraints = [y >= y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)
    opt_val = -np.log(x_val * np.exp(y_val))

    prob.x_prob.solve(solver=cp.SCS)
    assert np.isclose(prob.x_prob.value, opt_val)

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, opt_val)


@pytest.mark.parametrize(
    "x_val,y_val,c", [(1, 1, 1), (1, 0.5, 1), (3.01, 1.4, 2), (3, 2, 3)]
)
def test_wlse_multi_var_switching(x_val, y_val, c):
    y1 = cp.Variable()
    y2 = cp.Variable()

    x = cp.Variable(3, nonneg=True)

    a = 2
    wlse = weighted_log_sum_exp(a * y1 + a * y2 + c, cp.sum(x[1:]))

    obj = MinimizeMaximize(-wlse + cp.exp(x[0]) + cp.log(y1))
    x_constraints = [x == x_val]
    y_constraints = [y1 == y_val, y2 == y_val]

    prob = SaddleProblem(obj, x_constraints + y_constraints)

    opt_val = (
        -np.log(2 * x_val * np.exp(2 * a * y_val + c)) + np.exp(x_val) + np.log(y_val)
    )

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

    prob = SaddleProblem(obj, x_constraints + y_constraints)
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
    min_prob = cp.Problem(
        *minimax_to_min(K_switched, X_constraints, Y_constraints, [y], ltg)
    )

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
    min_prob_x_Gy = cp.Problem(
        *minimax_to_min(K_x_Gy, X_constraints, Y_constraints, [y], ltg)
    )
    min_prob_x_Gy.solve(solver=cp.SCS)
    assert min_prob_x_Gy.status == cp.OPTIMAL
    assert np.isclose(min_prob_x_Gy.value, -n * np.e)


def test_constant():
    obj = MinimizeMaximize(10)
    problem = SaddleProblem(obj)
    problem.solve(solver=cp.SCS)
    assert problem.value == 10


def test_variable():
    k = cp.Variable(name="k")
    constraints = [-10 <= k, k <= 10]

    obj = MinimizeMaximize(k)
    with pytest.raises(ValueError, match="Cannot split"):
        SaddleProblem(obj, constraints)

    with pytest.raises(AssertionError, match="Cannot resolve"):
        SaddleProblem(obj, [])

    problem = SaddleProblem(obj, constraints, minimization_vars={k})
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, -10)

    problem = SaddleProblem(obj, constraints, maximization_vars={k})
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 10)

    # No need to specify when convex/concave terms are present
    obj = MinimizeMaximize(k + 1e-10 * cp.pos(k))
    problem = SaddleProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, -10)

    obj = MinimizeMaximize(k - 1e-10 * cp.pos(k))
    problem = SaddleProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 10)


def test_sum():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    obj = MinimizeMaximize(x + y)
    constraints = [-1 <= x, x <= 1, -2 <= y, y <= 2]
    problem = SaddleProblem(
        obj, constraints, minimization_vars={x}, maximization_vars={y}
    )
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 1.0)


def test_mixed_curvature_affine():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    # TODO: parse @
    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + np.array([1, 2]) @ cp.vstack([x, y]))

    constraints = [x == 0, y == 1]

    problem = SaddleProblem(obj, constraints)
    problem.solve(solver=cp.SCS)

    assert np.isclose(problem.value, 1 + 2)


def test_indeterminate_problem():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")
    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + z)
    with pytest.raises(AssertionError, match="Cannot resolve"):
        SaddleProblem(obj)

    prob = SaddleProblem(
        obj, minimization_vars={z}, constraints=[y >= 0, x >= 0, z >= 0]
    )
    assert set(prob.x_prob.variables()) & set(prob.variables()) == {x, z}
    assert set(prob.y_prob.variables()) & set(prob.variables()) == {y}


def test_stacked_variables():
    x1 = cp.Variable(2)
    x2 = cp.Variable()
    y1 = cp.Variable()
    y2 = cp.Variable(2)

    obj = MinimizeMaximize(
        cp.sum(cp.exp(x1)) + cp.square(x2) + cp.log(y1) + cp.sum(cp.sqrt(y2))
    )
    constraints = [
        -1 <= x1,
        x1 <= 1,
        -2 <= x2,
        x2 <= 2,
        1 <= y1,
        y1 <= 3,
        1 <= y2,
        y2 <= 3,
    ]
    problem = SaddleProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 2 * np.exp(-1) + np.log(3) + 2 * np.sqrt(3))


def test_Gx_Fy():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    with pytest.warns(UserWarning, match="is non-positive"):
        obj = MinimizeMaximize(convex_concave_inner(cp.exp(x), cp.log(y)))

    low = 2
    constraints = [low <= x, x <= 4, 2 <= y, y <= 3]

    problem = SaddleProblem(obj, constraints)
    problem.solve(solver=cp.SCS)

    assert np.isclose(problem.value, 2 * np.exp(low) * np.log(3))


@pytest.mark.parametrize("x_val,y_val", [(1, 2 + np.e + 0.01), (2, 13.01), (1.3, 11.1)])
def test_wlse_comp(x_val, y_val):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    with pytest.warns(UserWarning, match="are non-positive"):
        wlse = weighted_log_sum_exp(cp.square(x), cp.log(cp.log(y)))

    obj = MinimizeMaximize(wlse)

    constraints = [x >= x_val, y <= y_val]

    problem = SaddleProblem(obj, constraints)
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

    problem = SaddleProblem(obj, constraints)
    opt_val = np.log(8 * np.exp(1)) + 2

    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, opt_val, atol=1e-4)


def test_robust_constraint():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    obj1 = MinimizeMaximize(cp.square(x))
    obj2 = MinimizeMaximize(x)

    _constraints = [x >= 1, y <= 1]

    constraints = _constraints + \
        RobustConstraints(weighted_log_sum_exp(x, y), 1.0, [y <= 1, x >= 0])

    # TODO, auto min vars
    with pytest.raises(ValueError, match="Cannot split"):
        SaddleProblem(obj2, constraints, maximization_vars=[y])  # objective is affine  in x

    problem = SaddleProblem(obj1, constraints, maximization_vars=[y])
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 1.0, atol=1e-4)

    # Idea: once we extend Constraint to RobustConstraint, we can avoid requring
    # min vars by providing an attribute to the constraint that indicates all
    # variables are minimization variables. Just kidding we probably dont want
    # to do this because then it couldn't be consumed by a regular cvxpy problem.

    # f = wlse(x,y)
    # constraints += [f > eta] + [f < eta]