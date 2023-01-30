import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import (
    conjugate,
    inner,
    saddle_inner,
    saddle_max,
    saddle_min,
    saddle_quad_form,
    weighted_log_sum_exp,
)
from dsp.cone_transforms import (
    K_repr_ax,
    K_repr_x_Gy,
    LocalToGlob,
    add_cone_constraints,
    get_cone_repr,
    minimax_to_min,
)
from dsp.local import LocalVariable, LocalVariableError
from dsp.parser import DSPError
from dsp.problem import (
    MinimizeMaximize,
    SaddlePointProblem,
    validate_all_saddle_extrema,
)


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

    prob = SaddlePointProblem(MinimizeMaximize(by), X_const + Y_const, minimization_vars={x})
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

    cone_constraints, _ = add_cone_constraints(Ax_b, cone_dims, dual=False)

    prob = cp.Problem(
        cp.Minimize(t_primal),
        [*cone_constraints, y == y_val],
    )
    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, y_val**4, atol=1e-3)

    u = cp.Variable((Q_bar.shape[0], 1))
    cone_constraints, u = add_cone_constraints(u, cone_dims, dual=True)
    max_prob = cp.Problem(
        cp.Maximize((R_bar.T @ u) * y_val - s_bar.T @ u),
        [-u.T @ p_bar == 1, Q_bar.T @ u == 0] + cone_constraints,
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

    FxGy = inner(F_x, G_y)

    objective = MinimizeMaximize(FxGy)
    constraints = [-1 <= x, x <= 1, -1 <= y, y <= 1]
    prob = SaddlePointProblem(objective, constraints)
    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, 0, atol=1e-4)
    assert np.isclose(prob.value, FxGy.value, atol=1e-4)


def test_overload_bilin():
    x = cp.Variable()
    y = cp.Variable()

    objective = MinimizeMaximize(x * y)
    constraints = [-1 <= x, x <= 1, -1.2 <= y, y <= -0.8]
    prob = SaddlePointProblem(objective, constraints, minimization_vars={x}, maximization_vars={y})

    with pytest.raises(DSPError, match="Use inner instead"):
        prob.solve()


@pytest.mark.parametrize(
    "obj",
    [lambda x, y: inner(x, 1 + y), lambda x, y: x + inner(x, y)],
)
def test_saddle_composition(obj):
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    objective = MinimizeMaximize(obj(x, y))

    constraints = [-1 <= x, x <= 1, -1.2 <= y, y <= -0.8]
    prob = SaddlePointProblem(objective, constraints, minimization_vars={x}, maximization_vars={y})

    prob.solve(solver=cp.SCS)
    assert prob.status == cp.OPTIMAL
    assert np.isclose(prob.value, 0)
    assert np.isclose(y.value, -1)


def test_constant():
    obj = MinimizeMaximize(10)
    problem = SaddlePointProblem(obj)
    problem.solve(solver=cp.SCS)
    assert problem.value == 10


def test_variable():
    k = cp.Variable(name="k")
    constraints = [-10 <= k, k <= 10]

    obj = MinimizeMaximize(k)
    with pytest.raises(DSPError, match="Cannot split"):
        SaddlePointProblem(obj, constraints).solve()

    with pytest.raises(DSPError, match="Cannot resolve"):
        SaddlePointProblem(obj, []).solve()

    problem = SaddlePointProblem(obj, constraints, minimization_vars={k})
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, -10)

    problem = SaddlePointProblem(obj, constraints, maximization_vars={k})
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 10)

    # No need to specify when convex/concave terms are present
    obj = MinimizeMaximize(k + 1e-10 * cp.pos(k))
    problem = SaddlePointProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, -10)

    obj = MinimizeMaximize(k - 1e-10 * cp.pos(k))
    problem = SaddlePointProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 10)


def test_sum():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    obj = MinimizeMaximize(x + y)
    constraints = [-1 <= x, x <= 1, -2 <= y, y <= 2]
    problem = SaddlePointProblem(obj, constraints, minimization_vars={x}, maximization_vars={y})
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 1.0)


def test_mixed_curvature_affine():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + np.array([1, 2]) @ cp.vstack([x, y]))

    constraints = [x == 0, y == 1]
    problem = SaddlePointProblem(obj, constraints)

    problem.solve()
    assert np.isclose(problem.value, 3.0)

    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + x + 2 * y)
    problem = SaddlePointProblem(obj, constraints)
    problem.solve()

    assert np.isclose(problem.value, 3)


def test_indeterminate_problem():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")
    obj = MinimizeMaximize(cp.exp(x) + cp.log(y) + z)
    with pytest.raises(DSPError, match="Cannot resolve"):
        SaddlePointProblem(obj).solve()

    prob = SaddlePointProblem(obj, minimization_vars={z}, constraints=[y >= 0, x >= 0, z >= 0])
    assert set(prob.x_prob.variables()) & set(prob.variables()) == {x, z}
    assert set(prob.y_prob.variables()) & set(prob.variables()) == {y}


def test_stacked_variables():
    x1 = cp.Variable(2)
    x2 = cp.Variable()
    y1 = cp.Variable()
    y2 = cp.Variable(2)

    obj = MinimizeMaximize(cp.sum(cp.exp(x1)) + cp.square(x2) + cp.log(y1) + cp.sum(cp.sqrt(y2)))
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
    problem = SaddlePointProblem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 2 * np.exp(-1) + np.log(3) + 2 * np.sqrt(3))


def test_Gx_Fy():
    x = cp.Variable(2, name="x")
    y = cp.Variable(2, name="y")

    with pytest.warns(UserWarning, match="is non-positive"):
        f = saddle_inner(cp.exp(x), cp.log(y))
    obj = MinimizeMaximize(f)
    assert f.is_nonneg()
    assert f.is_incr(0)
    assert f.is_incr(1)

    low = 2
    constraints = [low <= x, x <= 4, 2 <= y, y <= 3]

    problem = SaddlePointProblem(obj, constraints)
    problem.solve(solver=cp.SCS)

    assert np.isclose(problem.value, 2 * np.exp(low) * np.log(3))


def test_robust_constraint():
    x = cp.Variable(name="x")
    y = LocalVariable(name="y", nonneg=True)
    assert repr(y).startswith("LocalVariable")

    obj = MinimizeMaximize(cp.square(x))

    constraints = [x >= 1]

    constraints += [saddle_max(weighted_log_sum_exp(x, y), [y <= 1]) <= 1]

    invalid_saddle_problem = SaddlePointProblem(
        obj, constraints, minimization_vars=[x], maximization_vars=[y]
    )
    assert not invalid_saddle_problem.is_dsp()
    assert not dsp.is_dsp(invalid_saddle_problem)
    with pytest.raises(DSPError, match="Likely passed unused variables"):
        invalid_saddle_problem.solve()

    problem = SaddlePointProblem(obj, constraints)
    assert dsp.is_dsp(problem)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 1.0, atol=1e-4)


def test_robust_constraint_min():
    x = cp.Variable(name="x")
    y = LocalVariable(name="y", nonneg=True)

    obj = dsp.MinimizeMaximize(x)

    constraints = [x >= 1]

    constraints += [saddle_max(weighted_log_sum_exp(x, y), [y <= 1]) <= 1]

    problem = SaddlePointProblem(obj, constraints, minimization_vars=[x])
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 1.0, atol=1e-4)

    obj = cp.Minimize(x)
    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)
    validate_all_saddle_extrema(problem)
    assert np.isclose(problem.value, 1.0, atol=1e-4)


def test_robust_constraint_inf():
    """
    Test that we can handle robust constraint inf_x f(x,y) >= eta.
    """
    x = LocalVariable(name="x_dummy")
    y = cp.Variable(name="y", nonneg=True)

    x_val = 1.0
    y_val = 2.0

    saddle_min_expr = saddle_min(weighted_log_sum_exp(x, y), [x >= x_val])

    constraints = [y <= y_val, saddle_min_expr >= np.log(y_val * np.exp(x_val))]

    obj = cp.Maximize(y)

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)
    validate_all_saddle_extrema(problem)
    assert np.isclose(problem.value, y_val, atol=1e-4)
    assert set(saddle_min_expr.concave_variables()) == {y}
    assert set(saddle_min_expr.convex_variables()) == {x}
    assert np.isclose(saddle_min_expr.value, np.log(y_val * np.exp(x_val)), atol=1e-4)
    assert (
        saddle_min_expr.name() == "saddle_min(weighted_log_sum_exp(x_dummy, y), [1.0 <= x_dummy])"
    )


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize(
    "expr,expected",
    [
        (lambda Y: cp.log_det(Y), lambda n: n),
        (lambda Y: cp.trace(Y), lambda n: n * np.e),
        (lambda Y: -cp.lambda_max(Y), lambda n: -np.e),
    ],
)
def test_PSD_saddle(n, expr, expected):
    Y = cp.Variable((n, n), PSD=True, name="Y")
    f = expr(Y)

    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [Y == np.e * np.eye(n)], maximization_vars=[Y])

    prob.solve(solver=cp.SCS)
    assert np.isclose(prob.value, expected(n), atol=1e-4)


def test_psd_exp():
    n = 2
    Y = cp.Variable((n, n), PSD=True, name="Y")

    obj = MinimizeMaximize(cp.exp(cp.lambda_max(Y)))

    prob = SaddlePointProblem(obj, [Y >> np.eye(n)], minimization_vars=[Y])

    prob.solve()

    assert np.isclose(prob.value, np.e, atol=1e-4)


def test_worst_case_covariance():
    kappa = 0.5
    Sigma = np.array([[1, -0.3], [-0.3, 2]])
    delta = LocalVariable((2, 2), symmetric=True, name="delta")
    Sigma_pert = LocalVariable((2, 2), PSD=True, name="Sigma_pert")
    v = cp.Variable(2, name="v")
    obj = saddle_quad_form(v, Sigma_pert)
    constraints = [
        Sigma + delta == Sigma_pert,
        cp.abs(delta) <= kappa * np.sqrt(np.outer(Sigma.diagonal(), Sigma.diagonal())),
    ]

    worst_case_risk = saddle_max(obj, constraints)

    v_val = np.array([0.70348158, 0.29651842])  # This is the optimal value obtained via closed form

    wc_ref = v_val.T @ Sigma @ v_val + kappa * (np.sqrt(np.diag(Sigma)) @ np.abs(v_val)) ** 2

    v.value = v_val
    assert np.isclose(wc_ref, worst_case_risk.value, atol=1e-4)
    assert worst_case_risk.name().startswith("saddle_max(saddle_quad_form(")

    # TODO: solver does not find the optimal value without fixing the values to opt
    prob = cp.Problem(cp.Minimize(worst_case_risk), [cp.sum(v) == 1, v >= 0, v == v_val])

    prob.solve(solver=cp.SCS)

    wc_ob_ref = cp.quad_form(v, Sigma) + kappa * cp.square(np.sqrt(np.diag(Sigma)) @ cp.abs(v))
    ref_prob = cp.Problem(cp.Minimize(wc_ob_ref), [cp.sum(v) == 1, v >= 0])
    ref_prob.solve(solver=cp.SCS)

    assert np.isclose(ref_prob.value, prob.value, atol=1e-4)


def test_aux_variable_in_constraints():
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")

    with pytest.warns(UserWarning, match="Weights are non-positive"):
        wlse = weighted_log_sum_exp(x, y)

    saddle_prob = SaddlePointProblem(MinimizeMaximize(wlse), [x == 1, y <= z, z <= 1])
    assert saddle_prob.is_dsp()
    saddle_prob.solve()
    assert np.isclose(saddle_prob.value, 1, atol=1e-4)


def test_SE_variable_in_constraint():
    x = cp.Variable(name="x")
    y = LocalVariable(name="y")
    z = LocalVariable(name="z")
    z_nonlocal = cp.Variable(name="z_nonloacl")

    with pytest.warns(UserWarning, match="Weights are non-positive"):
        wlse = weighted_log_sum_exp(x, y)

    with pytest.raises(LocalVariableError):
        saddle_max(wlse, [y <= z_nonlocal, z_nonlocal <= 1])

    se = saddle_max(wlse, [y <= z, z <= 1])
    problem = cp.Problem(cp.Minimize(se), [x == 1])
    problem.solve()
    validate_all_saddle_extrema(problem)
    assert np.isclose(problem.value, 1, atol=1e-4)


def test_unconstrained():
    x = cp.Variable(1, name="x")
    y = cp.Variable(1, name="y")
    f = cp.square(x) - cp.square(y)
    obj = MinimizeMaximize(f)

    saddle_problem = SaddlePointProblem(obj)
    saddle_problem.is_dsp()

    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, 0)


def test_saddle_inner_neg():
    x = cp.Variable(1, name="x")
    y = cp.Variable(1, name="y", nonneg=True)
    f = saddle_inner(cp.square(x) - 1, y)
    obj = MinimizeMaximize(f)

    assert not obj.is_dsp()

    saddle_problem = SaddlePointProblem(obj)
    assert not saddle_problem.is_dsp()

    with pytest.raises(DSPError):
        saddle_problem.solve()


def test_scalar():
    x = cp.Variable()
    y = cp.Variable()
    f = inner(x, y)
    obj = MinimizeMaximize(f)
    prob = SaddlePointProblem(obj, [-1 <= x, x <= 1, -1 <= y, y <= 1])
    prob.solve()
    assert np.isclose(prob.value, 0)
    assert np.isclose(x.value, 0)
    assert np.isclose(y.value, 0)


def test_non_local_variable():
    x = cp.Variable(1, name="x")
    y = cp.Variable(1, name="y")

    with pytest.raises(LocalVariableError):
        conjugate(inner(x, y))


def test_pure_convex():
    x = cp.Variable(name="x")
    f = cp.square(x)
    obj = MinimizeMaximize(f)

    assert obj.is_dsp()

    saddle_problem = SaddlePointProblem(obj)
    assert saddle_problem.is_dsp()

    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, 0)


def test_pure_concave():
    y = cp.Variable(name="y")
    f = -cp.square(y)
    obj = MinimizeMaximize(f)

    assert obj.is_dsp()

    saddle_problem = SaddlePointProblem(obj)
    assert saddle_problem.is_dsp()

    saddle_problem.solve()
    assert np.isclose(saddle_problem.value, 0)


def test_local_variable_setter():
    x = LocalVariable(name="x")
    y = cp.Variable(name="y")
    se = saddle_min(y + x, [x <= 1])
    x.value = 1
    assert x.value == 1  # currently value only computes
