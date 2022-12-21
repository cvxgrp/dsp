import cvxpy as cp
import pytest

from dsp import is_dsp
from dsp.atoms import saddle_max, saddle_min, weighted_log_sum_exp
from dsp.local import LocalVariable, LocalVariableError
from dsp.parser import DSPError
from dsp.problem import MinimizeMaximize, SaddlePointProblem


def test_is_dsp():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)
    z = cp.Variable(name="z")
    f = weighted_log_sum_exp(x, x)

    obj = MinimizeMaximize(f)
    saddle_problem = SaddlePointProblem(obj, [cp.sum(x) == 1])

    assert (
        not saddle_problem.is_dsp()
    )  # AssertionError: Cannot add variables to both convex and concave set.

    obj = MinimizeMaximize(weighted_log_sum_exp(x, y) + z)
    saddle_problem = SaddlePointProblem(obj, [cp.sum(x) == 1, cp.sum(y) == 1])

    assert (
        not saddle_problem.is_dsp()
    )  # AssertionError: Cannot resolve curvature of variables ['z']. Specify curvature of these variables as SaddleProblem(obj, constraints, minimization_vars, maximization_vars).

    obj = MinimizeMaximize(weighted_log_sum_exp(x, y))
    saddle_problem = SaddlePointProblem(
        obj, [cp.sum(x) == 1, cp.sum(y) == 1], minimization_vars=[x, y]
    )

    assert (
        not saddle_problem.is_dsp()
    )  # AssertionError: Cannot add variables to both convex and concave set.

    with pytest.raises(DSPError, match="Cannot add variables to both convex and concave set."):
        saddle_problem.solve()


def test_concave_sadle_max():
    y_dummy = LocalVariable(2, name="y_dummy", nonneg=True)
    x = cp.Variable(name="x", nonneg=True)
    f = -cp.sqrt(cp.sum(y_dummy)) + cp.exp(x)
    F = saddle_max(f, [y_dummy], [cp.sum(y_dummy) == 1])
    # currently this fails on construction rather than via a "is_dsp" check


def test_saddle_double_dummy():
    x = cp.Variable(name="x", nonneg=True)
    y_local = LocalVariable(name="y", nonneg=True)

    y = cp.Variable(name="y", nonneg=True)
    x_local = LocalVariable(name="x", nonneg=True)

    f = cp.sqrt(y_local) + cp.exp(x)
    g = cp.sqrt(y) + cp.exp(x_local)

    F_1 = saddle_max(f, [y_local], [cp.sum(y_local) == 1])

    with pytest.raises(LocalVariableError):
        F_2 = saddle_max(f, [y_local], [cp.sum(y_local) == 1])

    G_1 = saddle_min(g, [x_local], [cp.sum(x_local) == 1])
    with pytest.raises(LocalVariableError):
        G_2 = saddle_min(g, [x_local], [cp.sum(x_local) == 1])

    F_1.is_dsp()
    G_1.is_dsp()

    prob = cp.Problem(cp.Minimize(F_1), [F_1 <= 0])
    assert is_dsp(prob)
    prob_2 = cp.Problem(cp.Maximize(G_1), [G_1 >= 0])
    assert is_dsp(prob_2)


def test_problem_is_dcp():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y", nonneg=True)

    f = weighted_log_sum_exp(x, y_local)
    sup_y_f = saddle_max(f, [y_local], [cp.sum(y_local) == 1])

    prob = cp.Problem(cp.Minimize(sup_y_f), [sup_y_f <= 0])

    assert is_dsp(prob)
    assert prob.is_dsp()

    y_local = LocalVariable(2, name="y", nonneg=True)
    f = weighted_log_sum_exp(x, y_local)
    f += cp.exp(y_local[1])
    sup_y_f = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    prob = cp.Problem(cp.Minimize(sup_y_f), [sup_y_f <= 0])

    assert not is_dsp(prob)
    assert not prob.is_dsp()


def test_expression_is_dcp():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y", nonneg=True)

    f = weighted_log_sum_exp(x, y_local)
    sup_y_f = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    F = sup_y_f + cp.exp(x[1])

    assert is_dsp(sup_y_f)
    assert sup_y_f.is_dsp()

    assert F.is_dsp()

    y_local = LocalVariable(2, name="y", nonneg=True)
    f = weighted_log_sum_exp(x, y_local)
    f += cp.exp(y_local[1])
    sup_y_f = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    F = sup_y_f + cp.exp(x[1])

    assert not is_dsp(sup_y_f)
    assert not sup_y_f.is_dsp()
    assert not F.is_dsp()


def test_saddle_fun_is_dsp():
    x = cp.Variable(2, name="x", nonneg=True)
    y = LocalVariable(2, name="y", nonneg=True)
    f = weighted_log_sum_exp(x, y)
    z = cp.Variable(name="z")

    f1 = f + z
    assert f1.is_dsp()
    assert saddle_max(f1, [y], [cp.sum(y) == 1]).is_dsp()

    assert (f - z).is_dsp()
    assert not (f * z).is_dsp()
    assert not (f + cp.exp(z) + cp.sqrt(z)).is_dsp()


def test_saddle_extremum_affine_is_dsp():
    x = cp.Variable(2, name="x", nonneg=True)
    y = LocalVariable(2, name="y", nonneg=True)
    wlse = weighted_log_sum_exp(x, y)
    z = cp.Variable(name="z")

    f = wlse + z
    assert f.is_dsp()

    F = saddle_max(f, [y], [cp.sum(y) == 1])
    assert F.is_dsp()

    x = LocalVariable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)
    wlse = weighted_log_sum_exp(x, y)
    z = cp.Variable(name="z")

    f = wlse + z
    assert f.is_dsp()

    F = saddle_min(f, [x], [cp.sum(x) == 1])
    assert F.is_dsp()


def test_saddle_extremum_local_affine():
    x = cp.Variable(2, name="x", nonneg=True)
    y = LocalVariable(2, name="y_local", nonneg=True)
    z = LocalVariable(name="z_local")
    f = weighted_log_sum_exp(x, y) + z

    assert f.is_dsp()

    F = saddle_max(f, [y], [cp.sum(y) == 1])
    assert (
        not F.is_dsp()
    )  # z is a local variable appearing in the objective, and not listed as a concave variable

    x = LocalVariable(2, name="x_local", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)
    z = LocalVariable(name="z_local")
    f = weighted_log_sum_exp(x, y) + z

    G = saddle_min(f, [x], [cp.sum(x) == 1])
    assert (
        not G.is_dsp()
    )  # z is a local variable appearing in the objective, and not listed as a convex variable


def test_saddle_extremum_missing_local():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local1", nonneg=True)
    z = cp.Variable(name="z")
    f = weighted_log_sum_exp(x, y_local) + z

    assert f.is_dsp()

    F = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    assert F.is_dsp()

    # max
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local2", nonneg=True)
    f = weighted_log_sum_exp(x, y_local) + cp.log(z)
    F = saddle_max(f, [y_local], [cp.sum(y_local) == 1])

    assert f.is_dsp()
    assert not F.is_dsp()

    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local2", nonneg=True)
    f = weighted_log_sum_exp(x, y_local) + cp.exp(z)
    F = saddle_max(f, [y_local], [cp.sum(y_local) == 1])

    assert f.is_dsp()
    assert F.is_dsp()


def test_bad_other():
    x_local = LocalVariable(2, name="x_local", nonneg=True)
    y = cp.Variable(2, name="y_new", nonneg=True)
    z = cp.Variable(name="z")
    f = weighted_log_sum_exp(x_local, y) + cp.log(z)

    F = saddle_min(f, [x_local], [cp.sum(x_local) == 1])
    assert f.is_dsp()
    assert F.is_dsp()

    x_local = LocalVariable(2, name="x_local", nonneg=True)
    f = weighted_log_sum_exp(x_local, y) + cp.exp(z)
    F = saddle_min(f, [x_local], [cp.sum(x_local) == 1])
    assert f.is_dsp()
    assert not F.is_dsp()


def test_bad_curvatures():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local", nonneg=True)
    y1_local = LocalVariable(name="y1_local", nonneg=True)
    z = cp.Variable(name="z")
    f = weighted_log_sum_exp(x, y_local) + z

    F = saddle_max(f, [y_local, y1_local], [cp.sum(y_local) == y1_local, y1_local == 1])
    assert f.is_dsp()
    assert F.is_dsp()


def test_affine_parts():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local", nonneg=True)
    z = cp.Variable(name="z")
    f = weighted_log_sum_exp(x, y_local) + z
    assert f.is_dsp()

    F = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    assert F.is_dsp()
    assert F.convex_vars == {x, z}

    y_local = LocalVariable(2, name="y_local", nonneg=True)
    z_local = LocalVariable(name="z_local")
    f = weighted_log_sum_exp(x, y_local) + z_local
    assert f.is_dsp()

    F = saddle_max(f, [y_local], [cp.sum(y_local) == 1])
    assert not F.is_dsp()

    y_local = LocalVariable(2, name="y_local", nonneg=True)
    z_local = LocalVariable(name="z_local")
    f = weighted_log_sum_exp(x, y_local) + z_local
    F = saddle_max(f, [y_local, z_local], [cp.sum(y_local) == 1])
    assert F.is_dsp()


def test_saddle_extremum_non_dcp_constraint():
    x = cp.Variable(2, name="x", nonneg=True)
    y_local = LocalVariable(2, name="y_local", nonneg=True)

    f = weighted_log_sum_exp(x, y_local)
    F = saddle_max(f, [y_local], [cp.sum(cp.abs(y_local)) == 1])
    assert not F.is_dsp()

    with pytest.raises(cp.DCPError):
        cp.Problem(cp.Minimize(F), [x == 0]).solve()
