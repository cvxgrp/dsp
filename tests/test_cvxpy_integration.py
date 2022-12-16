import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp.atoms import inner, saddle_max, saddle_min, weighted_log_sum_exp
from dsp.cvxpy_integration import extend_cone_canon_methods
from dsp.local import LocalVariable, LocalVariableError
from dsp.parser import DSPError
from dsp.problem import MinimizeMaximize, SaddlePointProblem, is_dsp

extend_cone_canon_methods()


def test_semi_infinite_matrix():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    x_d = LocalVariable(2, name="x_d", nonneg=True)
    y_d = LocalVariable(2, name="y_d", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y)

    # Saddle problem
    saddle_obj = MinimizeMaximize(inner_expr)
    saddle_problem = SaddlePointProblem(saddle_obj, [cp.sum(x) == 1, cp.sum(y) == 1])
    saddle_problem.solve()

    assert np.isclose(saddle_problem.value, 2.0, atol=1e-4)
    assert np.allclose(x.value, [1, 0], atol=1e-4)
    assert np.allclose(y.value, [0, 1], atol=1e-4)

    # Saddle max problem

    inner_expr = inner(x, A @ y_d)

    obj = cp.Minimize(saddle_max(inner_expr, [y_d], [cp.sum(y_d) == 1]))
    constraints = [cp.sum(x) == 1]

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)

    assert np.isclose(problem.value, 2.0, atol=1e-4)

    # Saddle min problem

    inner_expr = inner(x_d, A @ y)

    f = saddle_min(inner_expr, [x_d], [cp.sum(x_d) == 1])
    obj = cp.Maximize(f)
    constraints = [cp.sum(y) == 1]

    problem = cp.Problem(obj, constraints)
    problem.solve(solver=cp.SCS)
    assert np.isclose(problem.value, 2.0, atol=1e-4)
    assert np.allclose(x.value, [1, 0], atol=1e-4)
    assert np.allclose(y.value, [0, 1], atol=1e-4)


def test_dcp_concave_max_and_dummy():
    x = cp.Variable(2, name="x", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y)

    with pytest.raises(LocalVariableError):
        f_max = saddle_max(inner_expr, [y], [cp.sum(y) == 1])

    y2 = LocalVariable(2, name="y", nonneg=True)

    A = np.array([[1, 2], [3, 4]])
    inner_expr = inner(x, A @ y2)

    f_max = saddle_max(inner_expr, [y2], [cp.sum(y2) == 1])

    obj = cp.Maximize(f_max)
    assert not obj.is_dcp()


def test_semi_infinite_expr():
    x = cp.Variable(2, name="x", nonneg=True)
    y_dummy = LocalVariable(2, name="y_dummy", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    wlse = weighted_log_sum_exp(x, y)

    # Trying to create a saddle_max with a variable for y (instead of a dummy)
    with pytest.raises(LocalVariableError):
        sup_y_f = saddle_max(2 * wlse + y[1] + cp.exp(x[1]), [y], [y <= 1])

    wlse = weighted_log_sum_exp(x, y_dummy)

    # creating a valid saddle_max with a dummy variable
    sup_y_f = saddle_max(2 * wlse + y_dummy[1] + cp.exp(x[1]), [y_dummy], [y_dummy <= 1])
    assert sup_y_f.is_dcp()
    assert sup_y_f.is_dsp()

    # trying to use the same dummy variable in a new SE raises an error
    with pytest.raises(LocalVariableError):
        sup_y_f_reused_local = saddle_max(2 * wlse + 2 * cp.sum(y_dummy), [y_dummy], [y_dummy <= 1])

    # trying to get the value of the saddle_max before x has a value returns None
    assert sup_y_f.numeric(values=np.ones(1)) is None

    # trying to get the value of y_dummy before x has a value returns None
    assert y_dummy.value is None

    x.value = np.ones(2)

    assert np.allclose(y_dummy.value, np.ones(2))

    val = sup_y_f.numeric(values=np.ones(1))
    assert np.isclose(val, 2 * np.log(2 * np.e) + 1 + np.e)


def test_multiple_dummies():
    x = cp.Variable(2, name="x", nonneg=True)
    y1_local = LocalVariable(name="y1_local", nonneg=True)
    y2_local = LocalVariable(name="y2_local", nonneg=True)

    y = cp.Variable(name="y", nonneg=True)

    wlse = weighted_log_sum_exp(x, cp.hstack([y1_local, y2_local]))

    # only one dummy variable is specified. This raises a DSP error since not
    # all concave variables are specified, and not a local variable error,
    # because all of the concave variables are specified as local variables.
    sup_y_f = saddle_max(2 * wlse + y1_local + cp.exp(x[1]), [y1_local], [y1_local <= 1])
    assert not sup_y_f.is_dcp()
    assert not sup_y_f.is_dsp()

    # trying a mix of dummy and variable. This raises a local variable error on
    # construction.
    with pytest.raises(LocalVariableError):
        sup_y_f = saddle_max(2 * wlse + y + cp.exp(x[1]), [y1_local, y], [y1_local <= 1, y <= 1])

    sup_y_f = saddle_max(
        2 * wlse + y1_local + cp.exp(x[1]), [y1_local, y2_local], [y1_local <= 1, y2_local <= 1]
    )

    assert sup_y_f.numeric(values=np.ones(1)) is None
    assert y1_local.value is None
    assert y2_local.value is None

    x.value = np.ones(2)

    assert np.allclose(y1_local.value, np.ones(1))
    assert np.allclose(y2_local.value, np.ones(1))


def test_trivial_se():
    x = cp.Variable(name="x", nonneg=True)

    f = cp.exp(x)

    # trivial saddle max with no dummy variable
    F = saddle_max(f, [], [])

    prob = cp.Problem(cp.Minimize(F), [x >= 1])
    prob.solve()

    assert np.isclose(prob.value, np.e)

    y = cp.Variable(name="y", nonneg=True)
    f = cp.log(y)

    # trivial saddle min with no dummy variable
    F = saddle_min(f, [], [])

    prob = cp.Problem(cp.Maximize(F), [y <= 1])
    prob.solve()

    assert np.isclose(prob.value, 0)


def test_nested_saddle():
    x = cp.Variable(2, name="x", nonneg=True)
    y_1 = LocalVariable(2, name="y_1", nonneg=True)
    y_2 = LocalVariable(name="y_2", nonneg=True)

    f = weighted_log_sum_exp(x, y_1)

    sup_y_f = saddle_max(f, [y_1], [y_1 <= 1])

    g = weighted_log_sum_exp(sup_y_f, y_1[1])
    with pytest.raises(LocalVariableError):  # adding a local variable to multiple SEs
        sup_y_g = saddle_max(g, [y_1], [y_1 <= 1])

    g = weighted_log_sum_exp(sup_y_f, y_2)
    sup_y_g = saddle_max(g, [y_2], [y_2 <= 1])

    prob = cp.Problem(cp.Minimize(sup_y_g), [x >= 1])
    prob.solve()

    assert np.isclose(prob.value, np.log(2 * np.e))

    # TODO: prevent local variables from appearing in dcp/dsp problems


def test_saddle_max():
    x1 = LocalVariable(name="x1", nonneg=True)
    x2 = LocalVariable(name="x2", nonneg=True)
    y = cp.Variable(2, name="y", nonneg=True)

    x = cp.Variable(name="x", nonneg=True)

    wlse = weighted_log_sum_exp(cp.hstack([x1, x2]), y)

    # only one dummy variable is specified
    inf_x_f = saddle_min(2 * wlse + x1 + cp.log(y[1]), [x1], [x1 >= 1])
    assert not inf_x_f.is_dcp()
    assert not inf_x_f.is_dsp()

    # trying a mix of dummy and variable
    with pytest.raises(LocalVariableError):
        inf_x_f = saddle_min(2 * wlse + x + cp.log(y[1]), [x1, x], [x1 >= 1, x >= 1])

    inf_x_f = saddle_min(2 * wlse + x1 + cp.log(y[1]), [x1, x2], [x1 >= 1, x2 >= 1])

    # TODO: This is not working yet
    assert inf_x_f.numeric(values=np.ones(2)) is None
    assert x1.value is None
    assert x2.value is None

    y.value = np.ones(2)

    assert np.isclose(x1.value, 1)
    assert np.isclose(x2.value, 1)


def test_non_dsp_saddle():
    x = cp.Variable(2, name="x", nonneg=True)
    y = LocalVariable(2, name="y", nonneg=True)

    f = weighted_log_sum_exp(x, y) + cp.exp(y[1])

    sup_y_f = saddle_max(f, [y], [y <= 1])

    assert not sup_y_f.is_dsp()

    prob = cp.Problem(cp.Minimize(sup_y_f), [x >= 1])

    assert not is_dsp(prob)
    assert not prob.is_dcp()

    # currently a DCPError is raised. Do we want a DCPError or a DSPError?
    with pytest.raises(cp.error.DCPError):
        prob.solve()


def test_dsp_canon_error():
    x = cp.Variable(2, name="x", nonneg=True)
    y = LocalVariable(2, name="y", nonneg=True)

    f = weighted_log_sum_exp(x, y) + cp.exp(y[1])

    sup_y_f = saddle_max(f, [y], [y <= 1])

    prob = cp.Problem(cp.Minimize(sup_y_f), [x >= 1])

    with pytest.raises(cp.error.DCPError):
        prob.solve()
