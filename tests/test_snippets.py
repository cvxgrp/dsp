from dsp import *  # notational convenience


def test_creating_saddle_function_and_solve_matrix_game():

    # ## Begin snippet 1

    import cvxpy as cp
    import numpy as np

    # from dsp import * # notational convenience

    x = cp.Variable(2)
    y = cp.Variable(2)
    C = np.array([[1, 2], [3, 1]])

    f = inner(x, C @ y)

    f.is_dsp()  # True

    f.convex_variables()  # [x]
    f.concave_variables()  # [y]
    f.affine_variables()  # []

    # ## End snippet 1

    assert f.is_dsp()
    assert f.convex_variables() == [x]
    assert f.concave_variables() == [y]
    assert f.affine_variables() == []

    # ## Begin snippet 2

    obj = MinimizeMaximize(f)
    constraints = [x >= 0, cp.sum(x) == 1, y >= 0, cp.sum(y) == 1]
    prob = SaddlePointProblem(obj, constraints)

    prob.is_dsp()  # True
    prob.convex_variables()  # [x]
    prob.concave_variables()  # [y]
    prob.affine_variables()  # []

    prob.solve()  # solves the problem
    prob.value  # 1.6666666666666667
    x.value  # array([0.66666667, 0.33333333])
    y.value  # array([0.33333333, 0.66666667])

    # ## End snippet 2

    assert prob.is_dsp()
    assert prob.convex_variables() == [x]
    assert prob.concave_variables() == [y]
    assert prob.affine_variables() == []
    assert np.isclose(prob.value, 1.6666666666666667)
    assert np.allclose(x.value, [0.66666667, 0.33333333])
    assert np.allclose(y.value, [0.33333333, 0.66666667])


def test_creating_a_saddle_max_and_solve_problem():

    import cvxpy as cp
    import numpy as np

    C = np.array([[1, 2], [3, 1]])

    # ## Begin snippet 3

    # Creating variables
    x = cp.Variable(2)

    # Creating local variables
    y_loc = LocalVariable(2)

    # Convex in x, concave in y_loc
    f = saddle_inner(x, C @ y_loc)

    # maximizes over y_loc
    G = saddle_max(f, [y_loc >= 0, cp.sum(y_loc) == 1])

    # ## End snippet 3

    # ## Begin snippet 4

    prob = cp.Problem(cp.Minimize(G), [x >= 0, cp.sum(x) == 1])

    prob.is_dsp()  # True

    prob.solve()  # solving the problem
    prob.value  # 1.6666666666666667
    x.value  # array([0.66666667, 0.33333333])

    # ## End snippet 4

    assert prob.is_dsp()
    assert np.isclose(prob.value, 1.6666666666666667)
    assert np.allclose(x.value, [0.66666667, 0.33333333])
