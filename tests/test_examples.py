import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import LocalVariable, SaddlePointProblem, saddle_inner, saddle_min
from dsp.atoms import saddle_quad_form


def test_robust_bond():

    C = np.loadtxt("tests/example_data/robust_bond_portfolio/C.csv")
    p = np.loadtxt("tests/example_data/robust_bond_portfolio/p.csv")
    w_bar = np.loadtxt("tests/example_data/robust_bond_portfolio/target_weights.csv")
    y_nominal = np.loadtxt("tests/example_data/robust_bond_portfolio/y_nominal.csv")

    # Constants and parameters
    n, T = C.shape
    delta_max, kappa, omega = 0.05, 0.5, 0.1
    B = 100
    V_limit = 80

    # Creating variables
    h = cp.Variable(n, nonneg=True)
    w = cp.multiply(h, p) / B

    delta = LocalVariable(T)
    y = y_nominal + delta

    # Objective
    phi = 0.5 * cp.norm1(w - w_bar)

    # Creating saddle min function
    V = 0
    for i in range(n):
        t_plus_1 = np.arange(T) + 1  # Account for zero-indexing
        V += saddle_inner(cp.exp(cp.multiply(-t_plus_1, y)), h[i] * C[i])

    Y = [
        cp.norm_inf(delta) <= delta_max,
        cp.norm1(delta) <= kappa,
        cp.norm2(delta[1:] - delta[:-1]) <= omega,
    ]

    V_wc = saddle_min(V, Y)

    # Creating and solving the problem
    problem = cp.Problem(cp.Minimize(phi), [cp.sum(w) == 1, V_wc >= V_limit])
    problem.solve()  # 0.185

    assert problem.status == cp.OPTIMAL


def test_robust_markowitz():

    returns = np.loadtxt("tests/example_data/robust_portfolio_selection/ff_data.csv", delimiter=",")

    mu = np.mean(returns, axis=0)
    Sigma = np.cov(returns, rowvar=False)

    # Constants and parameters
    n = len(mu)
    rho, eta, gamma = 0.2, 0.2, 1

    # Creating variables
    w = cp.Variable(n, nonneg=True)

    delta_loc = LocalVariable(n)
    Sigma_perturbed = LocalVariable((n, n), PSD=True)
    Delta_loc = LocalVariable((n, n))

    # Creating saddle min function
    f = w @ mu + saddle_inner(delta_loc, w) - gamma * saddle_quad_form(w, Sigma_perturbed)

    Sigma_diag = Sigma.diagonal()
    local_constraints = [
        cp.abs(delta_loc) <= rho,
        Sigma_perturbed == Sigma + Delta_loc,
        cp.abs(Delta_loc) <= eta * np.sqrt(np.outer(Sigma_diag, Sigma_diag)),
    ]

    G = saddle_min(f, local_constraints)

    # Creating and solving the problem
    problem = cp.Problem(cp.Maximize(G), [cp.sum(w) == 1])
    problem.solve()  # 0.076

    assert problem.status == cp.OPTIMAL


def test_create_markowitz_plot():

    returns = np.loadtxt("tests/example_data/robust_portfolio_selection/ff_data.csv", delimiter=",")

    mu = np.mean(returns, axis=0)
    Sigma = np.cov(returns, rowvar=False)

    # Constants and parameters
    n = len(mu)
    rho, eta = 0.2, 0.2

    # Creating variables
    w = cp.Variable(n, nonneg=True)

    plot_data = []
    for gamma in np.linspace(0, 10, 20):
        delta_loc = LocalVariable(n)
        Sigma_perturbed = LocalVariable((n, n), PSD=True)
        Delta_loc = LocalVariable((n, n))

        # Creating saddle min function
        f = w @ mu + saddle_inner(delta_loc, w) - gamma * saddle_quad_form(w, Sigma_perturbed)

        Sigma_diag = Sigma.diagonal()
        local_constraints = [
            cp.abs(delta_loc) <= rho,
            Sigma_perturbed == Sigma + Delta_loc,
            cp.abs(Delta_loc) <= eta * np.sqrt(np.outer(Sigma_diag, Sigma_diag)),
        ]

        G = saddle_min(f, local_constraints)

        # Creating and solving the problem
        problem = cp.Problem(cp.Maximize(G), [cp.sum(w) == 1])
        problem.solve()  # 0.076
        assert problem.status == cp.OPTIMAL

        robust_utility = problem.value

        # Creating and solving the problem without robustness
        problem = cp.Problem(cp.Maximize(w @ mu - gamma * cp.quad_form(w, Sigma)), [cp.sum(w) == 1])
        problem.solve()
        assert problem.status == cp.OPTIMAL

        utility = G.value

        plot_data.append((gamma, utility, robust_utility))

    plot_data = np.array(plot_data)
    assert (plot_data[:, 2] - plot_data[:, 1] >= -1e-6).all()
    assert (plot_data[:, 2] - plot_data[:, 1]).max() >= 0.1

    plot = False
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3))
        plt.plot(plot_data[:, 0], plot_data[:, 2], label="Robust utility")
        plt.plot(plot_data[:, 0], plot_data[:, 1], label="Non-robust utility")
        plt.xlabel(r"$\gamma$")
        plt.ylabel("Utility")
        plt.legend()
        plt.tight_layout()
        plt.savefig("tests/example_data/robust_portfolio_selection/robust_markowitz_plot.pdf")
        plt.show()


def test_arbitrage():

    p = np.ones(2)

    V = np.array([[1.5, 0.5], [0.5, 1.5]])

    x = cp.Variable(2, name="x")

    # No arbitrage
    obj = cp.Minimize(p @ x)
    constraints = [V @ x >= 0]
    no_arb_problem = cp.Problem(obj, constraints)
    no_arb_problem.solve()
    assert np.isclose(no_arb_problem.value, 0)

    # Arbitrage
    p[0] = 0.1  # arb possible for p[0] < 1/3
    obj = cp.Minimize(p @ x)
    constraints = [V @ x >= 0]
    arb_problem = cp.Problem(obj, constraints)
    arb_problem.solve()
    assert arb_problem.status == cp.UNBOUNDED

    # Robust arbitrage - small perturbation
    p_tilde = cp.Variable(2, name="p_tilde")
    obj = dsp.MinimizeMaximize(dsp.inner(x, p_tilde))
    constraints = [V @ x >= 0, cp.abs(p_tilde - p) <= 0.1]
    saddle_problem = dsp.SaddlePointProblem(obj, constraints)
    with pytest.raises(AssertionError):
        saddle_problem.solve()

    # Robust arbitrage - large perturbation
    constraints = [V @ x >= 0, cp.abs(p_tilde - p) <= 1]
    saddle_problem = dsp.SaddlePointProblem(obj, constraints)
    saddle_problem.solve()
    assert saddle_problem.status == cp.OPTIMAL
    assert np.isclose(saddle_problem.value, 0)


def test_robust_model_fitting():
    import pandas as pd

    df = pd.read_stata("tests/example_data/robust_model_fitting/analysis_data_AEJ_pub.dta")
    df = df[df.survey == "Endline"]

    Y_ik = df.cluster
    T_ik = df.Treatment

    # Constants
    n = len(Y_ik)

    # Create variables
    beta_0 = cp.Variable()
    beta = cp.Variable()

    # OLS problem
    objective = cp.Minimize(cp.sum_squares(beta_0 + beta * T_ik - Y_ik))
    problem = cp.Problem(objective)
    problem.solve()
    assert problem.status == cp.OPTIMAL

    # Robust model fitting
    beta_promoted = cp.Variable(n)  # TODO: bugfix to handle promoted variables correctly
    loss = cp.square(beta_0 + cp.multiply(beta_promoted, T_ik) - Y_ik)
    weights = cp.Variable(n, nonneg=True)

    objective = dsp.MinimizeMaximize(saddle_inner(loss, weights))

    problem = SaddlePointProblem(
        objective, [cp.sum(weights) == n - 1, beta_promoted == beta, 0 * beta_0 == 0]
    )  # TODO: remove requirement for dummy constraint
    problem.solve(verbose=True)
    assert problem.status == cp.OPTIMAL
