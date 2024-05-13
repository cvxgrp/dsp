import cvxpy as cp
import numpy as np
import pytest

import dsp
from dsp import (
    LocalVariable,
    MinimizeMaximize,
    SaddlePointProblem,
    saddle_inner,
    saddle_min,
    saddle_quad_form,
)


@pytest.mark.skipif(cp.MOSEK not in cp.installed_solvers(), reason="MOSEK not installed")
def test_robust_bond():
    C = np.loadtxt("tests/example_data/robust_bond_portfolio/C.csv")
    p = np.loadtxt("tests/example_data/robust_bond_portfolio/p.csv")
    w_bar = np.loadtxt("tests/example_data/robust_bond_portfolio/target_weights.csv")
    y_nom = np.loadtxt("tests/example_data/robust_bond_portfolio/y_nominal.csv")
    h_mkt = (w_bar * 100) / p

    # Constants and parameters
    n, T = C.shape
    delta_max, kappa, omega = 0.02, 0.9, 1e-6
    B = 100
    V_lim = 90

    # Creating variables
    h = cp.Variable(n, nonneg=True)

    delta = LocalVariable(T)
    y = y_nom + delta

    # Objective
    phi = 0.5 * cp.norm1(cp.multiply(h, p) - cp.multiply(h_mkt, p))

    # Creating saddle min function
    V = 0
    for i in range(n):
        t_plus_1 = np.arange(T) + 1  # Account for zero-indexing
        V += saddle_inner(cp.exp(cp.multiply(-t_plus_1, y)), h[i] * C[i])

    Y = [
        cp.norm_inf(delta) <= delta_max,
        cp.norm1(delta) <= kappa,
        cp.sum_squares(delta[1:] - delta[:-1]) <= omega,
    ]

    V_wc = saddle_min(V, Y)

    # Creating and solving the problem
    problem = cp.Problem(cp.Minimize(phi), [h @ p == B, V_wc >= V_lim])
    problem.solve()  # 15.32

    assert problem.status == cp.OPTIMAL

    plotting = False

    if plotting:
        import matplotlib.pyplot as plt
        import pandas as pd

        # added tentative plotting, please review
        inds = (
            ((C > 0).astype(int) * np.arange(C.shape[1])).argmax(axis=1).argsort()
        )  # sort by maturity
        df = pd.DataFrame({"h": h.value[inds], "h_mkt": (h_mkt)[inds]})
        df.plot(kind="bar")
        plt.xlabel("Bond index (increasing maturity)")
        plt.ylabel("Holdings")
        plt.savefig("tests/example_data/robust_bond.pdf")
        # plt.show()

        df = pd.DataFrame({"y_nom": y_nom, "y_wc": y.value})
        df.plot()
        plt.xlabel(r"$t$")
        plt.ylabel("yield")
        plt.savefig("tests/example_data/yield.pdf")
        # plt.show()



def test_robust_bond_full():
    Cash_flows = np.loadtxt("data/C.csv")
    target_weights = np.loadtxt("data/target_weights.csv")
    F = np.loadtxt("data/F.csv")
    P_hat_inv = np.loadtxt("data/sigma_joint_inverse.csv")
    x_hat = np.loadtxt("data/mu_joint.csv")
    prices = np.loadtxt("data/p.csv")


    alpha_modest = 0.5
    ellipse_callable = partial(
        in_ellipse, P_hat_inv=P_hat_inv, x_hat=x_hat, F=F, alpha=alpha_modest
    )
    U = UncertaintySet(y_s_const=[ellipse_callable])

    n, T = Cash_flows.shape
    y_tilde = cp.Variable(T, name="y_tilde")
    s_tilde = cp.Variable(n, name="s_tilde")
    h = cp.Variable(n, name="h")

    # Objective
    phi = cp.norm1(target_weights - cp.multiply(h, prices)) / 2

    weight_map, y_tilde_map, s_tilde_map, exponent_offset = get_cashflow_mapping(Cash_flows)
    n_exponents = len(exponent_offset)

    exponents = cp.Variable(n_exponents, name="exponents")
    weights = cp.Variable(n_exponents, nonneg=True, name="weights")

    exponent_constraints = [
        *U.contains((y_tilde, s_tilde)),
        exponents == y_tilde_map @ y_tilde + s_tilde_map @ s_tilde + exponent_offset,
    ]

    B = 1

    weight_constraints = [h >= 0, h @ prices == B, weights == weight_map @ h]

    res = []
    for lambda_val in np.linspace(1, 20.0, 41):
        try:
            Delta = dsp.weighted_log_sum_exp(exponents, weights)
            saddle_problem = dsp.SaddlePointProblem(
                dsp.MinimizeMaximize(phi - lambda_val * Delta),
                weight_constraints + exponent_constraints,
            )
            saddle_problem.solve(solver=cp.MOSEK)
            assert saddle_problem.status == cp.OPTIMAL

            print(f"{lambda_val:.2f}, " f"{phi.value=:.2f}")
            res.append(
                {
                    "lambda": lambda_val,
                    "phi": phi.value,
                    "weights": h.value * prices,
                }
            )
        except Exception as e:
            print(f"Failed for lambda = {lambda_val}, {e}")
        break

def get_cashflow_mapping(
    Cash_flows: np.ndarray,
) -> tuple[sp.coo_matrix, sp.coo_matrix, sp.coo_matrix, np.ndarray]:
    n, T = Cash_flows.shape

    sprs_C = sp.coo_matrix(Cash_flows)
    nnz = sprs_C.nnz

    h_inds, t_inds, C_vals = sprs_C.row, sprs_C.col, sprs_C.data

    weight_map = sp.coo_matrix((np.ones(nnz), (np.arange(nnz), h_inds)), shape=(nnz, n))

    neg_t_plus_1 = -(t_inds + 1)

    y_tilde_map = sp.coo_matrix((neg_t_plus_1, (np.arange(nnz), t_inds)), shape=(nnz, T))
    s_tilde_map = sp.coo_matrix((neg_t_plus_1, (np.arange(nnz), h_inds)), shape=(nnz, n))
    exponent_offset = np.log(C_vals)

    return weight_map, y_tilde_map, s_tilde_map, exponent_offset

@dataclass
class UncertaintySet:
    y_const: list[Callable] | None = None
    s_const: list[Callable] | None = None
    y_s_const: list[Callable] | None = None

    def __post_init__(self) -> None:
        assert any([self.y_const is not None, self.s_const is not None, self.y_s_const is not None])

    def contains(self, ys: tuple[cp.Variable, cp.Variable]) -> list[Constraint]:
        y_tilde, s_tilde = ys
        c_constraints = [c(y_tilde) for c in self.y_const] if self.y_const is not None else []
        s_constraints = [c(s_tilde) for c in self.s_const] if self.s_const is not None else []
        y_s_constraints = (
            [c(y_tilde, s_tilde) for c in self.y_s_const] if self.y_s_const is not None else []
        )
        all_constraints = c_constraints + s_constraints + y_s_constraints
        all_iterable_constraints = [make_iterable(x) for x in all_constraints]
        flat_constraints = list(itertools.chain.from_iterable(all_iterable_constraints))
        return flat_constraints

    def has_maximum_element(self, ys: tuple[cp.Variable, cp.Variable]) -> bool:
        y_tilde, s_tilde = ys
        T, n = y_tilde.shape[0], s_tilde.shape[0]

        y_tilde_componentwise_max = np.zeros(T)
        for t in range(T):
            problem = cp.Problem(cp.Maximize(y_tilde[t]), self.contains((y_tilde, s_tilde)))
            problem.solve()
            y_tilde_componentwise_max[t] = y_tilde.value[t]

        s_tilde_componentwise_max = np.zeros(n)
        for i in range(n):
            problem = cp.Problem(cp.Maximize(s_tilde[i]), self.contains((y_tilde, s_tilde)))
            problem.solve()
            s_tilde_componentwise_max[i] = s_tilde.value[i]

        problem = cp.Problem(
            cp.Maximize(0),
            self.contains((y_tilde, s_tilde))
            + [y_tilde == y_tilde_componentwise_max, s_tilde == s_tilde_componentwise_max],
        )

        problem.solve()
        assert problem.status in {cp.OPTIMAL, cp.INFEASIBLE}
        return problem.status != cp.INFEASIBLE

    def get_A_c(self, ys: tuple[cp.Variable, cp.Variable]) -> tuple[np.ndarray, np.ndarray]:
        y_tilde, s_tilde = ys
        prob = cp.Problem(cp.Minimize(0), self.contains((y_tilde, s_tilde)))
        problem_data = prob.get_problem_data(solver=cp.SCS)
        return problem_data[0]["A"].toarray(), problem_data[0]["b"]


def make_iterable(x: Any | Iterable[Any]) -> Iterable[Any]:
    if not isinstance(x, Iterable):
        return [x]
    else:
        return x


def in_ellipse(
    y_tilde: cp.Variable,
    s_tilde: cp.Variable,
    P_hat_inv: np.ndarray,
    x_hat: np.ndarray,
    F: np.ndarray,
    alpha: float = 0.05,
) -> list[Constraint]:
    rhs = chi2.ppf(1 - alpha, df=P_hat_inv.shape[0])
    x = cp.Variable(P_hat_inv.shape[0])
    return [cp.quad_form((x - x_hat), P_hat_inv) <= rhs, F @ x == cp.hstack([y_tilde, s_tilde])]


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
    problem.solve(solver=cp.SCS)  # 0.076

    nominal_objective = cp.Maximize(w @ mu - gamma * cp.quad_form(w, Sigma))

    robust_wc_utility = G.value
    robust_non_wc_utility = nominal_objective.value

    assert problem.status == cp.OPTIMAL

    # Creating and solving the problem without robustness
    problem = cp.Problem(nominal_objective, [cp.sum(w) == 1])
    problem.solve(solver=cp.SCS)  # 0.065
    assert problem.status == cp.OPTIMAL

    nominal_wc_utility = G.value
    nominal_non_wc_utility = nominal_objective.value

    assert robust_wc_utility > nominal_wc_utility
    assert robust_non_wc_utility < nominal_non_wc_utility


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


@pytest.mark.skipif(cp.MOSEK not in cp.installed_solvers(), reason="MOSEK not installed")
def test_robust_model_fitting():
    # Load data
    data = np.loadtxt("tests/example_data/robust_model_fitting/data.csv", delimiter=",", skiprows=1)

    p = data[:, 0]
    p = (p - np.mean(p)) / np.std(p)

    A = data[:, 1:]
    A = (A - np.mean(A, axis=0)) / np.std(A, axis=0)
    A = np.hstack((np.ones((len(p), 1)), A))

    # Constants
    m, n = A.shape
    k = 0.2 * m
    eta = 0.1

    # Creating variables
    theta = cp.Variable(n)
    weights = cp.Variable(m, nonneg=True)

    # Defining the loss function and the weight constraints
    loss = cp.square(A @ theta - p)
    regularizer = eta * cp.norm1(theta)
    objective = MinimizeMaximize(saddle_inner(loss, weights) + regularizer)
    constraints = [cp.sum(weights) == k, weights <= 1]

    # Creating and solving the problem
    problem = SaddlePointProblem(objective, constraints)
    problem.solve()  # XXX
    assert problem.status == cp.OPTIMAL

    robust_obj_robust_weights = problem.value

    # OLS problem
    ols_objective = cp.Minimize(cp.sum_squares(A @ theta - p) + regularizer)

    ols_obj_robust_weights = ols_objective.value

    problem = cp.Problem(ols_objective)
    problem.solve()
    assert problem.status == cp.OPTIMAL

    ols_obj_ols_weights = problem.value
    robust_obj_ols_weights = np.sum(np.sort(np.square(A @ theta.value - p))[-int(k) :])

    assert ols_obj_ols_weights < ols_obj_robust_weights
    assert robust_obj_robust_weights < robust_obj_ols_weights

    # Using sum_largest
    loss = cp.sum_largest(cp.square(A @ theta - p), k)
    objective = cp.Minimize(loss + regularizer)
    problem = cp.Problem(objective)
    problem.solve()
    assert problem.status == cp.OPTIMAL

    assert np.isclose(problem.value, robust_obj_robust_weights)


def test_svm():
    import pandas as pd

    one_hot = True  # Use one-hot encoding for pclass and 5 age bins
    intercept = False and one_hot  # Include intercept in model
    bins = 3  # Number of age bins
    train_port = ["Q"]  # C, Q, S: the port to use for training
    without_train = True  # Dont include training data in eval

    df = pd.read_csv("tests/example_data/robust_model_fitting/titanic.csv")

    df["sex"] = df["sex"] == "male"

    df["intercept"] = 1

    features = ["pclass", "sex", "age"] + (["intercept"] if intercept else [])

    df = df.dropna(subset=features)

    class_hot = pd.get_dummies(df["pclass"], prefix="pclass")

    age_bins = pd.get_dummies(pd.cut(df["age"], bins), prefix="age")

    df = pd.concat([df, class_hot, age_bins], axis=1)

    df["survived_bool"] = df["survived"].copy()
    df["survived"] = 2 * df["survived"] - 1

    if one_hot:
        features = ["sex"] + age_bins.columns.tolist() + class_hot.columns.tolist()

    df_short = df[df.embarked.isin(train_port)]

    y_train = df_short["survived"].values.astype(float)
    A_train = df_short[features].values.astype(float)

    surv = df_short["survived_bool"].values.astype(float)

    # Constants and parameters
    m, n = A_train.shape
    inds_0 = surv == 0
    inds_1 = surv == 1
    eta = 0.05

    # Creating variables
    theta = cp.Variable(n)
    beta_0 = cp.Variable()
    weights = cp.Variable(m, nonneg=True)
    surv_weight_0 = cp.Variable()
    surv_weight_1 = cp.Variable()

    # Defining the loss function and the weight constraints
    y_hat = A_train @ theta + beta_0
    loss = cp.pos(1 - cp.multiply(y_train, y_hat))
    objective = MinimizeMaximize(saddle_inner(loss, weights) + eta * cp.sum_squares(theta))

    constraints = [
        cp.sum(weights) == 1,
        0.408 - 0.05 <= weights @ surv,
        weights @ surv <= 0.408 + 0.05,
        weights[inds_0] == surv_weight_0,
        weights[inds_1] == surv_weight_1,
    ]

    # Creating and solving the problem
    problem = SaddlePointProblem(objective, constraints)
    problem.solve()

    assert problem.status == cp.OPTIMAL
    robust_theta = theta.value

    # nominal SVM problem
    const_weights = np.ones(m)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(loss, const_weights)) + eta * cp.sum_squares(theta))
    )
    problem.solve()
    assert problem.status == cp.OPTIMAL
    nominal_theta = theta.value

    # Full sample
    if without_train:
        y = df[~df.embarked.isin(train_port)]["survived"].values.astype(float)
        A = df[~df.embarked.isin(train_port)][features].values.astype(float)
    else:
        y = df["survived"].values.astype(float)
        A = df[features].values.astype(float)

    print_results = False
    if print_results:
        print("Train accuracy nom.: ", accuracy(A_train @ nominal_theta, y_train))
        print("Train accuracy rob.: ", accuracy(A_train @ robust_theta, y_train))
        print("Test accuracy nom.: ", accuracy(A @ nominal_theta, y))
        print("Test accuracy rob.: ", accuracy(A @ robust_theta, y))

        print("-" * 80)

        print("Train loss nom.: ", avg_svm_loss_numpy(A_train @ nominal_theta, y_train))
        print("Train loss rob.: ", avg_svm_loss_numpy(A_train @ robust_theta, y_train))
        print("Test loss nom.: ", avg_svm_loss_numpy(A @ nominal_theta, y))
        print("Test loss rob.: ", avg_svm_loss_numpy(A @ robust_theta, y))

        print("-" * 80)

        print(
            "\n".join(
                ["robust theta"]
                + [
                    f"{name:20s}: {val:>10.4f}"
                    for name, val in zip(df_short[features].columns, robust_theta)
                ]
            )
        )
        print()
        print(
            "\n".join(
                ["nom theta"]
                + [
                    f"{name:20s}: {val:>10.4f}"
                    for name, val in zip(df_short[features].columns, nominal_theta)
                ]
            )
        )


def accuracy(scores, labels):
    scores[scores > 0] = 1
    scores[scores <= -0] = -1
    return np.mean(scores == labels)


def avg_svm_loss_numpy(y_hat, y):
    losses = np.maximum(0, 1 - y * y_hat)
    return np.mean(losses)
