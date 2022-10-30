from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import SOC
from cvxpy.constraints import ExpCone
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective


@dataclass
class KRepresentation:
    f: cp.Variable
    t: cp.Variable
    u_or_Q: cp.Variable | sp.base
    x: cp.Variable
    y: cp.Variable
    constraints: list[Constraint]


def switch_convex_concave(K_in: KRepresentation) -> KRepresentation:
    # Turn phi(x,y) into \bar{phi}(\bar{x},\bar{y}) = -phi(x,y)
    # with \bar{x} = y, \bar{y} = x

    assert isinstance(K_in.u_or_Q, (cp.Variable, np.ndarray))

    var_list = [K_in.f,
                K_in.t,
                K_in.x]
    if isinstance(K_in.u_or_Q, cp.Variable):
        var_list.append(K_in.u_or_Q)

    var_to_mat_mapping, const_vec, cone_dims = get_cone_repr(K_in.constraints,
                                                             var_list
                                                             )
    u_bar = cp.Variable(len(const_vec))
    u_bar_const = add_cone_constraints(u_bar, cone_dims)

    if isinstance(K_in.u_or_Q, cp.Variable):
        assert var_to_mat_mapping['eta'].size == 0
        Q = var_to_mat_mapping[K_in.u_or_Q.id]
    else:
        Q = K_in.u_or_Q
        assert var_to_mat_mapping['eta'].shape == Q.shape

    P = var_to_mat_mapping[K_in.f.id]
    p = var_to_mat_mapping[K_in.t.id].flatten()
    R = var_to_mat_mapping[K_in.x.id]
    s = const_vec

    f_bar = cp.Variable(K_in.x.size, name='f_bar')
    t_bar = cp.Variable(name='t_bar')
    x_bar = K_in.y
    y_bar = K_in.x

    constraints = [
        f_bar == -R.T @ u_bar,
        t_bar == s @ u_bar,
        Q.T @ u_bar == 0,
        p @ u_bar + 1 == 0,
        P.T @ u_bar + x_bar == 0,
        *u_bar_const
    ]

    return KRepresentation(
        f=f_bar,
        t=t_bar,
        u_or_Q=u_bar,
        x=x_bar,
        y=y_bar,
        constraints=constraints
    )


def log_sum_exp_K_repr(x: cp.Variable, y: cp.Variable):
    # log sum_i y_i exp(x_i)
    # min. f, t, u    f @ y + t
    # subject to      f >= exp(x+u)
    #                 t >= -u-1

    assert len(x.shape) == 1
    assert x.shape == y.shape

    f = cp.Variable(y.size, name='f')
    t = cp.Variable(name='t')
    u = cp.Variable(name='u')

    constraints = [
        ExpCone(x + u, np.ones(f.size), f),
        t >= -u - 1
    ]

    return KRepresentation(
        f=f,
        t=t,
        u_or_Q=u,
        x=x,
        y=y,
        constraints=constraints
    )


def minimax_to_min(K: KRepresentation,
                   X_constraints: list[Constraint],
                   Y_constraints: list[Constraint]) -> (Objective, list[Constraint]):
    var_id_to_mat, e, cone_dims = get_cone_repr(Y_constraints, [K.y])
    lamb = cp.Variable(len(e))
    lamb_const = add_cone_constraints(lamb, cone_dims)

    C = var_id_to_mat[K.y.id]
    D = var_id_to_mat['eta']

    obj = cp.Minimize(e @ lamb + K.t)

    constraints = [
        *K.constraints,
        C.T @ lamb == K.f,
        *lamb_const,
        *X_constraints,
    ]
    if D.shape[1] > 0:
        constraints.append(D.T @ lamb == 0)

    return obj, constraints


def K_repr_y_Fx(F: cp.Expression, y: cp.Variable) -> KRepresentation:
    assert F.is_convex()
    assert len(F.variables()) == 1
    x = F.variables()[0]
    assert F.ndim == 1
    assert y.ndim == 1
    assert F.shape == y.shape
    # assert y.is_nonneg()

    f = cp.Variable(F.size, name='f_bilin')
    t = cp.Variable(name='t_bilin_y_Fx')

    constraints = [
        t == 0,
        f >= F
    ]

    var_to_mat_mapping, _, _, = get_cone_repr(constraints, [x, f, t])
    T_bar = var_to_mat_mapping['eta']

    return KRepresentation(
        f=f,
        t=t,
        u_or_Q=T_bar,
        x=x,
        y=y,
        constraints=constraints
    )


def K_repr_x_Gy(G: cp.Expression, x: cp.Variable) -> KRepresentation:
    assert G.is_concave()
    assert len(G.variables()) == 1
    y = G.variables()[0]
    assert G.ndim == 1
    assert x.ndim == 1
    assert G.shape == x.shape
    # assert x.is_nonneg()

    w = cp.Variable(G.size, name='w_bilin')

    constraints = [
        w <= G
    ]

    var_to_mat_mapping_dual, s, cone_dims, = get_cone_repr(constraints, [y, w])
    Q_bar = var_to_mat_mapping_dual['eta']
    S_bar = var_to_mat_mapping_dual[w.id]
    R_bar = var_to_mat_mapping_dual[y.id]

    lamb = cp.Variable(len(s))
    lamb_constr = add_cone_constraints(lamb, cone_dims)

    f = cp.Variable(y.size, name='f_bilin_x_Gy')
    t = cp.Variable(name='t_bilin_x_Gy')

    K_constr = [
        f + R_bar.T @ lamb == 0,
        s @ lamb == t,
        S_bar.T @ lamb == x,
        *lamb_constr
    ]

    if Q_bar.shape[1] > 0:
        K_constr.append(Q_bar.T @ lamb == 0)

    var_to_mat_mapping_final, _, _ = get_cone_repr(K_constr, [f, t, x])
    Q = var_to_mat_mapping_final['eta']

    return KRepresentation(
        f=f,
        t=t,
        u_or_Q= Q,
        x=x,
        y=y,
        constraints=K_constr
    )


def K_repr_ax(a: cp.Expression, y: cp.Variable) -> KRepresentation:
    assert len(a.variables()) == 1
    assert a.is_convex()
    x = a.variables()[0]
    assert a.ndim == 0

    f = cp.Variable(a.size, name='f_ax')
    t = cp.Variable(name='t_ax')

    constraints = [
        t >= a,
        f == 0
    ]

    var_to_mat_mapping, _, _, = get_cone_repr(constraints, [x, f, t])
    Q_bar = var_to_mat_mapping['eta']

    return KRepresentation(
        f=f,
        t=t,
        u_or_Q=Q_bar,
        x=x,
        y=y,
        constraints=constraints
    )


def K_repr_by(b_neg: cp.Expression, x: cp.Variable) -> KRepresentation:
    assert len(b_neg.variables()) == 1
    assert b_neg.is_concave()
    y = b_neg.variables()[0]
    assert b_neg.ndim == 0

    b = -b_neg

    f = cp.Variable(b.size, name='f_by')
    t_primal = cp.Variable(name='t_by_primal')

    constraints = [
        t_primal >= b,
        f == 0
    ]

    var_to_mat_mapping, s_bar, _, = get_cone_repr(constraints, [x, f, t_primal])
    R_bar = var_to_mat_mapping[y.id]
    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping['eta']

    f = cp.Variable(R_bar.shape[1])
    u = cp.Variable(p_bar.shape[0])
    t = cp.Variable(name='t_by')

    K_constr = [
        f == R_bar.T @ u,
        t == s_bar @ u,
        p_bar.T @ u + 1 == 0,
    ]

    if Q_bar.shape[1] > 0:
        K_constr.append(Q_bar.T @ u == 0)

    var_to_mat_mapping_final, _, _ = get_cone_repr(K_constr, [f, t, x])
    assert var_to_mat_mapping_final[x.id].shape[1] == 0
    Q = var_to_mat_mapping_final['eta']

    return KRepresentation(
        f=f,
        t=t,
        u_or_Q=Q,
        x=x,
        y=y,
        constraints=K_constr
    )


def get_cone_repr(const: list[Constraint], vars: list[cp.Variable]):
    assert set(vars) <= {v for c in const for v in c.variables()}
    aux_prob = cp.Problem(cp.Minimize(0), const)
    solver_opts = {"use_quad_obj": False}
    chain = aux_prob._construct_chain(solver_opts=solver_opts)
    chain.reductions = chain.reductions[:-1]  # skip solver reduction
    prob_canon = chain.apply(aux_prob)[0]  # grab problem instance
    # get cone representation of c, A, and b for some problem.

    problem_data = aux_prob.get_problem_data(solver_opts=solver_opts, solver=cp.SCS)

    Ab = problem_data[0]['param_prob'].A.toarray().reshape((-1, prob_canon.x.size + 1),
                                                           order="F")  # TODO: keep sparsity
    A, const_vec = Ab[:, :-1], Ab[:, -1]
    unused_mask = np.ones(A.shape[1], dtype=bool)

    var_id_to_col = problem_data[0]['param_prob'].var_id_to_col

    var_to_mat_mapping = {}
    for v in vars:
        start_ind = var_id_to_col[v.id]
        end_ind = start_ind + v.size
        original_cols = np.arange(start_ind, end_ind)
        var_to_mat_mapping[v.id] = -A[:, original_cols]
        unused_mask[original_cols] = 0

    var_to_mat_mapping['eta'] = -A[:, unused_mask]

    cone_dims = problem_data[0]['dims']

    return var_to_mat_mapping, const_vec, cone_dims


def add_cone_constraints(s, cone_dims) -> list[Constraint]:
    s_const = []

    offset = 0
    if cone_dims.zero > 0:
        offset += cone_dims.zero
    if cone_dims.nonneg > 0:
        s_const.append(s[offset:offset + cone_dims.nonneg] >= 0)
        offset += cone_dims.nonneg
    if len(cone_dims.soc) > 0:
        for soc_dim in cone_dims.soc:
            s_const.append(SOC(t=s[offset], X=s[offset + 1:offset + soc_dim + 1]))
            offset += soc_dim
    if cone_dims.exp > 0:
        tau = s[offset + 2::3]  # z (in cvxpy) -> t -> tau
        sigma = s[offset + 1::3]  # y (in cvxpy) -> s -> sigma
        rho = -s[offset::3]  # x (in cvxpy) -> r -> -rho
        s_const.extend([
            tau >= 0,
            rho >= 0,
            sigma >= cp.rel_entr(rho, tau) - rho
        ])
    if len(cone_dims.p3d) > 0 or len(cone_dims.psd) > 0:
        raise NotImplementedError

    return s_const


def get_original_variable_cols(variables, prob_canon) -> list[int]:
    end_inds = sorted(prob_canon.var_id_to_col.values()) + [prob_canon.x.shape[0]]

    main_var_inds = []

    for var in variables:
        start_ind = prob_canon.var_id_to_col[var.id]
        end_ind = end_inds[end_inds.index(start_ind) + 1]
        main_var_inds.append(range(start_ind, end_ind))

    return np.hstack(main_var_inds)
