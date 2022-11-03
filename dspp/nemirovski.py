from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import SOC
from cvxpy.constraints import ExpCone
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective


class KRepresentation:

    def __init__(self, f: cp.Expression | cp.Variable,
                 t: cp.Expression | cp.Variable,
                 x: cp.Expression | cp.Variable,
                 y: cp.Expression | cp.Variable,
                 constraints: list[Constraint],
                 offset: float = 0.0):
        self.f = f
        self.t = t
        self.x = x
        self.y = y
        self.constraints = constraints
        self.offset = offset

    @classmethod
    def sum_of_K_reprs(cls, reprs: list[KRepresentation]) -> KRepresentation:
        assert len(reprs) >= 1

        f = cp.sum([K.f for K in reprs])
        t = cp.sum([K.t for K in reprs])

        all_constraints = [K.constraints for K in reprs]
        constraints = list(itertools.chain.from_iterable(all_constraints))

        x = cp.sum([K.x for K in reprs])
        y = cp.sum([K.y for K in reprs])
        offset = np.sum([K.offset for K in reprs])

        return KRepresentation(
            f=f,
            t=t,
            x=x,
            y=y,
            constraints=constraints,
            offset=offset
        )

    @classmethod
    def constant_repr(cls, value: float | int) -> KRepresentation:
        return KRepresentation(
            f=cp.Constant(0),
            t=cp.Constant(0),
            x=cp.Constant(0),
            y=cp.Constant(0),
            constraints=[],
            offset=float(value),
        )


@dataclass
class SwitchableKRepresentation(KRepresentation):

    def __init__(self, f: cp.Expression | cp.Variable,
                 t: cp.Expression | cp.Variable,
                 x: cp.Expression | cp.Variable,
                 y: cp.Expression | cp.Variable,
                 constraints: list[Constraint],
                 u_or_Q: cp.Variable | sp.base,
                 offset: float = 0.0):
        super().__init__(f, t, x, y, constraints, offset)
        self.u_or_Q = u_or_Q


def switch_convex_concave(K_in: SwitchableKRepresentation) -> KRepresentation:
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
    u_bar_const = add_cone_constraints(u_bar, cone_dims, dual=True)

    if isinstance(K_in.u_or_Q, cp.Variable):
        assert var_to_mat_mapping['eta'].size == 0
        Q = var_to_mat_mapping[K_in.u_or_Q.id]
    else:
        Q = K_in.u_or_Q
        assert var_to_mat_mapping['eta'].shape == Q.shape

    P = var_to_mat_mapping[K_in.f.id]
    R = var_to_mat_mapping[K_in.x.id]
    s = const_vec

    f_bar = cp.Variable(K_in.x.size, name='f_bar')
    t_bar = cp.Variable(name='t_bar')
    x_bar = K_in.y
    y_bar = K_in.x

    constraints = [
        f_bar == -R.T @ u_bar,
        t_bar == s @ u_bar,
        P.T @ u_bar + x_bar == 0,
        *u_bar_const
    ]

    if len(K_in.t.variables()) > 0:
        p = var_to_mat_mapping[K_in.t.id].flatten()
        constraints.append(p @ u_bar + 1 == 0,)

    if Q.shape[1] > 0:
        constraints.append(Q.T @ u_bar == 0)

    return KRepresentation(
        f=f_bar,
        t=t_bar,
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

    return SwitchableKRepresentation(
        f=f,
        t=t,
        x=x,
        y=y,
        constraints=constraints,
        u_or_Q=u
    )


def minimax_to_min(K: KRepresentation,
                   X_constraints: list[Constraint],
                   Y_constraints: list[Constraint]) -> (Objective, list[Constraint]):

    # Convex part
    obj = K.t + K.offset

    constraints = [
        *K.constraints,
        *X_constraints,
    ]

    # Concave part
    # this case is only skipped if K.y is zero, i.e., if it's a purely convex problem
    if len(K.y.variables()) > 0:
        var_id_to_mat, e, cone_dims = get_cone_repr(Y_constraints, [K.y])
        lamb = cp.Variable(len(e), name='lamb')
        lamb_const = add_cone_constraints(lamb, cone_dims, dual=True)

        C = var_id_to_mat[K.y.id]
        D = var_id_to_mat['eta']

        if D.shape[1] > 0:
            constraints.append(D.T @ lamb == 0)

        constraints += [C.T @ lamb == K.f,
                        *lamb_const]

        obj += lamb @ e

    return cp.Minimize(obj), constraints


def K_repr_y_Fx(F: cp.Expression, y: cp.Variable) -> KRepresentation:
    assert F.is_convex()
    assert len(F.variables()) == 1
    x = F.variables()[0]
    assert F.ndim == 1
    assert y.ndim == 1
    assert F.shape == y.shape
    # assert y.is_nonneg()

    f = cp.Variable(F.size, name='f_bilin')
    # t = cp.Variable(name='t_bilin_y_Fx')

    constraints = [
        f >= F
    ]

    return KRepresentation(
        f=f,
        t=cp.Constant(0),
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
    lamb_constr = add_cone_constraints(lamb, cone_dims, dual=True)

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

    return KRepresentation(
        f=f,
        t=t,
        x=x,
        y=y,
        constraints=K_constr
    )


def K_repr_ax(a: cp.Expression) -> KRepresentation:
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

    return KRepresentation(
        f=f,
        t=t,
        x=x,
        y=cp.Constant(0),
        constraints=constraints
    )


def K_repr_by(b_neg: cp.Expression) -> KRepresentation:
    assert len(b_neg.variables()) == 1
    assert b_neg.is_concave()
    y = b_neg.variables()[0]
    assert b_neg.ndim == 0

    b = -b_neg

    t_primal = cp.Variable(name='t_by_primal')

    constraints = [
        t_primal >= b,
    ]

    var_to_mat_mapping, s_bar, cone_dims, = get_cone_repr(constraints, [y, t_primal])

    R_bar = var_to_mat_mapping[y.id]
    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping['eta']

    f = cp.Variable(R_bar.shape[1], name='f_by')
    u = cp.Variable(p_bar.shape[0], name='u_by')
    t = cp.Variable(name='t_by')

    K_constr = [
        f == -R_bar.T @ u,
        t == s_bar @ u,
        p_bar.T @ u + 1 == 0,
        *add_cone_constraints(u, cone_dims, dual=True)
    ]

    if Q_bar.shape[1] > 0:
        K_constr.append(Q_bar.T @ u == 0)

    return KRepresentation(
        f=f,
        t=t,
        x=cp.Constant(0),
        y=y,
        constraints=K_constr
    )


def get_cone_repr(const: list[Constraint], exprs: list[cp.Variable | cp.Expression]):
    assert {v for e in exprs for v in e.variables()} <= {v for c in const for v in c.variables()}
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
    for e in exprs:
        if not e.variables():
            continue

        original_cols = np.array([], dtype=int)
        for v in e.variables():
            start_ind = var_id_to_col[v.id]
            end_ind = start_ind + v.size
            original_cols = np.append(original_cols, np.arange(start_ind, end_ind))

        var_to_mat_mapping[e.id] = -A[:, original_cols]
        unused_mask[original_cols] = 0

    var_to_mat_mapping['eta'] = -A[:, unused_mask]

    cone_dims = problem_data[0]['dims']

    return var_to_mat_mapping, const_vec, cone_dims


def add_cone_constraints(s, cone_dims, dual: bool) -> list[Constraint]:
    s_const = []

    offset = 0
    if cone_dims.zero > 0:
        if not dual:
            s_const.append(s[:cone_dims.zero] == 0)
        offset += cone_dims.zero
    if cone_dims.nonneg > 0:
        s_const.append(s[offset:offset + cone_dims.nonneg] >= 0)
        offset += cone_dims.nonneg
    if len(cone_dims.soc) > 0:
        for soc_dim in cone_dims.soc:
            s_const.append(SOC(t=s[offset], X=s[offset + 1:offset + soc_dim]))
            offset += soc_dim
    if cone_dims.exp > 0:
        if dual:
            tau = s[offset + 2::3]  # z (in cvxpy) -> t -> tau
            sigma = s[offset + 1::3]  # y (in cvxpy) -> s -> sigma
            rho = -s[offset::3]  # x (in cvxpy) -> r -> -rho
            s_const.extend([
                tau >= 0,
                rho >= 0,
                sigma >= cp.rel_entr(rho, tau) - rho
            ])
        else:
            x = s[offset::3]
            y = s[offset + 1::3]
            z = s[offset + 2::3]
            s_const.append(ExpCone(x, y, z))

        offset += 3 * cone_dims.exp
    if len(cone_dims.p3d) > 0 or len(cone_dims.psd) > 0:
        raise NotImplementedError

    assert offset == s.size

    return s_const
