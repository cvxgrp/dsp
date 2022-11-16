from __future__ import annotations
import abc

import itertools
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy import SOC
from cvxpy.constraints import ExpCone
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective

@dataclass
class KRepresentation:
    f: cp.Expression | cp.Variable
    t: cp.Expression | cp.Variable
    constraints: list[Constraint]
    offset: float = 0.0

    # def __init__(self,
    #              f: cp.Expression | cp.Variable,
    #              t: cp.Expression | cp.Variable,
    #              constraints: list[Constraint],
    #              offset: float = 0.0):
    #     self.f = f
    #     self.t = t
    #     self.constraints = constraints
    #     self.offset = offset

    @classmethod
    def sum_of_K_reprs(cls, reprs: list[KRepresentation]) -> KRepresentation:
        assert len(reprs) >= 1

        f = cp.sum([K.f for K in reprs])
        t = cp.sum([K.t for K in reprs])

        all_constraints = [K.constraints for K in reprs]
        constraints = list(itertools.chain.from_iterable(all_constraints))

        offset = np.sum([K.offset for K in reprs])

        return KRepresentation(
            f=f,
            t=t,
            constraints=constraints,
            offset=offset
        )

    def scalar_multiply(self, scalar: float) -> KRepresentation:
        assert scalar >= 0
        return KRepresentation(
            f=self.f * scalar,
            t=self.t * scalar,
            constraints=self.constraints,
            offset=self.offset * scalar
        )

    @classmethod
    def constant_repr(cls, value: float | int) -> KRepresentation:
        return KRepresentation(
            f=cp.Constant(0),
            t=cp.Constant(0),
            constraints=[],
            offset=float(value),
        )


def minimax_to_min(K: KRepresentation,
                   X_constraints: list[Constraint],
                   Y_constraints: list[Constraint],
                   y_vars: list[cp.Variable],
                   local_to_glob: LocalToGlob
                   ) -> (Objective, list[Constraint]):
    # Convex part
    obj = K.t + K.offset

    constraints = [
        *K.constraints,
        *X_constraints,
    ]

    # Concave part
    # this case is only skipped if K.y is zero, i.e., if it's a purely convex problem
    if len(y_vars) > 0:
        var_id_to_mat, e, cone_dims = get_cone_repr(Y_constraints, y_vars)
        lamb = cp.Variable(len(e), name='lamb')
        lamb_const = add_cone_constraints(lamb, cone_dims, dual=True)

        D = var_id_to_mat['eta']

        C = np.zeros((len(e), local_to_glob.size))
        for y in y_vars:
            start, end = local_to_glob.var_to_glob[y.id]
            C[:, start:end] = var_id_to_mat[y.id]

        if D.shape[1] > 0:
            constraints.append(D.T @ lamb == 0)

        constraints += [C.T @ lamb == K.f,
                        *lamb_const]

        obj += lamb @ e

    return cp.Minimize(obj), constraints


def K_repr_x_Gy(G: cp.Expression, x: cp.Variable, local_to_glob: LocalToGlob) -> KRepresentation:
    assert G.is_concave()
    assert G.shape == x.shape

    y_vars = G.variables()
    w = cp.Variable(G.size, name='w_bilin')

    constraints = [
        w <= G
    ]

    var_to_mat_mapping_dual, s, cone_dims, = get_cone_repr(constraints, [*y_vars, w])
    Q_bar = var_to_mat_mapping_dual['eta']
    S_bar = var_to_mat_mapping_dual[w.id]

    lamb = cp.Variable(len(s))
    lamb_constr = add_cone_constraints(lamb, cone_dims, dual=True)

    t = cp.Variable(name='t_bilin_x_Gy')

    R_bar = np.zeros((S_bar.shape[0], local_to_glob.size))
    f = cp.Variable(local_to_glob.size, name='f_xGy')

    for y in y_vars:
        start, end = local_to_glob.var_to_glob[y.id]
        R_bar[:, start:end] = var_to_mat_mapping_dual[y.id]

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
        constraints=K_constr
    )


def K_repr_ax(a: cp.Expression) -> KRepresentation:
    assert a.is_convex()
    assert a.size == 1

    f = cp.Variable(name='f_ax')  # TODO: can we remove this variable?
    t = cp.Variable(name='t_ax')

    constraints = [
        t >= a,
        f == 0
    ]

    return KRepresentation(
        f=f,
        t=t,
        constraints=constraints
    )


class LocalToGlob:
    def __init__(self, variables: list[cp.Variable]):

        self.size = sum(var.size for var in variables)
        # self.y_global = cp.Variable(self.size)
        # self.y_global_constraints = []
        self.var_to_glob = dict()

        offset = 0
        for var in variables:
            assert var.ndim <= 1 or (var.ndim == 2 and min(var.shape) == 1)
            # TODO: ensure matrix variables are flattened correctly

            self.var_to_glob[var.id] = (offset, offset + var.size)
            # self.y_global_constraints += [self.y_global[offset, offset+var.size] == var]
            offset += var.size


def K_repr_by(b_neg: cp.Expression, local_to_glob: LocalToGlob) -> KRepresentation:
    assert b_neg.is_concave()
    y_vars = b_neg.variables()
    assert b_neg.size == 1

    b = -b_neg

    t_primal = cp.Variable(name='t_by_primal')

    constraints = [
        t_primal >= b,
    ]

    var_to_mat_mapping, s_bar, cone_dims, = get_cone_repr(constraints, [*y_vars, t_primal])

    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping['eta']

    u = cp.Variable(p_bar.shape[0], name='u_by')
    t = cp.Variable(name='t_by')

    R_bar = np.zeros((p_bar.shape[0], local_to_glob.size))
    f = cp.Variable(local_to_glob.size, name='f_by')

    for y in y_vars:
        start, end = local_to_glob.var_to_glob[y.id]
        R_bar[:, start:end] = var_to_mat_mapping[y.id]

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
        constraints=K_constr
    )


def K_repr_FxGy(Fx: cp.Expression, Gy: cp.Expression, local_to_glob: LocalToGlob) -> KRepresentation:
    z = cp.Variable(Fx.shape)
    constraints = [z >= Fx]

    K_repr_zGy = K_repr_x_Gy(Gy, z, local_to_glob)

    return KRepresentation(
        f=K_repr_zGy.f,
        t=K_repr_zGy.t,
        constraints=constraints + K_repr_zGy.constraints
    )


def K_repr_bilin(Fx: cp.Expression, Gy: cp.Expression, local_to_glob: LocalToGlob) -> KRepresentation:
    # Fx = Ax + b, Gy = Cy + d
    # Fx@Gy = Fx.T @ (C y + d)

    C, d = affine_to_canon(Gy, local_to_glob)

    if Fx.shape == ():
        Fx = cp.reshape(Fx, (1,))

    return KRepresentation(
        f=C.T @ Fx,
        t=Fx.T @ d,
        constraints=[]
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


def affine_to_canon(y_expr: cp.Expression, local_to_glob: LocalToGlob) -> (np.ndarray, np.ndarray):
    y_vars = y_expr.variables()
    y_aux = cp.Variable(y_expr.shape)
    var_to_mat_mapping, c, cone_dims, = get_cone_repr([y_aux == y_expr], [*y_vars, y_aux])

    # get the equality constraints
    rows = cone_dims.zero
    assert rows == y_expr.size

    B = np.zeros((rows, local_to_glob.size))
    for y in y_vars:
        start, end = local_to_glob.var_to_glob[y.id]
        B[:, start:end] = -var_to_mat_mapping[y.id][:rows]

    c = c[:rows]

    return B, c


def split_K_repr_affine(expr, convex_vars, concave_vars):
    """
    Split an affine expression into a convex and concave part plus offset, i.e.
    A@x+b <=> C@convex_vars + D@concave_vars + b
    """
    assert expr.is_affine()
    aux = cp.Variable(expr.shape)
    var_to_mat_mapping, b, _ = get_cone_repr([aux == expr], [*convex_vars, *concave_vars])
    C = cp.Constant(0)
    D = cp.Constant(0)
    for v in convex_vars:
        C += -var_to_mat_mapping.get(v.id, 0) @ cp.vec(v)
    for v in concave_vars:
        D += -var_to_mat_mapping.get(v.id, 0) @ cp.vec(v)

    return C, D, cp.Constant(b)


def switch_convex_concave(K_in: KRepresentation,
                          x: cp.Variable,
                          y: cp.Expression,
                          Q: np.ndarray
                          ) -> KRepresentation:
    # Turn phi(x,y) into \bar{phi}(\bar{x},\bar{y}) = -phi(x,y)
    # with \bar{x} = y, \bar{y} = x

    assert isinstance(Q, np.ndarray)
    assert isinstance(x, cp.Variable)

    var_list = [K_in.f,
                K_in.t,
                x]

    var_to_mat_mapping, const_vec, cone_dims = get_cone_repr(K_in.constraints,
                                                             var_list
                                                             )
    u_bar = cp.Variable(len(const_vec))
    u_bar_const = add_cone_constraints(u_bar, cone_dims, dual=True)

    assert var_to_mat_mapping['eta'].shape == Q.shape

    P = var_to_mat_mapping[K_in.f.id]
    R = var_to_mat_mapping[x.id]
    s = const_vec

    f_bar = cp.Variable(x.size, name='f_bar')
    t_bar = cp.Variable(name='t_bar')
    x_bar = y

    constraints = [
        f_bar == -R.T @ u_bar,
        t_bar == s @ u_bar,
        P.T @ u_bar + x_bar == 0,
        *u_bar_const
    ]

    if len(K_in.t.variables()) > 0:
        p = var_to_mat_mapping[K_in.t.id].flatten()
        constraints.append(p @ u_bar + 1 == 0)

    if Q.shape[1] > 0:
        constraints.append(Q.T @ u_bar == 0)

    return KRepresentation(
        f=f_bar,
        t=t_bar,
        constraints=constraints
    )
