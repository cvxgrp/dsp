from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
from cvxpy import SOC
from cvxpy.constraints import ExpCone
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.psd import PSD
from cvxpy.problems.objective import Objective
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims


@dataclass
class KRepresentation:
    f: cp.Expression | cp.Variable
    t: cp.Expression | cp.Variable
    constraints: list[Constraint]
    offset: float = 0.0
    y_constraints: list[Constraint] = field(default_factory=list)
    concave_expr: callable = lambda x: 0

    @classmethod
    def sum_of_K_reprs(cls, reprs: list[KRepresentation]) -> KRepresentation:
        assert len(reprs) >= 1

        f = cp.sum([K.f for K in reprs])
        t = cp.sum([K.t for K in reprs])

        all_constraints = [K.constraints for K in reprs]
        constraints = list(itertools.chain.from_iterable(all_constraints))

        offset = np.sum([K.offset for K in reprs])

        def f_concave(x: np.ndarray) -> cp.Expression:
            nones = any([K.concave_expr(x) is None for K in reprs])
            return cp.sum([K.concave_expr(x) for K in reprs]) if not nones else None

        concave_expr = f_concave

        y_constraints = list(itertools.chain.from_iterable([K.y_constraints for K in reprs]))

        return KRepresentation(
            f=f,
            t=t,
            constraints=constraints,
            offset=offset,
            y_constraints=y_constraints,
            concave_expr=concave_expr,
        )

    def scalar_multiply(self, scalar: float) -> KRepresentation:
        assert scalar >= 0
        return KRepresentation(
            f=self.f * scalar,
            t=self.t * scalar,
            constraints=self.constraints,
            offset=self.offset * scalar,
            concave_expr=lambda x: self.concave_expr(x) * scalar
            if self.concave_expr(x) is not None
            else None,
        )

    @classmethod
    def constant_repr(cls, value: float | int) -> KRepresentation:
        return KRepresentation(
            f=cp.Constant(0),
            t=cp.Constant(0),
            constraints=[],
            offset=float(value),
            concave_expr=lambda x: float(value),
        )


def minimax_to_min(
    K: KRepresentation,
    X_constraints: list[Constraint],
    Y_constraints: list[Constraint],
    y_vars: list[cp.Variable],
    local_to_glob: LocalToGlob,
) -> tuple[Objective, list[Constraint]]:
    # Convex part
    obj = K.t + K.offset

    constraints = [
        *K.constraints,
        *X_constraints,
    ]

    # Concave part
    # this case is only skipped if K.y is zero, i.e., if it's a purely convex problem
    if len(y_vars) > 0:
        var_id_to_mat, e, cone_dims = get_cone_repr(Y_constraints + K.y_constraints, y_vars)
        lamb = cp.Variable(len(e), name="lamb")
        lamb_const = add_cone_constraints(lamb, cone_dims, dual=True)

        D = var_id_to_mat["eta"]

        C = np.zeros((len(e), local_to_glob.y_size))
        for y in y_vars:
            start, end = local_to_glob.var_to_glob[y.id]
            C[:, start:end] = var_id_to_mat[y.id]

        if D.shape[1] > 0:
            constraints.append(D.T @ lamb == 0)

        constraints += [C.T @ lamb == K.f, *lamb_const]

        obj += lamb @ e

    return cp.Minimize(obj), constraints


def K_repr_x_Gy(G: cp.Expression, x: cp.Variable, local_to_glob: LocalToGlob) -> KRepresentation:
    assert G.is_concave()
    assert G.shape == x.shape

    y_vars = G.variables()
    w = cp.Variable(G.size, name="w_bilin")

    constraints = [w <= G]

    (
        var_to_mat_mapping_dual,
        s,
        cone_dims,
    ) = get_cone_repr(constraints, [*y_vars, w])
    Q_bar = var_to_mat_mapping_dual["eta"]
    S_bar = var_to_mat_mapping_dual[w.id]

    lamb = cp.Variable(len(s))
    lamb_constr = add_cone_constraints(lamb, cone_dims, dual=True)

    t = cp.Variable(name="t_bilin_x_Gy")

    R_bar = np.zeros((S_bar.shape[0], local_to_glob.y_size))
    f = cp.Variable(local_to_glob.y_size, name="f_xGy")

    for y in y_vars:
        start, end = local_to_glob.var_to_glob[y.id]
        R_bar[:, start:end] = var_to_mat_mapping_dual[y.id]

    K_constr = [
        f + R_bar.T @ lamb == 0,
        s @ lamb == t,
        S_bar.T @ lamb == x,
        *lamb_constr,
    ]

    if Q_bar.shape[1] > 0:
        K_constr.append(Q_bar.T @ lamb == 0)

    return KRepresentation(f=f, t=t, constraints=K_constr)


def K_repr_ax(a: cp.Expression) -> KRepresentation:
    assert a.is_convex()
    assert a.size == 1

    f = cp.Variable(name="f_ax")  # TODO: can we remove this variable?
    t = cp.Variable(name="t_ax")

    constraints = [t >= a, f == 0]

    return KRepresentation(f=f, t=t, constraints=constraints, concave_expr=lambda x: a.value)


class LocalToGlob:
    def __init__(self, x_variables: list[cp.Variable], y_variables: list[cp.Variable]) -> None:

        self.y_size = sum(var.size for var in y_variables)
        self.x_size = sum(var.size for var in x_variables)
        self.outer_x_vars = x_variables
        self.var_to_glob: dict[int, tuple[int, int]] = {}

        self.add_vars_to_map(x_variables)
        self.add_vars_to_map(y_variables)

    def add_vars_to_map(self, variables: list[cp.Variable]) -> None:
        offset = 0
        for var in variables:
            assert (
                var.ndim <= 1
                or (var.ndim == 2 and min(var.shape) == 1)
                or (var.ndim == 2 and var.shape[0] == var.shape[1])
            )
            # TODO: ensure matrix variables are flattened correctly
            sz = (
                var.size
                if not (var.ndim > 1 and var.is_symmetric())
                else (var.shape[0] * (var.shape[0] + 1) // 2)
            )  # fix for symmetric variables

            self.var_to_glob[var.id] = (offset, offset + sz)
            offset += var.size


def K_repr_by(b_neg: cp.Expression, local_to_glob: LocalToGlob) -> KRepresentation:
    assert b_neg.is_concave()
    y_vars = b_neg.variables()
    assert b_neg.size == 1

    b = -b_neg

    t_primal = cp.Variable(name="t_by_primal")

    constraints = [
        t_primal >= b,
    ]

    (
        var_to_mat_mapping,
        s_bar,
        cone_dims,
    ) = get_cone_repr(constraints, [*y_vars, t_primal])

    p_bar = var_to_mat_mapping[t_primal.id]
    Q_bar = var_to_mat_mapping["eta"]

    u = cp.Variable(p_bar.shape[0], name="u_by")
    t = cp.Variable(name="t_by")

    R_bar = np.zeros((p_bar.shape[0], local_to_glob.y_size))
    f = cp.Variable(local_to_glob.y_size, name="f_by")

    for y in y_vars:
        start, end = local_to_glob.var_to_glob[y.id]
        R_bar[:, start:end] = var_to_mat_mapping[y.id]

    K_constr = [
        f == -R_bar.T @ u,
        t == s_bar @ u,
        p_bar.T @ u + 1 == 0,
        *add_cone_constraints(u, cone_dims, dual=True),
    ]

    if Q_bar.shape[1] > 0:
        K_constr.append(Q_bar.T @ u == 0)

    return KRepresentation(f=f, t=t, constraints=K_constr, concave_expr=lambda x: b_neg)


def K_repr_FxGy(
    Fx: cp.Expression,
    Gy: cp.Expression,
    local_to_glob: LocalToGlob,
    switched: bool = False,
) -> KRepresentation:

    # dummy_Fx = cp.Variable(Fx.size, name='dummy_Fx')
    # Fx = dummy_Fx if switched else Fx

    z = cp.Variable(Fx.shape)
    constraints = [z >= Fx]

    K_repr_zGy = K_repr_x_Gy(Gy, z, local_to_glob)

    K_unswitched = KRepresentation(
        f=K_repr_zGy.f, t=K_repr_zGy.t, constraints=constraints + K_repr_zGy.constraints
    )

    if switched:
        return switch_convex_concave(
            K_unswitched.constraints,
            K_repr_zGy.f,
            K_repr_zGy.t,
            Fx.variables(),
            local_to_glob,
        )
    else:
        return K_unswitched


def K_repr_bilin(
    Fx: cp.Expression, Gy: cp.Expression, local_to_glob: LocalToGlob
) -> KRepresentation:
    # Fx = Ax + b, Gy = Cy + d
    # Fx@Gy = Fx.T @ (C y + d)

    C, d = affine_to_canon(Gy, local_to_glob, switched=False)

    if Fx.shape == ():
        Fx = cp.reshape(Fx, (1,))

    return KRepresentation(f=C.T @ Fx, t=Fx.T @ d, constraints=[])


def get_cone_repr(
    const: list[Constraint], exprs: list[cp.Variable | cp.Expression]
) -> tuple[dict[int, np.ndarray], np.ndarray, ConeDims]:
    assert {v for e in exprs for v in e.variables()} <= {v for c in const for v in c.variables()}
    aux_prob = cp.Problem(cp.Minimize(0), const)
    solver_opts = {"use_quad_obj": False}
    chain = aux_prob._construct_chain(solver_opts=solver_opts)
    chain.reductions = chain.reductions[:-1]  # skip solver reduction
    prob_canon = chain.apply(aux_prob)[0]  # grab problem instance
    # get cone representation of c, A, and b for some problem.

    problem_data = aux_prob.get_problem_data(solver_opts=solver_opts, solver=cp.SCS)

    Ab = (
        problem_data[0]["param_prob"].A.toarray().reshape((-1, prob_canon.x.size + 1), order="F")
    )  # TODO: keep sparsity
    A, const_vec = Ab[:, :-1], Ab[:, -1]
    unused_mask = np.ones(A.shape[1], dtype=bool)

    var_id_to_col = problem_data[0]["param_prob"].var_id_to_col

    var_to_mat_mapping = {}
    for e in exprs:
        if not e.variables():
            continue

        original_cols = np.array([], dtype=int)
        for v in e.variables():
            start_ind = var_id_to_col[v.id]
            sz = (
                v.size
                if not (v.ndim > 1 and v.is_symmetric())
                else (v.shape[0] * (v.shape[0] + 1) // 2)
            )  # fix for symmetric variables
            end_ind = start_ind + sz
            original_cols = np.append(original_cols, np.arange(start_ind, end_ind))

        var_to_mat_mapping[e.id] = -A[:, original_cols]
        unused_mask[original_cols] = 0

    var_to_mat_mapping["eta"] = -A[:, unused_mask]

    cone_dims = problem_data[0]["dims"]

    return var_to_mat_mapping, const_vec, cone_dims


def add_cone_constraints(s: cp.Expression, cone_dims: ConeDims, dual: bool) -> list[Constraint]:
    s_const = []

    offset = 0
    if cone_dims.zero > 0:
        if not dual:
            s_const.append(s[: cone_dims.zero] == 0)
        offset += cone_dims.zero
    if cone_dims.nonneg > 0:
        s_const.append(s[offset : offset + cone_dims.nonneg] >= 0)
        offset += cone_dims.nonneg
    if len(cone_dims.soc) > 0:
        for soc_dim in cone_dims.soc:
            s_const.append(SOC(t=s[offset], X=s[offset + 1 : offset + soc_dim]))
            offset += soc_dim
    if cone_dims.exp > 0:
        end = offset + 3 * cone_dims.exp
        if dual:
            tau = s[offset + 2 : end : 3]  # z (in cvxpy) -> t -> tau
            sigma = s[offset + 1 : end : 3]  # y (in cvxpy) -> s -> sigma
            rho = -s[offset:end:3]  # x (in cvxpy) -> r -> -rho
            s_const.extend([tau >= 0, rho >= 0, sigma >= cp.rel_entr(rho, tau) - rho])
        else:
            x = s[offset:end:3]
            y = s[offset + 1 : end : 3]
            z = s[offset + 2 : end : 3]
            s_const.append(ExpCone(x, y, z))

        offset += 3 * cone_dims.exp

    if len(cone_dims.psd) > 0:
        for psd_dim in cone_dims.psd:
            z = s[offset : offset + psd_dim**2]
            s_const.append(PSD(z))
            offset += psd_dim**2

    if len(cone_dims.p3d) > 0:
        raise NotImplementedError

    assert offset == s.size

    return s_const


def affine_to_canon(
    expr: cp.Expression, local_to_glob: LocalToGlob, switched: bool
) -> tuple[np.ndarray, np.ndarray]:
    vars = expr.variables()
    aux = cp.Variable(expr.shape)

    (
        var_to_mat_mapping,
        c,
        cone_dims,
    ) = get_cone_repr([aux == expr], [*vars, aux])

    # get the equality constraints
    rows = cone_dims.zero
    assert rows == expr.size

    cols = local_to_glob.y_size if not switched else local_to_glob.x_size
    B = np.zeros((rows, cols))
    for v in vars:
        start, end = local_to_glob.var_to_glob[v.id]
        B[:, start:end] = var_to_mat_mapping[v.id][:rows]

    aux_columns = var_to_mat_mapping[aux.id][:rows]
    assert np.allclose(np.abs(aux_columns), np.eye(aux.size))
    assert (np.sign(np.diag(aux_columns)) == np.ones(rows)).all() or (
        np.sign(np.diag(aux_columns)) == -np.ones(rows)
    ).all()

    sgn = -np.sign(np.diag(aux_columns))[0]
    B = B * sgn

    c = c[:rows]

    return B, c


def split_K_repr_affine(
    expr: cp.Expression, convex_vars: list[cp.Variable], concave_vars: list[cp.Variable]
) -> tuple[cp.Expression, cp.Expression, cp.Constant]:
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


def switch_convex_concave(
    constraints: list[Constraint],
    f: cp.Variable,
    t: cp.Variable,
    x_vars: list[cp.Variable],
    local_to_glob: LocalToGlob,
    precomp: cp.Variable | None = None,
) -> KRepresentation:
    """
    Return switched global k_repr from k_repr constraints and x/y expressions
    """

    var_list = [f, t] + x_vars

    var_to_mat_mapping, s, cone_dims = get_cone_repr(constraints, var_list)

    u_bar = cp.Variable(len(s))
    u_bar_const = add_cone_constraints(u_bar, cone_dims, dual=True)

    P = var_to_mat_mapping[f.id]
    p = var_to_mat_mapping[t.id].flatten()
    Q = var_to_mat_mapping["eta"]

    R = np.zeros((len(s), local_to_glob.y_size))
    for v in x_vars:
        start, end = local_to_glob.var_to_glob[v.id]
        R[:, start:end] = var_to_mat_mapping[v.id]

    f_bar = cp.Variable(local_to_glob.y_size, name="f_bar")
    t_bar = cp.Variable(name="t_bar")

    constraints = [
        f_bar == -R.T @ u_bar,
        t_bar == s @ u_bar,
        # P.T @ u_bar + x_bar == 0,
        *u_bar_const,
        p @ u_bar + 1 == 0,
        Q.T @ u_bar == 0,
    ]

    if precomp is not None:
        constraints += [P.T @ u_bar + precomp == 0]
    else:
        for v in local_to_glob.outer_x_vars:
            start, end = local_to_glob.var_to_glob[v.id]
            # P.T @ u_bar + x_bar == 0,
            constraints += [(P.T @ u_bar)[start:end] + v == 0]

    return KRepresentation(f=f_bar, t=t_bar, constraints=constraints)
