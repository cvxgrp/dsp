from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import scipy.sparse as sp
from cvxpy import SOC
from cvxpy.atoms import reshape
from cvxpy.constraints import ExpCone
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import upper_tri_to_full
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
        all_Y_constraints = Y_constraints + K.y_constraints
        if len(all_Y_constraints) > 0:
            var_id_to_mat, e, cone_dims = get_cone_repr(all_Y_constraints, y_vars)

            n = len(e)
            lamb = cp.Variable(n, name="lamb")
            lamb_const, lamb = add_cone_constraints(lamb, cone_dims, dual=True)

            D = var_id_to_mat["eta"]

            C_shape = (len(e), local_to_glob.y_size)
            C = create_sparse_matrix_from_columns(C_shape, y_vars, local_to_glob, var_id_to_mat)

            if D.shape[1] > 0:
                constraints.append(D.T @ lamb == 0)

            constraints += [C.T @ lamb == K.f, *lamb_const]

            obj += lamb @ e
        else:
            constraints.append(K.f == 0)

    return cp.Minimize(obj), constraints


def scale_psd_dual(cone_dims: ConeDims, lamb: cp.Variable) -> cp.Variable:
    """
    Scale entries of the dual variable lamb corresponding to off-diagonal entries of PSD
    matrices by sqrt(2).
    """

    assert not cone_dims.p3d, "p3d cones not supported"

    if len(cone_dims.psd) > 0:
        exp_offset = 3 * cone_dims.exp
        n = lamb.shape[0]
        scaling_mat_diag = np.ones(n)
        n_psd_entries = sum([d * (d + 1) // 2 for d in cone_dims.psd])
        offset = n - n_psd_entries - exp_offset
        for psd_dim in cone_dims.psd:
            # scale_vec has sqrt(2) on entries corresponding to off-diagonal entries, 1 otherwise
            scale_vec = upper_tri_to_full(psd_dim).A.sum(axis=0) ** 0.5
            compressed_vars = len(scale_vec)
            scaling_mat_diag[offset : offset + compressed_vars] = scale_vec
            offset += compressed_vars
        lamb = np.diag(scaling_mat_diag) @ lamb
    return lamb


def K_repr_x_Gy(
    G: cp.Expression, x: cp.Variable, local_to_glob: LocalToGlob, switched: bool = False
) -> KRepresentation:
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
    lamb_constr, lamb = add_cone_constraints(lamb, cone_dims, dual=True)

    t = cp.Variable(name="t_bilin_x_Gy")
    f_size = local_to_glob.y_size if not switched else local_to_glob.x_size
    f = cp.Variable(f_size, name="f_xGy")

    R_bar_shape = (S_bar.shape[0], f_size)
    R_bar = create_sparse_matrix_from_columns(
        R_bar_shape, y_vars, local_to_glob, var_to_mat_mapping_dual
    )

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

        # self.y_size = sum(var.size for var in y_variables)
        # self.x_size = sum(var.size for var in x_variables)
        self.outer_x_vars = x_variables
        self.var_to_glob: dict[int, tuple[int, int]] = {}

        self.x_size = self.add_vars_to_map(x_variables)
        self.y_size = self.add_vars_to_map(y_variables)

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
            if var.attributes["diag"]:
                raise NotImplementedError("Diagonal variables are not supported yet")

            self.var_to_glob[var.id] = (offset, offset + sz)
            offset += sz

        return offset


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
    f = cp.Variable(local_to_glob.y_size, name="f_by")

    R_bar_shape = (p_bar.shape[0], local_to_glob.y_size)
    R_bar = create_sparse_matrix_from_columns(
        R_bar_shape, y_vars, local_to_glob, var_to_mat_mapping
    )

    cone_constraints, u = add_cone_constraints(u, cone_dims, dual=True)

    K_constr = [
        f == -R_bar.T @ u,
        t == s_bar @ u,
        p_bar.T @ u + 1 == 0,
        *cone_constraints,
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

    z = cp.Variable(Fx.shape)
    constraints = [z >= Fx]

    K_repr_zGy = K_repr_x_Gy(Gy, z, local_to_glob, switched=switched)

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


class NoYConstraintError(Exception):
    pass


def get_cone_repr(
    const: list[Constraint], exprs: list[cp.Variable | cp.Expression]
) -> tuple[dict[int, np.ndarray], np.ndarray, ConeDims]:
    if not {v for e in exprs for v in e.variables()} <= {v for c in const for v in c.variables()}:
        if len(const) == 0:
            raise NoYConstraintError(
                "No y constraints in problem. Variables are "
                + str([v.name() for e in exprs for v in e.variables()])
            )
        else:
            raise ValueError("Not all variables in exprs are constrained by const")

    # TODO: CVXPY does not have a stable API for getting the cone representation that is
    #  solver independent. We use SCS in line with the CVXPY documentation.
    #  Compare https://www.cvxpy.org/tutorial/advanced/index.html#getting-the-standard-form

    aux_prob = cp.Problem(cp.Minimize(0), const)
    solver_opts = {"use_quad_obj": False}

    # get cone representation of c, A, and b for some problem.
    problem_data = aux_prob.get_problem_data(solver_opts=solver_opts, solver=cp.SCS)

    A, const_vec = problem_data[0]["A"].tocsc(), problem_data[0]["b"]

    var_id_to_col = problem_data[0]["param_prob"].var_id_to_col

    unused_mask = np.ones(A.shape[1], dtype=bool)
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

        var_to_mat_mapping[e.id] = A[:, original_cols]
        unused_mask[original_cols] = 0

    var_to_mat_mapping["eta"] = A[:, unused_mask]

    cone_dims = problem_data[0]["dims"]

    return var_to_mat_mapping, const_vec, cone_dims


def add_cone_constraints(s: cp.Expression, cone_dims: ConeDims, dual: bool) -> list[Constraint]:
    assert len(s.shape) == 1 or s.shape[1] == 1, "s must be a vector"
    s = cp.reshape(s, (s.shape[0],))

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
        t_inds_dict = defaultdict(list)
        X_inds_dict = defaultdict(list)
        for soc_dim in cone_dims.soc:
            t_inds_dict[soc_dim].append(offset)
            X_inds_dict[soc_dim].extend(range(offset + 1, offset + soc_dim))
            offset += soc_dim

        for soc_dim, t_inds in t_inds_dict.items():
            X_slice = s[X_inds_dict[soc_dim]]
            X = cp.reshape(X_slice, (soc_dim - 1, len(t_inds)), order="F")
            s_const.append(SOC(t=s[t_inds], X=X, axis=0))

    if len(cone_dims.psd) > 0:
        for psd_dim in cone_dims.psd:
            m = psd_dim * (psd_dim + 1) // 2
            z = s[offset : offset + m]

            fill_coeff = Constant(upper_tri_to_full(psd_dim))
            flat_mat = fill_coeff @ z
            full_mat = reshape(flat_mat, (psd_dim, psd_dim))
            s_const.append(PSD(full_mat))
            offset += m

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

    if len(cone_dims.p3d) > 0:
        raise NotImplementedError

    assert offset == s.size

    lamb = scale_psd_dual(cone_dims, s)
    return s_const, lamb


def affine_to_canon(
    expr: cp.Expression, local_to_glob: LocalToGlob, switched: bool
) -> tuple[np.ndarray, np.ndarray]:
    variables = expr.variables()
    aux = cp.Variable(expr.shape)

    (
        var_to_mat_mapping,
        c,
        cone_dims,
    ) = get_cone_repr([aux == expr], [*variables, aux])

    # get the equality constraints
    rows_needed = cone_dims.zero
    assert rows_needed == expr.size

    cols = local_to_glob.y_size if not switched else local_to_glob.x_size

    rows_present = len(c)
    B_shape = (rows_present, cols)
    B = create_sparse_matrix_from_columns(B_shape, variables, local_to_glob, var_to_mat_mapping)
    B = B[:rows_needed]

    aux_columns = var_to_mat_mapping[aux.id][:rows_needed]
    assert (np.abs(aux_columns) - sp.eye(aux.size)).nnz == 0
    assert (np.sign(aux_columns.diagonal()) == np.ones(rows_needed)).all() or (
        np.sign(aux_columns.diagonal()) == -np.ones(rows_needed)
    ).all()

    sgn = -np.sign(aux_columns.diagonal())[0]
    B = B * sgn

    c = c[:rows_needed]

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


def create_sparse_matrix_from_columns(
    shape: tuple[int, int],
    variables: list[cp.Variable],
    local_to_glob: LocalToGlob,
    var_to_mat_mapping: dict[int, sp.csc_matrix],
) -> sp.csc_matrix:

    cols = []
    for v in variables:
        start, end = local_to_glob.var_to_glob[v.id]
        cols.append((start, end, v.id))

    cols = sorted(cols, key=lambda x: x[0])

    mats_to_stack = []
    current_col = 0
    for start, end, v_id in cols:
        if start == current_col:
            mats_to_stack.append(var_to_mat_mapping[v_id])
            current_col = end
        else:
            mats_to_stack.append(sp.csc_matrix((shape[0], start - current_col), dtype=float))
            mats_to_stack.append(var_to_mat_mapping[v_id])
            current_col = end
    if current_col < shape[1]:
        mats_to_stack.append(sp.csc_matrix((shape[0], shape[1] - current_col)))

    stacked_mat = sp.hstack(mats_to_stack, format="csc")
    assert stacked_mat.shape == shape
    return stacked_mat


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
    u_bar_const, u_bar = add_cone_constraints(u_bar, cone_dims, dual=True)

    P = var_to_mat_mapping[f.id]
    p = var_to_mat_mapping[t.id].toarray().flatten()
    Q = var_to_mat_mapping["eta"]

    R_shape = (len(s), local_to_glob.y_size)
    R = create_sparse_matrix_from_columns(R_shape, x_vars, local_to_glob, var_to_mat_mapping)

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

            if v.ndim > 1 and v.is_symmetric():
                n = v.shape[0]
                inds = np.triu_indices(n, k=0)  # includes diagonal
                v = v[inds]

            constraints += [(P.T @ u_bar)[start:end] + v == 0]

    return KRepresentation(f=f_bar, t=t_bar, constraints=constraints)
