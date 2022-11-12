from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import cvxpy as cp
from cvxpy.constraints import ExpCone
from cvxpy.atoms.atom import Atom
from dspp.cone_transforms import LocalToGlob, SwitchableKRepresentation, KRepresentation, add_cone_constraints, affine_to_canon, \
    get_cone_repr


class ConvexConcaveAtom(Atom, ABC):

    @abstractmethod
    def get_K_repr(self) -> SwitchableKRepresentation:
        pass

    @abstractmethod
    def get_convex_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def get_concave_variables(self) -> list[cp.Variable]:
        pass

    def is_atom_convex(self):
        return False

    def is_atom_concave(self):
        return False

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List['Constraint']]:
        raise NotImplementedError

    def _grad(self, values):
        raise NotImplementedError
    
class WeightedLogSumExp(ConvexConcaveAtom):

    def __init__(self, x: cp.Expression, y: cp.Expression):
        # log sum_i y_i exp(x_i)
        # min. f, t, u    f @ y + t
        # subject to      f >= exp(x+u)
        #                 t >= -u-1

        assert isinstance(x, cp.Expression)
        assert isinstance(y, cp.Expression)
        assert x.is_affine(), "x must be affine"
        assert y.is_affine(), "y must be affine"
        assert y.is_nonneg(), "y must be nonneg"

        self.x = x
        self.y = y

        assert (len(x.shape) <= 1 or (len(x.shape) == 2 and min(x.shape) == 1)) #TODO: implement matrix inputs
        assert (x.shape == y.shape or x.size == y.size)

        super().__init__(x, y)

    def get_K_repr(self, local_to_glob : LocalToGlob, switched = False) -> SwitchableKRepresentation:
        f_local = cp.Variable(self.y.size, name='f')
        t = cp.Variable(name='t')
        u = cp.Variable(name='u')

        constraints = [
            ExpCone(cp.reshape(self.x + u, (f_local.size,)), np.ones(f_local.size), f_local),
            t >= -u - 1
        ]


        if switched:
            var_to_mat_mapping, s, cone_dims, = get_cone_repr(constraints, [f_local, t, self.x])
            Q = var_to_mat_mapping['eta']

            K_repr_pre_switch = SwitchableKRepresentation(
                f = f_local,
                t = t,
                y = self.y,
                x = self.x,
                u_or_Q = Q,
                constraints=constraints
            )

            K_repr_local = switch_convex_concave(K_repr_pre_switch)
            local_constr = K_repr_local.constraints


            B, c = affine_to_canon(K_repr_local.y, local_to_glob)

            # f.T @ (B @ y_vars + c) = (B.T@f).T @ y_vars + f@c

            FuckMap = np.zeros((K_repr_local.y.size, sum(v.size for v in K_repr_local.y.variables())))
            offset = 0
            for v in K_repr_local.y.variables():
                start,end = local_to_glob.var_to_glob[v.id]
                FuckMap[:, offset:offset+v.size] = B[:,start:end] #TODO: wtf is B

            # entries in f correspond to unpacked variables in y
            # rows of B correspond to entries of y
            # columns of B correspond to all y variables

            # B.T       all var --> y
            # FuckMap   y --> v_var     B[:,[s1]]

            t_global = cp.Variable()
            local_constr += [t_global == K_repr_local.t + K_repr_local.f@FuckMap.T@c] # TODO: fix dimension of c

            f_global = cp.Variable(local_to_glob.size)

            local_constr += [f_global == B.T@FuckMap@K_repr_local.f]

            return KRepresentation(
                f=f_global,
                t=t_global,
                x=K_repr_local.x,
                y=K_repr_local.y,
                constraints=local_constr,    
            )
        else:
            B, c = affine_to_canon(self.y, local_to_glob) if not switched else ()

            # f.T @ (B @ y_vars + c) = (B.T@f).T @ y_vars + f@c

            t_global = cp.Variable()
            constraints += [t_global == t + f_local@c]

            f_global = cp.Variable(local_to_glob.size)
            constraints += [f_global == B.T@f_local]            

            return KRepresentation(
                f=f_global,
                t=t_global,
                x=self.x,
                y=self.y,
                constraints=constraints,
            )

    def get_convex_variables(self) -> list[cp.Variable]:
        return self.x.variables()

    def get_concave_variables(self) -> list[cp.Variable]:
        return self.y.variables()

    def shape_from_args(self) -> Tuple[int, ...]:
        return ()

    def sign_from_args(self) -> Tuple[bool, bool]:
        return (False, False)

    def is_incr(self, idx) -> bool:
        return True # increasing in both arguments since y nonneg

    def is_decr(self, idx) -> bool:
        return False

    


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

    f_bar = cp.Variable(sum(v.size for v in K_in.x.variables()), name='f_bar')
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
        constraints.append(p @ u_bar + 1 == 0)

    if Q.shape[1] > 0:
        constraints.append(Q.T @ u_bar == 0)

    return KRepresentation(
        f=f_bar,
        t=t_bar,
        x=x_bar,
        y=y_bar,
        constraints=constraints
    )


dspp_atoms = (WeightedLogSumExp, )
