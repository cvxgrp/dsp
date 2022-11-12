from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import cvxpy as cp
from cvxpy.constraints import ExpCone
from dspp.nemirovski import LocalToGlob, SwitchableKRepresentation, KRepresentation, add_cone_constraints, \
    get_cone_repr


class ConvexConcaveAtom(ABC):

    @abstractmethod
    def get_K_repr(self) -> SwitchableKRepresentation:
        pass

    @abstractmethod
    def get_convex_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def get_concave_variables(self) -> list[cp.Variable]:
        pass

    def is_convex(self):
        return False

    def is_concave(self):
        return False


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

    def get_K_repr(self, local_to_glob : LocalToGlob) -> SwitchableKRepresentation:
        f_local = cp.Variable(self.y.size, name='f')
        t = cp.Variable(name='t')
        u = cp.Variable(name='u')

        constraints = [
            ExpCone(self.x + u, np.ones(f_local.size), f_local),
            t >= -u - 1
        ]

        y_vars = self.y.variables()
        y_aux = cp.Variable(self.y.shape)
        var_to_mat_mapping, c, cone_dims, = get_cone_repr([y_aux == self.y], [*y_vars, y_aux])

        # get the equality constraints
        rows = cone_dims.zero
        assert rows == self.y.size
    
        B = np.zeros((rows,local_to_glob.size))
        for y in y_vars:
            start, end = local_to_glob.var_to_glob[y.id]
            B[:,start:end] = -var_to_mat_mapping[y.id][:rows]

        c = c[:rows]

        # f.T @ (B @ y_vars + c) = (B.T@f).T @ y_vars + f@c

        t_global = cp.Variable()
        constraints += [t_global == t + f_local@c]

        f_global = cp.Variable(local_to_glob.size)
        constraints += [f_global == B.T@f_local]


        # TODO: deal with affine x expr similar to y
        var_to_mat_mapping, s, cone_dims, = get_cone_repr(constraints, [f_global, t_global, self.x])
        Q = var_to_mat_mapping['eta']
        
        return SwitchableKRepresentation(
            f=f_global,
            t=t_global,
            x=self.x,
            y=self.y,
            constraints=constraints,
            u_or_Q=Q
        )

    def get_convex_variables(self) -> list[cp.Variable]:
        return [self.x]

    def get_concave_variables(self) -> list[cp.Variable]:
        return [self.y]


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
