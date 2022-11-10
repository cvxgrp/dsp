from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import cvxpy as cp
from cvxpy.constraints import ExpCone
from dspp.nemirovski import SwitchableKRepresentation, KRepresentation, add_cone_constraints, \
    get_cone_repr


class ConvexConcaveAtom(ABC):

    @abstractmethod
    def get_K_repr(self) -> KRepresentation:
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


class SwitchableConvexConcaveAtom(ConvexConcaveAtom, ABC):

    @abstractmethod
    def get_K_repr(self) -> SwitchableKRepresentation:
        pass


class WeightedLogSumExp(SwitchableConvexConcaveAtom):

    def __init__(self, x: cp.Variable, y: cp.Variable):
        # log sum_i y_i exp(x_i)
        # min. f, t, u    f @ y + t
        # subject to      f >= exp(x+u)
        #                 t >= -u-1

        assert isinstance(x, cp.Variable), "x must be a cvxpy.Variable, expressions can be used" \
                                           " but not supported yet"
        assert isinstance(y, cp.Variable), "y must be a cvxpy.Variable, expressions can be used" \
                                           " but not supported yet"

        self.x = x
        self.y = y

        assert len(x.shape) == 1
        assert x.shape == y.shape

    def get_K_repr(self) -> SwitchableKRepresentation:
        f = cp.Variable(self.y.size, name='f')
        t = cp.Variable(name='t')
        u = cp.Variable(name='u')

        constraints = [
            ExpCone(self.x + u, np.ones(f.size), f),
            t >= -u - 1
        ]

        return SwitchableKRepresentation(
            f=f,
            t=t,
            x=self.x,
            y=self.y,
            constraints=constraints,
            u_or_Q=u
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
        constraints.append(p @ u_bar + 1 == 0, )

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
