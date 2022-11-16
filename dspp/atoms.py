from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

import cvxpy as cp
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import ExpCone
from dspp.cone_transforms import K_repr_FxGy, K_repr_bilin, LocalToGlob, \
    KRepresentation, \
    affine_to_canon, \
    get_cone_repr, switch_convex_concave


class ConvexConcaveAtom(Atom, ABC):

    @abstractmethod
    def get_K_repr(self, local_to_glob: LocalToGlob, switched=False) -> KRepresentation:
        pass

    @abstractmethod
    def get_convex_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def get_concave_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def get_concave_objective(self, switched: bool) -> cp.Expression:
        pass

    def is_atom_convex(self):
        return False

    def is_atom_concave(self):
        return False

    def graph_implementation(self, arg_objs, shape: tuple[int, ...], data=None):
        raise NotImplementedError

    def _grad(self, values):
        raise NotImplementedError


class GeneralizedInnerProduct(ConvexConcaveAtom):

    def __init__(self, Fx : cp.Expression, Gy : cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)
        
        if not (Fx.is_affine() and Gy.is_affine()):
            assert Fx.is_convex(), "Fx must be convex"
            assert Fx.is_nonneg(), "Fx must be non-negative"
            assert Gy.is_concave(), "Gy must be concave"
            assert Gy.is_nonneg(), "Gy must be non-negative"
            self.bilinear = False
        else:
            self.bilinear = True

        self.Fx = Fx
        self.Gy = Gy

        assert (len(Fx.shape) <= 1 or (
            len(Fx.shape) == 2 and min(Fx.shape) == 1))  # TODO: implement matrix inputs
        assert (Fx.shape == Gy.shape or Fx.size == Gy.size)
        
        super().__init__(Fx, Gy)

    def get_concave_objective(self, switched=False) -> cp.Expression:
        Fx = self.Fx if not switched else self.Gy
        assert Fx.value is not None
        Fx = np.reshape(Fx.value, (Fx.size,), order='F')

        Gy = self.Gy if not switched else self.Fx
        Gy = cp.reshape(Gy, (Gy.size,), order='F')

        return Fx@Gy if not switched else -Fx@Gy
            

    def get_K_repr(self, local_to_glob: LocalToGlob, switched=False) -> KRepresentation:
        if self.bilinear:
            return K_repr_bilin(self.Fx, self.Gy, local_to_glob) if not switched else K_repr_bilin(-self.Gy, self.Fx, local_to_glob)
        else:
            return K_repr_FxGy(self.Fx, self.Gy, local_to_glob, switched)
        
    def get_convex_variables(self) -> list[cp.Variable]:
        return self.Fx.variables()

    def get_concave_variables(self) -> list[cp.Variable]:
        return self.Gy.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (True, False)

    def is_incr(self, idx) -> bool:
        return False  # increasing in both arguments since y nonneg

    def is_decr(self, idx) -> bool:
        return False

class Bilinear(GeneralizedInnerProduct):

    def __init__(self, Fx : cp.Expression, Gy : cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)
        
        assert Fx.is_affine() and Gy.is_affine(), "Bilinear must be affine. Use GeneralizedInnerProduct for non-affine bilinear terms."
        
        super().__init__(Fx, Gy)


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

        assert (len(x.shape) <= 1 or (
            len(x.shape) == 2 and min(x.shape) == 1))  # TODO: implement matrix inputs
        assert (x.shape == y.shape or x.size == y.size)

        super().__init__(x, y)

    def get_concave_objective(self, switched=False, eps=1e-6) -> cp.Expression:
        x = self.x if not switched else self.y
        assert x.value is not None
        x = np.reshape(x.value, (x.size,), order='F')

        y = self.y if not switched else self.x
        y = cp.reshape(y, (y.size,), order='F')

        if not switched:
            arg = y @ np.exp(x)
            return cp.log(arg)
        else:
            nonneg = np.where(x > eps)[0]
            arg = y[nonneg] + np.log(x)[nonneg]
            return -cp.log_sum_exp(arg)

    def get_K_repr(self, local_to_glob: LocalToGlob, switched=False) -> KRepresentation:
        f_local = cp.Variable(self.y.size, name='f')
        t = cp.Variable(name='t')
        u = cp.Variable(name='u')

        dummy_x = cp.Variable(self.x.size, name='dummy_x')
        x = self.x if not switched else dummy_x

        constraints = [
            ExpCone(cp.reshape(x + u, (f_local.size,)), np.ones(f_local.size), f_local),
            t >= -u - 1
        ]

        if switched:
            var_to_mat_mapping, s, cone_dims, = get_cone_repr(constraints, [f_local, t, x])
            Q = var_to_mat_mapping['eta']

            K_repr_pre_switch = KRepresentation(
                f=f_local,
                t=t,
                constraints=constraints
            )

            K_repr_local = switch_convex_concave(K_repr_pre_switch, x, self.y, Q)
            local_constr = K_repr_local.constraints

            B, c = affine_to_canon(self.x, local_to_glob)  # self.x is outer yexpr

            # f.T @ (B @ y_vars + c) = (B.T@f).T @ y_vars + f@c

            t_global = cp.Variable(name='t_wlse_switched')
            local_constr += [
                t_global == K_repr_local.t + K_repr_local.f @ c]  # TODO: fix dimension of c

            f_global = cp.Variable(local_to_glob.size, name='f_wlse_switched')

            local_constr += [f_global == B.T @ K_repr_local.f]

            return KRepresentation(
                f=f_global,
                t=t_global,
                constraints=local_constr,
            )
        else:
            B, c = affine_to_canon(self.y, local_to_glob) if not switched else ()

            # f.T @ (B @ y_vars + c) = (B.T@f).T @ y_vars + f@c

            t_global = cp.Variable()
            constraints += [t_global == t + f_local @ c]

            f_global = cp.Variable(local_to_glob.size)
            constraints += [f_global == B.T @ f_local]

            return KRepresentation(
                f=f_global,
                t=t_global,
                constraints=constraints,
            )

    def get_convex_variables(self) -> list[cp.Variable]:
        return self.x.variables()

    def get_concave_variables(self) -> list[cp.Variable]:
        return self.y.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (False, False)

    def is_incr(self, idx) -> bool:
        return True  # increasing in both arguments since y nonneg

    def is_decr(self, idx) -> bool:
        return False


