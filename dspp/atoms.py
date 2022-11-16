from __future__ import annotations

from abc import ABC, abstractmethod
import warnings

import numpy as np

import cvxpy as cp
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import ExpCone
from dspp.cone_transforms import (
    K_repr_FxGy,
    K_repr_bilin,
    LocalToGlob,
    KRepresentation,
    affine_to_canon,
    get_cone_repr,
    switch_convex_concave,
    switch_convex_concave,
)


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
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        if not (Fx.is_affine() and Gy.is_affine()):
            assert Fx.is_convex(), "Fx must be convex"
            assert Fx.is_nonneg(), "Fx must be non-negative"
            assert Gy.is_concave(), "Gy must be concave"

            if not Gy.is_nonneg():
                warnings.warn(
                    "Gy is non-positive. The y domain of GeneralizedBilinear is Gy >="
                    " 0. The implicit constraint Gy >= 0 will be added to the problem."
                )
            self.bilinear = False
        else:
            self.bilinear = True

        self.Fx = Fx
        self.Gy = Gy

        assert len(Fx.shape) <= 1 or (
            len(Fx.shape) == 2 and min(Fx.shape) == 1
        )  # TODO: implement matrix inputs
        assert Fx.shape == Gy.shape or Fx.size == Gy.size

        super().__init__(Fx, Gy)

    def get_concave_objective(self, switched=False) -> cp.Expression:
        Fx = self.Fx if not switched else self.Gy
        assert Fx.value is not None
        Fx = np.reshape(Fx.value, (Fx.size,), order="F")

        Gy = self.Gy if not switched else self.Fx
        Gy = cp.reshape(Gy, (Gy.size,), order="F")

        return Fx @ Gy if not switched else -Fx @ Gy

    def get_K_repr(self, local_to_glob: LocalToGlob, switched=False) -> KRepresentation:
        if self.bilinear:
            return (
                K_repr_bilin(self.Fx, self.Gy, local_to_glob)
                if not switched
                else K_repr_bilin(-self.Gy, self.Fx, local_to_glob)
            )
        else:
            K_out = K_repr_FxGy(self.Fx, self.Gy, local_to_glob, switched)
            if not self.Gy.is_nonneg():
                outer_constraints = K_out.y_constraints if not switched else K_out.constraints
                outer_constraints.append(self.Gy >= 0)

            return K_out

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
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        assert Fx.is_affine() and Gy.is_affine(), (
            "Bilinear must be affine. Use GeneralizedInnerProduct for non-affine" " bilinear terms."
        )

        super().__init__(Fx, Gy)


class WeightedLogSumExp(ConvexConcaveAtom):
    def __init__(self, exponents: cp.Expression, weights: cp.Expression):
        # log sum_i y_i exp(x_i)
        # min. f, t, u    f @ y + t
        # subject to      f >= exp(x+u)
        #                 t >= -u-1

        assert isinstance(exponents, cp.Expression)
        assert isinstance(weights, cp.Expression)
        assert exponents.is_convex(), "x must be convex"
        assert weights.is_affine(), "y must be affine"

        if not weights.is_nonneg():
            warnings.warn(
                "Weights are non-positive. The domain of weighted log-sum-exp is y >="
                " 0.                 The implicit constraint y >= 0 will be added to"
                " the problem."
            )

        self.exponents = exponents
        self.weights = weights

        assert len(exponents.shape) <= 1 or (
            len(exponents.shape) == 2 and min(exponents.shape) == 1
        )  # TODO: implement matrix inputs
        assert exponents.shape == weights.shape or exponents.size == weights.size

        super().__init__(exponents, weights)

    def get_concave_objective(self, switched=False, eps=1e-6) -> cp.Expression:
        x = self.exponents if not switched else self.weights
        assert x.value is not None
        x = np.reshape(x.value, (x.size,), order="F")

        y = self.weights if not switched else self.exponents
        y = cp.reshape(y, (y.size,), order="F")

        if not switched:
            arg = y @ np.exp(x)
            return cp.log(arg)
        else:
            nonneg = np.where(x > eps)[0]
            arg = y[nonneg] + np.log(x)[nonneg]
            return -cp.log_sum_exp(arg)

    def get_K_repr(self, local_to_glob: LocalToGlob, switched=False) -> KRepresentation:
        f_local = cp.Variable(self.weights.size, name="f_wlse")
        t = cp.Variable(name="t_wlse")
        u = cp.Variable(name="u_wlse")

        epi_exp = cp.Variable(self.exponents.size, name="exp_epi")
        constraints = [
            epi_exp >= self.exponents,  # handles composition in exponent
            ExpCone(cp.reshape(epi_exp + u, (f_local.size,)), np.ones(f_local.size), f_local),
            t >= -u - 1,
        ]

        B, c = affine_to_canon(self.weights, local_to_glob)  # TODO: dont assume weights are affine

        t_global = cp.Variable(name="t_global")
        constraints += [t_global == t + f_local @ c]

        f_global = cp.Variable(
            local_to_glob.y_size if not switched else local_to_glob.x_size,
            name="f_global",
        )
        constraints += [f_global == B.T @ f_local]

        K_repr = KRepresentation(
            f=f_global,
            t=t_global,
            constraints=constraints,
        )

        K_out = switch_convex_concave(constraints, f_global, t_global,
                           self.get_convex_variables(), local_to_glob) if switched else K_repr

        if not self.weights.is_nonneg():
            if switched:
                K_out.constraints += [self.weights >= 0]
            else:
                K_out.y_constraints += [self.weights >= 0]

        return K_out

    def get_convex_variables(self) -> list[cp.Variable]:
        return self.exponents.variables()

    def get_concave_variables(self) -> list[cp.Variable]:
        return self.weights.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (False, False)

    def is_incr(self, idx) -> bool:
        return True  # increasing in both arguments since y nonneg

    def is_decr(self, idx) -> bool:
        return False
