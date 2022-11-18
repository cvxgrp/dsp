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


class convex_concave_inner(ConvexConcaveAtom):
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        if not (Fx.is_affine() and Gy.is_affine()):
            assert Fx.is_convex(), "Fx must be convex"
            assert Fx.is_nonneg(), "Fx must be non-negative"
            assert Gy.is_concave(), "Gy must be concave"

            if not Gy.is_nonneg():
                warnings.warn(
                    "Gy is non-positive. The y domain of convex_concave_inner is Gy >="
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


class inner(convex_concave_inner):
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        assert Fx.is_affine() and Gy.is_affine(), (
            "inner must be affine. Use convex-concave-inner for non-affine" " bilinear terms."
        )

        super().__init__(Fx, Gy)


class weighted_log_sum_exp(ConvexConcaveAtom):
    def __init__(self, exponents: cp.Expression, weights: cp.Expression):
        """
        Implements the function f(x,y) = log(sum(exp(x_i) * y_i)) for vectors x and y.
        The weights, y, must be non-negative. If they are non recognized by
        cvxpy as nonnegative, an implicit domain constraint is added and a
        warning is provided.
        The exponents can be any convex cvxpy expression.

        The conic reprensentation is:

            log sum_i y_i exp(x_i)
            min. f, t, u    f @ y + t
            subject to      f >= exp(x+u)
                            t >= -u-1

        """

        assert isinstance(exponents, cp.Expression)
        assert isinstance(weights, cp.Expression)
        assert exponents.is_convex(), "exponents must be convex"

        # assert weights.is_affine(), "weights must be affine"
        assert weights.is_concave(), "weights must be concave"

        if not weights.is_nonneg():
            warnings.warn(
                "Weights are non-positive. The domain of weighted log-sum-exp is y >="
                " 0. The implicit constraint y >= 0 will be added to"
                " the problem."
            )

        self.concave_composition = not weights.is_affine()

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
        z = cp.Variable(self.weights.size, name="z_wlse") if self.concave_composition else None
        f_local = cp.Variable(self.weights.size, name="f_wlse")
        t = cp.Variable(name="t_wlse")
        u = cp.Variable(name="u_wlse")

        epi_exp = cp.Variable(self.exponents.size, name="exp_epi")
        constraints = [
            epi_exp >= self.exponents,  # handles composition in exponent
            ExpCone(cp.reshape(epi_exp + u, (f_local.size,)), np.ones(f_local.size), f_local),
            t >= -u - 1,
        ]

        t_global = cp.Variable(name="t_global")
        if self.concave_composition:
            constraints += [t_global == t]
            f_global = cp.Variable(self.weights.size, name="f_global_wlse_comp")
            constraints += [f_global == f_local]
        else:
            B, c = affine_to_canon(self.weights, local_to_glob)
            constraints += [t_global == t + f_local @ c]
            f_global = cp.Variable(
                local_to_glob.y_size if not switched else local_to_glob.x_size, name="f_global_wlse")
            constraints += [f_global == B.T @ f_local]

        K_repr = KRepresentation(
            f=f_global,
            t=t_global,
            constraints=constraints,
        )

        switching_variables = self.exponents.variables()  # if not self.concave_composition else [z]
        precomp = z if self.concave_composition else None
        K_out = switch_convex_concave(constraints, f_global, t_global,
                                      switching_variables, local_to_glob, precomp) if switched else K_repr

        if self.concave_composition:
            if not switched:
                x_vars_1 = self.get_convex_variables() if not switched else self.get_concave_variables()
                K_switch_1 = switch_convex_concave(
                    K_out.constraints, K_out.f, K_out.t, x_vars_1, local_to_glob, precomp=z)
                K_switch_1.constraints += [self.weights >= z]

                # dualize the outer concave exp variables if switched
                x_vars_2 = self.get_concave_variables() if not switched else self.get_convex_variables()
                K_out = switch_convex_concave(K_switch_1.constraints,
                                              K_switch_1.f, K_switch_1.t, x_vars_2, local_to_glob)

            else:
                K_out.constraints += [z <= self.weights]

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
