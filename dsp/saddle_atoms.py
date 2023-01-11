from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import cvxpy as cp
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.constraints import ExpCone

from dsp.cone_transforms import (
    K_repr_bilin,
    K_repr_FxGy,
    KRepresentation,
    LocalToGlob,
    affine_to_canon,
    switch_convex_concave,
)
from dsp.parser import DSPError
from dsp.utils import np_vec


class SaddleAtom(Atom, ABC):
    def get_K_repr(self, local_to_glob: LocalToGlob, switched: bool = False) -> KRepresentation:
        if not self.is_dsp():
            raise DSPError("This atom is not a DSP expression.")
        return self._get_K_repr(local_to_glob, switched)

    @abstractmethod
    def _get_K_repr(self, local_to_glob: LocalToGlob, switched: bool) -> KRepresentation:
        pass

    @abstractmethod
    def convex_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def concave_variables(self) -> list[cp.Variable]:
        pass

    def affine_variables(self) -> list[cp.Variable]:
        known_curvature_vars = set(self.convex_variables() + self.concave_variables())
        return [v for v in self.variables() if v not in known_curvature_vars]

    @abstractmethod
    def get_convex_expression(self) -> cp.Expression:
        pass

    @abstractmethod
    def get_concave_expression(self) -> cp.Expression:
        pass

    @abstractmethod
    def is_dsp(self) -> bool:
        pass

    @abstractmethod
    def numeric(self, values: list) -> np.ndarray | None:
        pass

    def is_atom_convex(self) -> bool:
        return False

    def is_atom_concave(self) -> bool:
        return False

    def graph_implementation(
        self, arg_objs: list, shape: tuple[int, ...], data=None  # noqa
    ):  # noqa
        raise NotImplementedError

    def _grad(self, values: list):  # noqa
        raise NotImplementedError


class saddle_inner(SaddleAtom):
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        if not (Fx.is_affine() and Gy.is_affine()):
            if not Gy.is_nonneg():
                warnings.warn(
                    "Gy is non-positive. The y domain of saddle_inner is Gy >="
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

    def is_dsp(self) -> bool:
        x_cvx = self.Fx.is_convex(), "Fx must be convex"
        x_nonneg = self.Fx.is_nonneg(), "Fx must be non-negative"
        y_ccv = self.Gy.is_concave(), "Gy must be concave"
        return all([x_cvx, x_nonneg, y_ccv])

    def get_concave_expression(self) -> cp.Expression:
        Fx = self.Fx
        assert Fx.value is not None
        Fx = np_vec(Fx.value, order="F")
        Gy = cp.vec(self.Gy, order="F")
        return Fx @ Gy

    def get_convex_expression(self) -> cp.Expression:
        Gy = self.Gy
        assert Gy.value is not None
        Gy = np_vec(Gy.value, order="F")
        Fx = cp.vec(self.Fx, order="F")
        return Fx @ Gy

    def _get_K_repr(self, local_to_glob: LocalToGlob, switched: bool = False) -> KRepresentation:
        if self.bilinear:
            K_out = (
                K_repr_bilin(self.Fx, self.Gy, local_to_glob)
                if not switched
                else K_repr_bilin(-self.Gy, self.Fx, local_to_glob)
            )
        else:
            K_out = K_repr_FxGy(self.Fx, self.Gy, local_to_glob, switched)
            if not self.Gy.is_nonneg():
                outer_constraints = K_out.y_constraints if not switched else K_out.constraints
                outer_constraints.append(self.Gy >= 0)

        if not switched:
            K_out.concave_expr = lambda x: self.get_concave_expression()
        else:
            K_out.concave_expr = lambda x: -self.get_convex_expression()
        return K_out

    def numeric(self, values: list) -> np.ndarray | None:
        if any([v is None for v in values]):
            return None
        return values[0] @ values[1]

    def convex_variables(self) -> list[cp.Variable]:
        return self.Fx.variables()

    def concave_variables(self) -> list[cp.Variable]:
        return self.Gy.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (True, False)

    def is_incr(self, idx: int) -> bool:
        return False  # increasing in both arguments since y nonneg

    def is_decr(self, idx: int) -> bool:
        return False


class inner(saddle_inner):
    def __init__(self, Fx: cp.Expression, Gy: cp.Expression) -> None:
        assert isinstance(Fx, cp.Expression)
        assert isinstance(Gy, cp.Expression)

        assert Fx.is_affine() and Gy.is_affine(), (
            "inner must be affine. Use convex-concave-inner for non-affine" " bilinear terms."
        )

        super().__init__(Fx, Gy)

    def is_dsp(self) -> bool:
        x_aff = self.Fx.is_affine()
        y_aff = self.Gy.is_affine()
        return x_aff and y_aff

    def get_concave_expression(self) -> cp.Expression:
        return self.Fx.value @ self.Gy

    def get_convex_expression(self) -> cp.Expression:
        return self.Fx @ self.Gy.value


class weighted_log_sum_exp(SaddleAtom):
    def __init__(self, exponents: cp.Expression, weights: cp.Expression) -> None:
        """
        Implements the function f(x,y) = log(sum(exp(x_i) * y_i)) for vectors x and y.
        The weights, y, must be non-negative. If they are non recognized by
        cvxpy as nonnegative, an implicit domain constraint is added and a
        warning is provided.
        The exponents can be any convex cvxpy expression.

        The conic representation is:

            log sum_i y_i exp(x_i)
            min. f, t, u    f @ y + t
            subject to      f >= exp(x+u)
                            t >= -u-1

        """

        assert isinstance(exponents, cp.Expression)
        assert isinstance(weights, cp.Expression)

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

    def numeric(self, values: list) -> np.ndarray | None:
        if any([v is None for v in values]):
            return None
        x, y = values
        return np.log(np.sum(y * np.exp(x)))

    def is_dsp(self) -> bool:
        x_cvx = self.exponents.is_convex()
        y_ccv = self.weights.is_concave()
        return x_cvx and y_ccv

    def get_concave_expression(self, eps: float = 1e-6) -> cp.Expression:
        x = self.exponents
        if x.value is None:
            return None

        x = np_vec(x.value, order="F")
        y = cp.vec(self.weights, order="F")
        arg = y @ np.exp(x)
        return cp.log(arg)

    def get_convex_expression(self, eps: float = 1e-6) -> cp.Expression:
        x = self.weights
        if x.value is None:
            return None

        x = np_vec(x.value, order="F")
        y = cp.vec(self.exponents, order="F")

        nonneg = np.where(x > eps)[0]
        arg = y[nonneg] + np.log(x)[nonneg]
        return cp.log_sum_exp(arg)

    def _get_K_repr(self, local_to_glob: LocalToGlob, switched: bool = False) -> KRepresentation:
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
            B, c = affine_to_canon(self.weights, local_to_glob, switched)
            constraints += [t_global == t + f_local @ c]
            f_global = cp.Variable(
                local_to_glob.y_size if not switched else local_to_glob.x_size,
                name="f_global_wlse",
            )
            constraints += [f_global == B.T @ f_local]

        K_repr = KRepresentation(
            f=f_global,
            t=t_global,
            constraints=constraints,
        )

        switching_variables = self.exponents.variables()  # if not self.concave_composition else [z]
        precomp = (
            cp.Variable(self.weights.size, name="z_wlse") if self.concave_composition else None
        )
        K_out = (
            switch_convex_concave(
                constraints,
                f_global,
                t_global,
                switching_variables,
                local_to_glob,
                precomp,
            )
            if switched
            else K_repr
        )

        if self.concave_composition:
            if not switched:
                x_vars_1 = self.convex_variables() if not switched else self.concave_variables()
                K_switch_1 = switch_convex_concave(
                    K_out.constraints,
                    K_out.f,
                    K_out.t,
                    x_vars_1,
                    local_to_glob,
                    precomp=precomp,
                )
                K_switch_1.constraints += [self.weights >= precomp]

                # dualize the outer concave exp variables if switched
                x_vars_2 = self.concave_variables() if not switched else self.convex_variables()
                K_out = switch_convex_concave(
                    K_switch_1.constraints,
                    K_switch_1.f,
                    K_switch_1.t,
                    x_vars_2,
                    local_to_glob,
                )

            else:
                K_out.constraints += [precomp <= self.weights]

        if not self.weights.is_nonneg():
            if switched:
                K_out.constraints += [self.weights >= 0]
            else:
                K_out.y_constraints += [self.weights >= 0]

        concave_fun = (
            (lambda x: self.get_concave_expression())
            if not switched
            else (
                lambda x: -self.get_convex_expression()
                if self.get_convex_expression() is not None
                else None
            )
        )
        K_out.concave_expr = concave_fun

        return K_out

    def convex_variables(self) -> list[cp.Variable]:
        return self.exponents.variables()

    def concave_variables(self) -> list[cp.Variable]:
        return self.weights.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (False, False)

    def is_incr(self, idx: int) -> bool:
        return True  # increasing in both arguments since y nonneg


class saddle_quad_form(SaddleAtom):
    def __init__(self, x: cp.Expression, P: cp.Expression) -> None:
        assert isinstance(x, cp.Expression)
        assert isinstance(P, cp.Expression)

        assert x.size == P.shape[0] == P.shape[1]

        self.x = x
        self.P = P

        super().__init__(x, P)

    def numeric(self, values: list) -> np.ndarray | None:
        if any(v is None for v in values):
            return None
        x, P = values
        return x.T @ P @ x

    def is_dsp(self) -> bool:
        return self.P.is_psd() and self.x.is_affine()

    def get_concave_expression(self) -> cp.Expression:
        x = self.x
        assert x.value is not None
        x = np_vec(x.value, order="F")
        P = self.P
        return x.T @ P @ x

    def get_convex_expression(self) -> cp.Expression:
        P = self.P
        assert P.value is not None

        x = self.x
        return cp.quad_form(x, P)

    def _get_K_repr(self, local_to_glob: LocalToGlob, switched: bool = False) -> KRepresentation:
        n = self.x.size
        F_local = cp.Variable((n, n), name="F_quad_form_local", PSD=True)
        A = cp.Variable((n + 1, n + 1), name="A_quad_form", PSD=True)
        t = cp.Variable(name="t_quad_form")

        constraints = [
            A[:n, :n] == F_local,
            A[-1, -1] == 1,
            A[:n, -1] == self.x,
            A[-1, :n] == self.x.T,
            t == 0,
        ]

        B, c = affine_to_canon(self.P, local_to_glob, switched)
        F_global = cp.Variable(
            local_to_glob.y_size if not switched else local_to_glob.x_size,
            name="f_global_saddle_quad_form",
        )
        constraints += [F_global == B.T @ cp.vec(F_local)]

        K_repr = KRepresentation(
            f=F_global,
            t=t,
            constraints=constraints,
            concave_expr=lambda x: self.get_concave_expression(),
        )

        if switched:
            K_repr = switch_convex_concave(
                constraints, F_global, t, self.convex_variables(), local_to_glob
            )
            K_repr.concave_expr = lambda x: self.get_convex_expression()

        return K_repr

    def name(self) -> str:
        return "saddle_quad_form(" + self.x.name() + ", " + self.P.name() + ")"

    def convex_variables(self) -> list[cp.Variable]:
        return self.x.variables()

    def concave_variables(self) -> list[cp.Variable]:
        return self.P.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (True, False)

    def is_incr(self, idx: int) -> bool:
        return False

    def is_decr(self, idx: int) -> bool:
        return False


class weighted_norm2(SaddleAtom):
    def __init__(self, x: cp.Expression, y: cp.Expression) -> None:
        """
        Implements the function f(x,y) = (sum_i y_i x_i^2)^(1/2) for vectors x and y.
        The weights, y, must be non-negative. If they are non recognized by
        cvxpy as nonnegative, an implicit domain constraint is added and a
        warning is provided.
        The exponents can be any convex cvxpy expression.

        The conic representation is:
            (sum_i y_i x_i^2)^(1/2)
            min. f, t       f @ y + t
            subject to      f >= 0
                            t >= 0
                            sqrt(t*f) >= 0.5 * abs(x)
        """

        assert isinstance(x, cp.Expression)
        assert isinstance(y, cp.Expression)

        if not y.is_nonneg():
            warnings.warn(
                "Weights are non-positive. The domain of weighted_norm2 is y >="
                " 0. The implicit constraint y >= 0 will be added to"
                " the problem."
            )

        self.concave_composition = not y.is_affine()

        self.x = x
        self.y = y

        assert len(x.shape) == 1 and len(y.shape) == 1
        assert x.shape == y.shape

        super().__init__(x, y)

    def numeric(self, values: list) -> np.ndarray | None:
        if any(v is None for v in values):
            return None
        x, y = values
        return np.sqrt(np.sum(y * x**2))

    def is_dsp(self) -> bool:
        x_cvx = cp.norm2(self.x).is_convex()
        y_ccv = self.y.is_concave()
        return x_cvx and y_ccv

    def _get_K_repr(self, local_to_glob: LocalToGlob, switched: bool) -> KRepresentation:
        f_local = cp.Variable(self.y.size, name="f_wnorm2")
        t = cp.Variable(name="t_wnorm2")

        # See https://www2.isye.gatech.edu/~nemirovs/LMCOLN2022Fall.pdf 2.3.5
        tau = cp.Variable(self.x.size, name="tau")
        sum_half = (f_local + t) / 2
        diff_half = (f_local - t) / 2
        stack = cp.vstack([tau, diff_half])

        constraints = [
            t >= 0,
            f_local >= 0,
            cp.norm2(stack, axis=0) <= sum_half,
            tau >= 0.5 * cp.abs(self.x),
        ]

        t_global = cp.Variable(name="t_global")
        if self.concave_composition:
            constraints += [t_global == t]
            f_global = cp.Variable(self.y.size, name="f_global_wnorm2_comp")
            constraints += [f_global == f_local]
        else:
            B, c = affine_to_canon(self.y, local_to_glob, switched)
            constraints += [t_global == t + f_local @ c]
            f_global = cp.Variable(
                local_to_glob.y_size if not switched else local_to_glob.x_size,
                name="f_global_wnorm2",
            )
            constraints += [f_global == B.T @ f_local]

        K_repr = KRepresentation(
            f=f_global,
            t=t_global,
            constraints=constraints,
        )

        switching_variables = self.x.variables()

        precomp = cp.Variable(self.y.size, name="z_wnorm2") if self.concave_composition else None
        K_out = (
            switch_convex_concave(
                constraints,
                f_global,
                t_global,
                switching_variables,
                local_to_glob,
                precomp,
            )
            if switched
            else K_repr
        )

        if self.concave_composition:
            if not switched:
                x_vars_1 = self.convex_variables() if not switched else self.concave_variables()
                K_switch_1 = switch_convex_concave(
                    K_out.constraints,
                    K_out.f,
                    K_out.t,
                    x_vars_1,
                    local_to_glob,
                    precomp=precomp,
                )
                K_switch_1.constraints += [self.y >= precomp]

                # dualize the outer concave x variables if switched
                x_vars_2 = self.concave_variables() if not switched else self.convex_variables()
                K_out = switch_convex_concave(
                    K_switch_1.constraints,
                    K_switch_1.f,
                    K_switch_1.t,
                    x_vars_2,
                    local_to_glob,
                )

            else:
                K_out.constraints += [precomp <= self.y]

        if not self.y.is_nonneg():
            if switched:
                K_out.constraints += [self.y >= 0]
            else:
                K_out.y_constraints += [self.y >= 0]

        concave_fun = (
            (lambda x: self.get_concave_expression())
            if not switched
            else (
                lambda x: -self.get_convex_expression()
                if self.get_convex_expression() is not None
                else None
            )
        )
        K_out.concave_expr = concave_fun

        return K_out

    def get_concave_expression(self) -> cp.Expression:
        x = self.x
        assert x.value is not None

        y = self.y

        return cp.sqrt(y.T @ np.square(x))

    def get_convex_expression(self) -> cp.Expression:
        y = self.y
        assert y.value is not None

        x = self.x

        return cp.norm2(x * np.sqrt(y))

    def convex_variables(self) -> list[cp.Variable]:
        return self.x.variables()

    def concave_variables(self) -> list[cp.Variable]:
        return self.y.variables()

    def shape_from_args(self) -> tuple[int, ...]:
        return ()

    def sign_from_args(self) -> tuple[bool, bool]:
        return (True, False)

    def is_incr(self, idx: int) -> bool:
        return self.x.is_nonneg()