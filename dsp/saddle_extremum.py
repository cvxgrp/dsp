from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.atoms.atom import Atom

import dsp
from dsp.cone_transforms import LocalToGlob
from dsp.local import LocalVariableError
from dsp.parser import DSPError, Parser, initialize_parser


class SaddleExtremum(Atom):
    def __init__(
        self,
        f: cp.Expression,
        constraints: Iterable[cp.Constraint],
    ) -> None:

        self.f = f
        self._parser = None

        self._validate_arguments(constraints)
        self.constraints = list(constraints)
        self.other_variables = [v for v in f.variables() if not isinstance(v, dsp.LocalVariable)]

        super().__init__(*self.other_variables)

    def _validate_arguments(self, constraints: list[cp.Constraint]) -> None:
        assert self.f.size == 1
        assert isinstance(constraints, Iterable)
        for c in constraints:
            for v in c.variables():
                if not isinstance(v, dsp.LocalVariable):
                    raise LocalVariableError(
                        "All variables in constraints must be instances of LocalVariable."
                    )
        assert isinstance(self.f, cp.Expression)

    def is_dsp(self) -> bool:
        try:
            self.parser  # noqa
            return all([c.is_dcp() for c in self.constraints])
        except DSPError:
            return False

    def is_incr(self, idx: int) -> bool:
        return False

    def is_decr(self, idx: int) -> bool:
        return False

    def shape_from_args(self) -> tuple[int, ...]:
        """Returns the (row, col) shape of the expression."""
        return self.f.shape

    def sign_from_args(self) -> tuple[bool, bool]:
        is_positive = self.f.is_nonneg()
        is_negative = self.f.is_nonpos()

        return is_positive, is_negative

    @abstractmethod
    def convex_variables(self) -> list[cp.Variable]:
        pass

    @abstractmethod
    def concave_variables(self) -> list[cp.Variable]:
        pass


class saddle_max(SaddleExtremum):
    r"""sup_{y\in Y}f(x,y)"""

    def __init__(
        self,
        f: cp.Expression,
        constraints: Iterable[cp.Constraint],
    ) -> None:

        super().__init__(f, constraints)

        self._concave_vars = set(filter(lambda v: isinstance(v, dsp.LocalVariable), f.variables()))
        self._concave_vars |= set(itertools.chain.from_iterable(c.variables() for c in constraints))

        for v in self._concave_vars:
            v.expr = self

    def convex_variables(self) -> list[cp.Variable]:
        return sorted(self.parser.convex_vars, key=lambda x: x.id)

    def concave_variables(self) -> list[cp.Variable]:
        return sorted(self._concave_vars, key=lambda x: x.id)

    @property
    def parser(self) -> Parser:
        if self._parser is None:

            parser = initialize_parser(
                self.f,
                minimization_vars=self.other_variables,
                maximization_vars=self._concave_vars,
                constraints=self.constraints,
            )

            all_concave_vars_local = all(
                [isinstance(v, dsp.LocalVariable) for v in self._concave_vars]
            )

            if not all_concave_vars_local:
                raise LocalVariableError(
                    "All concave variables must be instances of" "LocalVariable."
                )

            for v in self._concave_vars:
                v.expr = self

            self._parser = parser
            return parser
        else:
            return self._parser

    def name(self) -> str:
        return (
            "saddle_max("
            + self.f.name()
            + ", ["
            + "".join([str(c) for c in self.constraints])
            + "])"
        )

    def numeric(self, values: list) -> np.ndarray | None:
        r"""
        Compute sup_{y\in Y}f(x,y) numerically
        """

        local_to_glob_y = LocalToGlob(self.convex_variables(), self.concave_variables())

        K_repr = self.parser.parse_expr_repr(self.f, switched=False, local_to_glob=local_to_glob_y)

        ccv = K_repr.concave_expr(values)
        if ccv is None:
            return None
        else:
            aux_problem = cp.Problem(cp.Maximize(ccv), self.constraints)
            aux_problem.solve()
            return aux_problem.value

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return self.is_dsp()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return False


class saddle_min(SaddleExtremum):
    r"""inf_{x\in X}f(x,y)"""

    def __init__(
        self,
        f: cp.Expression,
        constraints: Iterable[cp.Constraint],
    ) -> None:

        super().__init__(f, constraints)

        self._convex_vars = set(filter(lambda v: isinstance(v, dsp.LocalVariable), f.variables()))
        self._convex_vars |= set(itertools.chain.from_iterable(c.variables() for c in constraints))

        for v in self._convex_vars:
            v.expr = self

    def concave_variables(self) -> list[cp.Variable]:
        return sorted(self.parser.concave_vars, key=lambda x: x.id)

    def convex_variables(self) -> list[cp.Variable]:
        return sorted(self._convex_vars, key=lambda x: x.id)

    @property
    def parser(self) -> Parser:
        if self._parser is None:
            parser = initialize_parser(
                self.f,
                minimization_vars=self._convex_vars,
                maximization_vars=self.other_variables,
                constraints=self.constraints,
            )

            all_convex_vars_local = all(
                [isinstance(v, dsp.LocalVariable) for v in self._convex_vars]
            )

            if not all_convex_vars_local:
                raise LocalVariableError(
                    "All convex variables must be instances of" "LocalVariable."
                )

            for v in self._convex_vars:
                v.expr = self

            self._parser = parser
            return parser
        else:
            return self._parser

    def name(self) -> str:
        return (
            "saddle_min("
            + self.f.name()
            + ", ["
            + "".join([str(c) for c in self.constraints])
            + "])"
        )

    def numeric(self, values: list) -> np.ndarray | None:
        r"""
        Compute inf_{x\in X}f(x,y) numerically via -sup_{x\in X}-f(x,y)
        """
        neg_parser = initialize_parser(
            -self.f,
            minimization_vars=[],
            maximization_vars=self.convex_variables(),
            constraints=self.constraints,
        )
        neg_local_to_glob = LocalToGlob(neg_parser.convex_vars, neg_parser.concave_vars)
        neg_K_repr = neg_parser.parse_expr_repr(
            -self.f, switched=False, local_to_glob=neg_local_to_glob
        )

        ccv = neg_K_repr.concave_expr(values)
        if ccv is None:
            return None
        else:
            aux_problem = cp.Problem(cp.Minimize(-ccv), self.constraints)
            aux_problem.solve()
            return aux_problem.value

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return self.is_dsp()


class conjugate(saddle_max):
    r"""f^*(y) = sup_{x\in \dom f} (y^Tx - f(x))"""

    def __init__(self, f: cp.Expression, B: float = 1e6) -> None:
        assert isinstance(f, cp.Expression)

        x_vars = f.variables()

        if not all(isinstance(x, dsp.LocalVariable) for x in x_vars):
            raise LocalVariableError("All conjugate variables must be local variables.")

        y_vars = [cp.Variable(name=f"{x.name()}_conjugate", shape=x.shape) for x in x_vars]

        obj = -f
        for x, y in zip(x_vars, y_vars):
            obj += dsp.inner(cp.vec(y), cp.vec(x))

        constraints = [x <= B for x in x_vars] + [x >= -B for x in x_vars]

        super().__init__(obj, constraints)
