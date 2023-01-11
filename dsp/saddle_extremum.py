from __future__ import annotations

import itertools
from typing import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.atoms.atom import Atom

import dsp
from dsp.cone_transforms import LocalToGlob
from dsp.local import LocalVariableError
from dsp.parser import DSPError, Parser, initialize_parser


def init_parser_wrapper(
    expr: cp.Expression,
    constraints: list[cp.Constraint],
    variables: set[cp.Variable],
    mode: str,
    other_variables: list[cp.Variable] | None = None,
) -> Parser:
    assert mode in ["sup", "inf"]

    expr = expr if mode == "sup" else -expr

    other_variables = other_variables if other_variables is not None else []

    parser = initialize_parser(
        expr,
        minimization_vars=other_variables,
        maximization_vars=variables,
        constraints=constraints,
    )

    return parser


class SaddleExtremum(Atom):
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
            self.parser
            return all([c.is_dcp() for c in self.constraints])
        except DSPError:
            return False


class saddle_max(SaddleExtremum):
    r"""sup_{y\in Y}f(x,y)"""

    def __init__(
        self,
        f: cp.Expression,
        constraints: Iterable[cp.Constraint],
    ) -> None:
        self.f = f

        self._validate_arguments(constraints)
        self.constraints = list(constraints)

        self.concave_vars = set(filter(lambda v: isinstance(v, dsp.LocalVariable), f.variables()))
        self.concave_vars |= set(itertools.chain.from_iterable(c.variables() for c in constraints))

        self._parser = None
        self.other_variables = [v for v in f.variables() if not isinstance(v, dsp.LocalVariable)]

        for v in self.concave_vars:
            v.expr = self

        super().__init__(*self.other_variables)

    @property
    def convex_vars(self) -> list[cp.Variable]:
        return self.parser.convex_vars

    @property
    def parser(self) -> Parser:
        if self._parser is None:
            parser = init_parser_wrapper(
                self.f,
                self.constraints,
                set(self.concave_vars),
                mode="sup",
                other_variables=self.other_variables,
            )

            # all_concave_vars_specified = set(self.concave_vars) == set(parser.concave_vars)
            all_concave_vars_local = all(
                [isinstance(v, dsp.LocalVariable) for v in self.concave_vars]
            )

            if not all_concave_vars_local:
                raise LocalVariableError(
                    "All concave variables must be instances of" "LocalVariable."
                )

            # if not all_concave_vars_specified:
            #     raise DSPError(
            #         "Must specify all concave variables, which all must be instances of"
            #         "LocalVariable."
            #     )

            for v in self.concave_vars:
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

        local_to_glob_y = LocalToGlob(self.convex_vars, self.concave_vars)

        K_repr = self.parser.parse_expr_repr(self.f, switched=False, local_to_glob=local_to_glob_y)

        ccv = K_repr.concave_expr(values)
        if ccv is None:
            return None
        else:
            aux_problem = cp.Problem(cp.Maximize(ccv), self.constraints)
            aux_problem.solve()
            return aux_problem.value

    def sign_from_args(self) -> tuple[bool, bool]:
        is_positive = self.f.is_nonneg()
        is_negative = self.f.is_nonpos()

        return is_positive, is_negative

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return self.is_dsp()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return False

    def is_incr(self, idx: int) -> bool:
        return False

    def is_decr(self, idx: int) -> bool:
        return False

    def shape_from_args(self) -> tuple[int, ...]:
        """Returns the (row, col) shape of the expression."""
        return self.f.shape


class saddle_min(SaddleExtremum):
    r"""inf_{x\in X}f(x,y)"""

    def __init__(
        self,
        f: cp.Expression,
        constraints: Iterable[cp.Constraint],
    ) -> None:
        self.f = f

        self._validate_arguments(constraints)
        self.constraints = list(constraints)

        self.convex_vars = set(filter(lambda v: isinstance(v, dsp.LocalVariable), f.variables()))
        self.convex_vars |= set(itertools.chain.from_iterable(c.variables() for c in constraints))

        self._parser = None
        self.other_variables = [v for v in f.variables() if not isinstance(v, dsp.LocalVariable)]

        for v in self.convex_vars:
            v.expr = self

        super().__init__(*self.other_variables)

    @property
    def concave_vars(self) -> list[cp.Variable]:
        return self.parser.concave_vars

    @property
    def parser(self) -> Parser:
        if self._parser is None:
            print(f"{self.other_variables=}")
            parser = init_parser_wrapper(
                self.f,
                self.constraints,
                set(self.convex_vars),
                mode="inf",
                other_variables=self.other_variables,
            )

            # all_convex_vars_specified = set(self.convex_vars) == set(parser.concave_vars)
            all_convex_vars_local = all(
                [isinstance(v, dsp.LocalVariable) for v in self.convex_vars]
            )

            if not all_convex_vars_local:
                raise LocalVariableError(
                    "All convex variables must be instances of" "LocalVariable."
                )

            # if not all_convex_vars_specified:
            #     raise DSPError(
            #         "Must specify all convex variables, which all must be instances of"
            #         "LocalVariable."
            #     )

            for v in self.convex_vars:
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
        neg_parser = init_parser_wrapper(
            -self.f, self.constraints, set(self.convex_vars), mode="sup"
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

    def sign_from_args(self) -> tuple[bool, bool]:
        is_positive = self.f.is_nonneg()
        is_negative = self.f.is_nonpos()

        return is_positive, is_negative

    def is_atom_convex(self) -> bool:
        """Is the atom convex?"""
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?"""
        return self.is_dsp()

    def is_incr(self, idx: int) -> bool:
        return False

    def is_decr(self, idx: int) -> bool:
        return False

    def shape_from_args(self) -> tuple[int, ...]:
        """Returns the (row, col) shape of the expression."""
        return self.f.shape


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
