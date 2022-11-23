from __future__ import annotations

import cvxpy as cp

import itertools
from typing import Iterable
from cvxpy import multiply
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective
from dspp.atoms import ConvexConcaveAtom
from dspp.cone_transforms import (
    K_repr_ax,
    K_repr_bilin,
    K_repr_by,
    KRepresentation,
    LocalToGlob,
    minimax_to_min,
    split_K_repr_affine,
)

class Parser:
    def __init__(self, convex_vars: set[cp.Variable], concave_vars: set[cp.Variable]) -> None:

        self.convex_vars: set[cp.Variable] = convex_vars
        self.concave_vars: set[cp.Variable] = concave_vars
        assert not (self.convex_vars & self.concave_vars)
        self.affine_vars: set[int] = set()

    def split_up_variables(self, expr: cp.Expression) -> None:
        if expr.is_affine():
            self.affine_vars |= set(expr.variables())
        else:
            self.convex_vars |= set(expr.variables())

        if isinstance(expr, cp.Constant) or isinstance(expr, (float, int)):
            return
        elif isinstance(expr, ConvexConcaveAtom):
            self.add_to_convex_vars(expr.get_convex_variables())
            self.add_to_concave_vars(expr.get_concave_variables())
        elif isinstance(expr, AddExpression):
            for arg in expr.args:
                self.split_up_variables(arg)
        elif expr.is_affine():
            self.affine_vars |= set(expr.variables()) - self.convex_vars - self.concave_vars
        elif expr.is_convex():
            self.add_to_convex_vars(expr.variables())
        elif expr.is_concave():
            self.add_to_concave_vars(expr.variables())
        elif isinstance(expr, NegExpression):
            if isinstance(expr.args[0], ConvexConcaveAtom):
                dspp_atom = expr.args[0]
                assert isinstance(dspp_atom, ConvexConcaveAtom)
                self.add_to_concave_vars(dspp_atom.get_convex_variables())
                self.add_to_convex_vars(dspp_atom.get_concave_variables())
            elif isinstance(expr.args[0], AddExpression):
                for arg in expr.args[0].args:
                    self.split_up_variables(-arg)
            elif isinstance(expr.args[0], NegExpression):  # double negation
                dspp_atom = expr.args[0].args[0]
                assert isinstance(dspp_atom, ConvexConcaveAtom)
                self.split_up_variables(dspp_atom)
            elif isinstance(expr.args[0], multiply):  # negated multiplication of dspp atom
                mult = expr.args[0]
                s = mult.args[0]
                assert isinstance(s, cp.Constant)
                self.split_up_variables(-s.value * mult.args[1])
            else:
                raise ValueError
        elif isinstance(expr, multiply):
            s = expr.args[0]
            assert isinstance(s, cp.Constant)
            dspp_atom = expr.args[1]
            assert isinstance(dspp_atom, ConvexConcaveAtom)
            if s.is_nonneg():
                self.add_to_convex_vars(dspp_atom.get_convex_variables())
                self.add_to_concave_vars(dspp_atom.get_concave_variables())
            else:
                self.add_to_concave_vars(dspp_atom.get_convex_variables())
                self.add_to_convex_vars(dspp_atom.get_concave_variables())
        else:
            raise ValueError(f"Cannot parse {expr=} with {expr.curvature=}.")

    def add_to_convex_vars(self, variables: Iterable[cp.Variable]) -> None:
        variables = set(variables)
        assert not (variables & self.concave_vars), (
            "Cannot add variables to both " "convex and concave set."
        )
        self.affine_vars -= variables
        self.convex_vars |= variables

    def add_to_concave_vars(self, variables: Iterable[cp.Variable]) -> None:
        variables = set(variables)
        assert not (variables & self.convex_vars), (
            "Cannot add variables to both " "convex and concave set."
        )
        self.affine_vars -= variables
        self.concave_vars |= variables

    def parse_scalar_mul(
        self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs: dict
    ) -> KRepresentation | None:
        assert expr.args[0].is_constant() or expr.args[1].is_constant()
        const_ind = 0 if isinstance(expr.args[0], cp.Constant) else 1
        var_ind = 1 - const_ind

        s = expr.args[const_ind]
        var_expr = expr.args[var_ind]

        if s.is_nonneg():
            return_val = self._parse_expr(var_expr, switched, repr_parse, **kwargs)
        else:
            return_val = self._parse_expr(var_expr, not switched, repr_parse, **kwargs)

        if repr_parse:
            assert return_val is not None
            return return_val.scalar_multiply(abs(s.value))

    def parse_known_curvature_repr(
        self, expr: cp.Expression, local_to_glob: LocalToGlob
    ) -> KRepresentation:
        if set(expr.variables()) <= self.convex_vars:
            assert expr.is_convex()
            return K_repr_ax(expr)
        elif set(expr.variables()) <= self.concave_vars:
            assert expr.is_concave()
            return K_repr_by(expr, local_to_glob)
        else:
            raise ValueError

    def parse_known_curvature_vars(self, expr: cp.Expression, switched: bool) -> None:
        vars = expr.variables()
        if expr.is_affine():
            self.affine_vars |= set(expr.variables()) - self.convex_vars - self.concave_vars
        elif expr.is_convex():
            self.add_to_convex_vars(vars) if not switched else self.add_to_concave_vars(vars)
        elif expr.is_concave():
            self.add_to_concave_vars(vars) if not switched else self.add_to_convex_vars(vars)
        else:
            raise ValueError(f"Cannot parse {expr=} with {expr.curvature=}.")

    def parse_add(
        self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs: dict
    ) -> KRepresentation | None:
        assert isinstance(expr, AddExpression)
        if repr_parse:
            K_reprs = [self.parse_expr_repr(arg, switched, **kwargs) for arg in expr.args]
            return KRepresentation.sum_of_K_reprs(K_reprs)
        else:
            for arg in expr.args:
                self.parse_expr_variables(arg, switched, **kwargs)

    def parse_dspp_atom(
        self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs: dict
    ) -> KRepresentation | None:
        assert isinstance(expr, ConvexConcaveAtom)
        if repr_parse:
            return expr.get_K_repr(**kwargs, switched=switched)
        else:
            convex_vars = (
                expr.get_convex_variables() if not switched else expr.get_concave_variables()
            )
            concave_vars = (
                expr.get_concave_variables() if not switched else expr.get_convex_variables()
            )
            self.add_to_convex_vars(convex_vars)
            self.add_to_concave_vars(concave_vars)

    def parse_bilin(
        self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs: dict
    ) -> KRepresentation | None:
        if repr_parse:
            raise ValueError("Unexpected; should fail in variable parse.")
        else:
            if all(arg.is_affine() for arg in expr.args):
                raise ValueError("Use inner instead for bilinear forms.")
            else:
                raise ValueError("Use convex_concave_inner instead.")

    def parse_expr_variables(self, expr: cp.Expression, switched: bool, **kwargs: dict) -> None:
        self._parse_expr(expr, switched, repr_parse=False, **kwargs)

    def parse_expr_repr(
        self, expr: cp.Expression, switched: bool, local_to_glob: LocalToGlob
    ) -> KRepresentation:

        K_repr = self._parse_expr(expr, switched, repr_parse=True, local_to_glob=local_to_glob)
        assert isinstance(K_repr, KRepresentation)
        return K_repr

    def _parse_expr(
        self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs: dict
    ) -> KRepresentation | None:

        # constant
        if not expr.variables():
            assert expr.size == 1
            return (
                KRepresentation.constant_repr(expr.value * (1 if not switched else -1))
                if repr_parse
                else None
            )

        # known curvature
        elif repr_parse and (
            (set(expr.variables()) <= self.convex_vars)
            or (set(expr.variables()) <= self.concave_vars)
        ):
            return self.parse_known_curvature_repr(expr * (1 if not switched else -1), **kwargs)
        elif (not repr_parse) and (expr.is_convex() or expr.is_concave()):
            return self.parse_known_curvature_vars(expr, switched)

        # convex and concave variables
        elif isinstance(expr, AddExpression):
            return self.parse_add(expr, switched, repr_parse, **kwargs)
        elif isinstance(expr, NegExpression):
            return self._parse_expr(expr.args[0], not switched, repr_parse, **kwargs)
        elif isinstance(expr, multiply):
            if expr.args[0].is_constant() or expr.args[1].is_constant():
                return self.parse_scalar_mul(expr, switched, repr_parse, **kwargs)
            else:
                return self.parse_bilin(expr, switched, repr_parse, **kwargs)
        elif isinstance(expr, ConvexConcaveAtom):
            return self.parse_dspp_atom(expr, switched, repr_parse, **kwargs)
        elif isinstance(expr, MulExpression):
            if expr.is_affine() and repr_parse:
                split_up_affine = split_K_repr_affine(expr, self.convex_vars, self.concave_vars)
                K_reprs = [self.parse_expr_repr(arg, switched, **kwargs) for arg in split_up_affine]
                return KRepresentation.sum_of_K_reprs(K_reprs)
            else:
                return self.parse_bilin(expr, switched, repr_parse, **kwargs)
        else:
            raise ValueError