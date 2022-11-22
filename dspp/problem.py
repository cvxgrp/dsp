from __future__ import annotations

import itertools
from typing import Iterable

import cvxpy as cp
import numpy as np
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


class AffineVariableError(Exception):
    pass


def affine_error_message(affine_vars: list[cp.Variable]) -> str:
    return (
        f"Cannot resolve curvature of variables {[v.name() for v in affine_vars]}. "
        f"Specify curvature of these variables as "
        f"SaddleProblem(obj, constraints, minimization_vars, maximization_vars)."
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
            if all(arg.is_affine() for arg in expr.args):
                expr.args[0] * (-1 if switched else 1)
                conv_ind = 0 if (set(expr.args[0].variables()) <= self.convex_vars) else 1
                Fx = expr.args[conv_ind]
                Gy = expr.args[1 - conv_ind]
                assert set(Gy.variables()) <= self.concave_vars
                return K_repr_bilin(Fx, Gy, **kwargs)
            else:
                raise ValueError("Use GeneralBilinAtom instead.")
        else:
            if all(arg.is_affine() for arg in expr.args):
                return None
            else:
                raise NotImplementedError

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


class MinimizeMaximize:
    def __init__(self, expr: cp.Expression) -> None:
        self.expr = Atom.cast_to_const(expr)
        self._validate_arguments(expr)

    @staticmethod
    def _validate_arguments(expr: cp.Expression | float | int) -> None:
        if isinstance(expr, cp.Expression):
            assert expr.size == 1
        elif isinstance(expr, (float, int)):
            pass
        else:
            raise TypeError(f"Cannot parse {expr=}")


class SaddleProblem(cp.Problem):
    def __init__(
        self,
        minmax_objective: MinimizeMaximize | Objective,
        constraints: list[Constraint | RobustConstraint] | None = None,
        minimization_vars: Iterable[cp.Variable] | None = None,
        maximization_vars: Iterable[cp.Variable] | None = None,
    ) -> None:
        self._validate_arguments(minmax_objective)
        self.obj_expr = minmax_objective.expr

        # Optional inputs
        minimization_vars = set(minimization_vars) if minimization_vars is not None else set()
        maximization_vars = set(maximization_vars) if maximization_vars is not None else set()
        self._constraints_ = list(constraints) if constraints is not None else []  # copy

        # Handle explicit minimization and maximization objective
        minimization_vars, maximization_vars = self.handle_single_curvature_objective(
            minmax_objective, minimization_vars, maximization_vars
        )
        self._minimization_vars = minimization_vars
        self._maximization_vars = maximization_vars

        # constraints_x, single_obj_x = self.dualized_problem(
        #     self.obj_expr, constraints, minimization_vars, maximization_vars
        # )

        # note the variables are switched, and the problem value will be negated
        # constraints_y, single_obj_y = self.dualized_problem(
        #     -self.obj_expr, constraints, maximization_vars, minimization_vars
        # )

        # self.x_prob = cp.Problem(single_obj_x, constraints_x)
        # self.y_prob = cp.Problem(single_obj_y, constraints_y)

        self._x_prob = None
        self._y_prob = None

        self._value: float | None = None
        self._status: str | None = None
        super().__init__(cp.Minimize(self.obj_expr))

    @property
    def x_prob(self) -> cp.Problem:
        if self._x_prob is None:
            constraints_x, single_obj_x = self.dualized_problem(
                self.obj_expr, self._constraints_, self._minimization_vars, self._maximization_vars
            )
            self._x_prob = cp.Problem(single_obj_x, constraints_x)
        return self._x_prob

    @property
    def y_prob(self) -> cp.Problem:
        if self._y_prob is None:
            # note the variables are switched, and the problem value will be negated
            constraints_y, single_obj_y = self.dualized_problem(
                -self.obj_expr, self._constraints_, self._maximization_vars, self._minimization_vars
            )
            self._y_prob = cp.Problem(single_obj_y, constraints_y)
        return self._y_prob

    @staticmethod
    def handle_single_curvature_objective(
        objective: MinimizeMaximize | Objective,
        minimization_vars: list[cp.Variable],
        maximization_vars: list[cp.Variable],
    ) -> tuple[list[cp.Variable], list[cp.Variable]]:
        if isinstance(objective, cp.Minimize):
            vars = set(objective.variables())
            assert not vars & maximization_vars
            minimization_vars |= vars
        elif isinstance(objective, cp.Maximize):
            vars = set(objective.variables())
            assert not vars & minimization_vars
            maximization_vars |= vars

        return minimization_vars, maximization_vars

    def dualized_problem(
        self,
        obj_expr: cp.Expression,
        constraints: list[Constraint],
        minimization_vars: Iterable[cp.Variable],
        maximization_vars: Iterable[cp.Variable],
    ) -> tuple[list[Constraint], Objective]:

        parser = Parser(minimization_vars, maximization_vars)
        parser.parse_expr_variables(obj_expr, switched=False)

        constraints = list(constraints)  # make copy

        x_constraints, y_constraints = self._split_constraints(constraints, parser)

        assert not parser.affine_vars, affine_error_message(parser.affine_vars)

        x_constraint_vars = set(
            itertools.chain.from_iterable(constraint.variables() for constraint in x_constraints)
        )
        y_constraint_vars = set(
            itertools.chain.from_iterable(constraint.variables() for constraint in y_constraints)
        )
        prob_vars = x_constraint_vars | y_constraint_vars | set(obj_expr.variables())
        assert (
            parser.convex_vars | parser.concave_vars == prob_vars
        ), "Likely passed unused variables"

        local_to_glob_y = LocalToGlob(parser.convex_vars, parser.concave_vars)

        K_repr = parser.parse_expr_repr(obj_expr, switched=False, local_to_glob=local_to_glob_y)

        single_obj, constraints = minimax_to_min(
            K_repr, x_constraints, y_constraints, parser.concave_vars, local_to_glob_y
        )

        return constraints, single_obj

    @staticmethod
    def _validate_arguments(minmax_objective: MinimizeMaximize) -> None:
        assert isinstance(minmax_objective, (MinimizeMaximize, Objective))

    def _split_constraints(
        self, constraints: list[Constraint | RobustConstraint], parser: Parser
    ) -> tuple[list[Constraint], list[Constraint]]:
        n_constraints = len(constraints)
        x_constraints = []
        y_constraints = []

        while constraints:
            con_len = len(constraints)
            for c in list(constraints):
                if isinstance(c, RobustConstraint):
                    c_vars = c.vars
                    constraints += c.robust_constraints
                    constraints.remove(c)
                    n_constraints = n_constraints - 1 + len(c.robust_constraints)
                    break

                c_vars = set(c.variables())
                if c_vars & parser.convex_vars:
                    assert not (c_vars & parser.concave_vars)
                    x_constraints.append(c)
                    constraints.remove(c)
                    parser.convex_vars |= c_vars
                    parser.affine_vars -= c_vars
                elif c_vars & parser.concave_vars:
                    assert not (c_vars & parser.convex_vars)
                    y_constraints.append(c)
                    constraints.remove(c)
                    parser.concave_vars |= c_vars
                    parser.affine_vars -= c_vars

            if con_len == len(constraints):
                raise ValueError(
                    "Cannot split constraints, specify minimization_vars and " "maximization_vars"
                )

        assert len(x_constraints) + len(y_constraints) == n_constraints
        assert not (parser.convex_vars & parser.concave_vars)

        return x_constraints, y_constraints

    def solve(self, eps: float = 1e-3, *args, **kwargs: dict) -> None:  # noqa
        self.x_prob.solve(*args, **kwargs)
        assert self.x_prob.status == cp.OPTIMAL

        self.y_prob.solve(*args, **kwargs)
        assert self.y_prob.status == cp.OPTIMAL

        diff = self.x_prob.value + self.y_prob.value  # y_prob.value is negated
        assert np.isclose(
            diff, 0, atol=eps
        ), f"Difference between x and y problem is {diff}, (should be 0)."

        self._status = cp.OPTIMAL
        self._value = self.x_prob.value

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def value(self) -> float | None:
        return self._value

    def is_dspp(self) -> bool:
        raise NotImplementedError
        # Currently if I construct a SaddleProblem with a non-DSPP objective, it
        # will break on asserts, rather than on solve.


def form_robust_constraints(
    expr: cp.Expression, eta: cp.Constant | float, constraints: list[Constraint], mode: str = "sup"
) -> list[Constraint]:
    assert isinstance(expr, cp.Expression)
    # TODO: better handling of DSPP-ness of constraint
    # TODO: handle requiring y constraints (fails without any y constraints)

    # expr = expr if mode == 'sup' else -expr
    aux_prob = SaddleProblem(MinimizeMaximize(expr), constraints)

    prob = aux_prob.x_prob if mode == "sup" else aux_prob.y_prob
    obj = prob.objective.expr
    constraints = prob.constraints

    return constraints + ([obj <= eta] if mode == "sup" else [obj <= -eta])


class RobustConstraint:  # TODO: rename?
    r"""
    Implements the robust constraints :math:`\sup_{y_\mathcal{Y}}f(x,y)
    \leq eta` or :math:`\inf_{x_\mathcal{Y}}f(x,y) \geq eta` where
    :math:`f(x,y) =` `expr` is a DSPP expression, and robust_constraints are the constraints
    on :math:`y_\mathcal{Y}`. The direction of the inequality depends on the
    argument `mode`, which takes on values `inf` or `sup`. The default is `sup`.
    """

    def __init__(
        self,
        expr: cp.Expression,
        eta: cp.Constant | float,
        constraints: list[Constraint],
        mode: str = "sup",
    ) -> None:
        self.expr = expr
        self.eta = eta
        assert mode in ["inf", "sup"], "Not a valid robust constraint mode."
        self.mode = mode
        self.robust_constraints = form_robust_constraints(expr, eta, constraints, mode)
        self.vars = set(
            itertools.chain.from_iterable([c.variables() for c in self.robust_constraints])
        )
