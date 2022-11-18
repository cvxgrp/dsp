from __future__ import annotations

from typing import Iterable

import numpy as np

import cvxpy as cp
from cvxpy import multiply
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from dspp.atoms import ConvexConcaveAtom, convex_concave_inner
from dspp.cone_transforms import K_repr_bilin, minimax_to_min, KRepresentation, K_repr_by, K_repr_ax, LocalToGlob, \
    split_K_repr_affine
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.binary_operators import MulExpression


class AffineVariableError(Exception):
    pass


def affine_error_message(affine_vars) -> str:
    return f'Cannot resolve curvature of variables {[v.name() for v in affine_vars]}. ' \
           f'Specify curvature of these variables as ' \
           f'SaddleProblem(obj, constraints, minimization_vars, maximization_vars).'


class Parser:
    def __init__(self, convex_vars: set[cp.Variable],
                 concave_vars: set[cp.Variable]):

        self.convex_vars: set[cp.Variable] = convex_vars if convex_vars is not None else set()
        self.concave_vars: set[cp.Variable] = concave_vars if concave_vars is not None else set()
        assert not (self.convex_vars & self.concave_vars)
        self.affine_vars = set()

    def split_up_variables(self, expr: cp.Expression):

        if isinstance(expr, cp.Constant) or isinstance(expr, (float, int)):
            return
        elif isinstance(expr, ConvexConcaveAtom):
            self.add_to_convex_vars(expr.get_convex_variables())
            self.add_to_concave_vars(expr.get_concave_variables())
        elif isinstance(expr, AddExpression):
            for arg in expr.args:
                self.split_up_variables(arg)
        elif expr.is_affine():
            self.affine_vars |= (set(expr.variables()) - self.convex_vars - self.concave_vars)
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
            raise ValueError(f'Cannot parse {expr=} with {expr.curvature=}.')

    def add_to_convex_vars(self, variables: Iterable[cp.Variable]):
        variables = set(variables)
        assert not (variables & self.concave_vars), 'Cannot add variables to both ' \
                                                    'convex and concave set.'
        self.affine_vars -= variables
        self.convex_vars |= variables

    def add_to_concave_vars(self, variables: Iterable[cp.Variable]):
        variables = set(variables)
        assert not (variables & self.convex_vars), 'Cannot add variables to both ' \
                                                   'convex and concave set.'
        self.affine_vars -= variables
        self.concave_vars |= variables

    def parse_scalar_mul(self, expr: cp.Expression, switched: bool, repr_parse, **kwargs):
        assert expr.args[0].is_constant() or expr.args[1].is_constant()
        const_ind = 0 if isinstance(expr.args[0], cp.Constant) else 1
        var_ind = 1 - const_ind

        s = expr.args[const_ind]
        var_expr = expr.args[var_ind]

        if s.is_nonneg():
            return_val = self.parse_expr(var_expr, switched, repr_parse, **kwargs)
        else:
            return_val = self.parse_expr(var_expr, not switched, repr_parse, **kwargs)

        return return_val.scalar_multiply(abs(s.value)) if repr_parse else return_val

    def parse_known_curvature_repr(self, expr: cp.Expression, local_to_glob: LocalToGlob):
        if set(expr.variables()) <= self.convex_vars:
            assert expr.is_convex()
            return K_repr_ax(expr)
        elif set(expr.variables()) <= self.concave_vars:
            assert expr.is_concave()
            return K_repr_by(expr, local_to_glob)
        else:
            raise ValueError

    def parse_known_curvature_vars(self, expr: cp.Expression, switched):
        vars = expr.variables()
        if expr.is_affine():
            self.affine_vars |= (set(expr.variables()) - self.convex_vars - self.concave_vars)
        elif expr.is_convex():
            self.add_to_convex_vars(vars) if not switched else self.add_to_concave_vars(vars)
        elif expr.is_concave():
            self.add_to_concave_vars(vars) if not switched else self.add_to_convex_vars(vars)
        else:
            raise ValueError(f'Cannot parse {expr=} with {expr.curvature=}.')

    def parse_add(self, expr: cp.Expression, switched, repr_parse: bool, **kwargs):
        assert isinstance(expr, AddExpression)
        if repr_parse:
            K_reprs = [self.parse_expr(arg, switched, repr_parse, **kwargs) for arg in expr.args]
            return KRepresentation.sum_of_K_reprs(K_reprs)
        else:
            for arg in expr.args:
                self.parse_expr(arg, switched, repr_parse, **kwargs)

    def parse_dspp_atom(self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs):
        assert isinstance(expr, ConvexConcaveAtom)
        if repr_parse:
            return expr.get_K_repr(**kwargs, switched=switched)
        else:
            convex_vars = expr.get_convex_variables() if not switched else expr.get_concave_variables()
            concave_vars = expr.get_concave_variables() if not switched else expr.get_convex_variables()
            self.add_to_convex_vars(convex_vars)
            self.add_to_concave_vars(concave_vars)

    def parse_bilin(self, expr: cp.Expression, switched: bool, repr_parse: bool, **kwargs):
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

    def parse_expr(self, expr: cp.Expression, switched, repr_parse, **kwargs) -> KRepresentation:

        # constant
        if not expr.variables():
            assert expr.size == 1
            return KRepresentation.constant_repr(expr.value * (1 if not switched else -1)) if repr_parse else None

        # known curvature
        elif repr_parse and (set(expr.variables()) <= self.convex_vars or set(expr.variables()) <= self.concave_vars):
            return self.parse_known_curvature_repr(expr * (1 if not switched else -1), **kwargs)
        elif (not repr_parse) and (expr.is_convex() or expr.is_concave()):
            return self.parse_known_curvature_vars(expr, switched)

        # convex and concave variables
        elif isinstance(expr, AddExpression):
            return self.parse_add(expr, switched, repr_parse, **kwargs)
        elif isinstance(expr, NegExpression):
            return self.parse_expr(expr.args[0], not switched, repr_parse, **kwargs)
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
                K_reprs = [self.parse_expr(arg, switched, repr_parse, **kwargs)
                           for arg in split_up_affine]
                return KRepresentation.sum_of_K_reprs(K_reprs)
            else:
                return self.parse_bilin(expr, switched, repr_parse, **kwargs)
        else:
            raise ValueError

    def get_concave_objective(self, expr: cp.Expression) -> cp.Expression:
        if isinstance(expr, (float, int, cp.Constant)):
            return expr
        elif isinstance(expr, cp.Variable):
            if expr in self.convex_vars:
                return expr.value
            elif expr in self.concave_vars:
                return expr
            else:
                raise ValueError
        elif isinstance(expr, ConvexConcaveAtom):
            return expr.get_concave_objective()
        elif isinstance(expr, cp.Expression):
            if isinstance(expr, AddExpression):
                return cp.sum([self.get_concave_objective(arg) for arg in expr.args])
            elif expr.is_affine():
                if set(expr.variables()) <= self.convex_vars:
                    return expr.value
                elif set(expr.variables()) <= self.concave_vars:
                    return expr
                else:
                    raise ValueError('Affine expressions may not contain variables of mixed '
                                     'curvature.')
            elif expr.is_convex():
                return expr.value
            elif expr.is_concave():
                return expr
            elif isinstance(expr, NegExpression):
                return expr.args[0].get_concave_objective(switched=True)
            elif isinstance(expr, multiply):
                if isinstance(expr.args[0], cp.Constant):
                    return abs(expr.args[0].value) * expr.args[1].get_concave_objective(
                        switched=expr.args[0].is_nonpos())
                else:
                    raise ValueError
            else:
                raise TypeError(f'Cannot parse {expr=}')
        else:
            raise TypeError(f'Cannot parse {expr=}')


class MinimizeMaximize:

    def __init__(self, expr: cp.Expression):
        self.expr = Atom.cast_to_const(expr)
        self._validate_arguments(expr)

    @staticmethod
    def _validate_arguments(expr):
        if isinstance(expr, cp.Expression):
            assert expr.size == 1
        elif isinstance(expr, (float, int)):
            pass
        else:
            raise TypeError(f'Cannot parse {expr=}')


class SaddleProblem(cp.Problem):
    def __init__(self, minmax_objective: MinimizeMaximize, constraints=None, minimization_vars=None,
                 maximization_vars=None):
        self._validate_arguments(minmax_objective)
        self.obj_expr = minmax_objective.expr

        constraints_x, single_obj_x = self.dualized_problem(
            self.obj_expr, constraints, minimization_vars, maximization_vars)

        # note the variables are switched, and the problem value will be negated
        constraints_y, single_obj_y = self.dualized_problem(
            -self.obj_expr, constraints, maximization_vars, minimization_vars)

        self.x_prob = cp.Problem(single_obj_x, constraints_x)
        self.y_prob = cp.Problem(single_obj_y, constraints_y)

        self._value = None
        self._status = None
        super().__init__(cp.Minimize(self.obj_expr), constraints)

    def dualized_problem(self, obj_expr, constraints, minimization_vars, maximization_vars):
        parser = Parser(minimization_vars, maximization_vars)
        # parser.split_up_variables(obj_expr)
        parser.parse_expr(obj_expr, switched=False, repr_parse=False)

        constraints = list(constraints) if constraints is not None else []  # copy

        x_constraints, y_constraints = self._split_constraints(constraints, parser)

        assert not parser.affine_vars, affine_error_message(parser.affine_vars)

        local_to_glob_y = LocalToGlob(parser.convex_vars, parser.concave_vars)

        K_repr = parser.parse_expr(obj_expr, switched=False,
                                   repr_parse=True, local_to_glob=local_to_glob_y)

        single_obj, constraints = minimax_to_min(K_repr,
                                                 x_constraints,
                                                 y_constraints,
                                                 parser.concave_vars,
                                                 local_to_glob_y
                                                 )

        return constraints, single_obj

    @staticmethod
    def _validate_arguments(minmax_objective):
        assert isinstance(minmax_objective, MinimizeMaximize)

    def _split_constraints(self, constraints: list[Constraint], parser: Parser) -> \
            tuple[list[Constraint], list[Constraint]]:
        n_constraints = len(constraints)
        x_constraints = []
        y_constraints = []

        while constraints:
            con_len = len(constraints)
            for c in list(constraints):
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
                raise ValueError("Cannot split constraints, specify minimization_vars and "
                                 "maximization_vars")

        assert len(x_constraints) + len(y_constraints) == n_constraints
        assert not (parser.convex_vars & parser.concave_vars)

        return x_constraints, y_constraints

    def solve(self, eps=1e-3, **args):
        self.x_prob.solve(**args)
        assert self.x_prob.status == cp.OPTIMAL

        self.y_prob.solve(**args)
        assert self.y_prob.status == cp.OPTIMAL

        diff = self.x_prob.value + self.y_prob.value  # y_prob.value is negated
        assert np.isclose(diff, 0, atol=eps), \
            f"Difference between x and y problem is {diff}, (should be 0)."

        self._status = cp.OPTIMAL
        self._value = self.x_prob.value

    @property
    def status(self):
        return self._status

    @property
    def value(self):
        return self._value
