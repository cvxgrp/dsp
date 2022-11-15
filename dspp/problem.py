from __future__ import annotations

from typing import Iterable

import numpy as np

import cvxpy as cp
from cvxpy import multiply
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from dspp.atoms import ConvexConcaveAtom
from dspp.cone_transforms import minimax_to_min, KRepresentation, K_repr_by, K_repr_ax, LocalToGlob, \
    split_K_repr_affine
from cvxpy.atoms.affine.unary_operators import NegExpression


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

    def scalar_mul(self, expr: cp.Expression, local_to_glob: LocalToGlob, switched: bool):
        assert expr.args[0].is_constant() or expr.args[1].is_constant()
        const_ind = 0 if isinstance(expr.args[0], cp.Constant) else 1
        var_ind = 1 - const_ind

        s = expr.args[const_ind]
        var_expr = expr.args[var_ind]

        if s.is_nonneg():
            return self.parse_expr(var_expr, local_to_glob).scalar_multiply(s.value)
        else:
            return self.parse_expr(var_expr, local_to_glob, not switched).scalar_multiply(abs(s.value))

    def parse_known_curvature(self, expr: cp.Expression, local_to_glob: LocalToGlob):
        if set(expr.variables()) <= self.convex_vars:
            assert expr.is_convex()
            return K_repr_ax(expr)
        elif set(expr.variables()) <= self.concave_vars:
            assert expr.is_concave()
            return K_repr_by(expr, local_to_glob)
        else:
            raise ValueError 
    
    def parse_expr(self, expr: cp.Expression, local_to_glob: LocalToGlob, switched = False) -> KRepresentation:
        
        print(expr, switched)
        
        # constant
        if not expr.variables():
            assert isinstance(expr, cp.Constant)
            assert expr.size == 1
            return KRepresentation.constant_repr(expr.value * (1 if not switched else -1))
        
        # known curvature
        elif set(expr.variables) <= self.convex_vars or set(expr.variables) <= self.concave_vars:
            return self.parse_known_curvature(expr * (1 if not switched else -1), local_to_glob)
        
        # convex and concave variables
        elif set(expr.variables()) & self.convex_vars & self.concave_vars:
            if isinstance(expr, AddExpression):
                K_reprs = [self.parse_expr(arg, local_to_glob, switched) for arg in expr.args]
                return KRepresentation.sum_of_K_reprs(K_reprs)
            elif isinstance(expr, NegExpression):
                return self.parse_expr(expr.args[0], local_to_glob, not switched)
            elif isinstance(expr, multiply):
                if expr.args[0].is_constant() or expr.args[1].is_constant():
                    return self.scalar_mul(expr, local_to_glob, switched)
            elif isinstance(expr, ConvexConcaveAtom):
                return expr.get_K_repr(local_to_glob, switched)
            else:
                raise ValueError
        else:
            raise ValueError


        if isinstance(expr, cp.Constant):
            assert expr.size == 1
            return KRepresentation.constant_repr(expr.value)

        elif isinstance(expr, cp.Variable):
            assert expr.shape == ()
            if expr in self.convex_vars:
                return KRepresentation(
                    f=cp.Constant(0),
                    t=expr,
                    constraints=[],
                )
            elif expr in self.concave_vars:
                return K_repr_by(expr, local_to_glob)
            else:
                raise ValueError
        elif isinstance(expr, ConvexConcaveAtom):
            return expr.get_K_repr(local_to_glob)
        elif isinstance(expr, cp.Expression):
            if isinstance(expr, AddExpression):
                K_reprs = [self.parse_expr(arg, local_to_glob) for arg in expr.args]
                return KRepresentation.sum_of_K_reprs(K_reprs)
            elif expr.is_affine():
                if set(expr.variables()) <= self.convex_vars:
                    assert not (set(expr.variables()) & self.concave_vars)
                    return K_repr_ax(expr)
                elif set(expr.variables()) <= self.concave_vars:
                    assert not (set(expr.variables()) & self.convex_vars)
                    return K_repr_by(expr, local_to_glob)
                else:
                    split_up_affine = split_K_repr_affine(expr, self.convex_vars, self.concave_vars)
                    K_reprs = [self.parse_expr(arg, local_to_glob) for arg in split_up_affine]
                    return KRepresentation.sum_of_K_reprs(K_reprs)
            elif expr.is_convex():
                return K_repr_ax(expr)
            elif expr.is_concave():
                return K_repr_by(expr, local_to_glob)
            elif isinstance(expr, NegExpression):
                if isinstance(expr.args[0], ConvexConcaveAtom):
                    dspp_atom = expr.args[0]
                    return dspp_atom.get_K_repr(local_to_glob, switched=True)
                elif isinstance(expr.args[0], AddExpression):
                    K_reprs = [self.parse_expr(-arg, local_to_glob) for arg in expr.args[0].args]
                    return KRepresentation.sum_of_K_reprs(K_reprs)
                elif isinstance(expr.args[0], NegExpression):  # double negation
                    dspp_atom = expr.args[0].args[0]
                    return self.parse_expr(dspp_atom, local_to_glob)
                elif isinstance(expr.args[0], multiply):  # negated multiplication of dspp atom
                    mult = expr.args[0]
                    s = mult.args[0]
                    assert isinstance(s, cp.Constant)
                    return self.parse_expr(-s.value * mult.args[1], local_to_glob)
                else:
                    raise ValueError
            elif isinstance(expr, multiply):
                assert expr.shape == ()
                assert len(expr.args) == 2
                assert expr.args[0].is_constant()
                assert isinstance(expr.args[1], ConvexConcaveAtom)
                dspp_atom = expr.args[1]
                if expr.args[0].is_nonneg():
                    return dspp_atom.get_K_repr(local_to_glob).scalar_multiply(expr.args[0].value)
                elif expr.args[0].is_nonpos():
                    return dspp_atom.get_K_repr(local_to_glob, switched=True).scalar_multiply(
                        abs(expr.args[0].value))
                else:
                    raise ValueError
            else:
                raise TypeError(f'Cannot parse {expr=}')
        else:
            raise TypeError(f'Cannot parse {expr=}')

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
        parser.split_up_variables(obj_expr)

        constraints = list(constraints) if constraints is not None else []  # copy

        x_constraints, y_constraints = self._split_constraints(constraints, parser)

        assert not parser.affine_vars, affine_error_message(parser.affine_vars)

        local_to_glob_y = LocalToGlob(parser.concave_vars)

        K_repr = parser.parse_expr(obj_expr, local_to_glob_y)

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
            (list[Constraint], list[Constraint]):
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

    def solve(self, eps=1e-4):
        self.x_prob.solve()
        assert self.x_prob.status == cp.OPTIMAL

        self.y_prob.solve()
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
