from __future__ import annotations

from typing import Iterable

import numpy as np

import cvxpy as cp
from cvxpy import multiply
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.constraint import Constraint
from dspp.atoms import dspp_atoms, ConvexConcaveAtom, switch_convex_concave
from dspp.cone_transforms import minimax_to_min, KRepresentation, K_repr_by, K_repr_ax, LocalToGlob
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

        self.convex_vars = convex_vars if convex_vars is not None else set()
        self.concave_vars = concave_vars if concave_vars is not None else set()
        assert not (self.convex_vars & self.concave_vars)
        self.affine_vars = set()

    def split_up_variables(self, expr: cp.Expression | ConvexConcaveAtom):
        if isinstance(expr, cp.Constant) or isinstance(expr, (float, int)):
            return
        elif isinstance(expr, dspp_atoms):
            assert isinstance(expr, ConvexConcaveAtom)
            self.add_to_convex_vars(expr.get_convex_variables())
            self.add_to_concave_vars(expr.get_concave_variables())
        elif isinstance(expr, AddExpression):
            for arg in expr.args:
                self.split_up_variables(arg)
        elif expr.is_affine():
            if set(expr.variables()) & self.convex_vars:
                self.add_to_convex_vars(expr.variables())
            elif set(expr.variables()) & self.concave_vars:
                self.add_to_concave_vars(expr.variables())
            else:
                self.affine_vars |= set(expr.variables())
        elif expr.is_convex():
            self.add_to_convex_vars(expr.variables())
        elif expr.is_concave():
            self.add_to_concave_vars(expr.variables())
        elif isinstance(expr, NegExpression):
            dspp_atom = expr.args[0]
            assert isinstance(dspp_atom, ConvexConcaveAtom)
            self.add_to_concave_vars(dspp_atom.get_convex_variables())
            self.add_to_convex_vars(dspp_atom.get_concave_variables())
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

    def parse_expr(self, expr: cp.Expression, local_to_glob: LocalToGlob) -> KRepresentation:
        if isinstance(expr, cp.Constant):
            assert expr.shape == ()
            return KRepresentation.constant_repr(expr.value)
        elif isinstance(expr, (float, int)):
            return KRepresentation.constant_repr(expr)
        elif isinstance(expr, cp.Variable):
            assert expr.shape == ()
            if expr in self.convex_vars:
                return KRepresentation(
                    f=cp.Constant(0),
                    t=expr,
                    x=cp.Constant(0),
                    y=cp.Constant(0),
                    constraints=[],
                )
            elif expr in self.concave_vars:
                return K_repr_by(expr, local_to_glob)
            else:
                raise ValueError
        elif isinstance(expr, dspp_atoms):
            assert isinstance(expr, ConvexConcaveAtom)
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
                    raise ValueError('Affine expressions may not contain variables of mixed '
                                     'curvature.')
            elif expr.is_convex():
                return K_repr_ax(expr)
            elif expr.is_concave():
                return K_repr_by(expr, local_to_glob)
            elif isinstance(expr, NegExpression):
                dspp_atom = expr.args[0]
                assert isinstance(dspp_atom, ConvexConcaveAtom)
                return dspp_atom.get_K_repr(local_to_glob, switched=True)
            elif isinstance(expr, multiply):
                assert expr.shape == ()
                assert len(expr.args) == 2
                assert expr.args[0].is_constant()
                assert isinstance(expr.args[1], ConvexConcaveAtom)
                if expr.args[0].is_nonneg():
                    return self.parse_expr(expr.args[1], local_to_glob).scalar_multiply(
                        expr.args[0].value)
                elif expr.args[0].is_nonpos():
                    K_repr_pos = self.parse_expr(expr.args[1], local_to_glob)
                    switched_K_repr = switch_convex_concave(K_repr_pos)
                    return switched_K_repr.scalar_multiply(abs(expr.args[0].value))
                else:
                    raise ValueError
            else:
                raise TypeError(f'Cannot parse {expr=}')
        else:
            raise TypeError(f'Cannot parse {expr=}')


class MinimizeMaximize:

    def __init__(self, expr: cp.Expression):
        self.expr = expr
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
    def __init__(self, objective: MinimizeMaximize, constraints=None, minimization_vars=None,
                 maximization_vars=None):
        self.minmax_objective = objective
        self._validate_arguments()

        parser = Parser(minimization_vars, maximization_vars)
        parser.split_up_variables(self.minmax_objective.expr)

        constraints = list(constraints) if constraints is not None else []  # copy

        self.x_constraints, self.y_constraints = self._split_constraints(constraints, parser)

        assert not parser.affine_vars, affine_error_message(parser.affine_vars)
        # TODO assert convex + concave variables are all the variables of the problem

        self.x_vars = parser.convex_vars
        self.y_vars = parser.concave_vars

        local_to_glob_y = LocalToGlob(self.y_vars)

        K_repr = parser.parse_expr(objective.expr, local_to_glob_y)

        single_obj, constraints = minimax_to_min(K_repr,
                                                 self.x_constraints,
                                                 self.y_constraints
                                                 )

        super().__init__(single_obj, constraints)

    def _validate_arguments(self):
        assert isinstance(self.minmax_objective, MinimizeMaximize)

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

    def solve(self):
        super(SaddleProblem, self).solve()
        # TODO: populate y vals by solving additional optimization problem


class SaddlePointProblem:

    def __init__(self,
                 objective: cp.Expression,
                 constraints,
                 min_vars: list[cp.Variable],
                 max_vars: list[cp.Variable]
                 ):

        self.objective = objective
        self.constraints = constraints
        self.min_vars = min_vars
        self.max_vars = max_vars

        self._validate_inputs()

    @property
    def variables(self) -> list[cp.Variable]:
        return cp.Problem(cp.Minimize(self.objective), self.constraints).variables()

    def is_dspp(self) -> bool:
        min_obj, min_constr = self.fix_vars(self.max_vars)
        minimization_problem = cp.Problem(cp.Minimize(min_obj), min_constr)

        max_obj, max_constr = self.fix_vars(self.min_vars)
        maximization_problem = cp.Problem(cp.Maximize(max_obj), max_constr)

        return minimization_problem.is_dcp() and maximization_problem.is_dcp()

    def solve(self, method: str = 'DR', **kwargs) -> None:
        if method == 'DR':
            return self._solve_DR(**kwargs)
        else:
            raise ValueError

    def fix_vars(self, fixed_vars: list[cp.Variable]):

        assert set(fixed_vars) == set(self.min_vars) or set(fixed_vars) == set(self.max_vars)
        min_and_max_vars = self.min_vars + self.max_vars
        non_fixed_vars = [v for v in min_and_max_vars if not {v} & set(fixed_vars)]

        # assert set(self.objective.variables()) == set(min_and_max_vars)
        fixed_obj = self.fix_vars_in_expr(self.objective, fixed_vars)

        relevant_constraints = []
        for constraint in self.constraints:
            constraint_vars = set(constraint.variables())
            if set(fixed_vars) & constraint_vars:
                assert not set(non_fixed_vars) & constraint_vars
            elif set(non_fixed_vars) & constraint_vars:
                assert not set(fixed_vars) & constraint_vars
                relevant_constraints.append(constraint)
            else:
                raise ValueError

        return fixed_obj, relevant_constraints

    @staticmethod
    def fix_vars_in_expr(expr: cp.Expression, variables: list[cp.Variable]) -> cp.Expression:
        expr_copy = expr.copy()
        assert isinstance(expr_copy.args, list)
        if len(expr_copy.args) == 0:
            if isinstance(expr_copy, cp.Variable):
                if {expr_copy} & set(variables):
                    return cp.Parameter(expr_copy.shape, value=expr_copy.value,
                                        **expr_copy.attributes)
            return expr_copy
        else:
            expr_copy.args = [SaddlePointProblem.fix_vars_in_expr(arg, variables) for arg in
                              expr_copy.args]
            return expr_copy

    def _validate_inputs(self):
        assert len(self.min_vars) > 0
        assert len(self.max_vars) > 0

    def _solve_DR(self, max_iters: int = 50, alpha=1, eps: float = 1e-4):
        alpha = 2

        self.initialize_variables()

        plot_array = np.zeros((max_iters - 1, sum(v.size for v in self.variables)))

        for k in range(max_iters):

            # if k > 0:
            #     for min_var_i, hist in zip(self.min_vars, min_vars_hist):
            #         min_var_i.value = np.mean(hist, axis=0)

            # 1. Maximization
            max_obj, max_constr = self.fix_vars(self.min_vars)
            prox_terms = [cp.sum_squares(v - v.value) for v in self.max_vars]
            max_obj -= alpha * cp.sum(cp.hstack(prox_terms))
            maximization_problem = cp.Problem(cp.Maximize(max_obj), max_constr)
            maximization_problem.solve(verbose=False)
            assert maximization_problem.status == cp.OPTIMAL

            if k == 0:
                max_vars_hist = [[v.value] for v in self.max_vars]
            else:
                for hist, new_val in zip(max_vars_hist, self.max_vars):
                    hist.append(new_val.value)

            # 2. Minimization
            # if k > 0:
            #     for max_var_i, hist in zip(self.max_vars, max_vars_hist):
            #         max_var_i.value = np.mean(hist, axis=0)

            min_obj, min_constr = self.fix_vars(self.max_vars)
            prox_terms = [cp.sum_squares(v - v.value) for v in self.min_vars]
            min_obj += alpha * cp.sum(cp.hstack(prox_terms))
            minimization_problem = cp.Problem(cp.Minimize(min_obj), min_constr)
            minimization_problem.solve(verbose=False)
            assert minimization_problem.status == cp.OPTIMAL

            if k == 0:
                min_vars_hist = [[v.value] for v in self.min_vars]
            else:
                for hist, new_val in zip(min_vars_hist, self.min_vars):
                    hist.append(new_val.value)

            # 3. Check stopping criterion

            if k > 0:
                current_array = np.hstack(
                    [np.mean(v, axis=0).flatten() for v in min_vars_hist + max_vars_hist])

                plot_array[k - 1] = current_array

                if k > 1:
                    delta = np.linalg.norm(plot_array[k - 2] - current_array, np.inf)
                    print(k, delta)
                    if delta <= eps:
                        break

        for min_var_i, hist in zip(self.min_vars, min_vars_hist):
            min_var_i.value = np.mean(hist, axis=0)
        for max_var_i, hist in zip(self.max_vars, max_vars_hist):
            max_var_i.value = np.mean(hist, axis=0)
        self._validate_saddlepoint()

    def initialize_variables(self):
        _, min_constr = self.fix_vars(self.max_vars)
        # Make sure unconstrained variables are also in the problem
        zero_weighted_vars = 0 * cp.sum(cp.hstack([cp.vec(v) for v in self.min_vars]))
        prob = cp.Problem(
            cp.Minimize(0 + zero_weighted_vars), min_constr)
        prob.solve()
        assert prob.status == cp.OPTIMAL

        _, max_constr = self.fix_vars(self.min_vars)
        # Make sure unconstrained variables are also in the problem
        zero_weighted_vars = 0 * cp.sum(cp.hstack([cp.vec(v) for v in self.max_vars]))
        prob = cp.Problem(cp.Maximize(0 + zero_weighted_vars), max_constr)
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def _validate_saddlepoint(self):
        max_vars_pre_validation = [v.value for v in self.max_vars]
        min_vars_pre_validation = [v.value for v in self.min_vars]

        max_obj, max_constr = self.fix_vars(self.min_vars)
        maximization_problem = cp.Problem(cp.Maximize(max_obj), max_constr)
        maximization_problem.solve(verbose=False)
        assert maximization_problem.status == cp.OPTIMAL
        maximization_obj_value = maximization_problem.value

        for max_var_i, max_var_prev_i in zip(self.max_vars, max_vars_pre_validation):
            max_var_i.value = max_var_prev_i

        min_obj, min_constr = self.fix_vars(self.max_vars)
        minimization_problem = cp.Problem(cp.Minimize(min_obj), min_constr)
        minimization_problem.solve(verbose=False)
        assert minimization_problem.status == cp.OPTIMAL
        minimization_obj_value = minimization_problem.value

        for min_var_i, min_var_prev_i in zip(self.min_vars, min_vars_pre_validation):
            min_var_i.value = min_var_prev_i

        gap = abs(maximization_obj_value - minimization_obj_value)
        print(f'Saddle point objective gap: {gap:.6f},\n'
              f'max. obj.={maximization_obj_value:.6f},\n'
              f'min. obj.={minimization_obj_value:.6f}')
