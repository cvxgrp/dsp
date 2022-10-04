from __future__ import annotations

import cvxpy as cp


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

        all_variables = self.variables
        non_fixed_vars = [v for v in all_variables if not {v} & set(fixed_vars)]

        fixed_obj = self.fix_vars_in_expr(self.objective, fixed_vars)

        relevant_constraints = []
        for constraint in self.constraints:
            constraint_vars = set(constraint.variables())
            if set(fixed_vars) & constraint_vars:
                assert not set(non_fixed_vars) & constraint_vars
                # fixed_constraints.append(self.fix_vars_in_expr(constraint, fixed_vars))
            else:
                assert set(non_fixed_vars) & constraint_vars
                relevant_constraints.append(constraint)

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

    def _solve_DR(self, max_iters: int = 50, alpha=0.5):

        self.initialize_variables()

        for k in range(max_iters):

            # 1. Check stopping criterion
            max_vars_prev = [v.value for v in self.max_vars]

            # 2. Maximization
            max_obj, max_constr = self.fix_vars(self.min_vars)
            prox_terms = [cp.sum_squares(v - v.value) for v in self.max_vars]
            max_obj -= 1 / (2 * alpha) * cp.sum(cp.hstack(prox_terms))
            maximization_problem = cp.Problem(cp.Maximize(max_obj), max_constr)
            maximization_problem.solve()

            max_vars_post_solve = [v.value for v in self.max_vars]

            # 3. Minimization
            for max_var_i, max_var_prev_i in zip(self.max_vars, max_vars_prev):
                max_var_i.value = 2 * max_var_i.value - max_var_prev_i

            min_obj, min_constr = self.fix_vars(self.max_vars)
            prox_terms = [cp.sum_squares(v - v.value) for v in self.min_vars]
            min_obj += 1 / (2 * alpha) * cp.sum(cp.hstack(prox_terms))
            minimization_problem = cp.Problem(cp.Minimize(min_obj), min_constr)
            minimization_problem.solve()

            for max_var_i, max_var_post_solve_i in zip(self.max_vars, max_vars_post_solve):
                max_var_i.value = max_var_post_solve_i

            print(self.min_vars[0].value, self.max_vars[0].value)

    def initialize_variables(self):
        _, min_constr = self.fix_vars(self.max_vars)
        prob = cp.Problem(cp.Minimize(0), min_constr)
        prob.solve()

        _, max_constr = self.fix_vars(self.min_vars)
        prob = cp.Problem(cp.Maximize(0), max_constr)
        prob.solve()

