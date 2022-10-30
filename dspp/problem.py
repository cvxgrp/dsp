from __future__ import annotations

import numpy as np

import cvxpy as cp
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.constraints.constraint import Constraint
from dspp.nemirovski import minimax_to_min, K_repr_x_Gy


class MinimizeMaximize:

    def __init__(self, inner_product: cp.Expression):
        assert isinstance(inner_product, MulExpression)
        self.F_x: cp.Expression = inner_product.args[0]
        self.G_y: cp.Expression = inner_product.args[1]
        self.x_vars = self.F_x.variables()
        self.y_vars = self.G_y.variables()
        self._validate_arguments(inner_product)

    def _validate_arguments(self, inner_product):
        assert inner_product.shape == ()
        assert self.F_x.is_convex()
        assert self.G_y.is_concave()
        # assert (self.F_x.is_nonneg() and self.G_y.is_nonneg())
               # (self.F_x.is_nonpos() and self.F_y.is_nonpos())
        assert not (set(self.x_vars) & set(self.y_vars))
        assert len(self.x_vars) == 1
        assert len(self.y_vars) == 1


class SaddleProblem(cp.Problem):
    def __init__(self, objective: MinimizeMaximize, constraints=None):
        if constraints is None:
            constraints = []
        self.minmax_objective = objective
        self.x_vars = objective.x_vars
        self.y_vars = objective.y_vars
        self._validate_arguments()
        self.x_constraints, self.y_constraints = self._split_constraints(constraints)

        # x = self.x_vars[0]
        #
        # dual_constr_x = [self.minmax_objective.F_x <= z]
        #
        # var_to_mat_mapping, s_hat, cone_dims = get_cone_repr(dual_constr_x, [z, x])
        #
        # R_hat = var_to_mat_mapping[x.id]
        # S_hat = var_to_mat_mapping[z.id]
        # Q_hat = var_to_mat_mapping['eta']
        #
        # x_cone_repr = R_hat @ x + S_hat @ z + s_hat
        # if Q_hat.shape[1] > 0:
        #     v = cp.Variable(Q_hat.shape[1])
        #     x_cone_repr += Q_hat @ v
        #
        # x_cone_const = add_cone_constraints(x_cone_repr, cone_dims)


        z = cp.Variable(self.minmax_objective.F_x.shape)
        epigraph_constraint = [z >= self.minmax_objective.F_x]

        K = K_repr_x_Gy(self.minmax_objective.G_y, z)
        single_obj, constraints = minimax_to_min(K,
                                                 self.x_constraints + epigraph_constraint,
                                                 self.y_constraints
                                                 )

        super().__init__(single_obj, constraints)

    def _validate_arguments(self):
        assert isinstance(self.minmax_objective, MinimizeMaximize)

    def _split_constraints(self, constraints: list[Constraint]) -> (list[Constraint], list[Constraint]):
        n_constraints = len(constraints)
        x_constraints = []
        x_constraints_vars = set(self.x_vars)
        y_constraints = []
        y_constraints_vars = set(self.y_vars)

        while constraints:
            con_len = len(constraints)
            for c in list(constraints):
                c_vars = set(c.variables())
                if c_vars & x_constraints_vars:
                    assert not (c_vars & y_constraints_vars)
                    x_constraints.append(c)
                    constraints.remove(c)
                elif c_vars & y_constraints_vars:
                    assert not (c_vars & x_constraints_vars)
                    y_constraints.append(c)
                    constraints.remove(c)

            if con_len == len(constraints):
                raise ValueError

        assert len(x_constraints) + len(y_constraints) == n_constraints
        assert not (x_constraints_vars & y_constraints_vars)
        return x_constraints, y_constraints

    def solve(self):
        super(SaddleProblem, self).solve()


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

                plot_array[k-1] = current_array

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

        gap = abs(maximization_obj_value-minimization_obj_value)
        print(f'Saddle point objective gap: {gap:.6f},\n'
              f'max. obj.={maximization_obj_value:.6f},\n'
              f'min. obj.={minimization_obj_value:.6f}')
