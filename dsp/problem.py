from __future__ import annotations

from itertools import chain
from typing import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective
from cvxpy.utilities import Canonical

from dsp.cone_transforms import (
    LocalToGlob,
    add_cone_constraints,
    get_cone_repr,
    minimax_to_min,
)
from dsp.parser import DSPError, Parser, initialize_parser
from dsp.saddle_extremum import SaddleExtremum


class MinimizeMaximize(Canonical):

    NAME = "minimize-maximize"

    def __init__(self, expr: cp.Expression) -> None:
        self._validate_arguments(expr)
        self.args = [cp.Expression.cast_to_const(expr)]

    @staticmethod
    def _validate_arguments(expr: cp.Expression | float | int) -> None:
        if isinstance(expr, cp.Expression):
            assert expr.size == 1
        else:
            assert isinstance(expr, (float, int)), f"Cannot parse {expr=}"

    def is_dsp(self) -> bool:
        return self.expr.is_dsp()

    @property
    def value(self) -> float:
        return self.expr.value

    def __str__(self) -> str:
        return ' '.join([self.NAME, self.args[0].name()])


class SaddlePointProblem(cp.Problem):
    def __init__(
        self,
        minmax_objective: MinimizeMaximize,
        constraints: list[Constraint] | None = None,
        minimization_vars: Iterable[cp.Variable] | None = None,
        maximization_vars: Iterable[cp.Variable] | None = None,
    ) -> None:
        self._validate_arguments(minmax_objective)
        self.obj_expr = minmax_objective.expr

        # Optional inputs
        self._minimization_vars = set(minimization_vars) if minimization_vars is not None else set()
        self._maximization_vars = set(maximization_vars) if maximization_vars is not None else set()
        self._constraints_ = list(constraints) if constraints is not None else []  # copy

        self._x_prob = None
        self._y_prob = None

        self._value: float | None = None
        self._status: str | None = None
        super().__init__(cp.Minimize(self.obj_expr), constraints)

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

    def dualized_problem(
        self,
        obj_expr: cp.Expression,
        constraints: list[Constraint],
        minimization_vars: Iterable[cp.Variable],
        maximization_vars: Iterable[cp.Variable],
    ) -> tuple[list[Constraint], Objective]:
        parser = initialize_parser(obj_expr, minimization_vars, maximization_vars, constraints)

        local_to_glob_y = LocalToGlob(parser.convex_vars, parser.concave_vars)

        K_repr = parser.parse_expr_repr(obj_expr, switched=False, local_to_glob=local_to_glob_y)

        single_obj, constraints = minimax_to_min(
            K_repr,
            parser.x_constraints,
            parser.y_constraints,
            list(parser.concave_vars),
            local_to_glob_y,
        )

        return constraints, single_obj

    @staticmethod
    def _validate_arguments(minmax_objective: MinimizeMaximize) -> None:
        assert isinstance(minmax_objective, MinimizeMaximize)

    def solve(self, eps: float = 1e-3, *args, **kwargs: dict) -> float:  # noqa
        """
        Solves the saddle point problem.
        """
        self.x_prob.solve(*args, **kwargs)
        assert self.x_prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, self.x_prob.status

        self.y_prob.solve(*args, **kwargs)
        assert self.y_prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}, self.y_prob.status

        diff = self.x_prob.value + self.y_prob.value  # y_prob.value is negated
        assert np.isclose(self.x_prob.value, -self.y_prob.value, atol=eps) and np.isclose(
            -self.y_prob.value, self.x_prob.value, atol=eps
        ), f"Difference between x and y problem is {diff}, (should be 0)."

        self._status = (
            cp.OPTIMAL_INACCURATE
            if cp.OPTIMAL_INACCURATE in {self.x_prob.status, self.y_prob.status}
            else cp.OPTIMAL
        )
        self._value = self.x_prob.value
        return self._value

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def value(self) -> float | None:
        """
        Returns the value of the objective function at the solution, and None if the problem has not
        been successfully solved.
        """
        return self._value

    def is_dsp(self) -> bool:
        # try to form x_prob and catch the exception
        try:
            self.x_prob  # noqa
            return True
        except DSPError:
            return False

    def convex_variables(self) -> list[cp.Variable]:
        parser = initialize_parser(
            self.obj_expr, self._minimization_vars, self._maximization_vars, self._constraints_
        )
        return sorted(parser.convex_vars, key=lambda x: x.id)

    def concave_variables(self) -> list[cp.Variable]:
        parser = initialize_parser(
            self.obj_expr, self._minimization_vars, self._maximization_vars, self._constraints_
        )
        return sorted(parser.concave_vars, key=lambda x: x.id)

    def affine_variables(self) -> list[cp.Variable]:
        parser = initialize_parser(
            self.obj_expr, self._minimization_vars, self._maximization_vars, self._constraints_
        )
        return sorted(parser.affine_vars, key=lambda x: x.id)


def semi_infinite_epigraph(
    expr: cp.Expression, variables: list[cp.Variable], constraints: list[Constraint], mode: str
) -> tuple[cp.Expression, list[Constraint]]:
    assert mode in ["inf", "sup"], "Not a valid semi-infinite mode."

    minimization_vars = variables if mode == "inf" else []
    maximization_vars = variables if mode == "sup" else []

    aux_prob = SaddlePointProblem(
        MinimizeMaximize(expr), constraints, minimization_vars, maximization_vars
    )
    # aux_prob.solve()  # TODO: should we solve here to verify saddle point property?
    # assert aux_prob.status == cp.OPTIMAL
    prob = aux_prob.x_prob if mode == "sup" else aux_prob.y_prob
    obj = prob.objective.expr
    aux_constraints = prob.constraints  # aux_constraints may not be canonicalized

    variables = list(set(chain.from_iterable([c.variables() for c in aux_constraints])))
    var_id_map = {v.id: v for v in variables}
    var_to_mat_mapping, const_vec, cone_dims = get_cone_repr(aux_constraints, variables)

    # A @ [all variables]
    expr = 0
    aux_size = var_to_mat_mapping["eta"].shape[1]
    if aux_size > 0:  # used extra variables
        eta = cp.Variable(aux_size)
        expr += var_to_mat_mapping["eta"] @ eta
    var_to_mat_mapping.pop("eta")

    for v_id, A_ in var_to_mat_mapping.items():
        v = var_id_map[v_id]
        if v.ndim > 1 and v.is_symmetric():
            # create v_vec, the symmetric part of v
            assert v.shape[0] == v.shape[1]
            inds = np.triu_indices(v.shape[0], k=0)  # includes diagonal
            expr += A_ @ v[inds]
        else:
            expr += A_ @ cp.vec(v)

    z = const_vec - expr  # Ax + b in K

    cone_constraints, z = add_cone_constraints(z, cone_dims, dual=False)

    return obj, cone_constraints


def get_problem_SE_atoms(problem: cp.Problem) -> list[SaddleExtremum]:
    SE_atoms = []
    SE_atoms += get_SE_atoms(problem.objective)
    for constraint in problem.constraints:
        SE_atoms += get_SE_atoms(constraint)
    return SE_atoms


def get_SE_atoms(expr: Canonical) -> list[SaddleExtremum]:
    if isinstance(expr, SaddleExtremum):
        return [expr]
    elif not expr.args:
        return []
    else:
        return list(chain.from_iterable([get_SE_atoms(arg) for arg in expr.args]))


def validate_saddle_extremum(
    SE_atom: SaddleExtremum, problem_constraints: list[Constraint]
) -> None:
    aux_prob = SaddlePointProblem(
        MinimizeMaximize(SE_atom.f),
        constraints=problem_constraints + SE_atom.constraints,
        minimization_vars=SE_atom.convex_variables(),
        maximization_vars=SE_atom.concave_variables(),
    )
    aux_prob.solve()
    assert aux_prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def validate_all_saddle_extrema(problem: cp.Problem) -> None:
    SE_atoms = get_problem_SE_atoms(problem)
    for SE_atom in SE_atoms:
        validate_saddle_extremum(SE_atom, problem.constraints)


def is_dsp_expr(obj: cp.Expression) -> bool:
    if obj.is_dcp():
        return True
    try:
        parser = Parser(set(), set())
        parser.parse_expr_variables(obj, switched=False)
        return True
    except DSPError:
        return False


def is_dsp(obj: cp.Problem | SaddlePointProblem | cp.Expression) -> bool:
    if isinstance(obj, SaddlePointProblem):
        return obj.is_dsp()
    elif isinstance(obj, cp.Problem):
        all_SE_atoms = get_problem_SE_atoms(obj)
        return obj.is_dcp() and all(atom.is_dsp() for atom in all_SE_atoms)
    elif isinstance(obj, cp.Expression):
        return is_dsp_expr(obj)
    else:
        return False
