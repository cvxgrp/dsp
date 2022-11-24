from __future__ import annotations

from itertools import chain
from typing import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective

from dspp.cone_transforms import (
    LocalToGlob,
    add_cone_constraints,
    get_cone_repr,
    minimax_to_min,
)
from dspp.parser import initialize_parser


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
        constraints: list[Constraint] | None = None,
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

        parser = initialize_parser(obj_expr, minimization_vars, maximization_vars, constraints)

        local_to_glob_y = LocalToGlob(parser.convex_vars, parser.concave_vars)

        K_repr = parser.parse_expr_repr(obj_expr, switched=False, local_to_glob=local_to_glob_y)

        single_obj, constraints = minimax_to_min(
            K_repr, parser.x_constraints, parser.y_constraints, parser.concave_vars, local_to_glob_y
        )

        return constraints, single_obj

    @staticmethod
    def _validate_arguments(minmax_objective: MinimizeMaximize) -> None:
        assert isinstance(minmax_objective, (MinimizeMaximize, Objective))

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


def semi_infinite_epigraph(
    expr: cp.Expression, variables: list[cp.Variable], constraints: list[Constraint], mode: str
) -> tuple[cp.Expression, list[Constraint]]:
    assert mode in ["inf", "sup"], "Not a valid semi-infinite mode."

    minimization_vars = variables if mode == "inf" else []
    maximization_vars = variables if mode == "sup" else []

    aux_prob = SaddleProblem(
        MinimizeMaximize(expr), constraints, minimization_vars, maximization_vars
    )
    prob = aux_prob.x_prob if mode == "sup" else aux_prob.y_prob
    obj = prob.objective.expr
    aux_constraints = prob.constraints  # aux_constraints may not be canonicalized

    vars = list(set(chain.from_iterable([c.variables() for c in aux_constraints])))
    var_id_map = {v.id: v for v in vars}
    var_to_mat_mapping, const_vec, cone_dims = get_cone_repr(aux_constraints, vars)

    # A @ [all variablea]
    expr = 0
    aux_size = var_to_mat_mapping["eta"].shape[1]
    if aux_size > 0:  # used extra variables
        eta = cp.Variable(aux_size)
        expr += var_to_mat_mapping["eta"] @ eta
    var_to_mat_mapping.pop("eta")

    for v_id, A_ in var_to_mat_mapping.items():
        expr += A_ @ cp.vec(var_id_map[v_id])

    z = const_vec - expr  # Ax + b in K

    cone_constraints = add_cone_constraints(z, cone_dims, dual=False)

    return obj, cone_constraints
