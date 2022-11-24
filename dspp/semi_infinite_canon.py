from typing import Any

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint

from dspp.atoms import concave_inf, convex_sup
from dspp.problem import semi_infinite_epigraph

# TODO: handle arbitrary DSPP expressions


def concave_max_canon(expr: convex_sup, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    mode = "sup"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.concave_vars, expr.constraints, mode)
    return obj, aux_constraints


def convex_min_canon(expr: concave_inf, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    mode = "inf"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.convex_vars, expr.constraints, mode)
    return -obj, aux_constraints  # -obj because we want a hypograph variable
