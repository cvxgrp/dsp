from __future__ import annotations

from typing import Any

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint

from dsp.atoms import saddle_max, saddle_min
from dsp.problem import semi_infinite_epigraph

# TODO: handle arbitrary DSP expressions


def concave_max_canon(expr: saddle_max, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    mode = "sup"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.concave_vars, expr.constraints, mode)
    return obj, aux_constraints


def convex_min_canon(expr: saddle_min, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    mode = "inf"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.convex_vars, expr.constraints, mode)
    return -obj, aux_constraints  # -obj because we want a hypograph variable
