from __future__ import annotations

from typing import Any

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint

from dsp.parser import DSPError
from dsp.problem import semi_infinite_epigraph
from dsp.saddle_extremum import saddle_max, saddle_min

# TODO: handle arbitrary DSP expressions


def saddle_max_canon(expr: saddle_max, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    if not expr.is_dsp():
        raise DSPError("The objective function must be a DSP expression.")
    mode = "sup"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.concave_vars, expr.constraints, mode)
    return obj, aux_constraints


def saddle_min_canon(expr: saddle_min, args: Any) -> tuple[cp.Expression, list[Constraint]]:
    if not expr.is_dsp():
        raise DSPError("The objective function must be a DSP expression.")
    mode = "inf"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.convex_vars, expr.constraints, mode)
    return -obj, aux_constraints  # -obj because we want a hypograph variable
