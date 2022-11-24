from dspp.atoms import concave_max, convex_min
from dspp.problem import semi_infinite_epigraph

# TODO: handle arbitrary DSPP expressions


def concave_max_canon(expr: concave_max, args):
    mode = "sup"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.concave_vars, expr.constraints, mode)
    return obj, aux_constraints


def convex_min_canon(expr: convex_min, args):
    mode = "inf"
    obj, aux_constraints = semi_infinite_epigraph(expr.f, expr.convex_vars, expr.constraints, mode)
    return -obj, aux_constraints  # -obj because we want a hypograph variable
