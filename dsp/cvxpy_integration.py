from cvxpy import Expression, Problem
from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS

from dsp.atoms import saddle_max, saddle_min
from dsp.problem import is_dsp
from dsp.semi_infinite_canon import concave_max_canon, convex_min_canon


def extend_cone_canon_methods() -> None:
    """Extend the cone_canon_methods dictionary with methods from the
    semi_infinite_canon module.
    """

    CANON_METHODS.update(
        {
            saddle_max: concave_max_canon,
            saddle_min: convex_min_canon,
        }
    )


def add_is_dsp() -> None:
    """Add the is_dsp method to the cvxpy Problem class."""
    Problem.is_dsp = lambda self: is_dsp(self)
    Expression.is_dsp = lambda self: is_dsp(self)
