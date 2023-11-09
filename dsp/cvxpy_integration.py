from cvxpy import Expression, Problem

try:
    from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
except ModuleNotFoundError:
    from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS

from dsp.problem import is_dsp
from dsp.saddle_extremum import conjugate, saddle_max, saddle_min
from dsp.semi_infinite_canon import saddle_max_canon, saddle_min_canon


def extend_cone_canon_methods() -> None:
    """Extend the cone_canon_methods dictionary with methods from the
    semi_infinite_canon module.
    """

    CANON_METHODS.update(
        {
            saddle_max: saddle_max_canon,
            saddle_min: saddle_min_canon,
            conjugate: saddle_max_canon,
        }
    )


def add_is_dsp() -> None:
    """Add the is_dsp method to the cvxpy Problem class."""
    Problem.is_dsp = lambda self: is_dsp(self)
    Expression.is_dsp = lambda self: is_dsp(self)
