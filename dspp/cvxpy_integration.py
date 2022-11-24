from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS

from dspp.atoms import concave_inf, convex_sup
from dspp.semi_infinite_canon import concave_max_canon, convex_min_canon


def extend_cone_canon_methods() -> None:
    """Extend the cone_canon_methods dictionary with methods from the
    semi_infinite_canon module.
    """

    CANON_METHODS.update(
        {
            convex_sup: concave_max_canon,
            concave_inf: convex_min_canon,
        }
    )
