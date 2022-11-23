from dspp.atoms import concave_max, convex_min


def extend_cone_canon_methods():
    """Extend the cone_canon_methods dictionary with methods from the
    semi_infinite_canon module.
    """

    from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS

    from dspp.semi_infinite_canon import concave_max_canon, convex_min_canon

    CANON_METHODS.update({
        concave_max: concave_max_canon,
        convex_min: convex_min_canon,
    })