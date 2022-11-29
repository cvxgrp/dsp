import cvxpy as cp
import numpy as np
from dspp.cone_transforms import LocalToGlob, affine_to_canon

def test_affine_to_canon():
    x = cp.Variable(2, name="x", nonneg=True)
    aff = cp.sum(x-1)

    ltg = LocalToGlob([],[x])

    B, c = affine_to_canon(aff, ltg, False)
    assert np.allclose(B, np.ones(2)[None,:])
    assert np.allclose(c, -2)