from __future__ import annotations

from itertools import chain
from typing import Iterable

import cvxpy as cp
import numpy as np
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from cvxpy.problems.objective import Objective

import dspp

class Dummy(cp.Variable):
    def __init__(self, shape: int | Iterable[int, ...] = (), name: str | None = None, var_id: int | None = None, **kwargs: Any):
        self._saddle_expr = None
        super().__init__(shape, name, var_id, **kwargs)
    
    @property
    def expr(self) -> dspp.saddle_min | dspp.saddle_max:
        assert self._saddle_expr is not None, "Must be assosciated with an SE."
        assert isinstance(self._saddle_expr, (dspp.saddle_min, dspp.saddle_max))
        return self._saddle_expr

    @expr.setter
    def expr(self,  new_value : dspp.saddle_min | dspp.saddle_max) -> None:
        assert isinstance(new_value, (dspp.saddle_min, dspp.saddle_max))
        assert self._saddle_expr is None, "Cannot assign a Dummy to multiple SEs."
        self._saddle_expr = new_value

    @property
    def value(self):
        expr = self.expr
        if self._value is None:
            expr.numeric(None) # TODO: allow numeric to take an arbirary value rather than requiring a solve.
        if self._value is None:
            raise ValueError("Dummy variable has no value after numeric() call.")
        else:
            return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def __repr__(self) -> str:
        return f"Dummy{super().__repr__()}"