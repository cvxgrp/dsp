from __future__ import annotations

from typing import Any, Iterable

import cvxpy as cp
import numpy as np

import dspp


class Dummy(cp.Variable):
    def __init__(
        self,
        shape: int | Iterable[int, ...] = (),
        name: str | None = None,
        var_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._saddle_expr = None
        super().__init__(shape, name, var_id, **kwargs)

    @property
    def expr(self) -> dspp.saddle_min | dspp.saddle_max:
        assert self._saddle_expr is not None, "Must be assosciated with an SE."
        assert isinstance(self._saddle_expr, (dspp.saddle_min, dspp.saddle_max))
        return self._saddle_expr

    @expr.setter
    def expr(self, new_value: dspp.saddle_min | dspp.saddle_max) -> None:
        assert isinstance(new_value, (dspp.saddle_min, dspp.saddle_max))
        assert self._saddle_expr is None, "Cannot assign a Dummy to multiple SEs."
        self._saddle_expr = new_value

    @property
    def value(self) -> np.ndarray:
        expr = self.expr
        if self._value is None:
            expr.numeric(
                None
            )  # TODO: allow numeric to take an arbirary value rather than requiring a solve.
        return self._value  # can be none if numeric had no variable value

    @value.setter
    def value(self, new_value: np.ndarray) -> None:
        self._value = new_value

    def __repr__(self) -> str:
        return f"Dummy{super().__repr__()}"
