from __future__ import annotations

from collections.abc import Iterable

import cvxpy as cp
import numpy as np

import dsp


class LocalVariableError(Exception):
    pass


class LocalVariable(cp.Variable):
    def __init__(
        self,
        shape: int | Iterable[int, ...] = (),
        name: str | None = None,
        var_id: int | None = None,
        **kwargs: dict,
    ) -> None:
        self._saddle_expr = None
        super().__init__(shape, name, var_id, **kwargs)

    @property
    def expr(self) -> dsp.saddle_min | dsp.saddle_max:
        assert self._saddle_expr is not None, "Must be associated with an SE."
        assert isinstance(self._saddle_expr, dsp.saddle_min | dsp.saddle_max)
        return self._saddle_expr

    @expr.setter
    def expr(self, new_value: dsp.saddle_min | dsp.saddle_max) -> None:
        assert isinstance(new_value, dsp.saddle_min | dsp.saddle_max)
        if self._saddle_expr is not None:
            if self._saddle_expr is not new_value:
                raise LocalVariableError("Cannot assign a local variable to multiple SEs.")
            else:
                return
        self._saddle_expr = new_value

    @property
    def value(self) -> np.ndarray:
        expr = self.expr
        if self._value is None:
            expr.numeric(
                None
            )  # TODO: allow numeric to take an arbitrary value rather than requiring a solve.
        return self._value  # can be none if numeric had no variable value

    @value.setter
    def value(self, new_value: np.ndarray) -> None:
        self._value = new_value

    def __repr__(self) -> str:
        return f"Local{super().__repr__()}"
