from __future__ import annotations

from typing import ClassVar

from pymoors.expression import Expression, MathOperator
from pymoors.modeling.constant import Constant


class Pow(MathOperator):
    is_pow: ClassVar[bool] = True
    identity: ClassVar[Constant] = Constant(value=1)
    anihilator: ClassVar[Constant] = Constant(value=0)

    _base: Expression
    _exp: float

    __slots__ = ("_base", "_exp")

    def __new__(cls, base: Expression, exp: Expression) -> None:
        obj = MathOperator.__new__(cls, base, exp)
        if isinstance(obj, Pow):
            obj._base = base
            obj._exp = exp
        return obj

    def __repr__(self) -> str:
        return f"{self.base}**{self._exp}"

    @property
    def base(self) -> Expression:
        return self._base

    @property
    def exp(self) -> Expression:
        return self._exp
