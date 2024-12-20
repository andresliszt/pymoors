from __future__ import annotations

from typing import ClassVar

from pymoors.expression import Expression, MathOperator
from pymoors.modeling.constant import Constant

# pylint: disable=C0415


class Pow(MathOperator):
    is_pow: ClassVar[bool] = True
    identity: ClassVar[Constant] = Constant(value=1)
    anihilator: ClassVar[Constant] = Constant(value=0)

    _base: Expression
    _exp: Expression

    __slots__ = ("_base", "_exp")

    def __new__(cls, base: Expression, exp: Expression) -> None:
        obj = MathOperator.__new__(cls, base, exp)
        if isinstance(obj, Pow):
            obj._base = base
            obj._exp = exp
        return obj

    def __repr__(self) -> str:
        if self.base.is_add:
            base = f"({self.base})"
        else:
            base = f"{self.base}"
        return f"{base}**{self.exp}"

    @property
    def base(self) -> Expression:
        return self._base

    @property
    def exp(self) -> Expression:
        return self._exp

    def expand(self) -> Expression:
        from pymoors.helpers import mul_expression

        Mul = mul_expression()
        if self.base.is_add:
            return Mul(*(self.base for _ in range(self.exp.value))).expand()
        return self

    @classmethod
    def _simplfy_expressions(
        cls, base: Expression, exp: Expression
    ) -> tuple[Expression, Expression]:
        if base.is_pow and exp.is_constant:
            exp = exp + base.exp
            base = base.base
        return base, exp
