from __future__ import annotations
from typing import List
from pydantic import PrivateAttr

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant


class AddExpression(Expression):
    _expressions: List[Expression] = PrivateAttr(default=None)
    _size: int = PrivateAttr(default=None)

    def __init__(self, var1: Expression, var2: Expression, **kwargs) -> None:
        if len(var1) != len(var2) and not (
            isinstance(var2, Constant)
            and var2.size == 1
            or isinstance(var1, Constant)
            and var1.size == 1
        ):
            raise ValueError(
                f"Two expressions with different sizes cannot be added. Got sizes {len(var1)} and {len(var2)}"
            )
        if isinstance(var1, AddExpression) and isinstance(var2, AddExpression):
            expressions = [*var1.expressions, *var2.expressions]
        elif isinstance(var1, AddExpression):
            expressions = [*var1.expressions, var2]
        elif isinstance(var2, AddExpression):
            expressions = [var1 * var2.expressions]
        else:
            expressions = [var1, var2]

        super().__init__(**kwargs)
        # We set the size of this expression. We use max to handle constants with length = 1
        self._size = max(var1.size, var2.size)
        self._expressions = expressions

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self._size

    @property
    def name(self) -> str:
        return " + ".join([expr.name for expr in self.expressions])

    @property
    def expressions(self) -> List[Expression]:
        return self._expressions

    @property
    def constant(self) -> Constant:
        return sum(expr.constant for expr in self.expressions if isinstance(expr, Constant))
