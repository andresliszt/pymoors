from __future__ import annotations

from pydantic import PrivateAttr

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant
from pymoors.core.modeling.index import Index


class MultiplyExpression(Expression):
    """For now it only supports multiply by scalar"""

    expression: Expression
    scalar: Constant
    _size: int = PrivateAttr(default=None)

    def __init__(self, scalar: float, expression: Expression, **kwargs):
        super().__init__(expression=expression, scalar=Constant(value=scalar), **kwargs)
        # We set the size of this expression.
        self._size = expression.size

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self._size

    @property
    def name(self) -> str:
        return f"{self.scalar.value}*{self.expression.name}"
