from pydantic import PrivateAttr, model_validator

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant


class MultiplyExpression(Expression):
    """For now it only supports multiply by scalar"""

    expression: Expression
    scalar: Constant
    _size: int = PrivateAttr(default=None)

    def __init__(self, scalar: float, expression: Expression, **kwargs):
        super().__init__(expression=expression, scalar=Constant(value=scalar), **kwargs)
        # We set the size of this expression.
        self._size = expression.size

    @model_validator(mode="after")
    def simplfy_mul_expression(self):
        if isinstance(self.expression, MultiplyExpression):
            self.scalar = Constant(value=self.scalar * self.expression.scalar)
            self.expression = self.expression.expression
        return self

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self._size

    @property
    def name(self) -> str:
        return f"{self.scalar.value}*({self.expression.name})"
