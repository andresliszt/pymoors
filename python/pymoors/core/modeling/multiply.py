from typing import List

from pydantic import PrivateAttr, model_validator

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant


class MultiplyExpression(Expression):
    """For now it only supports multiply by scalar"""

    _expressions: List[Expression] = PrivateAttr(default=None)
    _size: int = PrivateAttr(default=None)

    def __init__(self, var1: Expression, var2: Expression, **kwargs):
        if not any(isinstance(var1, Constant), isinstance(var2, Constant)):
            raise TypeError(
                "Currently only multiplication beetwen constant and expression is supported"
            )
        # Get constant and expression
        constant = var1 if isinstance(var1, Constant) else var2
        expression = var1 if isinstance(var2, Constant) else var2
        # Only scalar multiplication or inner dot product is supported
        if constant.size > 1 and constant.size != expression.size:
            raise ValueError(
                "An expression can only be multiplied by a scalar constant or by an array constant. "
                "The last implies that the inner dot product will be performed, therefore the array size "
                f"must be equal to expression size. Got constant size {constant.size} and expression size {expression.size}"
            )
        # Now simplify constants
        if isinstance(expression, MultiplyExpression):
            constant = expression.constant * constant

        self._expressions = [constant, *expression.non_constant_expressions]

        super().__init__(**kwargs)
        # We set the size of this expression.
        self._size = expression.size

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self._size

    @property
    def name(self) -> str:
        return f"{self.scalar.value}*({self.expression.name})"

    @property
    def expressions(self) -> List[Expression]:
        return self._expressions

    @property
    def constant(self) -> Constant:
        """Constant is always placed in the first position of expressions"""
        return self.expressions[0]
