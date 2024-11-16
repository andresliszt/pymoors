from typing import List, Tuple

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant


class MultiplyExpression(Expression):
    """Currently it only supports multiply by coefficient"""

    coefficient: Constant
    expression: Expression

    def __init__(self, var1, var2, **kwargs):
        const = var1 if isinstance(var1, Constant) else var2
        expr = var1 if var1 is not const else var2
        super().__init__(coefficient=const, expression=expr, **kwargs)

    def _shape_from_expressions(self) -> Tuple[int]:
        if len(self.coefficient.shape) == 2:
            raise ValueError(
                "Trying to perform matrix multiplication. Use the `@` operator instead"
            )

        if self.coefficient.is_scalar:
            return self.expression.shape
        if self.expression.is_scalar:
            raise ValueError("Can not multiply a scalar by a vector coefficient on the left.")
        if self.coefficient.shape[0] == self.expression.shape[0]:
            # Inner product
            return ()
        raise ValueError(
            "Can't perform inner product got different shapes: \n"
            f"Constant shape {self.coefficient.shape} and expression shape {self.expression.shape}"
        )

    @property
    def name(self) -> str:
        return f"{self.coefficient.value}*({self.expression.name})"

    @property
    def factors(self) -> List[Expression]:
        return [self.coefficient, self.expression]
