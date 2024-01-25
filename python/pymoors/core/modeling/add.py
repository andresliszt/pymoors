from __future__ import annotations
from typing import List, Dict

import numpy as np
from pydantic import PrivateAttr

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.constant import Constant
from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.index import Index
from pymoors.typing import OneDArray


class AddExpression(Expression):
    expressions: List[Expression]
    _size: int = PrivateAttr(default=None)

    def __init__(self, var1: Expression, var2: Expression, **kwargs) -> None:
        if var1.size != var2.size and not (var1.is_constant or var2.is_constant):
            raise ValueError(
                f"Two expressions with different sizes cannot be added. Got sizes {var1.size} and {var2.size}"
            )
        if isinstance(var1, AddExpression) and isinstance(var2, AddExpression):
            expressions = [*var1.expressions, *var2.expressions]
        elif isinstance(var1, AddExpression):
            expressions = [*var1.expressions, var2]
        elif isinstance(var2, AddExpression):
            expressions = [var1, *var2.expressions]
        else:
            expressions = [var1, var2]

        super().__init__(expressions=expressions, **kwargs)
        # We set the size of this expression.
        self._size = var1.size

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self._size

    @property
    def name(self) -> str:
        return " + ".join([expr.name for expr in self.expressions])

    @property
    def constant(self) -> Constant:
        """Returns constant from expressions"""
        return Constant(
            value=sum(expr.value for expr in self.expressions if expr.is_constant)
        )

    @property
    def pure_expressions(self) -> List[Expression]:
        return [expr for expr in self.expressions if not expr.is_constant]
