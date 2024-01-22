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
        if var1.size != var2.size and not (var2.is_constant or var1.is_constant):
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


    def decode(self) -> OneDArray:
        """Concatenates all sums into a 1D coefficient vector

        For example in a three variable problem (x,y,z), if add expression
        contains [x, z], then this decoding must return np.array([1, 0, 1]).

        Order is performed using expression_id

        """
        # Sort by expression_id
        expressions = sorted(self.expression_id, key = lambda expr: expr.expression_id)
        













class AddVariableDecoder:
    """Handles the decoding of sum expressions"""

    def __init__(self, add_expression: AddExpression) -> None:
        self.add_expression = add_expression

    @staticmethod
    def _add_two_variables(var1: Variable, var2: Variable) -> Variable:
        return Variable(
            number_variables=var1.number_variables,
            expression_id=var1.expression_id,
            coefficients=var1.coefficients + var2.coefficients,
        )

    @staticmethod
    def _add_variable_with_index(var1: Variable, var2: Index) -> Variable:
        coefficients = var1.coefficients
        coefficients[var2.index] += var2.expression.coefficients[var2.index]
        return Variable(
            number_variables=var1.number_variables,
            expression_id=var1.expression_id,
            coefficients=coefficients,
        )

    @staticmethod
    def _add_two_indexes(var1: Index, var2: Index) -> Variable:
        index_variable = Variable(
            number_variables=var1.expression.number_variables,
            coefficients=np.zeros(var1.expression.size),
            expression_id=var1.expression_id,
        )
        coefficients = index_variable.coefficients
        coefficients[var1.index] += var1.expression.coefficients[var1.index]
        coefficients[var2.index] += var2.expression.coefficients[var2.index]
        return index_variable

    def _add(self, var1, var2) -> Variable:
        if isinstance(var1, Variable) and isinstance(var2, Variable):
            return self._add_two_variables(var1, var2)
        if isinstance(var1, Variable) and isinstance(var2, Index):
            return self._add_variable_with_index(var1, var2)
        return self._add_two_indexes(var1, var2)

    def _decode_single_expression(self, expressions: List[Expression]) -> Variable:
        """Handles decoding of expressions that have the same id"""
        if len(expressions) == 1:
            return expressions[0]
        variable: Variable = self._add(expressions[0], expressions[1])
        return self._decode_single_expression([variable, *expressions[2:]])

    def decode(self) -> Dict[int, Variable]:
        return {
            expression_id: self._decode_single_expression(expression_list)
            for expression_id, expression_list in self.add_expression.expressions.items()
        }
