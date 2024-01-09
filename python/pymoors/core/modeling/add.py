from __future__ import annotations
from typing import List, Union


from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.index import Index


class AddExpression(Expression):
    expressions: List[Expression]

    def __init__(self, var1, var2, **kwargs):
        if var1.size != var2.size:
            raise ValueError(
                f"Two expressions with different sizes cannot be added. Got sizes {var1.size} and {var2.size}"
            )
        if isinstance(var1, AddExpression):
            if isinstance(var2, AddExpression):
                expressions = var1.expressions + var2.expressions
            else:
                expressions = var1.expressions
                expressions.append(var2)
        else:
            expressions = [var1, var2]
        super().__init__(expressions=expressions, **kwargs)

    @property
    def size(self) -> int:
        """All expressions have the same size"""
        return self.expressions[0].size

    @property
    def name(self) -> str:
        return " + ".join([e.name for e in self.expressions])


class AddExpressionDecoder:
    def __init__(self, var1, var2, **kwargs):
        if isinstance(var1, Variable) and isinstance(var2, Variable):
            expressions = self._add_two_variables(var1, var2)
        elif isinstance(var1, Index) and isinstance(var2, Index):
            expressions = self._add_index_with_index(var1, var2)
        elif isinstance(var1, Variable) and isinstance(var2, Index):
            expressions = self._add_variable_with_index(var1, var2)
        elif isinstance(var1, AddExpression) and isinstance(var2, (Variable, Index)):
            expressions = self._add_variable_with_add_expression(var1, var2)
        else:
            raise NotImplementedError

        super().__init__(expressions=expressions, **kwargs)

    @staticmethod
    def _add_two_variables(var1: Variable, var2: Variable):
        if var1.expression_id == var2.expression_id:
            result = [
                Variable(
                    number_variables=var1.number_variables,
                    expression_id=var1.expression_id,
                    coefficients=var1.coefficients + var2.coefficients,
                )
            ]
        else:
            result = [var1, var2]
        return result

    @staticmethod
    def _add_variable_with_index(var1: Variable, var2: Index):
        if var1.expression_id == var2.expression_id:
            coefficients = var1.coefficients
            coefficients[var2.index] += var2.expression.coefficients[var2.index]
            result = [
                Variable(
                    number_variables=var1.number_variables,
                    expression_id=var1.expression_id,
                    coefficients=coefficients,
                )
            ]
        else:
            result = [var1, var2]
        return result

    @staticmethod
    def _add_index_with_index(var1: Index, var2: Index):
        """Index are stored in a result list and they will be decoded later"""
        return [var1, var2]

    @staticmethod
    def _add_variable_with_add_expression(
        var1: AddExpression, var2: Union[Variable, Index]
    ):
        for idx, v in enumerate(var1.variables):
            if v.expression_id == var2.expression_id:
                if isinstance(var2, Variable):
                    var1[idx] = AddExpression._add_variable_with_add_expression(v, var2)[
                        0
                    ]
                else:
                    var1[idx] = AddExpression._add_variable_with_index(v, var2)[0]
                break
        else:
            var1.variables.append(var2)
        return var1
