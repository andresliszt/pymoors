from typing import Dict, NewType, Optional, Union, List
import numpy as np
from pydantic import BaseModel, PrivateAttr

from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.index import Index
from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core.modeling.multiply import MultiplyExpression
from pymoors.core.constraints.inequality import Inequality
from pymoors.core.modeling.constant import Constant
from pymoors.typing import OneDArray, TwoDArray


ExpressionID = NewType("UserID", int)


class LinearDecoder(BaseModel):
    """Gets coefficients from linear expressions"""

    variables: Union[Variable, List[Variable]]
    _ordered_variable_mapping: Optional[Dict[ExpressionID, int]] = PrivateAttr(default=None)

    @property
    def ordered_variable_mapping(self) -> Optional[Dict[ExpressionID, int]]:
        if self._ordered_variable_mapping is None:
            if isinstance(self.variables, list):
                self._ordered_variable_mapping = {
                    var.expression_id: index for index, var in enumerate(self.variables)
                }
        return self._ordered_variable_mapping

    @property
    def number_variables(self) -> int:
        return (
            sum(var.size for var in self.variables)
            if isinstance(self.variables, list)
            else self.variables.size
        )

    def _coeff_index(self, index: Index) -> TwoDArray:
        variable = index.expression
        if not isinstance(variable, Variable):
            raise TypeError("Decoder on index from arbitrary expression is not supported yet")

        coeff = np.zeros(self.number_variables)
        coeff[index.index] = 1
        return coeff.reshape(1, -1)

    def _coeff_variable(self, variable: Variable) -> TwoDArray:
        if variable.length == self.number_variables:
            return np.eye(self.number_variables)
        if variable.length == 1:
            if self.ordered_variable_mapping is None:
                raise ValueError(
                    "When variables have length one `ordered_variable_mapping` must be passed at initialization"
                )
            coeff = np.zeros(self.number_variables)
            # pylint: disable=E1136
            coeff[self.ordered_variable_mapping[variable.expression_id]] = 1
            return coeff.reshape(1, -1)

        raise ValueError(
            "Decoder on variables with lenght different from one or the total number of variables is not supported"
        )

    # def _constant_from_add_expression(self, add_expr: AddExpression) -> OneDArray:
    #     """Gets coefficients vector from `AddExpression` object"""
    #     constant: List[Constant] = add_expr.constant
    #     pure_expressions: List[Expression] = add_expr.pure_expressions

    #     if all(isinstance(expr.expression, Index))



    def variable_coefficients(self, expr: Expression) -> TwoDArray:
        if isinstance(expr, Index):
            return self._coeff_index(expr)
        if isinstance(expr, Variable):
            return self._coeff_variable(expr)
        if isinstance(expr, MultiplyExpression):
            return expr.constant.value * self.variable_coefficients(expr.expression)
        if isinstance(expr, AddExpression):
            return sum(self.variable_coefficients(e) for e in expr.non_constant_expressions)
        raise TypeError(f"Decoder acting on a non-supported type {type(expr)}")
