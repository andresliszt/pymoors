from typing import Dict, NewType, Optional, Union, List
import numpy as np
from pydantic import BaseModel, PrivateAttr

from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.index import Index
from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core.modeling.multiply import MultiplyExpression
from pymoors.typing import OneDArray


ExpressionID = NewType("UserID", int)


class LinearDecoder(BaseModel):
    """Gets coefficients from linear expressions"""

    variables: Union[Variable, List[Variable]]
    _ordered_variable_mapping: Optional[Dict[ExpressionID, int]] = PrivateAttr(
        default=None
    )

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

    def _decode_index(self, index: Index) -> OneDArray:
        variable = index.expression
        if not isinstance(variable, Variable):
            raise TypeError(
                "Decoder on index from arbitrary expression is not supported yet"
            )

        coeff = np.zeros(self.number_variables)
        coeff[index.index] = 1
        return coeff

    def _decode_variable(self, variable: Variable) -> OneDArray:
        if variable.length == self.number_variables:
            return np.ones(self.number_variables)
        if variable.length == 1:
            if self.ordered_variable_mapping is None:
                raise ValueError(
                    "When variables have length one `ordered_variable_mapping` must be passed at initialization"
                )
            coeff = np.zeros(self.number_variables)
            # pylint: disable=E1136
            coeff[self.ordered_variable_mapping[variable.expression_id]] = 1
            return coeff

        raise ValueError(
            "Decoder on variables with lenght different from one or the total number of variables is not supported"
        )

    def decode(self, expr: Expression) -> OneDArray:
        if isinstance(expr, Index):
            return self._decode_index(expr)
        if isinstance(expr, Variable):
            return self._decode_variable(expr)
        if isinstance(expr, MultiplyExpression):
            return expr.scalar.value * self.decode(expr.expression)
        if isinstance(expr, AddExpression):
            return sum(self.decode(e) for e in expr.pure_expressions)
        raise TypeError(f"Decoder acting on a non-supported type {type(expr)}")
