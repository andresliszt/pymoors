from typing import Dict, Optional, Union, List, Tuple
import numpy as np
from pydantic import BaseModel, PrivateAttr

from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.index import Index
from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core.modeling.multiply import MultiplyExpression
from pymoors.core.constraints.inequality import Inequality
from pymoors.typing import OneDArray, TwoDArray



class LinearDecoder(BaseModel):
    """Gets coefficients from linear expressions"""

    variables: Union[Variable, List[Variable]]
    _ordered_variable_mapping: Optional[Dict[int, int]] = PrivateAttr(default=None)

    @property
    def ordered_variable_mapping(self) -> Optional[Dict[int, int]]:
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

    def _coefficient_matrix(self, indices: List[int]) -> TwoDArray:
        H = np.zeros((len(indices), self.number_variables))
        for row, idx in enumerate(indices):
            H[row, idx] = 1
        return H

    def _coeff_index(self, index: Index) -> TwoDArray:
        return self._coefficient_matrix(indices=index.index)

    def _coeff_variable(self, variable: Variable) -> TwoDArray:
        # TODO: Add support to variables with 1 < length < n_variables
        if variable.length == self.number_variables:
            indices = list(range(self.number_variables))
        else:
            indices = [self.ordered_variable_mapping[variable.expression_id]]
        return self._coefficient_matrix(indices=indices)

    def _intercept(self, add_expr: AddExpression) -> OneDArray:
        # We consider -1 because all constant are moved to left
        value = -1 * (add_expr.constant.value)
        return value if isinstance(value, np.ndarray) else np.array(add_expr.size * [value])

    def _variable_coefficients(self, expr: Expression) -> TwoDArray:
        """Given an expression this method returns a matrix"""
        if isinstance(expr, Index):
            return self._coeff_index(expr)
        if isinstance(expr, Variable):
            return self._coeff_variable(expr)
        if isinstance(expr, MultiplyExpression):
            # TODO: Currently multiplication is supported for a constant and a single expression
            assert (
                len(expr.non_constant_expressions) == 1
            ), "Decoder is acting on a non supported multiply expression having more than one expression"
            return expr.constant.value * self._variable_coefficients(
                expr.non_constant_expressions[0]
            )
        if isinstance(expr, AddExpression):
            return sum(self._variable_coefficients(e) for e in expr.non_constant_expressions)
        raise TypeError(f"Decoder acting on a non-supported type {type(expr)}")

    def decode(self, inequalities: List[Inequality]) -> Tuple[TwoDArray, OneDArray]:
        variable_coeffs = [
            self._variable_coefficients(expr=ineq.expression) for ineq in inequalities
        ]
        constant_coeffs = [self._intercept(add_expr=ineq.expression) for ineq in inequalities]
        return np.vstack(variable_coeffs), np.hstack(constant_coeffs)
