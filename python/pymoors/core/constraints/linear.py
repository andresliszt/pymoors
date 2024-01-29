from typing import Callable, List, Union

import numpy as np
from pydantic import BaseModel, PrivateAttr

from pymoors.typing import OneDArray, TwoDArray
from pymoors.core.constraints.inequality import Inequality
from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.decoder import LinearDecoder
from pymoors.typing import PopulationCallable


class AffineConstraints(BaseModel):
    variables: Union[Variable, List[Variable]]
    affine_constraints: List[Inequality]
    _decoder = PrivateAttr(default=None)

    @property
    def decoder(self) -> LinearDecoder:
        if self._decoder is None:
            self._decoder = LinearDecoder(variables=self.variables)
        return self._decoder

    def _ineq_coefficients(self, inequality: Inequality) -> OneDArray:
        # expression is lhs - rhs
        expr: AddExpression = inequality.expression
        coeff = self.decoder.decode(expr)
        return coeff

    @staticmethod
    def _ineq_constant(inequality: Inequality) -> float:
        # expression is lhs - rhs
        add_expression: AddExpression = inequality.expression
        return add_expression.constant.value

    @property
    def constraint_matrix(self) -> TwoDArray:
        return np.vstack([self._ineq_coefficients(ine) for ine in self.affine_constraints])

    @property
    def constant_vector(self) -> OneDArray:
        return np.array([self._ineq_constant(ine) for ine in self.affine_constraints])

    @property
    def function(self) -> PopulationCallable:
        G = self.constraint_matrix
        B = self.constant_vector[:, np.newaxis]

        def _function(population: TwoDArray) -> TwoDArray:
            return np.dot(G, population) - B

        return _function
