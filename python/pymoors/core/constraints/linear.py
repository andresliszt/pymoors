from typing import Callable, List, Union

import numpy as np
from pydantic import BaseModel

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

    def _get_coefficient_from_inequality(self, inequality: Inequality) -> OneDArray:
        # expression is lhs - rhs
        expr: AddExpression = inequality.expression
        # Create decoder object
        decoder = LinearDecoder(variables=self.variables)
        # Get coefficient vector
        coeff = decoder.decode(expr)
        return coeff

    @staticmethod
    def _get_constant_from_inequality(inequality: Inequality) -> float:
        # expression is lhs - rhs
        add_expression: AddExpression = inequality.expression
        return add_expression.constant.value

    @property
    def constraint_matrix(self) -> TwoDArray:
        return np.array(
            [
                self._get_coefficient_from_inequality(ine)
                for ine in self.affine_constraints
            ]
        )

    @property
    def constant_vector(self) -> OneDArray:
        return np.array(
            [self._get_constant_from_inequality(ine) for ine in self.affine_constraints]
        )

    @property
    def function(self) -> PopulationCallable:
        G = self.constraint_matrix
        B = self.constant_vector[:, np.newaxis]

        def _function(population: TwoDArray) -> TwoDArray:
            return np.dot(G, population) - B

        return _function
