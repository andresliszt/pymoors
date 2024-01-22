from typing import Callable, List, Optional

import numpy as np
from pydantic import BaseModel

from pymoors.typing import OneDArray, TwoDArray
from pymoors.core.constraints.inequality import Inequality
from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core.modeling.constant import Constant
from pymoors.core.modeling.variable import Variable

ConstraintIndividualCallable = Callable[[OneDArray], OneDArray]
ConstraintPopulationCallable = Callable[[TwoDArray], TwoDArray]


def affine_inequality_factory(
    coefficients: OneDArray, rhs: float
) -> ConstraintIndividualCallable:
    def inequality_function(ind: OneDArray) -> OneDArray:
        return np.dot(coefficients, ind) - rhs

    return inequality_function


class AffineConstraint(BaseModel):
    affine_constraints: List[Inequality]

    def decode_inequality(self, inequality: Inequality):
        # expression is lhs - rhs
        add_expression: AddExpression = inequality.expression
        # We get the sum of all constant involved in AddExpression
        constant: Constant = add_expression.constant
        # Get non constant expressions
        pure_expressions: List[Expression] = add_expression.pure_expressions

        if len(pure_expressions) == 0:
            # This case is when the inequality includes all the variables at the same time, e.g x >= 1
            if isinstance(pure_expressions[0], Variable):
                return affine_inequality_factory(coefficients=pure_expressions[0].coefficients, rhs = constant.value)

    def constraints_function_from_affine(self) -> ConstraintPopulationCallable:
        ...
