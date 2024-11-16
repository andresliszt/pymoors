from typing import Optional, List, Union, Callable

import numpy as np
from pydantic import BaseModel, model_validator, PrivateAttr

from pymoors.typing import TwoDArray
from pymoors.core.constraints.linear import AffineConstraints
from pymoors.core.modeling.variable import Variable
from pymoors.core.constraints.inequality import Inequality, Equality
from pymoors.typing import PopulationCallable


class Constraints(BaseModel):
    variables: Union[Variable, List[Variable]]
    constraints: List[Union[Inequality, Equality, PopulationCallable]]
    _affine_constraints: Optional[AffineConstraints] = PrivateAttr(default=None)
    _function: PopulationCallable = PrivateAttr(default=None)

    @model_validator(mode="after")
    def set_affine_constraints(self) -> "Constraints":
        expr = [constr for constr in self.constraints if not isinstance(constr, Callable)]
        if expr:
            self._affine_constraints = AffineConstraints(
                variables=self.variables, affine_constraints=expr
            )
        return self

    @property
    def affine_constraints(self) -> AffineConstraints:
        return self._affine_constraints

    @property
    def function(self) -> PopulationCallable:
        if self._function is None:
            callable_constraints = [f for f in self.constraints if isinstance(f, Callable)]

            # Define function from callable constraints first
            def _function(population: TwoDArray) -> TwoDArray:
                return np.column_stack([f(population) for f in callable_constraints])

            if self._affine_constraints is None:
                # If expression constraints are given, extend _function definition
                def _function(population: TwoDArray) -> TwoDArray:
                    return np.concatenate(
                        [self.affine_constraints.function(population), _function(population)],
                        axis=1,
                    )

            self._function = _function

        return self._function
