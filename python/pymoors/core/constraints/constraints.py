from typing import Callable, List, Union, Optional

import numpy as np
from pydantic import BaseModel, model_validator, PrivateAttr

from pymoors.typing import TwoDArray
from pymoors.core.constraints.linear import AffineConstraints
from pymoors.typing import PopulationCallable


class Constraints(BaseModel):
    constraints_function: Optional[PopulationCallable]
    affine_constraints: Optional[AffineConstraints]

    _function: PopulationCallable = PrivateAttr(default=None)

    @model_validator(mode="after")
    def validate_constraints(self):
        if self.constraints_function is None and self.affine_constraints is None:
            raise TypeError(
                "At least one of `constraints_function` and `affine_constraints` must be given"
            )
        return self

    @property
    def function(self) -> PopulationCallable:
        if self._function is None:
            if self.constraints_function is not None and self.affine_constraints is not None:
                aff_function: PopulationCallable = self.affine_constraints.function
                custom_function: PopulationCallable = self.constraints_function

                def _function(population: TwoDArray) -> TwoDArray:
                    return np.concatenate(
                        (aff_function(population), custom_function(population)), axis=1
                    )

                self._function = _function
            elif self.constraints_function is not None:
                self._function = self.constraints_function
            else:
                self._function = self.affine_constraints.function
        return self._function

    def evaluate(self, population: TwoDArray) -> TwoDArray:
        return self.function(population)
