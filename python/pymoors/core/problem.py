import abc
from typing import final, Optional

import numpy as np
from pydantic import BaseModel, Field, ConfigDict, model_validator

from pymoors._typing import OneDArray, TwoDArray


class Evaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def evaluate(self):
        ...


class Problem(BaseModel, abc.ABC):
    number_objectives: int = Field(ge=1)
    number_variables: int = Field(ge=1)
    lower_bounds: Optional[OneDArray] = None
    upper_bounds: Optional[OneDArray] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "Problem":
        if (
            self.lower_bounds.ndim is not None
            and (self.lower_bounds.ndim != 0 or len(self.lower_bounds) != self.number_variables)
        ) or (
            self.upper_bounds.ndim is not None
            and (self.upper_bounds_bounds.ndim != 0 or len(self.upper_bounds_bounds) != self.number_variables)
        ):
            raise ValueError(
                "upper_bounds and lower_bounds if provided must be 1D `numpy.ndarray` and their "
                f"length must be equal to the number of variables. Got upper_bounds `{self.upper_bounds}`, "
                f"lower_bounds `{self.lower_bounds}` and number_variables `{self.number_variables}`"
            )
        return self

    @abc.abstractmethod
    def fitness(self, population: TwoDArray) -> TwoDArray:
        pass

    @abc.abstractmethod
    def constraints(self, population: TwoDArray) -> TwoDArray:
        pass
