from typing import Dict, Any, Callable
from numbers import Number

import numpy as np
from pydantic import BaseModel, model_validator

from pymoors.typing import TwoDArray

# from pymoors.core.expression import Expression, Index


class Inequality(BaseModel):
    ...
    # lhs: Expression
    # rhs: Expression

    # @model_validator(mode="before")
    # @classmethod
    # def cast_number_to_constant(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    #     lhs = data.get("lhs")
    #     rhs = data.get("rhs")
    #     if isinstance(lhs, Number):
    #         data["lhs"] = Constant(value=lhs)
    #     if isinstance(rhs, Number):
    #         data["rhs"] = Constant(value=rhs)
    #     if not (isinstance(lhs, Constant) ^ isinstance(rhs, Constant)):
    #         raise TypeError("Only one between 'lhs' and 'rhs' must be a constant value")
    #     return data

    # def decode(self) -> Callable[[TwoDArray], TwoDArray]:
    #     if isinstance(self.lhs, Variable):
    #         return lambda population:
